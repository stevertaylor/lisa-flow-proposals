"""V5 deployment MCMC: 6mo re-analysis with per-rung flows.

For each rung k in [0..NTEMPS-1], use flow_6mo_v5_rung{k:02d}.pkl as the
independence proposal at that temperature. The ladder is FIXED to the betas
the training run settled on (flow_6mo_v5_betas.npy), with adaptive=False,
so rung k in deployment targets exactly the distribution rung k's flow
was trained on.

Baselines for comparison at 6mo (24w x 10t x (1000 burn + 2000 sampling)):
  * StretchMove                                  -> chain_v5_06mo_default.h5
  * PerRungFlowSlabMove(95/5)                    -> chain_v5_06mo_perrung_95_05.h5
  * v4 single-flow FlowSlabMove(95/5) reference  -> read existing chain_v4_06mo_*.h5

Per-rung metrics (acceptance, IACT, ESS) are recorded for every temperature so
we can see how each rung's flow behaves -- not just the cold chain.

All results pickled to data/flow_benchmark/flow_v5_results.pkl.
"""

# %% Imports
import os
import time
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
if not hasattr(np, "in1d"):
    np.in1d = np.isin

from lisatools.utils.constants import YRSID_SI
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.datacontainer import DataResidualArray
from lisatools.sensitivity import AE1SensitivityMatrix
from gbgpu.gbgpu import GBGPU

import eryn.moves as eryn_moves
from eryn.ensemble import EnsembleSampler
from eryn.state import State
from eryn.prior import uniform_dist, ProbDistContainer
from eryn.backends import HDFBackend

from coppuccino import load_flow, sample, log_prob


# %% Config
SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
RESULTS_DIR = SCRIPT_DIR.parents[1] / "data" / "flow_benchmark"

NWALKERS, NTEMPS, NLEAVES, NDIM = 24, 10, 1, 6
NSTEPS, BURN = 2000, 1000
DT, N_WAVE = 5.0, 256
MONTH_SECONDS = YRSID_SI / 12.0
WINDOW = 50  # Sokal window for IACT estimator

TOBS_MONTHS = 6
FLOW_WEIGHT, SLAB_WEIGHT = 0.95, 0.05

BETAS_PATH = RESULTS_DIR / "flow_6mo_v5_betas.npy"


# %% Injection / priors  (identical to v4)
amp, f0, fdot, fddot = 2e-23, 3e-3, 7.538331e-18, 0.0
phi0, inc, psi = 0.1, 0.2, 0.3
lam_inj, beta_inj = 4.13, -0.15
injection_params = np.array([amp, f0, fdot, fddot, phi0, inc, psi, lam_inj, beta_inj])
default_values = np.array([lam_inj, beta_inj])

priors = {"gb": ProbDistContainer({
    0: uniform_dist(1e-23, 1e-21),
    1: uniform_dist(0.00299995, 0.00300005),
    2: uniform_dist(1e-18, 1e-17),
    3: uniform_dist(0.0, 2 * np.pi),
    4: uniform_dist(0.0, np.pi),
    5: uniform_dist(0.0, np.pi),
})}

gb = GBGPU()


# %% Likelihood + data
def like_wrap(x, data, fd, df_local, Tobs, dt, default_values, n_wave=N_WAVE):
    inp = np.zeros(9)
    inp[np.array([0, 1, 2, 4, 5, 6])] = x
    inp[np.array([7, 8])] = default_values
    gb.run_wave(*inp, T=Tobs, dt=dt, N=n_wave)
    A, E = gb.A[0], gb.E[0]
    start = int(gb.freqs[0][0] / df_local)
    tmpl = DataResidualArray(np.array([A, E]), f_arr=gb.freqs[0])
    data_tmp = DataResidualArray(data[:, start:start + n_wave],
                                 f_arr=fd[start:start + n_wave])
    return AnalysisContainer(
        data_tmp, AE1SensitivityMatrix(data_tmp.f_arr)
    ).template_likelihood(tmpl)


def make_data(Tobs_seconds):
    N = int(Tobs_seconds / DT)
    Tobs_eff = N * DT
    df_local = 1.0 / Tobs_eff
    f_arr = np.arange(0.0, 1.0 / (2.0 * DT) + df_local, df_local)
    data_fd = np.asarray(gb.inject_signal(*injection_params, T=Tobs_eff, dt=DT, N=N_WAVE))
    return data_fd, f_arr, Tobs_eff, df_local


# %% PerRungFlowSlabMove
class PerRungFlowSlabMove(eryn_moves.MHMove):
    """Independence flow proposal that dispatches a different flow per
    temperature index.

    Args:
        flows: dict {branch_name: list of flows length ntemps} OR list of flows
               length ntemps (treated as the single-branch case). Flow at index
               k is used for points at temperature index k.
        slab_dist: prior-like distribution (dict-of-name or single object) used
                   as the slab in the flow/slab mixture proposal.
        flow_weight, slab_weight: mixture weights (renormalized).

    The temperature index for each active point is taken from
    np.where(branches_inds == True)[0] in get_proposal, which is what eryn's
    MHMove guarantees as the leading axis of `branches_coords`.
    """

    def __init__(self, flows, slab_dist, flow_weight=0.95, slab_weight=0.05,
                 return_gpu=False, **kwargs):
        self.flows = flows
        self.slab_dist = slab_dist
        self.return_gpu = return_gpu
        w = np.asarray([flow_weight, slab_weight], dtype=float)
        if np.any(w < 0.0):
            raise ValueError("weights must be non-negative")
        if not np.any(w > 0.0):
            raise ValueError("at least one weight must be positive")
        w /= w.sum()
        self.flow_weight, self.slab_weight = w
        self.log_flow_weight = -np.inf if self.flow_weight == 0.0 else np.log(self.flow_weight)
        self.log_slab_weight = -np.inf if self.slab_weight == 0.0 else np.log(self.slab_weight)
        super().__init__(**kwargs)

    def _get_flow_list(self, name):
        return self.flows[name] if isinstance(self.flows, dict) else self.flows

    def _get_slab(self, name):
        return self.slab_dist[name] if isinstance(self.slab_dist, dict) else self.slab_dist

    def _log_mixture(self, name, pts, t_idx):
        out = np.empty(pts.shape[0])
        flow_list = self._get_flow_list(name)
        slab = self._get_slab(name)
        for t in np.unique(t_idx):
            m = t_idx == t
            sub = pts[m]
            if self.slab_weight == 0.0:
                out[m] = np.asarray(log_prob(flow_list[int(t)], sub))
                continue
            if self.flow_weight == 0.0:
                out[m] = np.asarray(slab.logpdf(sub))
                continue
            lf = np.asarray(log_prob(flow_list[int(t)], sub))
            ls = np.asarray(slab.logpdf(sub))
            out[m] = np.logaddexp(self.log_flow_weight + lf,
                                  self.log_slab_weight + ls)
        return out

    def _draw_mixture(self, name, shape, t_idx, random):
        n, d = shape
        out = np.empty((n, d))
        flow_list = self._get_flow_list(name)
        slab = self._get_slab(name)
        use_flow = random.rand(n) < self.flow_weight  # row-wise mixture pick
        for t in np.unique(t_idx):
            m = t_idx == t
            mf = m & use_flow
            ms = m & ~use_flow
            if mf.any():
                fd = np.asarray(sample(
                    flow_list[int(t)],
                    n_samples=int(mf.sum()),
                    rng_seed=int(random.randint(1e10)),
                ))
                out[mf] = np.atleast_2d(fd).reshape(-1, d)
            if ms.any():
                sd = np.asarray(slab.rvs(size=int(ms.sum())))
                out[ms] = np.atleast_2d(sd).reshape(-1, d)
        return out

    def get_proposal(self, branches_coords, random, branches_inds=None, **kwargs):
        q, factors = {}, None
        for i, (name, coords) in enumerate(branches_coords.items()):
            ntemps, nwalkers, nleaves_max, ndim = coords.shape
            inds = (np.ones((ntemps, nwalkers, nleaves_max), dtype=bool)
                    if branches_inds is None else branches_inds[name])
            if i == 0:
                factors = np.zeros((ntemps, nwalkers))
            q[name] = coords.copy()
            idx = np.where(inds == True)
            if len(idx[0]) == 0:
                continue
            t_idx = idx[0]                # temperature index per active point
            old_pts = coords[idx]
            log_q_old = self._log_mixture(name, old_pts, t_idx)
            new_pts = self._draw_mixture(name, old_pts.shape, t_idx, random)
            q[name][idx] = new_pts
            log_q_new = self._log_mixture(name, new_pts, t_idx)
            factors[idx[:2]] += log_q_old - log_q_new

        if self.periodic is not None:
            q = self.periodic.wrap(
                {n: t.reshape((ntemps * nwalkers,) + t.shape[-2:]) for n, t in q.items()},
                xp=self.xp,
            )
            q = {n: t.reshape((ntemps, nwalkers) + t.shape[-2:]) for n, t in q.items()}

        if self.use_gpu and not self.return_gpu:
            for n, a in list(q.items()):
                q[n] = a.get()
            factors = factors.get()
        return q, factors


# %% IACT + per-rung metric helpers
def iact_walker(chain_w, window=WINDOW):
    """Sokal-window IACT for one walker, max over dims, clipped to >=1."""
    nsteps, ndim = chain_w.shape
    W = min(window, nsteps // 5)
    iacts = []
    for d in range(ndim):
        x = chain_w[:, d] - chain_w[:, d].mean()
        if x.var() == 0.0:
            iacts.append(1.0); continue
        f = np.fft.fft(x, n=2 * nsteps)
        acf = np.fft.ifft(f * np.conjugate(f))[:nsteps].real
        acf /= acf[0]
        iacts.append(max(1.0, 1.0 + 2.0 * acf[1:W + 1].sum()))
    return max(iacts)


def per_rung_metrics(backend_path):
    """Return per-rung acceptance (ntemps,) and per-rung IACT (ntemps, nwalkers)."""
    bk = HDFBackend(str(backend_path))
    af = np.asarray(bk.accepted, dtype=float) / max(bk.iteration, 1)  # (ntemps, nwalkers)
    chain = bk.get_chain()["gb"][:, :, :, 0, :]  # (nsteps, ntemps, nwalkers, ndim)
    nsteps, ntemps, nwalkers, ndim = chain.shape
    iacts = np.zeros((ntemps, nwalkers))
    for t in range(ntemps):
        for w in range(nwalkers):
            iacts[t, w] = iact_walker(chain[:, t, w, :])
    return af, iacts


def build_sampler(move, Tobs, data_fd, f_arr, df_local, backend_path, betas):
    if os.path.exists(backend_path):
        os.remove(backend_path)
    return EnsembleSampler(
        NWALKERS, {"gb": NDIM}, like_wrap, priors,
        branch_names=["gb"],
        args=(data_fd, f_arr, df_local, Tobs, DT, default_values),
        tempering_kwargs=dict(
            betas=betas,
            adaptive=False,
        ),
        nleaves_max={"gb": NLEAVES},
        moves=move,
        backend=backend_path,
    )


def initial_state(betas):
    return State({"gb": priors["gb"].rvs(size=(len(betas), NWALKERS, NLEAVES))})


def run_one(label, move, Tobs_seconds, betas):
    backend = RESULTS_DIR / f"chain_{label}.h5"
    data_fd, f_arr, Tobs_eff, df_local = make_data(Tobs_seconds)
    sampler = build_sampler(move, Tobs_eff, data_fd, f_arr, df_local,
                            str(backend), betas)
    t0 = time.perf_counter()
    sampler.run_mcmc(initial_state(betas), NSTEPS, burn=BURN, progress=True)
    wallclock_total = time.perf_counter() - t0
    wallclock_sampling = wallclock_total * NSTEPS / (NSTEPS + BURN)

    af, iacts = per_rung_metrics(backend)            # (ntemps,nw), (ntemps,nw)
    ess_per_n = 1.0 / iacts                          # (ntemps, nwalkers)
    ess_total_cold = float((NSTEPS / iacts[0]).sum())
    ess_per_sec_cold = ess_total_cold / wallclock_sampling

    return {
        "label":                  label,
        "betas":                  betas.copy(),
        "accept_per_rung":        af,                # (ntemps, nwalkers)
        "iact_per_rung":          iacts,             # (ntemps, nwalkers)
        "ess_per_n_per_rung":     ess_per_n,         # (ntemps, nwalkers)
        "wallclock_total_sec":    wallclock_total,
        "wallclock_sampling_sec": wallclock_sampling,
        "ess_total_cold":         ess_total_cold,
        "ess_per_sec_cold":       ess_per_sec_cold,
        "backend":                str(backend),
    }


def fmt(r):
    af_cold = r["accept_per_rung"][0]
    iact_cold = r["iact_per_rung"][0]
    return (f"  cold acc={af_cold.mean():.3f}+-{af_cold.std():.3f}  "
            f"IACT={iact_cold.mean():.2f}+-{iact_cold.std():.2f}  "
            f"ESS/N={(1.0/iact_cold).mean():.3f}  "
            f"wall_samp={r['wallclock_sampling_sec']:.0f}s  "
            f"ESS/sec={r['ess_per_sec_cold']:.2f}")


def fmt_per_rung(r):
    lines = [f"  rung-by-rung (mean over {NWALKERS} walkers):"]
    lines.append(f"    {'k':>3} {'beta':>10} {'acc':>7} {'IACT':>8} {'ESS/N':>8}")
    for k in range(len(r["betas"])):
        lines.append(
            f"    {k:>3d} {r['betas'][k]:>10.4e} "
            f"{r['accept_per_rung'][k].mean():>7.3f} "
            f"{r['iact_per_rung'][k].mean():>8.2f} "
            f"{(1.0/r['iact_per_rung'][k]).mean():>8.3f}"
        )
    return "\n".join(lines)


# %% Load frozen ladder + per-rung flows
print(f"Loading frozen ladder from {BETAS_PATH}", flush=True)
betas = np.load(BETAS_PATH)
assert len(betas) == NTEMPS, \
    f"betas has length {len(betas)}, expected {NTEMPS}"
print(f"Betas: {np.array2string(betas, precision=4)}", flush=True)

print("\nLoading per-rung flows...", flush=True)
flows_per_rung = []
for k in range(NTEMPS):
    p = RESULTS_DIR / f"flow_6mo_v5_rung{k:02d}.pkl"
    flows_per_rung.append(load_flow(str(p)))
    print(f"  loaded {p.name}", flush=True)


# %% Run both: StretchMove baseline, then PerRungFlowSlabMove
results = {}
Tobs_s = TOBS_MONTHS * MONTH_SECONDS

label = f"v5_{TOBS_MONTHS:02d}mo_default"
print(f"\n=== {label} ===", flush=True)
r = run_one(label, eryn_moves.StretchMove(), Tobs_s, betas)
results[label] = r
print(fmt(r), flush=True)
print(fmt_per_rung(r), flush=True)

label = f"v5_{TOBS_MONTHS:02d}mo_perrung_95_05"
print(f"\n=== {label} ===", flush=True)
move = PerRungFlowSlabMove(
    flows={"gb": flows_per_rung},
    slab_dist=priors["gb"],
    flow_weight=FLOW_WEIGHT, slab_weight=SLAB_WEIGHT,
)
r = run_one(label, move, Tobs_s, betas)
results[label] = r
print(fmt(r), flush=True)
print(fmt_per_rung(r), flush=True)


# %% Persist
with open(RESULTS_DIR / "flow_v5_results.pkl", "wb") as fh:
    pickle.dump(results, fh)
print(f"\nSaved {RESULTS_DIR / 'flow_v5_results.pkl'}", flush=True)


# %% Summary
print("\n=== v5 summary (cold chain) ===", flush=True)
print(f"{'label':<40} {'acc':>6} {'IACT':>7} {'ESS/N':>8} {'ESS/sec':>9}",
      flush=True)
for k in sorted(results.keys()):
    r = results[k]
    af = r["accept_per_rung"][0]
    iact = r["iact_per_rung"][0]
    print(f"{k:<40} {af.mean():>6.3f} {iact.mean():>7.2f} "
          f"{(1.0/iact).mean():>8.3f} {r['ess_per_sec_cold']:>9.2f}",
          flush=True)

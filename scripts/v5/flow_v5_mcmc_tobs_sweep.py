"""V5 Tobs sweep: deploy 6mo-trained per-rung flows at Tobs = 7..12 months.

Design choices:
  * Ladder is FROZEN at the 6mo training betas (flow_6mo_v5_betas.npy) for
    every Tobs. Rung k always targets prior * L_Tobs^beta_k with the same
    beta_k flow_k was trained on -- only the likelihood changes between
    training (Tobs=6mo) and deployment (Tobs in {7..12}). This isolates the
    "flow trained on the wrong likelihood" effect from any ladder mismatch.
  * StretchMove baseline at each Tobs uses the same frozen ladder for an
    apples-to-apples comparison.

For each Tobs in TOBS_MONTHS:
  * StretchMove                       -> chain_v5_{T}mo_default.h5
  * PerRungFlowSlabMove(95/5)         -> chain_v5_{T}mo_perrung_95_05.h5

All results appended to data/flow_benchmark/flow_v5_results.pkl (existing keys
from the 6mo proof-of-principle are preserved).
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

from coppuccino import load_flow

# Re-use the PerRungFlowSlabMove + helpers from the 6mo deployment script
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "flow_v5_mcmc",
    Path(__file__).resolve().parent / "flow_v5_mcmc.py",
)
# We import only the class; loading the full module would re-execute the 6mo
# proof-of-principle run.  So copy/redeclare here instead of import.

from coppuccino import sample, log_prob  # noqa: E402


# %% Config
SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
RESULTS_DIR = SCRIPT_DIR.parents[1] / "data" / "flow_benchmark"

NWALKERS, NTEMPS, NLEAVES, NDIM = 24, 10, 1, 6
NSTEPS, BURN = 2000, 1000
DT, N_WAVE = 5.0, 256
MONTH_SECONDS = YRSID_SI / 12.0
WINDOW = 50

TOBS_MONTHS = [7, 8, 9, 10, 11, 12]
FLOW_WEIGHT, SLAB_WEIGHT = 0.95, 0.05

BETAS_PATH = RESULTS_DIR / "flow_6mo_v5_betas.npy"


# %% Injection / priors
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


# %% PerRungFlowSlabMove  (copy of class in flow_v5_mcmc.py)
class PerRungFlowSlabMove(eryn_moves.MHMove):
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
        use_flow = random.rand(n) < self.flow_weight
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
            t_idx = idx[0]
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


# %% Metric helpers
def iact_walker(chain_w, window=WINDOW):
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
    bk = HDFBackend(str(backend_path))
    af = np.asarray(bk.accepted, dtype=float) / max(bk.iteration, 1)
    chain = bk.get_chain()["gb"][:, :, :, 0, :]
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
        tempering_kwargs=dict(betas=betas, adaptive=False),
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

    af, iacts = per_rung_metrics(backend)
    ess_per_n = 1.0 / iacts
    ess_total_cold = float((NSTEPS / iacts[0]).sum())
    ess_per_sec_cold = ess_total_cold / wallclock_sampling

    return {
        "label":                  label,
        "betas":                  betas.copy(),
        "accept_per_rung":        af,
        "iact_per_rung":          iacts,
        "ess_per_n_per_rung":     ess_per_n,
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
            f"ESS/sec={r['ess_per_sec_cold']:.2f}  "
            f"min_rung_acc={r['accept_per_rung'].mean(axis=1).min():.3f}")


# %% Load ladder + flows
print(f"Loading frozen 6mo ladder from {BETAS_PATH}", flush=True)
betas = np.load(BETAS_PATH)
assert len(betas) == NTEMPS
print(f"Betas: {np.array2string(betas, precision=4)}", flush=True)

print("\nLoading per-rung flows...", flush=True)
flows_per_rung = [
    load_flow(str(RESULTS_DIR / f"flow_6mo_v5_rung{k:02d}.pkl"))
    for k in range(NTEMPS)
]
print(f"  {NTEMPS} flows loaded.", flush=True)


# %% Sweep
results_path = RESULTS_DIR / "flow_v5_results.pkl"
results = {}
if results_path.exists():
    with open(results_path, "rb") as fh:
        results = pickle.load(fh)
    print(f"\nLoaded existing results ({len(results)} entries) -- new keys "
          f"will be appended.", flush=True)

for months in TOBS_MONTHS:
    Tobs_s = months * MONTH_SECONDS

    label = f"v5_{months:02d}mo_default"
    print(f"\n=== {label} ===", flush=True)
    r = run_one(label, eryn_moves.StretchMove(), Tobs_s, betas)
    results[label] = r
    print(fmt(r), flush=True)

    label = f"v5_{months:02d}mo_perrung_95_05"
    print(f"\n=== {label} ===", flush=True)
    move = PerRungFlowSlabMove(
        flows={"gb": flows_per_rung},
        slab_dist=priors["gb"],
        flow_weight=FLOW_WEIGHT, slab_weight=SLAB_WEIGHT,
    )
    r = run_one(label, move, Tobs_s, betas)
    results[label] = r
    print(fmt(r), flush=True)


# %% Persist
with open(results_path, "wb") as fh:
    pickle.dump(results, fh)
print(f"\nSaved {results_path}", flush=True)


# %% Summary
print("\n=== v5 Tobs sweep summary (cold chain + min-rung acceptance) ===",
      flush=True)
print(f"{'label':<35} {'cold_acc':>9} {'cold_IACT':>10} "
      f"{'min_rung_acc':>13} {'ESS/sec':>9}", flush=True)
for k in sorted(results.keys()):
    r = results[k]
    af = r["accept_per_rung"]
    iact = r["iact_per_rung"]
    print(f"{k:<35} {af[0].mean():>9.3f} {iact[0].mean():>10.2f} "
          f"{af.mean(axis=1).min():>13.3f} {r['ess_per_sec_cold']:>9.2f}",
          flush=True)

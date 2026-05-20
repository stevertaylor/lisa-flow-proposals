"""V4: MCMC validation across Tobs = 6..12 months using 6mo-trained flows.

For each Tobs in {6, 7, 8, 9, 10, 11, 12}:
  * StretchMove baseline           -> chain_v4_{T}mo_default.h5
  * FlowSlabMove(95/5, full flow)  -> chain_v4_{T}mo_flow_full_95_05.h5

Training-size ablation at Tobs = 12 only:
  * FlowSlabMove(95/5, 50k flow)   -> chain_v4_12mo_flow_50k_95_05.h5
  * FlowSlabMove(95/5, 200k flow)  -> chain_v4_12mo_flow_200k_95_05.h5

Metrics recorded per run (cold chain):
  - acceptance fraction (per walker)
  - integrated autocorrelation length (per walker, Sokal window, max-over-dims)
  - ESS / N            = 1 / IACT
  - wallclock (total = burn + sampling, then a sampling-only estimate
              wallclock_total * NSTEPS / (NSTEPS + BURN))
  - ESS per CPU-second = total_ESS / wallclock_sampling

All results pickled to data/flow_benchmark/flow_v4_results.pkl.
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

TOBS_MONTHS = [6, 7, 8, 9, 10, 11, 12]


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


# %% FlowSlabMove (copied from flow_v2_mcmc_validation.py / 6mo_12mo_runs.py)
class FlowSlabMove(eryn_moves.MHMove):
    def __init__(self, flow, slab_dist, flow_weight=0.95, slab_weight=0.05,
                 return_gpu=False, **kwargs):
        self.flow = flow
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

    def _get_flow(self, name):
        return self.flow[name] if isinstance(self.flow, dict) else self.flow

    def _get_slab(self, name):
        return self.slab_dist[name] if isinstance(self.slab_dist, dict) else self.slab_dist

    def _log_mixture(self, name, pts):
        if self.slab_weight == 0.0:
            return np.asarray(log_prob(self._get_flow(name), pts))
        if self.flow_weight == 0.0:
            return np.asarray(self._get_slab(name).logpdf(pts))
        lf = np.asarray(log_prob(self._get_flow(name), pts))
        ls = np.asarray(self._get_slab(name).logpdf(pts))
        return np.logaddexp(self.log_flow_weight + lf, self.log_slab_weight + ls)

    def _draw_mixture(self, name, shape, random):
        n, d = shape
        use_flow = random.rand(n) < self.flow_weight
        out = np.empty((n, d))
        if np.any(use_flow):
            fd = np.asarray(sample(
                self._get_flow(name),
                n_samples=int(use_flow.sum()),
                rng_seed=int(random.randint(1e10)),
            ))
            out[use_flow] = np.atleast_2d(fd).reshape(-1, d)
        if np.any(~use_flow):
            sd = np.asarray(self._get_slab(name).rvs(size=int((~use_flow).sum())))
            out[~use_flow] = np.atleast_2d(sd).reshape(-1, d)
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
            old_pts = coords[idx]
            log_q_old = self._log_mixture(name, old_pts)
            new_pts = self._draw_mixture(name, old_pts.shape, random)
            q[name][idx] = new_pts
            log_q_new = self._log_mixture(name, new_pts)
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


# %% IACT + metric helpers
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


def cold_metrics(backend_path):
    bk = HDFBackend(str(backend_path))
    af = np.asarray(bk.accepted, dtype=float) / max(bk.iteration, 1)
    chain = bk.get_chain()["gb"][:, 0, :, 0, :]  # (nsteps, nwalkers, ndim)
    nsteps, nwalkers, ndim = chain.shape
    iacts = np.array([iact_walker(chain[:, w, :]) for w in range(nwalkers)])
    return af[0], iacts


def build_sampler(move, Tobs, data_fd, f_arr, df_local, backend_path):
    if os.path.exists(backend_path):
        os.remove(backend_path)
    return EnsembleSampler(
        NWALKERS, {"gb": NDIM}, like_wrap, priors,
        branch_names=["gb"],
        args=(data_fd, f_arr, df_local, Tobs, DT, default_values),
        tempering_kwargs=dict(ntemps=NTEMPS),
        nleaves_max={"gb": NLEAVES},
        moves=move,
        backend=backend_path,
    )


def initial_state():
    return State({"gb": priors["gb"].rvs(size=(NTEMPS, NWALKERS, NLEAVES))})


def run_one(label, move, Tobs_seconds):
    backend = RESULTS_DIR / f"chain_{label}.h5"
    data_fd, f_arr, Tobs_eff, df_local = make_data(Tobs_seconds)
    sampler = build_sampler(move, Tobs_eff, data_fd, f_arr, df_local, str(backend))
    t0 = time.perf_counter()
    sampler.run_mcmc(initial_state(), NSTEPS, burn=BURN, progress=True)
    wallclock_total = time.perf_counter() - t0
    # Sampling-phase estimate: per-step cost is ~constant, so scale by step share.
    wallclock_sampling = wallclock_total * NSTEPS / (NSTEPS + BURN)

    af, iacts = cold_metrics(backend)
    ess_per_n_per_walker = 1.0 / iacts                       # (nwalkers,)
    ess_total = float((NSTEPS / iacts).sum())                # sum over walkers
    ess_per_sec = ess_total / wallclock_sampling

    return {
        "label":                  label,
        "accept_cold":            af,
        "iact_cold":              iacts,
        "ess_per_n":              ess_per_n_per_walker,
        "wallclock_total_sec":    wallclock_total,
        "wallclock_sampling_sec": wallclock_sampling,
        "ess_total":              ess_total,
        "ess_per_sec":            ess_per_sec,
        "backend":                str(backend),
    }


def fmt(r):
    return (f"  acc={r['accept_cold'].mean():.3f}+-{r['accept_cold'].std():.3f}  "
            f"IACT={r['iact_cold'].mean():.2f}+-{r['iact_cold'].std():.2f}  "
            f"ESS/N={r['ess_per_n'].mean():.3f}  "
            f"wall_samp={r['wallclock_sampling_sec']:.0f}s  "
            f"ESS/sec={r['ess_per_sec']:.2f}")


# %% Load flows
print("Loading 6mo-trained flows...", flush=True)
flows = {
    "full": load_flow(str(RESULTS_DIR / "flow_6mo_v2_full.pkl")),
    "50k":  load_flow(str(RESULTS_DIR / "flow_6mo_v2_50k.pkl")),
    "200k": load_flow(str(RESULTS_DIR / "flow_6mo_v2_200k.pkl")),
}


# %% Main loop
results = {}

for months in TOBS_MONTHS:
    Tobs_s = months * MONTH_SECONDS

    # StretchMove baseline
    label = f"v4_{months:02d}mo_default"
    print(f"\n=== {label} ===", flush=True)
    r = run_one(label, eryn_moves.StretchMove(), Tobs_s)
    results[label] = r
    print(fmt(r), flush=True)

    # FlowSlabMove(95/5, full flow)
    label = f"v4_{months:02d}mo_flow_full_95_05"
    print(f"\n=== {label} ===", flush=True)
    move = FlowSlabMove(flow=flows["full"], slab_dist=priors["gb"],
                        flow_weight=0.95, slab_weight=0.05)
    r = run_one(label, move, Tobs_s)
    results[label] = r
    print(fmt(r), flush=True)

# Training-size ablation at 12mo
for size_label in ["50k", "200k"]:
    label = f"v4_12mo_flow_{size_label}_95_05"
    print(f"\n=== {label} (ablation) ===", flush=True)
    move = FlowSlabMove(flow=flows[size_label], slab_dist=priors["gb"],
                        flow_weight=0.95, slab_weight=0.05)
    r = run_one(label, move, 12 * MONTH_SECONDS)
    results[label] = r
    print(fmt(r), flush=True)


# %% Persist
with open(RESULTS_DIR / "flow_v4_results.pkl", "wb") as fh:
    pickle.dump(results, fh)
print(f"\nSaved {RESULTS_DIR / 'flow_v4_results.pkl'}", flush=True)


# %% Summary
print("\n=== v4 summary ===", flush=True)
print(f"{'label':<40} {'acc':>6} {'IACT':>7} {'ESS/N':>8} {'ESS/sec':>9}",
      flush=True)
for k in sorted(results.keys()):
    r = results[k]
    print(f"{k:<40} {r['accept_cold'].mean():>6.3f} "
          f"{r['iact_cold'].mean():>7.2f} {r['ess_per_n'].mean():>8.3f} "
          f"{r['ess_per_sec']:>9.2f}", flush=True)

"""V5 training run: 6-month single-GB MCMC with StretchMove and a ladder
that is frozen after burn-in (stop_adaptation=BURN).

Why this differs from the v4 training run:
  * v4 ran fully-adaptive PT for 1000 burn + 20000 sampling iterations. eryn's
    Vousden-style update has lag=10000, so betas drift slowly throughout
    sampling. The per-rung samples in chain_train_6mo_v4.h5 are therefore not
    stationary w.r.t. a single beta -- each rung's chain is a mixture over a
    moving beta trajectory.
  * v5 sets stop_adaptation=BURN so the controller adapts only during burn-in.
    Once sampling begins the ladder is fixed, so the post-burn samples at rung
    k are stationary w.r.t. a single beta_k. These are the samples we will fit
    per-rung flows on.

Output:
  data/flow_benchmark/chain_train_6mo_v5.h5  -- eryn backend (incl. betas/iter)
  data/flow_benchmark/flow_6mo_v5_betas.npy  -- (ntemps,) final-iteration betas
"""

# %% Imports
import time
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


# %% Config
SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
RESULTS_DIR = SCRIPT_DIR.parents[1] / "data" / "flow_benchmark"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NWALKERS, NTEMPS, NLEAVES, NDIM = 24, 10, 1, 6
NSTEPS, BURN = 20000, 1000
DT, N_WAVE = 5.0, 256
MONTH_SECONDS = YRSID_SI / 12.0
TOBS_MONTHS = 6

BACKEND_PATH = RESULTS_DIR / "chain_train_6mo_v5.h5"
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


# %% Run
print(f"=== v5 training run: Tobs={TOBS_MONTHS}mo, "
      f"{NWALKERS} walkers x {NTEMPS} temps x ({BURN} burn + {NSTEPS} sampling) ===",
      flush=True)
print(f"     stop_adaptation={BURN} (ladder freezes when sampling begins)",
      flush=True)

if BACKEND_PATH.exists():
    BACKEND_PATH.unlink()

data_fd, f_arr, Tobs_eff, df_local = make_data(TOBS_MONTHS * MONTH_SECONDS)

sampler = EnsembleSampler(
    NWALKERS, {"gb": NDIM}, like_wrap, priors,
    branch_names=["gb"],
    args=(data_fd, f_arr, df_local, Tobs_eff, DT, default_values),
    tempering_kwargs=dict(
        ntemps=NTEMPS,
        adaptive=True,
        stop_adaptation=BURN,
    ),
    nleaves_max={"gb": NLEAVES},
    moves=eryn_moves.StretchMove(),
    backend=str(BACKEND_PATH),
)

init = State({"gb": priors["gb"].rvs(size=(NTEMPS, NWALKERS, NLEAVES))})

t0 = time.perf_counter()
sampler.run_mcmc(init, NSTEPS, burn=BURN, progress=True)
wallclock = time.perf_counter() - t0

print(f"\nWallclock (burn + sampling): {wallclock/60:.1f} min ({wallclock:.0f} s)",
      flush=True)
print(f"Backend: {BACKEND_PATH}", flush=True)


# %% Verify ladder froze + persist final betas
bk = HDFBackend(str(BACKEND_PATH))
betas_history = bk.get_betas()                           # (nsteps_total, ntemps)
final_betas = betas_history[-1].copy()
last_drift = float(np.max(np.abs(betas_history[-1] - betas_history[-100])))
print(f"\nBetas (final iter): {np.array2string(final_betas, precision=4)}",
      flush=True)
print(f"Max |beta_final - beta_(-100)| across rungs: {last_drift:.2e}",
      flush=True)
if last_drift > 1e-10:
    print("WARNING: ladder still drifting at end of run -- check stop_adaptation",
          flush=True)
else:
    print("Ladder is frozen.", flush=True)

np.save(BETAS_PATH, final_betas)
print(f"Saved final betas to {BETAS_PATH}", flush=True)

# Quick per-rung sample-count sanity check
chain = bk.get_chain()["gb"][:, :, :, 0, :]   # (nsteps, ntemps, nwalkers, ndim)
print(f"Per-rung samples available (nsteps*nwalkers): "
      f"{chain.shape[0] * chain.shape[2]} per temperature", flush=True)

"""V4 training run: 6-month single-GB MCMC with StretchMove.

24 walkers x 10 temps x (1000 burn + 20000 sampling) at Tobs = 6 months.
Produces ~480k cold-chain samples used to train three flows in flow_v4_fit.py.

Output backend: data/flow_benchmark/chain_train_6mo_v4.h5
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


# %% Config
SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
RESULTS_DIR = SCRIPT_DIR.parents[1] / "data" / "flow_benchmark"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NWALKERS, NTEMPS, NLEAVES, NDIM = 24, 10, 1, 6
NSTEPS, BURN = 20000, 1000
DT, N_WAVE = 5.0, 256
MONTH_SECONDS = YRSID_SI / 12.0
TOBS_MONTHS = 6

BACKEND_PATH = RESULTS_DIR / "chain_train_6mo_v4.h5"


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
print(f"=== v4 training run: Tobs={TOBS_MONTHS}mo, "
      f"{NWALKERS} walkers x {NTEMPS} temps x ({BURN} burn + {NSTEPS} sampling) ===",
      flush=True)

if BACKEND_PATH.exists():
    BACKEND_PATH.unlink()

data_fd, f_arr, Tobs_eff, df_local = make_data(TOBS_MONTHS * MONTH_SECONDS)

sampler = EnsembleSampler(
    NWALKERS, {"gb": NDIM}, like_wrap, priors,
    branch_names=["gb"],
    args=(data_fd, f_arr, df_local, Tobs_eff, DT, default_values),
    tempering_kwargs=dict(ntemps=NTEMPS),
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

chain = sampler.get_chain()["gb"][:, 0, :, 0].reshape(-1, NDIM)
print(f"Cold-chain shape: {chain.shape}  "
      f"(expected ({NSTEPS*NWALKERS}, {NDIM}))", flush=True)

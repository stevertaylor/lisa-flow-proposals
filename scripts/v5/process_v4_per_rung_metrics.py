"""Re-process the v4 deployment backends to extract per-rung acceptance and
IACT (v4's flow_v4_results.pkl only stored cold-chain numbers).

For each chain_v4_{T}mo_{default,flow_full_95_05}.h5 (Tobs = 6..12):
  - per-rung acceptance  (ntemps, nwalkers)
  - per-rung IACT        (ntemps, nwalkers, Sokal window matching v4)
  - per-rung beta (median over sampling, since v4 deployments were adaptive)

Output: data/flow_benchmark/flow_v4_per_rung_metrics.pkl
"""

# %% Imports
import pickle
from pathlib import Path

import numpy as np

from eryn.backends import HDFBackend


# %% Config
SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
RESULTS_DIR = SCRIPT_DIR.parents[1] / "data" / "flow_benchmark"
WINDOW = 50  # match v4

TOBS_MONTHS = [6, 7, 8, 9, 10, 11, 12]
LABELS_PER_TOBS = ["default", "flow_full_95_05"]


# %% IACT helper (identical to v4_mcmc.py)
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


def per_rung(backend_path):
    bk = HDFBackend(str(backend_path))
    af = np.asarray(bk.accepted, dtype=float) / max(bk.iteration, 1)  # (ntemps, nwalkers)
    chain = bk.get_chain()["gb"][:, :, :, 0, :]    # (nsteps, ntemps, nwalkers, ndim)
    betas = bk.get_betas()                          # (nsteps, ntemps)
    nsteps, ntemps, nwalkers, ndim = chain.shape
    iacts = np.zeros((ntemps, nwalkers))
    for t in range(ntemps):
        for w in range(nwalkers):
            iacts[t, w] = iact_walker(chain[:, t, w, :])
    return dict(
        accept_per_rung=af,
        iact_per_rung=iacts,
        beta_median=np.median(betas, axis=0),
        beta_final=betas[-1].copy(),
        nsteps=nsteps,
        nwalkers=nwalkers,
    )


# %% Process all v4 chains
results = {}
for months in TOBS_MONTHS:
    for label in LABELS_PER_TOBS:
        path = RESULTS_DIR / f"chain_v4_{months:02d}mo_{label}.h5"
        if not path.exists():
            print(f"  skip (missing): {path.name}", flush=True)
            continue
        print(f"Processing {path.name}...", flush=True)
        results[f"v4_{months:02d}mo_{label}"] = per_rung(path)

# Also process the v4 12mo size-ablation runs while we're here
for size_label in ["50k", "200k"]:
    path = RESULTS_DIR / f"chain_v4_12mo_flow_{size_label}_95_05.h5"
    if not path.exists():
        continue
    print(f"Processing {path.name}...", flush=True)
    results[f"v4_12mo_flow_{size_label}_95_05"] = per_rung(path)


# %% Persist
out = RESULTS_DIR / "flow_v4_per_rung_metrics.pkl"
with open(out, "wb") as fh:
    pickle.dump(results, fh)
print(f"\nSaved {out}", flush=True)


# %% Summary table: cold + median over all rungs + worst rung
print("\n=== v4 per-rung summary (mean over walkers) ===", flush=True)
print(f"{'label':<35} {'cold_acc':>9} {'mean_acc':>9} {'min_acc':>8} "
      f"{'cold_IACT':>10} {'mean_IACT':>10}", flush=True)
for k in sorted(results.keys()):
    r = results[k]
    af = r["accept_per_rung"]
    iact = r["iact_per_rung"]
    print(f"{k:<35} {af[0].mean():>9.3f} {af.mean():>9.3f} {af.mean(axis=1).min():>8.3f} "
          f"{iact[0].mean():>10.2f} {iact.mean():>10.2f}", flush=True)

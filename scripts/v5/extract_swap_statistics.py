"""Pull per-adjacent-rung swap acceptance from every v4 and v5 deployment
backend and compare. eryn's HDFBackend stores `swaps_accepted` as a cumulative
count over the run, so swap-acceptance fraction per pair = swaps_accepted /
(nwalkers * niter_total).

Output: data/flow_benchmark/flow_swap_statistics.pkl
   keys: same labels as flow_v4_per_rung_metrics.pkl / flow_v5_results.pkl,
         each value a dict {swap_accept: (ntemps-1,), nwalkers, niter,
                            betas_final: (ntemps,)}

Also writes a tidy console table for the v4-vs-v5 comparison at every Tobs.
"""

# %% Imports
import pickle
from pathlib import Path

import numpy as np

from eryn.backends import HDFBackend


# %% Config
SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
RESULTS_DIR = SCRIPT_DIR.parents[1] / "data" / "flow_benchmark"

TOBS_MONTHS = [6, 7, 8, 9, 10, 11, 12]


def pair_acceptance(backend_path):
    bk = HDFBackend(str(backend_path))
    swaps_acc = np.asarray(bk.swaps_accepted, dtype=float)   # (ntemps-1,)
    betas = bk.get_betas()                                    # (niter, ntemps)
    niter, ntemps = betas.shape
    # Each iteration proposes nwalkers swaps per adjacent pair (tempering.py:286).
    # Derive nwalkers from a chain read (bk.shape is per-branch dict).
    any_branch = next(iter(bk.get_chain().values()))
    nwalkers = any_branch.shape[2]
    proposed_per_pair = nwalkers * niter
    return dict(
        swap_accept=swaps_acc / proposed_per_pair,
        swaps_accepted=swaps_acc,
        nwalkers=nwalkers,
        niter=niter,
        betas_final=betas[-1].copy(),
    )


# %% Collect for every backend
labels = []
results = {}

for months in TOBS_MONTHS:
    for label in [f"v4_{months:02d}mo_default",
                  f"v4_{months:02d}mo_flow_full_95_05",
                  f"v5_{months:02d}mo_default",
                  f"v5_{months:02d}mo_perrung_95_05"]:
        path = RESULTS_DIR / f"chain_{label}.h5"
        if not path.exists():
            continue
        results[label] = pair_acceptance(path)
        labels.append(label)

# Save
out = RESULTS_DIR / "flow_swap_statistics.pkl"
with open(out, "wb") as fh:
    pickle.dump(results, fh)
print(f"Saved {out}", flush=True)


# %% Compact table — mean swap-accept across pairs + min-pair acceptance
print(f"\n{'label':<35} {'mean_swap_acc':>14} {'min_pair_acc':>13} "
      f"{'pair-by-pair acceptance (rungs 0<->1 ... 8<->9)':<60}", flush=True)
for label in sorted(results.keys()):
    sa = results[label]["swap_accept"]
    pair_str = " ".join(f"{x:.2f}" for x in sa)
    print(f"{label:<35} {sa.mean():>14.3f} {sa.min():>13.3f}  {pair_str}",
          flush=True)


# %% v4-vs-v5 head-to-head at each Tobs (mean + min over pairs)
print("\n=== v4 single-flow vs v5 per-rung — mean / min swap acceptance ===",
      flush=True)
print(f"{'Tobs':<5} {'v4 single (mean/min)':>22} {'v5 per-rung (mean/min)':>24}",
      flush=True)
for m in TOBS_MONTHS:
    k4 = f"v4_{m:02d}mo_flow_full_95_05"
    k5 = f"v5_{m:02d}mo_perrung_95_05"
    if k4 not in results or k5 not in results:
        continue
    s4 = results[k4]["swap_accept"]
    s5 = results[k5]["swap_accept"]
    print(f"{m:<5} {f'{s4.mean():.3f} / {s4.min():.3f}':>22} "
          f"{f'{s5.mean():.3f} / {s5.min():.3f}':>24}", flush=True)

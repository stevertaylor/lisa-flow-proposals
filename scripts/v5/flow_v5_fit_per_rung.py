"""V5: fit one coppuccino flow per temperature index using the 6mo training run.

For each rung k in [0..NTEMPS-1], extract post-burn samples at that rung from
chain_train_6mo_v5.h5 and fit a coppuccino flow with prior_bounds. Architecture
matches v2/v4 (6 spline layers, 4 knots, 200 marginal-CDF points).

Outputs (data/flow_benchmark/):
  flow_6mo_v5_rung{k:02d}.pkl              -- one per rung, indexed by temp idx
  flow_6mo_v5_per_rung_losses.npz          -- train/val loss curves per rung
  flow_6mo_v5_per_rung_training_curves.png -- val-loss curves, colour by beta
  flow_6mo_v5_per_rung_summary.npz         -- betas, sample counts, final losses
"""

# %% Imports
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
if not hasattr(np, "in1d"):
    np.in1d = np.isin

from eryn.backends import HDFBackend
import coppuccino
import coppuccino.copula_flows as cf
from coppuccino import save_flow


# %% Monkey-patch fit_to_data to capture losses
captured = {}
_orig_fit = cf.fit_to_data


def _patched_fit(*args, **kwargs):
    flow, losses = _orig_fit(*args, **kwargs)
    captured["losses"] = losses
    return flow, losses


cf.fit_to_data = _patched_fit


# %% Config
SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
RESULTS_DIR = SCRIPT_DIR.parents[1] / "data" / "flow_benchmark"

NTEMPS, NDIM = 10, 6
MAX_EPOCHS = 1500
RNG_SEED = 42

PRIOR_BOUNDS = np.array([
    [1e-23, 1e-21],
    [0.00299995, 0.00300005],
    [1e-18, 1e-17],
    [0.0, 2 * np.pi],
    [0.0, np.pi],
    [0.0, np.pi],
])

TRAIN_BACKEND = RESULTS_DIR / "chain_train_6mo_v5.h5"
BETAS_PATH = RESULTS_DIR / "flow_6mo_v5_betas.npy"


# %% Load training chain (all rungs) and the frozen ladder
print("Loading 6mo training chain...", flush=True)
bk = HDFBackend(str(TRAIN_BACKEND))
chain_all = bk.get_chain()["gb"][:, :, :, 0, :]  # (nsteps, ntemps, nwalkers, ndim)
nsteps, ntemps, nwalkers, ndim = chain_all.shape
assert ntemps == NTEMPS and ndim == NDIM, \
    f"unexpected shape {chain_all.shape}, expected ntemps={NTEMPS} ndim={NDIM}"
print(f"Chain shape: {chain_all.shape}  "
      f"(nsteps, ntemps, nwalkers, ndim)", flush=True)

betas = np.load(BETAS_PATH)
print(f"Frozen betas: {np.array2string(betas, precision=4)}", flush=True)


# %% Fit per-rung
loss_curves = {}
final_val_losses = np.full(NTEMPS, np.nan)
n_samples_per_rung = np.zeros(NTEMPS, dtype=int)

for k in range(NTEMPS):
    chain_k = chain_all[:, k, :, :].reshape(-1, NDIM)
    n_samples_per_rung[k] = chain_k.shape[0]

    print(f"\n--- rung {k:02d}  beta={betas[k]:.4e}  "
          f"({chain_k.shape[0]} samples) ---", flush=True)

    captured.clear()
    t0 = time.perf_counter()
    flow = coppuccino.normalizing_flows_fit(
        chain_k,
        max_epochs=MAX_EPOCHS,
        rng_seed=RNG_SEED,
        prior_bounds=PRIOR_BOUNDS,
    )
    t_fit = time.perf_counter() - t0

    losses = captured.get("losses", {"train": [], "val": []})
    train_l = np.asarray(losses["train"])
    val_l = np.asarray(losses["val"])
    n_epochs = len(train_l)
    best_ep = int(np.argmin(val_l)) + 1 if len(val_l) else -1
    stopped_early = n_epochs < MAX_EPOCHS
    if len(val_l):
        final_val_losses[k] = val_l.min()
        print(f"  epochs run        : {n_epochs}/{MAX_EPOCHS}  "
              f"({'EARLY STOP' if stopped_early else 'hit ceiling'})", flush=True)
        print(f"  fit wallclock     : {t_fit:.1f}s", flush=True)
        print(f"  val loss [first,last,min(epoch)] : "
              f"[{val_l[0]:.4f}, {val_l[-1]:.4f}, {val_l.min():.4f} (ep {best_ep})]",
              flush=True)

    out_path = RESULTS_DIR / f"flow_6mo_v5_rung{k:02d}.pkl"
    save_flow(flow, str(out_path))
    print(f"  saved {out_path}", flush=True)
    loss_curves[k] = (train_l, val_l)


# %% Plot training curves, colour-coded by beta
fig, ax = plt.subplots(figsize=(9, 5.5))
cmap = plt.cm.viridis
for k in range(NTEMPS):
    tl, vl = loss_curves[k]
    if len(vl) == 0:
        continue
    epochs = np.arange(1, len(tl) + 1)
    colour = cmap(k / max(NTEMPS - 1, 1))
    ax.plot(epochs, vl, lw=1.2, color=colour,
            label=f"rung {k:02d}  beta={betas[k]:.2e}")
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation NLL loss")
ax.set_title("V5 per-rung flow training (6mo, frozen ladder)")
ax.legend(fontsize=7, ncol=2, loc="best")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "flow_6mo_v5_per_rung_training_curves.png", dpi=160)
plt.close(fig)


# %% Persist loss arrays + summary
save_dict = {}
for k, (tl, vl) in loss_curves.items():
    save_dict[f"rung{k:02d}_train"] = tl
    save_dict[f"rung{k:02d}_val"] = vl
np.savez(RESULTS_DIR / "flow_6mo_v5_per_rung_losses.npz", **save_dict)

np.savez(
    RESULTS_DIR / "flow_6mo_v5_per_rung_summary.npz",
    betas=betas,
    final_val_losses=final_val_losses,
    n_samples_per_rung=n_samples_per_rung,
)

print(f"\nFit {NTEMPS} per-rung flows.", flush=True)
print(f"Final val losses: {np.array2string(final_val_losses, precision=3)}",
      flush=True)

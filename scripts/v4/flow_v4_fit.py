"""V4: fit three coppuccino flows on subsamples of the 6mo training chain.

Subsample sizes: 50k (matches the original v2 3mo training set), 200k, full (~480k).
Architecture: v2 (6 spline layers, 4 knots, 200 marginal-CDF points), prior_bounds
set from the GB uniform priors, max_epochs=1500, patience=30, rng_seed=42.

Outputs (data/flow_benchmark/):
  flow_6mo_v2_50k.pkl, flow_6mo_v2_200k.pkl, flow_6mo_v2_full.pkl
  flow_v4_training_losses.npz, flow_v4_training_curves.png
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

NDIM = 6
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

SUBSAMPLE_SIZES = {"50k": 50_000, "200k": 200_000, "full": None}


# %% Load 6mo training chain
print("Loading 6mo training chain...", flush=True)
chain_full = HDFBackend(
    str(RESULTS_DIR / "chain_train_6mo_v4.h5")
).get_chain()["gb"][:, 0, :, 0].reshape(-1, NDIM)
print(f"Full chain shape: {chain_full.shape}", flush=True)


# %% Fit each subsample
loss_curves = {}
for label, size in SUBSAMPLE_SIZES.items():
    if size is None or size >= chain_full.shape[0]:
        chain_sub = chain_full
        actual = chain_full.shape[0]
    else:
        rng = np.random.default_rng(RNG_SEED)
        idx = rng.choice(chain_full.shape[0], size=size, replace=False)
        chain_sub = chain_full[idx]
        actual = size

    print(f"\n--- Fitting flow on {label} subsample ({actual} samples) ---",
          flush=True)
    captured.clear()
    t0 = time.perf_counter()
    flow = coppuccino.normalizing_flows_fit(
        chain_sub,
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
    print(f"  epochs run        : {n_epochs}/{MAX_EPOCHS}  "
          f"({'EARLY STOP' if stopped_early else 'hit ceiling'})", flush=True)
    print(f"  fit wallclock     : {t_fit:.1f}s", flush=True)
    if len(val_l):
        print(f"  val loss [first,last,min(epoch)] : "
              f"[{val_l[0]:.4f}, {val_l[-1]:.4f}, {val_l.min():.4f} (ep {best_ep})]",
              flush=True)

    out_path = RESULTS_DIR / f"flow_6mo_v2_{label}.pkl"
    save_flow(flow, str(out_path))
    print(f"  saved {out_path}", flush=True)
    loss_curves[label] = (train_l, val_l)


# %% Plot training curves
fig, ax = plt.subplots(figsize=(8, 5))
for label, (tl, vl) in loss_curves.items():
    if len(vl) == 0:
        continue
    epochs = np.arange(1, len(tl) + 1)
    line, = ax.plot(epochs, vl, lw=1.4, label=f"{label} (val)")
    ax.plot(epochs, tl, lw=0.8, ls="--", color=line.get_color(), alpha=0.7,
            label=f"{label} (train)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Negative log-likelihood loss")
ax.set_title("V4 flow training: 6mo chain subsamples")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "flow_v4_training_curves.png", dpi=160)
plt.close(fig)

# Save loss arrays (variable length so use dict)
save_dict = {}
for k, (tl, vl) in loss_curves.items():
    save_dict[f"{k}_train"] = tl
    save_dict[f"{k}_val"] = vl
np.savez(RESULTS_DIR / "flow_v4_training_losses.npz", **save_dict)

print("\nAll three flows fit and saved.", flush=True)

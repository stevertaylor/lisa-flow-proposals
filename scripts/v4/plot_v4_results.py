"""V4 plots:
  1. 4-panel cold-chain metrics vs Tobs (acceptance, IACT, ESS/N, ESS/sec)
     for StretchMove vs FlowSlabMove(v4, full flow, 95/5).
  2. Bar plot: training-size ablation at Tobs=12mo (50k vs 200k vs full).
  3. Corner plot at 12mo: StretchMove target + FlowSlabMove chain + raw flow draws.
  4. Prints a numeric summary table for the chat log.
"""

# %% Imports
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eryn.backends import HDFBackend

try:
    import pandas as pd
    from chainconsumer import Chain, ChainConsumer
    HAVE_CC = True
except Exception as exc:
    print(f"ChainConsumer unavailable ({exc}); skipping corner plot.", flush=True)
    HAVE_CC = False


# %% Config
SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
RESULTS_DIR = SCRIPT_DIR.parents[1] / "data" / "flow_benchmark"

TOBS_MONTHS = [6, 7, 8, 9, 10, 11, 12]


# %% Load
with open(RESULTS_DIR / "flow_v4_results.pkl", "rb") as fh:
    results = pickle.load(fh)


# %% Build per-Tobs series
def series(key_template):
    out = dict(acc_mean=[], acc_std=[], iact_mean=[], iact_std=[],
               ess_n=[], ess_s=[])
    for m in TOBS_MONTHS:
        r = results[key_template.format(m=m)]
        out["acc_mean"].append(r["accept_cold"].mean())
        out["acc_std"].append(r["accept_cold"].std())
        out["iact_mean"].append(r["iact_cold"].mean())
        out["iact_std"].append(r["iact_cold"].std())
        out["ess_n"].append(r["ess_per_n"].mean())
        out["ess_s"].append(r["ess_per_sec"])
    return {k: np.asarray(v) for k, v in out.items()}


s_default = series("v4_{m:02d}mo_default")
s_flow    = series("v4_{m:02d}mo_flow_full_95_05")


# %% 4-panel figure
months = np.array(TOBS_MONTHS, dtype=float)
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

panels = [
    (axes[0, 0], "Cold-chain acceptance fraction", "acc_mean", "acc_std"),
    (axes[0, 1], "Cold-chain IACT",                "iact_mean", "iact_std"),
    (axes[1, 0], "ESS / N  (=1/IACT)",             "ess_n",    None),
    (axes[1, 1], "ESS per CPU-second",             "ess_s",    None),
]

for ax, title, key_mean, key_std in panels:
    if key_std is not None:
        ax.errorbar(months, s_default[key_mean], yerr=s_default[key_std],
                    fmt="-o", color="#222", capsize=3, ms=6, label="StretchMove")
        ax.errorbar(months, s_flow[key_mean], yerr=s_flow[key_std],
                    fmt="-^", color="#E07B00", capsize=3, ms=6,
                    label="FlowSlabMove(v4) 95/5")
    else:
        ax.plot(months, s_default[key_mean], "-o", color="#222", ms=6,
                label="StretchMove")
        ax.plot(months, s_flow[key_mean], "-^", color="#E07B00", ms=6,
                label="FlowSlabMove(v4) 95/5")
    ax.set_xlabel(r"$T_\mathrm{obs}$  (months)")
    ax.set_ylabel(title)
    ax.set_xticks(TOBS_MONTHS)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

fig.suptitle("V4: flow trained at 6mo (full, ~480k samples), reused 6 -> 12 mo")
fig.tight_layout()
fig.savefig(RESULTS_DIR / "cold_chain_vs_tobs_v4.png", dpi=160)
plt.close(fig)
print("Saved cold_chain_vs_tobs_v4.png", flush=True)


# %% Training-size ablation at 12mo
ab_labels = ["50k", "200k", "full"]
ab_keys = {
    "50k":  "v4_12mo_flow_50k_95_05",
    "200k": "v4_12mo_flow_200k_95_05",
    "full": "v4_12mo_flow_full_95_05",
}
ab_r = {lbl: results[ab_keys[lbl]] for lbl in ab_labels}

acc_mean  = [ab_r[l]["accept_cold"].mean() for l in ab_labels]
acc_std   = [ab_r[l]["accept_cold"].std()  for l in ab_labels]
iact_mean = [ab_r[l]["iact_cold"].mean()   for l in ab_labels]
iact_std  = [ab_r[l]["iact_cold"].std()    for l in ab_labels]
ess_n     = [ab_r[l]["ess_per_n"].mean()   for l in ab_labels]
ess_s     = [ab_r[l]["ess_per_sec"]        for l in ab_labels]

x = np.arange(len(ab_labels))
fig, axes = plt.subplots(1, 4, figsize=(15, 4))
for ax, vals, errs, title in [
    (axes[0], acc_mean,  acc_std,  "Cold acceptance"),
    (axes[1], iact_mean, iact_std, "Cold IACT"),
    (axes[2], ess_n,     None,     "ESS / N"),
    (axes[3], ess_s,     None,     "ESS per CPU-second"),
]:
    if errs is not None:
        ax.bar(x, vals, yerr=errs, capsize=4, color="#E07B00")
    else:
        ax.bar(x, vals, color="#E07B00")
    ax.set_xticks(x)
    ax.set_xticklabels(ab_labels)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

fig.suptitle("V4 training-size ablation at Tobs = 12mo  (FlowSlabMove 95/5)")
fig.tight_layout()
fig.savefig(RESULTS_DIR / "training_size_ablation_12mo.png", dpi=160)
plt.close(fig)
print("Saved training_size_ablation_12mo.png", flush=True)


# %% 12mo corner: target + flow-driven chain + raw flow draws
if HAVE_CC:
    from coppuccino import load_flow, sample as flow_sample
    flow_full = load_flow(str(RESULTS_DIR / "flow_6mo_v2_full.pkl"))
    flow_draws = np.asarray(flow_sample(flow_full, n_samples=20000, rng_seed=2026))

    bk_def = HDFBackend(str(RESULTS_DIR / "chain_v4_12mo_default.h5"))
    target_12mo = bk_def.get_chain()["gb"][:, 0, :, 0].reshape(-1, 6)

    bk_flow = HDFBackend(str(RESULTS_DIR / "chain_v4_12mo_flow_full_95_05.h5"))
    flow_chain = bk_flow.get_chain()["gb"][:, 0, :, 0].reshape(-1, 6)

    def make_df(arr):
        return pd.DataFrame({
            "log10_amp": np.log10(arr[:, 0]),
            "log10_f0":  np.log10(arr[:, 1]),
            "fdot":      arr[:, 2],
            "phi0":      arr[:, 3],
            "inc":       arr[:, 4],
            "psi":       arr[:, 5],
        })

    cc = ChainConsumer()
    cc.add_chain(Chain(samples=make_df(target_12mo), name="StretchMove 12mo"))
    cc.add_chain(Chain(samples=make_df(flow_chain),  name="FlowSlabMove(v4) 12mo"))
    cc.add_chain(Chain(samples=make_df(flow_draws),  name="Raw flow draws"))
    fig = cc.plotter.plot()
    fig.savefig(RESULTS_DIR / "corner_v4_12mo.png", dpi=140)
    plt.close(fig)
    print("Saved corner_v4_12mo.png", flush=True)


# %% Numeric summary table
print("\n=== V4 cold-chain metrics by Tobs ===", flush=True)
print(f"{'Tobs':>5} {'acc(F)':>8} {'acc(S)':>8} {'IACT(F)':>9} {'IACT(S)':>9} "
      f"{'ESS/N(F)':>10} {'ESS/N(S)':>10} {'ESS/s(F)':>10} {'ESS/s(S)':>10}",
      flush=True)
for i, m in enumerate(TOBS_MONTHS):
    print(f"{m:>5} "
          f"{s_flow['acc_mean'][i]:>8.3f} {s_default['acc_mean'][i]:>8.3f} "
          f"{s_flow['iact_mean'][i]:>9.2f} {s_default['iact_mean'][i]:>9.2f} "
          f"{s_flow['ess_n'][i]:>10.3f} {s_default['ess_n'][i]:>10.3f} "
          f"{s_flow['ess_s'][i]:>10.2f} {s_default['ess_s'][i]:>10.2f}",
          flush=True)

print("\n=== 12mo training-size ablation ===", flush=True)
print(f"{'size':>8} {'acc':>8} {'IACT':>8} {'ESS/N':>10} {'ESS/sec':>10}",
      flush=True)
for lbl, am, im, en, es in zip(ab_labels, acc_mean, iact_mean, ess_n, ess_s):
    print(f"{lbl:>8} {am:>8.3f} {im:>8.2f} {en:>10.3f} {es:>10.2f}", flush=True)

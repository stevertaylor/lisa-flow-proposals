# lisa-flow-proposals

Normalizing-flow MCMC proposals for LISA Galactic Binary recovery, benchmarked against `StretchMove` across observation times. Two staged experiments that built from v1-3 that were unsuccessful:

- **v4** — single flow trained on a 6-month MCMC chain, reused at 6 → 12 months.
- **v5** — per-temperature flow stack, frozen ladder, reused at 6 → 12 months. Fixes the hot-rung acceptance collapse seen in v4.

Setup: noise-free single-GB injection at `amp=2e-23, f0=3e-3 Hz, fdot=7.538e-18, phi0=0.1, inc=0.2, psi=0.3`; sky fixed at the Galactic centre (`λ=4.13, β=-0.15`). Eryn `EnsembleSampler` with 24 walkers × 10 temperatures (2000 sampling + 1000 burn). Narrow uniform priors (e.g. `f0 ± 5e-8 Hz`).

## Headline results

### v4 — single flow trained at 6 months, reused at 6 → 12 months

Cold-chain (β=1) mean ± std across 24 walkers:

| Tobs | StretchMove (acc / IACT / ESS/sec) | FlowSlabMove v4 95/5 |
|---:|---|---|
|  6 mo | 0.332 / 6.38 / 33.79 | **0.736 / 2.03 / 64.95** |
|  9 mo | 0.342 / 7.22 / 28.51 | 0.245 / 3.72 / 36.72 |
| 12 mo | 0.348 / 7.47 / 28.03 | 0.175 / 4.29 / 32.63 |

v4 beats `StretchMove` on ESS/sec at every Tobs even though cold acceptance falls — each accepted move is far better than a `StretchMove` step. But the **hot-rung acceptance collapses to ~0.002** because a single cold-mode flow is a poor proposal for tempered chains targeting the prior (lesson-learned #3 below).

### v5 — per-temperature flow stack, frozen ladder, reused at 6 → 12 months

Cold-chain / min-rung acceptance:

| Tobs | StretchMove (frozen ladder) | PerRungFlowSlabMove v5 95/5 |
|---:|---|---|
|  6 mo | 0.328 / 7.41 / 30.75 / min-rung 0.274 | **0.788 / 1.70 / 70.33 / min-rung 0.745** |
|  9 mo | 0.349 / 7.01 / 29.73 / min-rung 0.273 | 0.253 / 4.04 / 30.15 / min-rung 0.231 |
| 12 mo | 0.350 / 6.84 / 31.03 / min-rung 0.270 | 0.182 / 4.70 / 26.40 / min-rung 0.158 |

**Per-rung dispatch is a PT-mixing fix, not a cold-chain-efficiency fix.** Min-rung acceptance is 0.158–0.745 at every Tobs (vs ~0.002 in v4), but cold-chain numbers match v4. Use per-rung when ladder quality matters; use a single flow when only cold ESS/sec at the training Tobs matters.

## Working coppuccino flow recipe

```python
import numpy as np
from coppuccino import normalizing_flows_fit

PRIOR_BOUNDS = np.array([
    [1e-23, 1e-21],
    [0.00299995, 0.00300005],
    [1e-18, 1e-17],
    [0.0, 2*np.pi],
    [0.0, np.pi],
    [0.0, np.pi],
])
flow = normalizing_flows_fit(
    chain,                        # (n_samples, 6), e.g. cold chain post-burn
    max_epochs=1500,              # patience=30 early stop
    rng_seed=42,
    prior_bounds=PRIOR_BOUNDS,    # critical — see lessons learned 1
)
```

Defaults (6 spline layers, 4 knots, 200 marginal-CDF points) are fine.

## Lessons learned

1. **Always set `prior_bounds`** when fitting on a bounded-prior MCMC chain. Without it the flow's marginal CDF spans only the chain's observed `[min, max]`. Walker states near the actual prior edge then fall outside the flow's support → `log_q(x) = -inf` → MH rejects. This single argument moved cold MH acceptance from 0.097 → 0.516 in v2.

2. **More flow capacity ≠ better MCMC.** A high-capacity fit (12 layers, 8 knots, 500 points) gave 47× higher IS ESS/N but *worse* cold MH acceptance (0.344 vs 0.516). Peaky flows have thinner tails than the true posterior, so MH rejects when the chain transiently visits a tail. Validate any flow with a short MCMC; do not trust IS-ESS alone as a proposal-quality metric.

3. **Independence flow proposals fail at hot temperatures.** Same flow used at every β → cold chain (β=1) accepts well; hot chains target the prior but receive proposals from the cold-mode flow → low acceptance. All eryn independence moves (`FlowMove`, `GaussianMove`, `DistributionGenerate`, `PriorDraw`) share this. Only relative-position moves (`StretchMove` and other `RedBlueMove` subclasses) implicitly adapt because their proposal scale follows the walker ensemble. Fixes: `CombineMove([FlowMove, StretchMove])`, β-scaled flow weight, or per-rung flow stack (v5 — fully solves this), or skip PT.

4. **`coppuccino.normalizing_flows_fit` discards its `losses` array.** To inspect training curves, monkey-patch `coppuccino.copula_flows.fit_to_data`.

5. **ChainConsumer can hang on `plt.show()` with the macOS backend.** Use `matplotlib.use("Agg")` before any pyplot import in scripts that save corner plots.

## Report ESS, not just acceptance

For proposal benchmarking, report ESS/N **and** ESS/sec, not acceptance alone. In v4, flow acceptance fell below `StretchMove` from 8 months onward but ESS/sec stayed higher — because each accepted flow move was a near-independent sample whereas `StretchMove` produces correlated steps.

## Layout

```
lisa-flow-proposals/
├── scripts/
│   ├── v4/   flow_v4_6mo_training_run.py, flow_v4_fit.py, flow_v4_mcmc.py,
│   │        plot_v4_results.py, run_v4_pipeline.sh
│   └── v5/   flow_v5_6mo_training_run.py, flow_v5_fit_per_rung.py,
│            flow_v5_mcmc.py, flow_v5_mcmc_tobs_sweep.py,
│            process_v4_per_rung_metrics.py, extract_swap_statistics.py,
│            run_v5_pipeline.sh
├── notebooks/   flow_v4_plots.ipynb, flow_v5_plots.ipynb
└── data/flow_benchmark/   small numeric summaries + figures only
```

Every script resolves `RESULTS_DIR = SCRIPT_DIR.parents[1] / "data" / "flow_benchmark"`, so it can be launched from anywhere.

### What's in `data/flow_benchmark/`

Committed (small, ~1 MB total):

- Numeric summaries: `flow_v4_results.pkl`, `flow_v4_per_rung_metrics.pkl`, `flow_v5_results.pkl`, `flow_swap_statistics.pkl`
- Training curves: `flow_v4_training_losses.npz`, `flow_6mo_v5_per_rung_losses.npz`, `flow_6mo_v5_per_rung_summary.npz`
- Frozen ladder: `flow_6mo_v5_betas.npy`
- Figures: `cold_chain_vs_tobs_v4.png`, `corner_v4_12mo.png`, `flow_v4_training_curves.png`, `flow_6mo_v5_per_rung_training_curves.png`, `training_size_ablation_12mo.png`

Not committed (regenerate via the pipelines below — see `.gitignore`):

- MCMC backends (`chain_*.h5`, ~44 MB each, ~1.3 GB total)
- Training backends (`chain_train_6mo_v?.h5`, 438 MB each)
- Flow pickles (`flow_6mo_v2_*.pkl`, `flow_6mo_v5_rung*.pkl`, ~10–22 MB each)

## Install

```bash
conda env create -f environment.yml
conda activate lisa_env
```

This installs `gbgpu`, `eryn`, `coppuccino`, `flowjax`, `lisatools`, plus dependencies. Python 3.12.

## Reproduce from scratch

```bash
PY=$(which python) bash scripts/v4/run_v4_pipeline.sh   # ~3 h
PY=$(which python) bash scripts/v5/run_v5_pipeline.sh   # ~80 min
python scripts/v5/flow_v5_mcmc_tobs_sweep.py            # ~80 min
python scripts/v5/process_v4_per_rung_metrics.py        # ~1 min
```

Pipeline order:

1. `flow_v?_6mo_training_run.py` — produces the 6-month training backend.
2. `flow_v4_fit.py` (single flow with 50k / 200k / full subsample variants) or `flow_v5_fit_per_rung.py` (10 per-rung flows on the frozen ladder).
3. `flow_v?_mcmc.py` — 6→12 mo MCMC comparing default vs flow proposal.
4. `plot_v4_results.py` (v4 only) — figures + `flow_v4_results.pkl`.

Notebooks under `notebooks/` produce tweakable plots from the committed `.pkl` summaries; they do **not** require the large `.h5` backends.

## License

See `LICENSE`.

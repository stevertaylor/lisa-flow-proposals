#!/bin/bash
# V4 pipeline: training run -> three flow fits -> 7+2 MCMC validations -> plots.
# Detached via nohup from the calling shell; logs to data/flow_benchmark/v4_logs/.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$EXP_DIR/data/flow_benchmark/v4_logs"
PY="${PY:-python}"
mkdir -p "$LOG_DIR"
cd "$SCRIPT_DIR"

PIPELINE_LOG="$LOG_DIR/pipeline.log"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$PIPELINE_LOG"
}

log "v4 pipeline start  (pid=$$)"

log "Step 1/4: flow_v4_6mo_training_run.py"
"$PY" -u flow_v4_6mo_training_run.py 2>&1 | tee "$LOG_DIR/01_training.log"

log "Step 2/4: flow_v4_fit.py"
"$PY" -u flow_v4_fit.py 2>&1 | tee "$LOG_DIR/02_fit.log"

log "Step 3/4: flow_v4_mcmc.py"
"$PY" -u flow_v4_mcmc.py 2>&1 | tee "$LOG_DIR/03_mcmc.log"

log "Step 4/4: plot_v4_results.py"
"$PY" -u plot_v4_results.py 2>&1 | tee "$LOG_DIR/04_plot.log"

log "v4 pipeline complete"
touch "$LOG_DIR/.DONE"

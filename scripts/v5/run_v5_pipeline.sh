#!/bin/bash
# V5 pipeline: 6mo training run with frozen ladder -> per-rung flow fits -> 6mo
# deployment MCMC with PerRungFlowSlabMove. Detached via nohup; logs to
# data/flow_benchmark/v5_logs/.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$EXP_DIR/data/flow_benchmark/v5_logs"
PY="${PY:-python}"
mkdir -p "$LOG_DIR"
cd "$SCRIPT_DIR"

PIPELINE_LOG="$LOG_DIR/pipeline.log"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$PIPELINE_LOG"
}

log "v5 pipeline start  (pid=$$)"

log "Step 1/3: flow_v5_6mo_training_run.py"
"$PY" -u flow_v5_6mo_training_run.py 2>&1 | tee "$LOG_DIR/01_training.log"

log "Step 2/3: flow_v5_fit_per_rung.py"
"$PY" -u flow_v5_fit_per_rung.py 2>&1 | tee "$LOG_DIR/02_fit.log"

log "Step 3/3: flow_v5_mcmc.py"
"$PY" -u flow_v5_mcmc.py 2>&1 | tee "$LOG_DIR/03_mcmc.log"

log "v5 pipeline complete"
touch "$LOG_DIR/.DONE"

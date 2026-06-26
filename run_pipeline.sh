#!/usr/bin/env bash
# Full Enlighten-AI pipeline: scrape -> ingest -> EDA.
# Designed to run DETACHED (setsid) so it survives terminal/session disconnects.
#   setsid bash run_pipeline.sh </dev/null >/dev/null 2>&1 &
# Progress -> data/pipeline.log ; final status -> data/pipeline.status
set -uo pipefail
cd "$(dirname "$0")"

PYBIN=/home/m.a.kerkouri/.conda/envs/enlighten/bin/python
LOG=data/pipeline.log
STATUS=data/pipeline.status

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(stamp)] $*" >>"$LOG"; }

mkdir -p data
rm -f "$STATUS"
: >"$LOG"
log "PIPELINE START (pid $$)"

log "STAGE 1/3: scrape (build_dataset, resumes via idempotent skip)"
"$PYBIN" -m Scrapper.build_dataset >>"$LOG" 2>&1; sc=$?
log "scrape exit=$sc"

log "STAGE 2/3: ingest --rebuild"
"$PYBIN" -m DrK_Chat.ingest --rebuild >>"$LOG" 2>&1; ic=$?
log "ingest exit=$ic"

log "STAGE 3/3: EDA report"
"$PYBIN" -m data_analysis.eda >>"$LOG" 2>&1; ec=$?
log "eda exit=$ec"

echo "scrape=$sc ingest=$ic eda=$ec finished=$(stamp)" >"$STATUS"
log "PIPELINE COMPLETE -> $STATUS"

#!/bin/bash
# k49_dreams_emb on the local GPU in the Kaggle docker image (plan 10.4, 2026-10-24: DreaMS ssl_model.ckpt spectrum
# embeddings + ChemBERTa-100M-MLM molecule embeddings for k50; not pushed to Kaggle). Same notebook as the Kaggle build.
# No internet: the official ssl_model.ckpt (sha256 checked by the kernel) and the ChemBERTa snapshot (rev f5c45f44,
# HF_SHA file) are mounted under /kaggle/input, where the kernel looks for them before downloading.
#   tools/local/run_k49.sh            # full run, output in out/m9_k49
#   SMOKE=1 tools/local/run_k49.sh    # 3,000 training rows, 20 queries per set, 5,000 pool pids -> out/m9_k49_smoke
#                                     # (+ the five R2 queries without k30 candidates, for k50's empty-window path)
# The container runs detached; follow it with tail -f <out>/run.log (the log ends with "exit <rc>"). A rerun on the same
# output directory continues the spectrum embedding from emb_progress.json.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SMOKE=${SMOKE:-0}
OUT=${OUT_DIR:-$ROOT/out/m9_k49$([ "$SMOKE" = 1 ] && echo _smoke || true)}
IMG=gcr.io/kaggle-gpu-images/python:latest
U="$(id -u):$(id -g)"
NAME=${NAME:-casmi26-k49-$([ "$SMOKE" = 1 ] && echo smoke- || true)$(date +%m%d%H%M)}
DR=$ROOT/out/m9_msalign_in/dreams
CB=$ROOT/out/m9_msalign_in/chemberta
[ -f "$DR/ssl_model.ckpt" ] || { echo "missing $DR/ssl_model.ckpt"; exit 1; }
[ -f "$CB/HF_SHA" ] || { echo "missing $CB/HF_SHA"; exit 1; }
mkdir -p "$OUT"
if [ -e "$OUT/run.log" ]; then i=1; while [ -e "$OUT/run.log.$i" ]; do i=$((i + 1)); done; mv "$OUT/run.log" "$OUT/run.log.$i"; fi
py=$OUT/k49.py
python3 "$ROOT/tools/build_kernel.py" k49_dreams_emb > /dev/null
python3 "$ROOT/tools/smoke/nb2py.py" "$ROOT/kaggle/k49_dreams_emb/build/casmi26-k49-dreams-emb.ipynb" "$py"
for pat in "^MAX_TRAIN_ROWS = None " "^MAX_QUERIES = None " "^SMOKE_EXTRA_QIDS = \[\] " "^MAX_POOL = None " "^BUDGET_H = 8.5" "^CHUNK = 8192 " \
           "^EMB_THREADS = None " "^STOP_AFTER_CHUNKS = None "; do
  grep -q "$pat" "$py" || { echo "k49 script is not at the Kaggle defaults: $pat"; exit 1; }
done
[ "$SMOKE" = 1 ] && sed -i 's/^MAX_TRAIN_ROWS = None /MAX_TRAIN_ROWS = 3000 /; s/^MAX_QUERIES = None /MAX_QUERIES = 20 /; s/^SMOKE_EXTRA_QIDS = \[\] /SMOKE_EXTRA_QIDS = [5855, 6551, 7327, 7621, 8173] /; s/^MAX_POOL = None /MAX_POOL = 5000 /' "$py"
# test knobs (resume / threads): CHUNK=256 EMB_THREADS=2 STOP_AFTER_CHUNKS=5 OUT_DIR=out/... SMOKE=1 tools/local/run_k49.sh
[ -n "${CHUNK:-}" ] && sed -i "s/^CHUNK = 8192 /CHUNK = $CHUNK /" "$py"
[ -n "${EMB_THREADS:-}" ] && sed -i "s/^EMB_THREADS = None /EMB_THREADS = $EMB_THREADS /" "$py"
[ -n "${STOP_AFTER_CHUNKS:-}" ] && sed -i "s/^STOP_AFTER_CHUNKS = None /STOP_AFTER_CHUNKS = $STOP_AFTER_CHUNKS /" "$py"
MOUNTS=(-v "$ROOT/out/ds_wheels":/kaggle/input/casmi26-wheels:ro -v "$DR/ssl_model.ckpt":/kaggle/input/casmi26-dreams-ssl/ssl_model.ckpt:ro
        -v "$CB":/kaggle/input/casmi26-chemberta:ro)
# RESUME_FROM=<a previous k49 output dir>: mounted under /kaggle/input like an attached earlier version (resume test)
[ -n "${RESUME_FROM:-}" ] && MOUNTS+=(-v "$(cd "$RESUME_FROM" && pwd)":/kaggle/input/casmi26-k49-dreams-emb-prev:ro)
for s in casmi26-k01-cache casmi26-k05-pool casmi26-k05b-massfix; do
  [ -d "$ROOT/out/m6_in/$s" ] || { echo "missing input out/m6_in/$s"; exit 1; }
  MOUNTS+=(-v "$ROOT/out/m6_in/$s":/kaggle/input/$s:ro)
done
docker run -d --name "$NAME" --gpus all --network none --memory 16g --memory-swap 16g --shm-size 8g \
  -e KAGGLE_KERNEL_RUN_TYPE=Batch -e PYTHONUNBUFFERED=1 -e NUMBA_CACHE_DIR=/tmp/numba \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_DISABLE_PROGRESS_BARS=1 \
  "${MOUNTS[@]}" -v "$py":/run.py:ro -v "$OUT":/kaggle/working "$IMG" \
  bash -c "cd /kaggle/working && python /run.py > run.log 2>&1; rc=\$?; echo \"exit \$rc\" >> run.log; chown -R $U /kaggle/working; exit \$rc"
echo "started $NAME -> $OUT/run.log"

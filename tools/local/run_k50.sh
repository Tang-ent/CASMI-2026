#!/bin/bash
# k50_msalign on the local GPU in the Kaggle docker image (plan 10.4, 2026-10-24: six MSAlign-type variants on the k49
# embeddings, window scores on the k30 windows, gates S / W; not pushed to Kaggle). Same notebook as the Kaggle build.
# With one local GPU the variants train one after another (Kaggle: two at a time).
#   tools/local/run_k50.sh            # full run on out/m9_k49, output in out/m9_k50
#   SMOKE=1 tools/local/run_k50.sh    # 300 steps per variant (warmup 30, log every 100), 20 queries per set, on
#                                     # out/m9_k49_smoke, output in out/m9_k50_smoke
#   STOP_AT=150 SMOKE=1 tools/local/run_k50.sh   # stop every variant at step 150 (resume test: rerun without STOP_AT
#                                     # on the same output directory continues from msa_<variant>_state.pt)
#   K49=out/... tools/local/run_k50.sh  # another k49 output directory
#   RESUME_FROM=out/... tools/local/run_k50.sh  # a previous k50 output mounted under /kaggle/input (resume / refusal test)
#   (SMOKE=1 also adds the five R2 queries without k30 candidates, 5855 6551 7327 7621 8173: empty windows, RR 0)
#   OUT_DIR=out/... tools/local/run_k50.sh  # another output directory (a rerun on a finished one skips training)
# The container runs detached; follow it with tail -f <out>/run.log (the log ends with "exit <rc>").
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SMOKE=${SMOKE:-0}
SFX=$([ "$SMOKE" = 1 ] && echo _smoke || true)
K49=${K49:-$ROOT/out/m9_k49$SFX}
OUT=${OUT_DIR:-$ROOT/out/m9_k50$SFX}
IMG=gcr.io/kaggle-gpu-images/python:latest
U="$(id -u):$(id -g)"
NAME=${NAME:-casmi26-k50-$([ "$SMOKE" = 1 ] && echo smoke- || true)$(date +%m%d%H%M)}
[ -e "$K49/k49_info.json" ] || { echo "missing k49 output $K49/k49_info.json"; exit 1; }
mkdir -p "$OUT"
if [ -e "$OUT/run.log" ]; then i=1; while [ -e "$OUT/run.log.$i" ]; do i=$((i + 1)); done; mv "$OUT/run.log" "$OUT/run.log.$i"; fi
py=$OUT/k50.py
python3 "$ROOT/tools/build_kernel.py" k50_msalign > /dev/null
python3 "$ROOT/tools/smoke/nb2py.py" "$ROOT/kaggle/k50_msalign/build/casmi26-k50-msalign.ipynb" "$py"
for pat in "^MAX_QUERIES = None " "^SMOKE_EXTRA_QIDS = \[\] " "^MAX_STEPS = 30000$" "^WARMUP = 2000$" "^LOG_EVERY = 500$" "^STOP_AT = None " "^N_BOOT = 2000$"; do
  grep -q "$pat" "$py" || { echo "k50 script is not at the Kaggle defaults: $pat"; exit 1; }
done
[ "$SMOKE" = 1 ] && sed -i 's/^MAX_QUERIES = None /MAX_QUERIES = 20 /; s/^SMOKE_EXTRA_QIDS = \[\] /SMOKE_EXTRA_QIDS = [5855, 6551, 7327, 7621, 8173] /; s/^MAX_STEPS = 30000$/MAX_STEPS = 300/; s/^WARMUP = 2000$/WARMUP = 30/; s/^LOG_EVERY = 500$/LOG_EVERY = 100/; s/^N_BOOT = 2000$/N_BOOT = 200/' "$py"
[ -n "${STOP_AT:-}" ] && sed -i "s/^STOP_AT = None /STOP_AT = $STOP_AT /" "$py"
MOUNTS=(-v "$K49":/kaggle/input/casmi26-k49-dreams-emb:ro -v "$ROOT/out/ds_mspred_flat":/kaggle/input/casmi26-mspred:ro)
# RESUME_FROM=<a previous k50 output dir>: mounted under /kaggle/input like an attached earlier version (resume test;
# its msa_<variant>_state.pt are used only when their cfg["tag"] matches the mounted k49)
[ -n "${RESUME_FROM:-}" ] && MOUNTS+=(-v "$(cd "$RESUME_FROM" && pwd)":/kaggle/input/casmi26-k50-msalign-prev:ro)
for s in casmi26-k01-cache casmi26-k05-pool casmi26-k05b-massfix casmi26-k26-fp2; do
  [ -d "$ROOT/out/m6_in/$s" ] || { echo "missing input out/m6_in/$s"; exit 1; }
  MOUNTS+=(-v "$ROOT/out/m6_in/$s":/kaggle/input/$s:ro)
done
for s in casmi26-k30-features4-s0 casmi26-k30-features4-s1 casmi26-k30-features4-s2 casmi26-k30-features4-s3; do
  [ -d "$ROOT/out/p3_in/$s" ] || { echo "missing input out/p3_in/$s"; exit 1; }
  MOUNTS+=(-v "$ROOT/out/p3_in/$s":/kaggle/input/$s:ro)
done
docker run -d --name "$NAME" --gpus all --network none --memory 16g --memory-swap 16g --shm-size 8g \
  -e KAGGLE_KERNEL_RUN_TYPE=Batch -e PYTHONUNBUFFERED=1 -e NUMBA_CACHE_DIR=/tmp/numba \
  "${MOUNTS[@]}" -v "$py":/run.py:ro -v "$OUT":/kaggle/working "$IMG" \
  bash -c "cd /kaggle/working && python /run.py > run.log 2>&1; rc=\$?; echo \"exit \$rc\" >> run.log; chown -R $U /kaggle/working; exit \$rc"
echo "started $NAME -> $OUT/run.log"

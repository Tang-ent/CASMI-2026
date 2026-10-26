#!/bin/bash
# k47_recols2 on the local GPU in the Kaggle docker image (plan 9.2, 2026-10-23: per-spectrum statistics and consensus
# columns on the k30 windows; not pushed to Kaggle). Same notebook as the Kaggle build; only MAX_QUERIES differs for a
# smoke run.
#   tools/local/run_k47.sh            # full run, output in out/m9_k47
#   SMOKE=1 tools/local/run_k47.sh    # 20 queries per set, output in out/m9_k47_smoke
# The container runs detached; follow it with tail -f <out>/run.log (or docker wait <name>; the log ends with "exit <rc>")
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SMOKE=${SMOKE:-0}
OUT=$ROOT/out/m9_k47$([ "$SMOKE" = 1 ] && echo _smoke || true)
IMG=gcr.io/kaggle-gpu-images/python:latest
U="$(id -u):$(id -g)"
NAME=${NAME:-casmi26-k47-$([ "$SMOKE" = 1 ] && echo smoke- || true)$(date +%m%d%H%M)}
mkdir -p "$OUT"
if [ -e "$OUT/run.log" ]; then i=1; while [ -e "$OUT/run.log.$i" ]; do i=$((i + 1)); done; mv "$OUT/run.log" "$OUT/run.log.$i"; fi
py=$OUT/k47.py
python3 "$ROOT/tools/build_kernel.py" k47_recols2 > /dev/null
python3 "$ROOT/tools/smoke/nb2py.py" "$ROOT/kaggle/k47_recols2/build/casmi26-k47-recols2.ipynb" "$py"
for pat in "^MAX_QUERIES = None " "^N_CHECK = 300 " "^REQUIRE_CUDA = True "; do
  grep -q "$pat" "$py" || { echo "k47 script is not at the Kaggle defaults: $pat"; exit 1; }
done
[ "$SMOKE" = 1 ] && sed -i 's/^MAX_QUERIES = None /MAX_QUERIES = 20 /' "$py"
MOUNTS=(-v "$ROOT/out/ds_wheels":/kaggle/input/casmi26-wheels:ro -v "$ROOT/out/m7_in/casmi26-g2":/kaggle/input/casmi26-g2:ro)
for s in casmi26-k01-cache casmi26-k05-pool casmi26-k05b-massfix casmi26-k21-frag; do
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

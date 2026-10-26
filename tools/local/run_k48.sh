#!/bin/bash
# k48_eval9 on CPU in the Kaggle docker image (plan 9.2, 2026-10-23): k30 features + the k47 columns vs the reference v12
# (not pushed to Kaggle). Inputs as the k42 smoke step of tools/smoke/run.sh, plus the k47 output as
# casmi26-k47-recols2. Runs in the foreground (--rm); the log is <out>/run.log.
#   tools/local/run_k48.sh            # full run on out/m9_k47, output in out/m9_k48
#   SMOKE=1 tools/local/run_k48.sh    # 20 queries per set, N_BOOT 200, on out/m9_k47_smoke, output in out/m9_k48_smoke
#   K47=out/... tools/local/run_k48.sh  # another k47 output directory
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SMOKE=${SMOKE:-0}
SFX=$([ "$SMOKE" = 1 ] && echo _smoke || true)
K47=${K47:-$ROOT/out/m9_k47$SFX}
OUT=$ROOT/out/m9_k48$SFX
IMG=gcr.io/kaggle-gpu-images/python:latest
U="$(id -u):$(id -g)"
NAME=${NAME:-casmi26-k48-$([ "$SMOKE" = 1 ] && echo smoke- || true)$(date +%m%d%H%M)}
[ -e "$K47/recols2_info.json" ] || { echo "missing k47 output $K47/recols2_info.json"; exit 1; }
mkdir -p "$OUT"
if [ -e "$OUT/run.log" ]; then i=1; while [ -e "$OUT/run.log.$i" ]; do i=$((i + 1)); done; mv "$OUT/run.log" "$OUT/run.log.$i"; fi
py=$OUT/k48.py
python3 "$ROOT/tools/build_kernel.py" k48_eval9 > /dev/null
python3 "$ROOT/tools/smoke/nb2py.py" "$ROOT/kaggle/k48_eval9/build/casmi26-k48-eval9.ipynb" "$py"
for pat in "^MAX_QUERIES = None " "^N_BOOT = 2000"; do
  grep -q "$pat" "$py" || { echo "k48 script is not at the Kaggle defaults: $pat"; exit 1; }
done
[ "$SMOKE" = 1 ] && sed -i 's/^N_BOOT = 2000/N_BOOT = 200/; s/^MAX_QUERIES = None /MAX_QUERIES = 20 /' "$py"
MOUNTS=(-v "$ROOT/out/m6_in/casmi26-k26-fp2":/kaggle/input/k26:ro -v "$ROOT/out/m6_in/casmi26-k05-pool":/kaggle/input/k05:ro
        -v "$ROOT/out/ds_mspred_flat":/kaggle/input/casmi26-mspred:ro -v "$K47":/kaggle/input/casmi26-k47-recols2:ro)
for s in casmi26-k30-features4-s0 casmi26-k30-features4-s1 casmi26-k30-features4-s2 casmi26-k30-features4-s3 casmi26-k35-eval7; do
  [ -d "$ROOT/out/p3_in/$s" ] || { echo "missing input out/p3_in/$s"; exit 1; }
  MOUNTS+=(-v "$ROOT/out/p3_in/$s":/kaggle/input/$s:ro)
done
echo "running $NAME -> $OUT/run.log"
docker run --rm --name "$NAME" --network none --memory 16g --memory-swap 16g \
  -e KAGGLE_KERNEL_RUN_TYPE=Batch -e PYTHONUNBUFFERED=1 -e NUMBA_CACHE_DIR=/tmp/numba \
  "${MOUNTS[@]}" -v "$py":/run.py:ro -v "$OUT":/kaggle/working "$IMG" \
  bash -c "cd /kaggle/working && python /run.py > run.log 2>&1; rc=\$?; echo \"exit \$rc\" >> run.log; chown -R $U /kaggle/working; exit \$rc" \
  || { tail -30 "$OUT/run.log"; exit 1; }
grep -v -i warn "$OUT/run.log" | tail -"${TAIL:-25}"

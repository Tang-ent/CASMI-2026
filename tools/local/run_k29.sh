#!/bin/bash
# Full k29_g2 run on the local GPU in the Kaggle docker image (user decision 2026-10-12: keep the Kaggle GPU quota for
# P1; k29 is CPU-bound, so Kaggle's second T4 would not help). Same notebook as the Kaggle build, without any smoke
# patch; only the hardware differs.
#   tools/local/run_k29.sh            # inputs in out/m6_in/<slug>/, rdkit from out/ds_wheels, output in out/m6_k29
#   RESUME=1 tools/local/run_k29.sh   # continue an interrupted run from out/m6_k29/g2_state.pt (same cfg; the old
#                                     # run.log is kept as run.log.<n>)
# The container runs detached (docker run -d) so that it outlives the session; follow it with
#   tail -f out/m6_k29/run.log ; docker stats <name printed at the start> ; nvidia-smi
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
IN=$ROOT/out/m6_in
OUT=$ROOT/out/m6_k29
IMG=gcr.io/kaggle-gpu-images/python:latest
U="$(id -u):$(id -g)"
NAME=${NAME:-casmi26-k29-local-$(date +%m%d%H%M)}  # a new name per start (a stopped container keeps the old one)
RESUME=${RESUME:-0}
SLUGS="casmi26-k01-cache casmi26-k05-pool casmi26-k05b-massfix casmi26-k14-fpnet-probe casmi26-k15-fpnet2-s1 casmi26-k20-fwd
       casmi26-k26-fp2"

if [ "$RESUME" = 1 ]; then
  [ -f "$OUT/g2_state.pt" ] || { echo "no $OUT/g2_state.pt to resume from"; exit 1; }
  i=1; while [ -e "$OUT/run.log.$i" ]; do i=$((i + 1)); done
  [ -e "$OUT/run.log" ] && mv "$OUT/run.log" "$OUT/run.log.$i"
else
  [ -e "$OUT/run.log" ] && { echo "$OUT/run.log exists; move it away first (or RESUME=1)"; exit 1; }
fi
mkdir -p "$OUT"
py=$OUT/k29.py
python3 "$ROOT/tools/smoke/nb2py.py" "$ROOT/kaggle/k29_g2/build/casmi26-k29-g2.ipynb" "$py"
# the Kaggle defaults, not the smoke values
for pat in "^TOTAL_STEPS = None " "^BATCH, K_NEG, WORKERS = None, None, None " "^FIT = None " "^MIN_AGREE = 0.98 " \
           "^PPM = 10.0$" "^N_BOOT = 2000$" "^TOTAL_H = 9.0 " "^MAX_QUERIES = None " "^EVAL_PROBE = 30 " "^RESUME = None "; do
  grep -q "$pat" "$py" || { echo "k29 script is not at the Kaggle defaults: $pat"; exit 1; }
done
if [ "$RESUME" = 1 ]; then
  sed -i 's|^RESUME = None |RESUME = "/kaggle/working/g2_state.pt" |' "$py"
  grep -q '^RESUME = "/kaggle/working/g2_state.pt" ' "$py" || { echo "RESUME patch did not apply"; exit 1; }
fi
MOUNTS=(-v "$ROOT/out/ds_wheels":/kaggle/input/casmi26-wheels:ro)
for s in $SLUGS; do
  [ -d "$IN/$s" ] || { echo "missing input $IN/$s"; exit 1; }
  MOUNTS+=(-v "$IN/$s":/kaggle/input/$s:ro)
done
docker run -d --name "$NAME" --gpus all --network none --memory 14g --memory-swap 14g \
  -e KAGGLE_KERNEL_RUN_TYPE=Batch -e PYTHONUNBUFFERED=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${MOUNTS[@]}" -v "$py":/run.py:ro -v "$OUT":/kaggle/working "$IMG" \
  bash -c "cd /kaggle/working && python /run.py > run.log 2>&1; rc=\$?; echo \"exit \$rc\" >> run.log; chown -R $U /kaggle/working; exit \$rc"
echo "started $NAME -> $OUT/run.log"

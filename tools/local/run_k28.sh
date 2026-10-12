#!/bin/bash
# Full k28_fprich run on the local GPU in the Kaggle docker image (user decision 2026-10-12: the Kaggle GPU quota is
# too tight for k28). Same notebook as the Kaggle build, without any smoke patch; only the hardware differs.
#   tools/local/run_k28.sh            # inputs in out/m6_in/<slug>/ (out/m6_in/fetch.sh), output and run.log in out/m6_k28
#   RESUME=1 tools/local/run_k28.sh   # continue an interrupted run from out/m6_k28/fprich_state.pt (same cfg; the old
#                                     # run.log is kept as run.log.<n>)
# The container runs detached (docker run -d) so that it outlives the session; follow it with
#   tail -f out/m6_k28/run.log ; docker stats <name printed at the start> ; nvidia-smi
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
IN=$ROOT/out/m6_in
OUT=$ROOT/out/m6_k28
IMG=gcr.io/kaggle-gpu-images/python:latest
U="$(id -u):$(id -g)"
NAME=${NAME:-casmi26-k28-local-$(date +%m%d%H%M)}  # a new name per start (a stopped container keeps the old one)
RESUME=${RESUME:-0}
SLUGS="casmi26-k01-cache casmi26-k05-pool casmi26-k05b-massfix casmi26-k14-fpnet-probe casmi26-k15-fpnet2-s1 casmi26-k26-fp2"

if [ "$RESUME" = 1 ]; then
  [ -f "$OUT/fprich_state.pt" ] || { echo "no $OUT/fprich_state.pt to resume from"; exit 1; }
  i=1; while [ -e "$OUT/run.log.$i" ]; do i=$((i + 1)); done
  [ -e "$OUT/run.log" ] && mv "$OUT/run.log" "$OUT/run.log.$i"
else
  [ -e "$OUT/run.log" ] && { echo "$OUT/run.log exists; move it away first (or RESUME=1)"; exit 1; }
fi
mkdir -p "$OUT"
py=$OUT/k28.py
python3 "$ROOT/tools/smoke/nb2py.py" "$ROOT/kaggle/k28_fprich/build/casmi26-k28-fprich.ipynb" "$py"
# the Kaggle defaults, not the smoke values
for pat in "^TOTAL_STEPS = None " "^BATCH = None " "^MIN_AGREE = 0.98 " "^PPM = 10.0$" "^N_BOOT = 2000$" "^BUDGET_H = 10.75$"; do
  grep -q "$pat" "$py" || { echo "k28 script is not at the Kaggle defaults: $pat"; exit 1; }
done
if [ "$RESUME" = 1 ]; then
  sed -i 's|^RESUME = None |RESUME = "/kaggle/working/fprich_state.pt" |' "$py"
  grep -q '^RESUME = "/kaggle/working/fprich_state.pt" ' "$py" || { echo "RESUME patch did not apply"; exit 1; }
fi
MOUNTS=()
for s in $SLUGS; do
  [ -d "$IN/$s" ] || { echo "missing input $IN/$s"; exit 1; }
  MOUNTS+=(-v "$IN/$s":/kaggle/input/$s:ro)
done
docker run -d --name "$NAME" --gpus all --network none --memory 14g --memory-swap 14g \
  -e KAGGLE_KERNEL_RUN_TYPE=Batch -e PYTHONUNBUFFERED=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${MOUNTS[@]}" -v "$py":/run.py:ro -v "$OUT":/kaggle/working "$IMG" \
  bash -c "cd /kaggle/working && python /run.py > run.log 2>&1; rc=\$?; echo \"exit \$rc\" >> run.log; chown -R $U /kaggle/working; exit \$rc"
echo "started $NAME -> $OUT/run.log"

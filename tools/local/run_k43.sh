#!/bin/bash
# k43_submit8 on the local GPU with the real inputs (M7 P3-2 check before the Kaggle push): builds the notebook with
# VARIANT, runs it in the Kaggle docker image, and leaves submission.csv / variant.json in the output directory.
#   tools/local/run_k43.sh '<variant>' <k42 output dir> <out dir>
# e.g. the v12 variant must reproduce out/m7_k36/10160835/submission.csv (no ext); an ext variant runs ms-pred.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
VARIANT=${1:?variant}; K42=$(cd "${2:?k42 output dir}" && pwd); OUT=$3
IMG=gcr.io/kaggle-gpu-images/python:latest
U="$(id -u):$(id -g)"
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
python3 "$ROOT/tools/build_kernel.py" k43_submit8 --define "VARIANT = \"$VARIANT\"" > /dev/null
py=$OUT/k43.py
python3 "$ROOT/tools/smoke/nb2py.py" "$ROOT/kaggle/k43_submit8/build/casmi26-k43-submit8.ipynb" "$py"
MOUNTS=(-v "$ROOT/out/ds_wheels":/kaggle/input/casmi26-wheels:ro -v "$ROOT/out/ds_mspred_flat":/kaggle/input/casmi26-mspred:ro
        -v "$ROOT/out/m7_in/casmi26-fprich":/kaggle/input/casmi26-fprich:ro -v "$ROOT/out/m7_in/casmi26-g2":/kaggle/input/casmi26-g2:ro
        -v "$ROOT/enveda-CASMI26-molecule-id-mass-spectra":/kaggle/input/enveda-CASMI26-molecule-id-mass-spectra:ro
        -v "$ROOT/out/p3_in/casmi26-k35-eval7":/kaggle/input/casmi26-k35-eval7:ro -v "$K42":/kaggle/input/casmi26-k42-eval8:ro)
for s in casmi26-k01-cache casmi26-k05-pool casmi26-k05b-massfix casmi26-k15-fpnet2-s1 casmi26-k20-fwd casmi26-k21-frag \
         casmi26-k25-fwd2 casmi26-k26-fp2; do
  MOUNTS+=(-v "$ROOT/out/m6_in/$s":/kaggle/input/$s:ro)
done
docker run --rm --name "casmi26-k43-$(date +%m%d%H%M%S)" --gpus all --network none --memory 16g --memory-swap 16g --shm-size 2g \
  -e KAGGLE_KERNEL_RUN_TYPE=Batch -e PYTHONUNBUFFERED=1 -e NUMBA_CACHE_DIR=/tmp/numba \
  "${MOUNTS[@]}" -v "$py":/run.py:ro -v "$OUT":/kaggle/working "$IMG" \
  bash -c "cd /kaggle/working && python /run.py > run.log 2>&1; rc=\$?; echo \"exit \$rc\" >> run.log; chown -R $U /kaggle/working; exit \$rc" \
  || { tail -30 "$OUT/run.log"; exit 1; }
grep -v -iE "warn|lightning automatically|re.findall|nested_tensor|cloud_io" "$OUT/run.log" | tail -12

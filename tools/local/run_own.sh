#!/bin/bash
# M8 (2026-10-19): run own_runner.py (tools/build_fuse.py: k44_own + embedded casmi26) on the local GPU in the Kaggle docker
# image, the way the fusion notebook runs it: ROOT = the engine inputs laid out as in casmi26_engine_bundle.tar
# (<slug>/<files>; here the local copies are mounted), COMP = the competition data, a fresh RDKit site in the container.
#   tools/local/run_own.sh [out dir, default out/m8_own_check] [tar]
# With a second argument (a local casmi26_engine_bundle.tar), ROOT is that tar unpacked inside the container instead.
# The visible test is the same as k36's, so own_submission.csv must equal out/m7_k36/10160835/submission.csv (v12).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${1:-$ROOT/out/m8_own_check}; TAR=${2:-}
IMG=gcr.io/kaggle-gpu-images/python:latest
U="$(id -u):$(id -g)"
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
python3 "$ROOT/tools/build_fuse.py" > /dev/null
cp "$ROOT/kaggle/k45_fuse/build/own_runner.py" "$OUT/own_runner.py"
M=(-v "$ROOT/enveda-CASMI26-molecule-id-mass-spectra":/comp:ro -v "$OUT":/out -v "$OUT/own_runner.py":/run/own_runner.py:ro)
if [ -n "$TAR" ]; then
  M+=(-v "$(cd "$(dirname "$TAR")" && pwd)/$(basename "$TAR")":/bundle.tar:ro)
  PREP="mkdir -p /tmp/own_bundle && tar -xf /bundle.tar -C /tmp/own_bundle && "
  R=/tmp/own_bundle
else
  for s in casmi26-k01-cache casmi26-k05-pool casmi26-k05b-massfix casmi26-k15-fpnet2-s1 casmi26-k20-fwd casmi26-k21-frag \
           casmi26-k25-fwd2 casmi26-k26-fp2; do
    M+=(-v "$ROOT/out/m6_in/$s":/bundle/$s:ro)
  done
  M+=(-v "$ROOT/out/m7_in/casmi26-fprich":/bundle/casmi26-fprich:ro -v "$ROOT/out/m7_in/casmi26-g2":/bundle/casmi26-g2:ro
      -v "$ROOT/out/p3_in/casmi26-k35-eval7":/bundle/casmi26-k35-eval7:ro -v "$ROOT/out/ds_wheels":/bundle/casmi26-wheels:ro)
  PREP=""
  R=/bundle
fi
docker run --rm --name "casmi26-own-$(date +%m%d%H%M%S)" --gpus all --network none --memory 16g --memory-swap 16g \
  --shm-size 2g -e PYTHONUNBUFFERED=1 "${M[@]}" "$IMG" \
  bash -c "${PREP}cd /out && python /run/own_runner.py $R /comp /out /tmp/own_site > run.log 2>&1; rc=\$?; echo \"exit \$rc\" >> run.log; chown -R $U /out; exit \$rc" \
  || { tail -30 "$OUT/run.log"; exit 1; }
grep -E "^rdkit|^torch|^variant|molecules,|^OWN_STATS|^exit" "$OUT/run.log" | cut -c1-300
cmp "$OUT/own_submission.csv" "$ROOT/out/m7_k36/10160835/submission.csv" && echo "own_submission.csv == k36 v12 submission"

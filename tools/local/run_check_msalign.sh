#!/bin/bash
# Plan 10.2 / 10.3 checks of src/casmi26/msalign.py (tools/smoke/check_msalign.py) in the Kaggle docker image on the
# local GPU: real k01 cache, k05 pool, k05b mass fix and k26 formula table; synthetic spectrum / molecule tables.
# fpnet.py of the reference commit (REF, default c57f500) goes to <out>/head for the default provenance_coef check.
#   tools/local/run_check_msalign.sh          # output in out/m9_check_msalign (run.log; scratch weights in tmp*/)
#   REF=c57f500 tools/local/run_check_msalign.sh
#   OUT_DIR=out/... tools/local/run_check_msalign.sh   # another output directory
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
REF=${REF:-c57f500}
OUT=${OUT_DIR:-$ROOT/out/m9_check_msalign}
IMG=gcr.io/kaggle-gpu-images/python:latest
U="$(id -u):$(id -g)"
mkdir -p "$OUT/head"
git -C "$ROOT" show "$REF:src/casmi26/fpnet.py" > "$OUT/head/fpnet.py"
MOUNTS=(-v "$ROOT/src":/src:ro -v "$OUT/head":/head:ro -v "$ROOT/tools/smoke/check_msalign.py":/c.py:ro)
for s in casmi26-k01-cache casmi26-k05-pool casmi26-k05b-massfix casmi26-k26-fp2; do
  [ -d "$ROOT/out/m6_in/$s" ] || { echo "missing input out/m6_in/$s"; exit 1; }
  MOUNTS+=(-v "$ROOT/out/m6_in/$s":/kaggle/input/$s:ro)
done
docker run --rm --gpus all --network none --memory 16g --memory-swap 16g --shm-size 4g "${MOUNTS[@]}" -v "$OUT":/w \
  -e PYTHONUNBUFFERED=1 -e NUMBA_CACHE_DIR=/tmp/numba "$IMG" \
  bash -c "python /c.py > /w/run.log 2>&1; rc=\$?; chown -R $U /w; exit \$rc" || { tail -40 "$OUT/run.log"; exit 1; }
grep -v -i warn "$OUT/run.log" | grep -E "PASS|FAIL|speed|checks passed|window MRR" | tail -60

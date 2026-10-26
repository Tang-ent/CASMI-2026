#!/bin/bash
# Plan 9.2 checks of tools/smoke/check_g2.py (G2_REAL=1) on the real local inputs, in the Kaggle docker image on CPU:
# per-spectrum outputs, consensus scores, training-path cosine and the default outputs against a reference commit
# (REF, default c57f500 = the code before the 9.2 changes; its g2 / frag / spectra / search.py go to <out>/head).
#   tools/local/run_check_g2_real.sh            # output in out/m9_check_g2 (run.log)
#   N_PER=8 MAX_WIN=300 THREADS=8 REF=c57f500   # queries per kind, largest window, torch threads, reference commit
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
REF=${REF:-c57f500}
OUT=$ROOT/out/m9_check_g2
IMG=gcr.io/kaggle-gpu-images/python:latest
U="$(id -u):$(id -g)"
mkdir -p "$OUT/head"
for m in g2 frag spectra search; do git -C "$ROOT" show "$REF:src/casmi26/$m.py" > "$OUT/head/$m.py"; done
MOUNTS=(-v "$ROOT/src":/src:ro -v "$OUT/head":/head:ro -v "$ROOT/tools/smoke/check_g2.py":/c.py:ro
        -v "$ROOT/out/ds_wheels":/wheels:ro -v "$ROOT/out/m7_in/casmi26-g2":/kaggle/input/casmi26-g2:ro)
for s in casmi26-k01-cache casmi26-k05-pool casmi26-k05b-massfix casmi26-k21-frag; do
  [ -d "$ROOT/out/m6_in/$s" ] || { echo "missing input out/m6_in/$s"; exit 1; }
  MOUNTS+=(-v "$ROOT/out/m6_in/$s":/kaggle/input/$s:ro)
done
docker run --rm --network none --memory 16g --memory-swap 16g "${MOUNTS[@]}" -v "$OUT":/w \
  -e G2_REAL=1 -e N_PER="${N_PER:-8}" -e MAX_WIN="${MAX_WIN:-300}" -e THREADS="${THREADS:-8}" \
  -e PYTHONUNBUFFERED=1 -e NUMBA_CACHE_DIR=/tmp/numba "$IMG" \
  bash -c "pip install -q --no-index --find-links /wheels rdkit > /dev/null 2>&1; python /c.py > /w/run.log 2>&1; \
  rc=\$?; chown -R $U /w; exit \$rc" || { tail -40 "$OUT/run.log"; exit 1; }
grep -v -i warn "$OUT/run.log" | tail -30

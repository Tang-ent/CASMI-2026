#!/bin/bash
# casmi26.molenc checks (plan 10.2; tools/smoke/check_molenc.py) on the real k05 pool in the Kaggle docker image:
# canon_text on all pool SMILES (CPU, WORKERS processes), the tautomer pass on a sample, ChemBERTa token lengths,
# embedding throughput / correctness on the local GPU, fp_bits. ChemBERTa is mounted at the slug k49 will use.
#   tools/local/run_check_molenc.sh                 # output in out/m9_check_molenc (run.log, molenc_check.json)
#   LIMIT=20000 tools/local/run_check_molenc.sh     # only the first LIMIT pool rows (quick run)
#   WORKERS=14 N_SAMPLE=20000 N_SINGLE=1000 BATCH=256
#   OUT_DIR=out/... tools/local/run_check_molenc.sh   # another output directory
# The container runs detached; follow it with tail -f <out>/run.log (the log ends with "exit <rc>")
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${OUT_DIR:-$ROOT/out/m9_check_molenc$([ "${LIMIT:-0}" != 0 ] && echo _limit || true)}
IMG=gcr.io/kaggle-gpu-images/python:latest
U="$(id -u):$(id -g)"
NAME=${NAME:-casmi26-molenc-$(date +%m%d%H%M)}
mkdir -p "$OUT"
if [ -e "$OUT/run.log" ]; then i=1; while [ -e "$OUT/run.log.$i" ]; do i=$((i + 1)); done; mv "$OUT/run.log" "$OUT/run.log.$i"; fi
MOUNTS=(-v "$ROOT/src":/src:ro -v "$ROOT/tools/smoke/check_molenc.py":/c.py:ro -v "$ROOT/out/ds_wheels":/wheels:ro
        -v "$ROOT/out/m9_msalign_in/chemberta":/kaggle/input/casmi26-chemberta:ro)
for s in casmi26-k05-pool casmi26-k05b-massfix; do
  [ -d "$ROOT/out/m6_in/$s" ] || { echo "missing input out/m6_in/$s"; exit 1; }
  MOUNTS+=(-v "$ROOT/out/m6_in/$s":/kaggle/input/$s:ro)
done
docker run -d --name "$NAME" --gpus all --network none --memory 16g --memory-swap 16g --shm-size 4g \
  -e WORKERS="${WORKERS:-14}" -e N_SAMPLE="${N_SAMPLE:-20000}" -e N_SINGLE="${N_SINGLE:-1000}" \
  -e BATCH="${BATCH:-256}" -e LIMIT="${LIMIT:-0}" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_DISABLE_PROGRESS_BARS=1 \
  -e PYTHONUNBUFFERED=1 -e NUMBA_CACHE_DIR=/tmp/numba "${MOUNTS[@]}" -v "$OUT":/w "$IMG" \
  bash -c "pip install -q --no-index --find-links /wheels rdkit > /dev/null 2>&1; cd /w && python /c.py > run.log 2>&1; \
  rc=\$?; echo \"exit \$rc\" >> run.log; chown -R $U /w; exit \$rc"
echo "started $NAME -> $OUT/run.log"

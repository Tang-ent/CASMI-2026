#!/bin/bash
# Plan 10.2 checks of src/casmi26/dreams.py (tools/smoke/check_dreams.py) in the Kaggle docker image on the local GPU:
# (0) code-free loading of ssl_model.ckpt, (1) equivalence with the official DreaMS module of the clone dbec3a0,
# (2) SSL masked-peak self-check per polarity, (3) fp16 blocks vs fp32 cosine and throughput at batch 128.
#   tools/local/run_check_dreams.sh          # output in out/m9_check_dreams (run.log, check_dreams.json)
#   N_EQ=64 N_SSL=2000 BATCH=128              # equivalence spectra, SSL spectra per polarity, batch size
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$ROOT/out/m9_check_dreams
IMG=gcr.io/kaggle-gpu-images/python:latest
U="$(id -u):$(id -g)"
DR=$ROOT/out/m9_msalign_in/dreams
[ -f "$DR/ssl_model.ckpt" ] || { echo "missing $DR/ssl_model.ckpt"; exit 1; }
[ -d "$DR/repo_dbec3a0/dreams" ] || { echo "missing $DR/repo_dbec3a0"; exit 1; }
mkdir -p "$OUT"
MOUNTS=(-v "$ROOT/src":/src:ro -v "$ROOT/tools/smoke/check_dreams.py":/c.py:ro -v "$ROOT/out/ds_wheels":/wheels:ro
        -v "$DR/repo_dbec3a0":/repo:ro -v "$DR/ssl_model.ckpt":/ckpt/ssl_model.ckpt:ro)
for s in casmi26-k01-cache casmi26-k05-pool; do
  [ -d "$ROOT/out/m6_in/$s" ] || { echo "missing input out/m6_in/$s"; exit 1; }
  MOUNTS+=(-v "$ROOT/out/m6_in/$s":/kaggle/input/$s:ro)
done
docker run --rm --gpus all --network none --memory 20g --memory-swap 20g --shm-size 4g "${MOUNTS[@]}" -v "$OUT":/w \
  -e N_EQ="${N_EQ:-64}" -e N_SSL="${N_SSL:-2000}" -e BATCH="${BATCH:-128}" \
  -e PYTHONUNBUFFERED=1 -e NUMBA_CACHE_DIR=/tmp/numba -e MPLCONFIGDIR=/tmp/mpl -e WANDB_MODE=disabled "$IMG" \
  bash -c "pip install -q --no-index --find-links /wheels rdkit > /dev/null 2>&1; cd /tmp && python /c.py > /w/run.log 2>&1; \
  rc=\$?; chown -R $U /w; exit \$rc" || { grep -v -i warn "$OUT/run.log" | tail -40; exit 1; }
grep -v -i warn "$OUT/run.log" | tail -30

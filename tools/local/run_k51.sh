#!/bin/bash
# k51_eval10 on CPU in the Kaggle docker image (plan 10.3, 2026-10-24): k30 features + the msa columns of the k50
# variants that passed gates S and W vs the reference v12 (not pushed to Kaggle). Inputs as tools/local/run_k48.sh,
# with the k50 output as casmi26-k50-msalign. Runs in the foreground (--rm); the log is <out>/run.log.
#   tools/local/run_k51.sh            # full run on out/m9_k50, output in out/m9_k51
#   SMOKE=1 tools/local/run_k51.sh    # 20 queries per set, N_BOOT 200, on out/m9_k50_smoke, output in out/m9_k51_smoke;
#                                     # the four non-blind variants are forced in (FORCE_VARIANTS) to test the plumbing
#   K50=out/... tools/local/run_k51.sh  # another k50 output directory
#   OUT_DIR=out/... tools/local/run_k51.sh  # another output directory
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SMOKE=${SMOKE:-0}
SFX=$([ "$SMOKE" = 1 ] && echo _smoke || true)
K50=${K50:-$ROOT/out/m9_k50$SFX}
OUT=${OUT_DIR:-$ROOT/out/m9_k51$SFX}
IMG=gcr.io/kaggle-gpu-images/python:latest
U="$(id -u):$(id -g)"
NAME=${NAME:-casmi26-k51-$([ "$SMOKE" = 1 ] && echo smoke- || true)$(date +%m%d%H%M)}
[ -e "$K50/gates.json" ] || { echo "missing k50 output $K50/gates.json"; exit 1; }
mkdir -p "$OUT"
if [ -e "$OUT/run.log" ]; then i=1; while [ -e "$OUT/run.log.$i" ]; do i=$((i + 1)); done; mv "$OUT/run.log" "$OUT/run.log.$i"; fi
py=$OUT/k51.py
python3 "$ROOT/tools/build_kernel.py" k51_eval10 > /dev/null
python3 "$ROOT/tools/smoke/nb2py.py" "$ROOT/kaggle/k51_eval10/build/casmi26-k51-eval10.ipynb" "$py"
for pat in "^MAX_QUERIES = None " "^N_BOOT = 2000" "^FORCE_VARIANTS = None "; do
  grep -q "$pat" "$py" || { echo "k51 script is not at the Kaggle defaults: $pat"; exit 1; }
done
[ "$SMOKE" = 1 ] && sed -i 's/^N_BOOT = 2000/N_BOOT = 200/; s/^MAX_QUERIES = None /MAX_QUERIES = 20 /; s/^FORCE_VARIANTS = None /FORCE_VARIANTS = ["cb_mass", "cb_formula", "fp_mass", "fp_formula"] /' "$py"
# the k50 output holds model weights / training state the eval does not read: mount only gates.json and the msa tables
MOUNTS=(-v "$ROOT/out/m6_in/casmi26-k26-fp2":/kaggle/input/k26:ro -v "$ROOT/out/m6_in/casmi26-k05-pool":/kaggle/input/k05:ro
        -v "$ROOT/out/ds_mspred_flat":/kaggle/input/casmi26-mspred:ro -v "$K50/gates.json":/kaggle/input/casmi26-k50-msalign/gates.json:ro)
for f in "$K50"/msa_*_*.parquet; do
  MOUNTS+=(-v "$f":/kaggle/input/casmi26-k50-msalign/$(basename "$f"):ro)
done
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

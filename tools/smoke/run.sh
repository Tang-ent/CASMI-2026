#!/bin/bash
# Local smoke test of the M2 kernels (k07 -> k08 -> k09) in the Kaggle docker image.
#   tools/smoke/run.sh            # all steps; setup / model are skipped when their output already exists
#   tools/smoke/run.sh k08 k09    # only these steps
# Build the notebooks first (python tools/build_kernel.py k07_features2:3 k08_eval2 k09_submit2).
# Data and outputs go to out/smoke/ (gitignored). Kernel steps run with --network none; setup needs internet (rdkit).
# k07 runs with a wide mass window (4000 ppm) and max_cand_train=10 so that ranking and the R truncation are
# exercised on the tiny pool. The resulting MRRs are meaningless (mass_ppm alone finds the truth); this only
# checks that every kernel runs end to end and writes well-formed outputs.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
D=$ROOT/out/smoke
IMG=gcr.io/kaggle-gpu-images/python:latest
U="$(id -u):$(id -g)"
COMP=$ROOT/enveda-CASMI26-molecule-id-mass-spectra
STEPS=${*:-setup model k07 k08 k09}
mkdir -p "$D"

# run_kernel <label> <script> <outdir> <docker args...>: run a converted notebook, keep its log in <outdir>/run.log
run_kernel() {
  local label=$1 script=$2 out=$3
  shift 3
  mkdir -p "$out"
  echo "== $label"
  docker run --rm --network none -e KAGGLE_KERNEL_RUN_TYPE=Batch "$@" -v "$script":/run.py:ro \
    -v "$out":/kaggle/working "$IMG" \
    bash -c "cd /kaggle/working && python /run.py > run.log 2>&1; rc=\$?; chown -R $U /kaggle/working; exit \$rc" \
    || { tail -30 "$out/run.log"; exit 1; }
  grep -v -i warn "$out/run.log" | tail -"${TAIL:-15}"
}
nb() { python3 "$ROOT/tools/smoke/nb2py.py" "$1" "$2"; }
IN_DATA=(-v "$D/data/work":/kaggle/input/k01:ro -v "$D/data/pool":/kaggle/input/k05:ro
         -v "$D/data/k05b":/kaggle/input/k05b:ro -v "$D/model":/kaggle/input/fpnet:ro
         -v "$COMP":/kaggle/input/enveda-CASMI26-molecule-id-mass-spectra:ro)

for step in $STEPS; do
  case $step in
    setup)
      if [ -f "$D/data/pool/queries.csv" ]; then echo "== setup: exists"; continue; fi
      mkdir -p "$D/data"
      docker run --rm -v "$ROOT":/data:ro -v "$D/data":/smoke "$IMG" \
        bash -c "python /data/tools/smoke/setup_data.py 2>&1 | tail -1; chown -R $U /smoke" ;;
    model)
      if [ -f "$D/model/fpnet.pt" ]; then echo "== model: exists"; continue; fi
      mkdir -p "$D/model"
      docker run --rm --network none -v "$ROOT/src":/src:ro -v "$D/data":/d:ro \
        -v "$ROOT/tools/smoke/train_model.py":/t.py:ro -v "$D/model":/kaggle/working "$IMG" \
        bash -c "python /t.py 2>&1 | tail -1; chown -R $U /kaggle/working" ;;
    k07)
      for i in 0 1 2; do
        py=$D/k07_s$i.py
        nb "$ROOT/kaggle/k07_features2/build_s$i/casmi26-k07-features2-s$i.ipynb" "$py"
        sed -i 's/S = search.Searcher(C, P, model=MODEL)/S = search.Searcher(C, P, cfg=search.SearchCfg(ppm_win=4000, ppm_fallback=8000), model=MODEL)/; s/keep_rule=keep_rule/keep_rule=keep_rule, max_cand_train=10/g' "$py"
        grep -q "ppm_win=4000" "$py" && grep -q "max_cand_train=10" "$py" \
          || { echo "smoke patch did not apply to k07 (source changed?)"; exit 1; }
        TAIL=2 run_kernel "k07 shard $i" "$py" "$D/k07_s$i" "${IN_DATA[@]}"
      done ;;
    k08)
      nb "$ROOT/kaggle/k08_eval2/build/casmi26-k08-eval2.ipynb" "$D/k08.py"
      TAIL=${TAIL:-40} run_kernel k08 "$D/k08.py" "$D/k08" -v "$D/k07_s0":/kaggle/input/s0:ro \
        -v "$D/k07_s1":/kaggle/input/s1:ro -v "$D/k07_s2":/kaggle/input/s2:ro -v "$D/data/pool":/kaggle/input/k05:ro ;;
    k09)
      nb "$ROOT/kaggle/k09_submit2/build/casmi26-k09-submit2.ipynb" "$D/k09.py"
      TAIL=8 run_kernel k09 "$D/k09.py" "$D/k09" "${IN_DATA[@]}" -v "$D/k08":/kaggle/input/k08:ro
      python3 - "$D/k09/submission.csv" <<'PY'
import sys
import pandas as pd
s = pd.read_csv(sys.argv[1])
n = s.smiles.str.split(";").map(len)
dup = s.smiles.str.split(";").map(lambda x: len(x) - len(set(x))).sum()
print(f"submission: {len(s)} rows, unique ids {s.molecule_id.is_unique}, candidates {n.min()}-{n.max()}, dups {dup}")
assert len(s) == 400 and s.molecule_id.is_unique and n.max() <= 25 and dup == 0
PY
      ;;
    *) echo "unknown step $step"; exit 1 ;;
  esac
done

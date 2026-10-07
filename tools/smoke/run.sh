#!/bin/bash
# Local smoke test of the M2 kernels (k07 -> k08 -> k09) and the PubChem probe (k10) in the Kaggle docker image.
#   tools/smoke/run.sh            # all steps; setup / model are skipped when their output already exists
#   tools/smoke/run.sh k08 k09    # only these steps
#   tools/smoke/run.sh pubchem k10 # PubChem probe (pubchem builds out/smoke/pubchem from the heads of the real files)
#   tools/smoke/run.sh k05p; PC=1 tools/smoke/run.sh k07 k13     # yardstick: k07 v7 (levels 0 and 1) and k13
#   (k08 / k11 read the pre-v7 feature names feats_<set>_s<i> and no longer run on k07 v7 output)
# Build the notebooks first (python tools/build_kernel.py k07_features2:3 k08_eval2 k09_submit2 k10_pubchem_probe).
# Data and outputs go to out/smoke/ (gitignored). Kernel steps run with --network none; setup needs internet (rdkit).
# k07 runs with a wide mass window (4000 ppm) and a truncation cap of 10 so that ranking and the R truncation are
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
check_sub() {
  python3 - "$1" <<'PY'
import sys
import pandas as pd
s = pd.read_csv(sys.argv[1])
n = s.smiles.str.split(";").map(len)
dup = s.smiles.str.split(";").map(lambda x: len(x) - len(set(x))).sum()
print(f"submission: {len(s)} rows, unique ids {s.molecule_id.is_unique}, candidates {n.min()}-{n.max()}, dups {dup}")
assert len(s) == 400 and s.molecule_id.is_unique and n.max() <= 25 and dup == 0
PY
}
# rdkit wheel for offline `pip install rdkit` (k10 and setup_pubchem.py), fetched once
WHEELS=(-v "$D/wheels":/wheels:ro -e PIP_NO_INDEX=1 -e PIP_FIND_LINKS=/wheels)
wheels() {
  ls "$D"/wheels/rdkit-*.whl > /dev/null 2>&1 && return
  mkdir -p "$D/wheels"
  docker run --rm -v "$D/wheels":/wheels "$IMG" bash -c "pip download -q rdkit -d /wheels --no-deps; chown -R $U /wheels"
}
IN_DATA=(-v "$D/data/work":/kaggle/input/k01:ro -v "$D/data/pool":/kaggle/input/k05:ro
         -v "$D/data/k05b":/kaggle/input/k05b:ro -v "$D/model":/kaggle/input/fpnet:ro
         -v "$COMP":/kaggle/input/enveda-CASMI26-molecule-id-mass-spectra:ro)
# PC=1: k07 / k08 / k09 also see the smoke PubChem part (run the k05p step first)
[ "${PC:-0}" = 1 ] && IN_DATA+=(-v "$D/k05p/pubchem":/kaggle/input/k05p:ro)

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
        sed -i 's/S = search.Searcher(C, P, model=MODEL)/S = search.Searcher(C, P, cfg=search.SearchCfg(ppm_win=4000, ppm_fallback=8000), model=MODEL)/; s/^LEVELS = {0: 300, 1: 1000}/LEVELS = {0: 10, 1: 10}/' "$py"
        grep -q "ppm_win=4000" "$py" && grep -q "^LEVELS = {0: 10, 1: 10} " "$py" \
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
      check_sub "$D/k09/submission.csv" ;;
    k11)
      nb "$ROOT/kaggle/k11_eval3/build/casmi26-k11-eval3.ipynb" "$D/k11.py"
      TAIL=${TAIL:-40} run_kernel k11 "$D/k11.py" "$D/k11" -v "$D/k07_s0":/kaggle/input/s0:ro \
        -v "$D/k07_s1":/kaggle/input/s1:ro -v "$D/k07_s2":/kaggle/input/s2:ro -v "$D/data/pool":/kaggle/input/k05:ro \
        -v "$D/k10":/kaggle/input/k10:ro ;;
    k12)
      nb "$ROOT/kaggle/k12_submit3/build/casmi26-k12-submit3.ipynb" "$D/k12.py"
      # needs PC=1 (the PubChem part in IN_DATA); k12 stops if it is not attached
      TAIL=8 run_kernel k12 "$D/k12.py" "$D/k12" "${IN_DATA[@]}" -v "$D/k11":/kaggle/input/k11:ro
      check_sub "$D/k12/submission.csv" ;;
    pubchem)
      P=$D/pubchem
      if [ -f "$P/CID-Mass.gz" ]; then echo "== pubchem: exists"; continue; fi
      mkdir -p "$P/raw"
      B=https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras
      for f in CID-Mass CID-SMILES CID-InChI-Key CID-PMID CID-Synonym-filtered CID-Patent; do
        [ -s "$P/raw/$f.txt" ] || { curl -s "$B/$f.gz" | gunzip 2>/dev/null | head -30000 > "$P/raw/$f.txt" || true; }
      done
      wheels
      docker run --rm --network none "${WHEELS[@]}" -v "$ROOT":/data "$IMG" \
        bash -c "python /data/tools/smoke/setup_pubchem.py 2>&1 | tail -8; chown -R $U /data/out/smoke/pubchem" ;;
    k10)
      # offline: rdkit from out/smoke/wheels, PubChem files from out/smoke/pubchem (no download, checked below)
      wheels
      py=$D/k10.py
      nb "$ROOT/kaggle/k10_pubchem_probe/build/casmi26-k10-pubchem-probe.ipynb" "$py"
      sed -i 's/^N_SAMPLE = 3_000_000/N_SAMPLE = 3_000/; s/^N_POS = 300_000/N_POS = 300/; s/^N_SPEED = 20_000/N_SPEED = 200/; s/^BUDGETS = {"S": 1_500_000, "M": 5_000_000, "L": 15_000_000}/BUDGETS = {"S": 1_000, "M": 3_000, "L": 8_000}/' "$py"
      grep -q "^N_SAMPLE = 3_000 " "$py" && grep -q '^BUDGETS = {"S": 1_000,' "$py" \
        || { echo "smoke patch did not apply to k10 (source changed?)"; exit 1; }
      TAIL=${TAIL:-60} run_kernel k10 "$py" "$D/k10" "${WHEELS[@]}" -v "$D/data/work":/kaggle/input/k01:ro \
        -v "$D/data/pool":/kaggle/input/k05:ro -v "$D/data/k05b":/kaggle/input/k05b:ro -v "$D/pubchem":/kaggle/input/pubchem:ro
      if grep -qE "downloaded |zenodo" "$D/k10/run.log"; then echo "k10 smoke downloaded from the internet"; exit 1; fi
      ;;
    k05p)
      # offline: PubChem part from the smoke k10 output (pubchem_new.parquet with n_pmid >= 1) and the smoke CID-SMILES
      wheels
      nb "$ROOT/kaggle/k05p_pubchem/build/casmi26-k05p-pubchem.ipynb" "$D/k05p.py"
      TAIL=${TAIL:-20} run_kernel k05p "$D/k05p.py" "$D/k05p" "${WHEELS[@]}" -v "$D/data/pool":/kaggle/input/k05:ro \
        -v "$D/data/k05b":/kaggle/input/k05b:ro -v "$D/k10":/kaggle/input/k10:ro -v "$D/pubchem":/kaggle/input/pubchem:ro
      if grep -qE "downloaded " "$D/k05p/run.log"; then echo "k05p smoke downloaded from the internet"; exit 1; fi
      ;;
    k13)
      nb "$ROOT/kaggle/k13_yardstick/build/casmi26-k13-yardstick.ipynb" "$D/k13.py"
      sed -i 's/^N_BOOT = 2000/N_BOOT = 200/' "$D/k13.py"
      TAIL=${TAIL:-40} run_kernel k13 "$D/k13.py" "$D/k13" -v "$D/k07_s0":/kaggle/input/s0:ro \
        -v "$D/k07_s1":/kaggle/input/s1:ro -v "$D/k07_s2":/kaggle/input/s2:ro -v "$D/data/pool":/kaggle/input/k05:ro \
        -v "$D/k10":/kaggle/input/k10:ro ;;
    *) echo "unknown step $step"; exit 1 ;;
  esac
done

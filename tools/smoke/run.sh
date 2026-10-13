#!/bin/bash
# Local smoke test of the M2 kernels (k07 -> k08 -> k09) and the PubChem probe (k10) in the Kaggle docker image.
#   tools/smoke/run.sh            # all steps; setup / model are skipped when their output already exists
#   tools/smoke/run.sh k08 k09    # only these steps
#   tools/smoke/run.sh pubchem k10 # PubChem probe (pubchem builds out/smoke/pubchem from the heads of the real files)
#   tools/smoke/run.sh k05p; PC=1 tools/smoke/run.sh k07 k13     # yardstick: k07 v7 (levels 0 and 1) and k13
#   tools/smoke/run.sh k14 k15 k07 k16 k17   # M4: probe, FPNet v2 (CPU, 20 steps), k07 v8 features, eval, submit
#   (k07 v8 reads the k15 smoke output; k08 / k11 / k13 read older feature names and no longer run on it)
#   tools/smoke/run.sh k18 k19 k20           # M5 probes (after k14 k15 k07): isomer ceiling, generation, forward model
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

# GPU=1: give the containers the local GPU (exercises the CUDA paths: autocast, GradScaler, fp16 attention bias)
GPU_ARGS=()
[ "${GPU:-0}" = 1 ] && GPU_ARGS=(--gpus all)
# run_kernel <label> <script> <outdir> <docker args...>: run a converted notebook, keep its log in <outdir>/run.log
run_kernel() {
  local label=$1 script=$2 out=$3
  shift 3
  mkdir -p "$out"
  echo "== $label"
  docker run --rm --network none -e KAGGLE_KERNEL_RUN_TYPE=Batch "${GPU_ARGS[@]}" "$@" -v "$script":/run.py:ro \
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
      # k07 v8: FPNet v2 from the k15 smoke output (s0 = v2a; run the k14 and k15 steps first), level 0 only
      for i in 0 1 2; do
        py=$D/k07_s$i.py
        nb "$ROOT/kaggle/k07_features2/build_s$i/casmi26-k07-features2-s$i.ipynb" "$py"
        sed -i 's/S = search.Searcher(C, P, model=MODEL)/S = search.Searcher(C, P, cfg=search.SearchCfg(ppm_win=4000, ppm_fallback=8000), model=MODEL)/; s/^LEVELS = {0: 300}/LEVELS = {0: 10}/' "$py"
        grep -q "ppm_win=4000" "$py" && grep -q "^LEVELS = {0: 10} " "$py" \
          || { echo "smoke patch did not apply to k07 (source changed?)"; exit 1; }
        TAIL=2 run_kernel "k07 shard $i" "$py" "$D/k07_s$i" "${IN_DATA[@]}" -v "$D/k15_s0":/kaggle/input/k15:ro
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
    k14)
      # offline: rdkit from out/smoke/wheels; wide window so that the tiny pool gives several candidates
      wheels
      py=$D/k14.py
      nb "$ROOT/kaggle/k14_fpnet_probe/build/casmi26-k14-fpnet-probe.ipynb" "$py"
      sed -i 's/^PPM = 10.0/PPM = 4000.0/; s/^N_BOOT = 2000/N_BOOT = 200/' "$py"
      grep -q "^PPM = 4000.0" "$py" || { echo "smoke patch did not apply to k14 (source changed?)"; exit 1; }
      TAIL=${TAIL:-80} run_kernel k14 "$py" "$D/k14" "${WHEELS[@]}" -v "$D/data/work":/kaggle/input/k01:ro \
        -v "$D/data/pool":/kaggle/input/k05:ro -v "$D/data/k05b":/kaggle/input/k05b:ro -v "$D/model":/kaggle/input/fpnet:ro ;;
    k15)
      # CPU, 40 steps (2 epochs, so the last epoch passes the min_epoch=1 guard of k07 / k17); s0 = v2a, then
      # s1 = v2b (reads the s0 output); needs the k14 smoke output
      for i in 0 1; do
        py=$D/k15_s$i.py
        nb "$ROOT/kaggle/k15_fpnet2/build_s$i/casmi26-k15-fpnet2-s$i.ipynb" "$py"
        sed -i 's/^TOTAL_STEPS = None /TOTAL_STEPS = 40 /; s/^N_BOOT = 2000/N_BOOT = 200/' "$py"
        grep -q "^TOTAL_STEPS = 40 " "$py" || { echo "smoke patch did not apply to k15 (source changed?)"; exit 1; }
        extra=()
        [ $i = 1 ] && extra=(-v "$D/k15_s0":/kaggle/input/k15s0:ro)
        TAIL=${TAIL:-30} run_kernel "k15 s$i" "$py" "$D/k15_s$i" -v "$D/data/work":/kaggle/input/k01:ro \
          -v "$D/data/pool":/kaggle/input/k05:ro -v "$D/data/k05b":/kaggle/input/k05b:ro -v "$D/k14":/kaggle/input/k14:ro \
          "${extra[@]}"
      done ;;
    k16)
      # needs the k07 (L0 feature names) and k13 smoke outputs
      nb "$ROOT/kaggle/k16_eval4/build/casmi26-k16-eval4.ipynb" "$D/k16.py"
      sed -i 's/^N_BOOT = 2000/N_BOOT = 200/' "$D/k16.py"
      TAIL=${TAIL:-50} run_kernel k16 "$D/k16.py" "$D/k16" -v "$D/k07_s0":/kaggle/input/s0:ro \
        -v "$D/k07_s1":/kaggle/input/s1:ro -v "$D/k07_s2":/kaggle/input/s2:ro -v "$D/data/pool":/kaggle/input/k05:ro \
        -v "$D/k13":/kaggle/input/k13:ro -v "$D/k14":/kaggle/input/k14:ro -v "$D/k15_s0":/kaggle/input/k15:ro ;;
    k17)
      # needs the k15 (s0) and k16 smoke outputs; the smoke k16 chooses nothing, so force a routed variant with a mass cut
      nb "$ROOT/kaggle/k17_submit4/build/casmi26-k17-submit4.ipynb" "$D/k17.py"
      # k17 keeps an explicitly set VARIANT; otherwise force one (the smoke k16 chooses nothing)
      sed -i 's/^VARIANT = None /VARIANT = "route:soft:2|ppm8.5" /' "$D/k17.py"
      grep -q '^VARIANT = "' "$D/k17.py" || { echo "smoke patch did not apply to k17"; exit 1; }
      TAIL=8 run_kernel k17 "$D/k17.py" "$D/k17" "${IN_DATA[@]}" -v "$D/k15_s0":/kaggle/input/k15:ro \
        -v "$D/k16":/kaggle/input/k16:ro
      check_sub "$D/k17/submission.csv" ;;
    k18)
      # M5 isomer probe on the k07 smoke output (v2a FPNet, 4000 ppm windows); no k16 v2 reference check
      py=$D/k18.py
      nb "$ROOT/kaggle/k18_isomer_probe/build/casmi26-k18-isomer-probe.ipynb" "$py"
      sed -i 's/^N_BOOT = 2000/N_BOOT = 200/; s/^PPM = 10.0 /PPM = 4000.0 /; s/^EXPECT_VARIANT = "v2b"/EXPECT_VARIANT = "v2a"/; s/^REF_TOL = 0.005 /REF_TOL = None /' "$py"
      grep -q "^REF_TOL = None " "$py" && grep -q '^EXPECT_VARIANT = "v2a"' "$py" && grep -q "^PPM = 4000.0 " "$py" \
        || { echo "smoke patch did not apply to k18 (source changed?)"; exit 1; }
      TAIL=${TAIL:-70} run_kernel k18 "$py" "$D/k18" -v "$D/k07_s0":/kaggle/input/s0:ro -v "$D/k07_s1":/kaggle/input/s1:ro \
        -v "$D/k07_s2":/kaggle/input/s2:ro -v "$D/data/pool":/kaggle/input/k05:ro -v "$D/data/k05b":/kaggle/input/k05b:ro ;;
    k19)
      # M5 generation probe: offline rdkit from out/smoke/wheels, FPNet from the k15 smoke output (s0 = v2a), wide
      # search window on the tiny pool (the generation tolerance stays 10 ppm)
      wheels
      py=$D/k19.py
      nb "$ROOT/kaggle/k19_gen_probe/build/casmi26-k19-gen-probe.ipynb" "$py"
      sed -i 's/^N_BOOT = 2000/N_BOOT = 200/; s/^EXPECT_VARIANT = "v2b"/EXPECT_VARIANT = "v2a"/; s/^SEARCH_CFG = search.SearchCfg() /SEARCH_CFG = search.SearchCfg(ppm_win=4000, ppm_fallback=8000) /' "$py"
      grep -q '^EXPECT_VARIANT = "v2a"' "$py" && grep -q "^SEARCH_CFG = search.SearchCfg(ppm_win=4000" "$py" \
        || { echo "smoke patch did not apply to k19 (source changed?)"; exit 1; }
      TAIL=${TAIL:-60} run_kernel k19 "$py" "$D/k19" "${WHEELS[@]}" -v "$D/data/work":/kaggle/input/k01:ro \
        -v "$D/data/pool":/kaggle/input/k05:ro -v "$D/data/k05b":/kaggle/input/k05b:ro -v "$D/k15_s0":/kaggle/input/k15:ro ;;
    k20)
      # M5 G1: CPU, 2 epochs x 20 steps; wide window as in the k14 smoke (so the fz ranks differ from k15's 10 ppm
      # ones and the agreement check is off); needs the k14 and k15 (s0 = v2a) smoke outputs
      py=$D/k20.py
      nb "$ROOT/kaggle/k20_fwd/build/casmi26-k20-fwd.ipynb" "$py"
      sed -i 's/^MAX_STEPS = None /MAX_STEPS = 20 /; s/^PPM = 10.0/PPM = 4000.0/; s/^N_BOOT = 2000/N_BOOT = 200/; s/^FPNET_VARIANT = "v2b"/FPNET_VARIANT = "v2a"/; s/^N_VAL = 300 /N_VAL = 20 /; s/^MIN_AGREE = 0.98 /MIN_AGREE = 0.0 /' "$py"
      grep -q "^MAX_STEPS = 20 " "$py" && grep -q "^MIN_AGREE = 0.0 " "$py" && grep -q '^FPNET_VARIANT = "v2a"' "$py" \
        || { echo "smoke patch did not apply to k20 (source changed?)"; exit 1; }
      TAIL=${TAIL:-50} run_kernel k20 "$py" "$D/k20" -v "$D/data/work":/kaggle/input/k01:ro \
        -v "$D/data/pool":/kaggle/input/k05:ro -v "$D/data/k05b":/kaggle/input/k05b:ro -v "$D/k14":/kaggle/input/k14:ro \
        -v "$D/k15_s0":/kaggle/input/k15:ro ;;
    k21)
      # M5 P1 fragment table of the smoke pool (offline rdkit from out/smoke/wheels)
      wheels
      nb "$ROOT/kaggle/k21_frag/build/casmi26-k21-frag.ipynb" "$D/k21.py"
      TAIL=${TAIL:-8} run_kernel k21 "$D/k21.py" "$D/k21" "${WHEELS[@]}" -v "$D/data/pool":/kaggle/input/k05:ro ;;
    k22)
      # M5 P1 features: 4 shards, wide window and a cut of 10 as in the k07 smoke; needs the k15 (s0), k20, k21 and
      # k25 smoke outputs; rdkit from out/ds_wheels (the casmi26-wheels dataset); 20 public-test molecules for the timing
      for i in 0 1 2 3; do
        py=$D/k22_s$i.py
        nb "$ROOT/kaggle/k22_features3/build_s$i/casmi26-k22-features3-s$i.ipynb" "$py"
        sed -i 's/^SEARCH_CFG = search.SearchCfg() /SEARCH_CFG = search.SearchCfg(ppm_win=4000, ppm_fallback=8000) /; s/^MAX_CAND_TRAIN = 300/MAX_CAND_TRAIN = 10/; s/^N_TEST = 400 /N_TEST = 20 /; s/^REQUIRE_CUDA = True /REQUIRE_CUDA = False /' "$py"
        grep -q "^SEARCH_CFG = search.SearchCfg(ppm_win=4000" "$py" && grep -q "^MAX_CAND_TRAIN = 10" "$py" \
          || { echo "smoke patch did not apply to k22 (source changed?)"; exit 1; }
        TAIL=3 run_kernel "k22 shard $i" "$py" "$D/k22_s$i" "${IN_DATA[@]}" -v "$ROOT/out/ds_wheels":/kaggle/input/casmi26-wheels:ro \
          -v "$D/k15_s0":/kaggle/input/k15:ro \
          -v "$D/k20":/kaggle/input/k20:ro -v "$D/k21":/kaggle/input/k21:ro -v "$D/k25":/kaggle/input/k25:ro
      done ;;
    k30)
      # M6 P1 features: k22 + fprich / G2 columns, same smoke patches as k22 (wide window, cut of 10, CPU); the smoke
      # fprich / G2 models did not pass their gates (REQUIRE_GATES False); needs the k15 (s0), k20, k21, k25, k26, k28 and
      # k29 smoke outputs. Then the first 41 columns must be byte-identical to the k22 smoke output (same FPNet, CPU)
      for i in 0 1 2 3; do
        py=$D/k30_s$i.py
        nb "$ROOT/kaggle/k30_features4/build_s$i/casmi26-k30-features4-s$i.ipynb" "$py"
        sed -i 's/^SEARCH_CFG = search.SearchCfg() /SEARCH_CFG = search.SearchCfg(ppm_win=4000, ppm_fallback=8000) /; s/^MAX_CAND_TRAIN = 300/MAX_CAND_TRAIN = 10/; s/^N_TEST = 400 /N_TEST = 20 /; s/^REQUIRE_CUDA = True /REQUIRE_CUDA = False /; s/^REQUIRE_GATES = True /REQUIRE_GATES = False /; s/^GRAPH_WORKERS = 4/GRAPH_WORKERS = 2/' "$py"
        grep -q "^SEARCH_CFG = search.SearchCfg(ppm_win=4000" "$py" && grep -q "^MAX_CAND_TRAIN = 10" "$py" \
          && grep -q "^REQUIRE_GATES = False " "$py" || { echo "smoke patch did not apply to k30 (source changed?)"; exit 1; }
        TAIL=3 run_kernel "k30 shard $i" "$py" "$D/k30_s$i" "${IN_DATA[@]}" -v "$ROOT/out/ds_wheels":/kaggle/input/casmi26-wheels:ro \
          -v "$D/k15_s0":/kaggle/input/k15:ro -v "$D/k20":/kaggle/input/k20:ro -v "$D/k21":/kaggle/input/k21:ro \
          -v "$D/k25":/kaggle/input/k25:ro -v "$D/k26":/kaggle/input/k26:ro -v "$D/k28":/kaggle/input/k28:ro \
          -v "$D/k29":/kaggle/input/k29:ro
      done
      python3 - "$D" <<'PY'
import glob, os, sys
import numpy as np, pandas as pd
D = sys.argv[1]
n_files = n_bad = 0
for f4 in sorted(glob.glob(f"{D}/k30_s*/feats4_*.parquet")):
    f22 = f4.replace("/k30_s", "/k22_s").replace("feats4_", "feats_")
    a, b = pd.read_parquet(f4), pd.read_parquet(f22)
    same = a.shape[0] == b.shape[0] and all(np.array_equal(a[c].values, b[c].values, equal_nan=True) for c in b.columns)
    n_files += 1
    n_bad += not same
    if not same:
        print("differs:", os.path.basename(f4), [c for c in b.columns if not np.array_equal(a[c].values, b[c].values, equal_nan=True)][:8])
print(f"k30 vs k22 smoke: {n_files - n_bad}/{n_files} feature files byte-identical in the k22 columns")
assert n_files and not n_bad
PY
      ;;
    k31)
      # M6 P1 evaluation on the k30 smoke output (v2a FPNet); the k22 smoke output for the check; CPU
      nb "$ROOT/kaggle/k31_eval6/build/casmi26-k31-eval6.ipynb" "$D/k31.py"
      sed -i 's/^N_BOOT = 2000/N_BOOT = 200/; s/^EXPECT_VARIANT = "v2b"/EXPECT_VARIANT = "v2a"/' "$D/k31.py"
      grep -q '^EXPECT_VARIANT = "v2a"' "$D/k31.py" || { echo "smoke patch did not apply to k31"; exit 1; }
      M=(); for i in 0 1 2 3; do M+=(-v "$D/k30_s$i":/kaggle/input/k30s$i:ro -v "$D/k22_s$i":/kaggle/input/k22s$i:ro); done
      TAIL=${TAIL:-25} run_kernel k31 "$D/k31.py" "$D/k31" --memory 6g "${M[@]}" -v "$D/data/pool":/kaggle/input/k05:ro ;;
    k32)
      # M6 P1 submission: the k31 smoke keeps k24 v1, so CHOSEN forces fprich + G2 + generation (S3, 0.7, rk) to run
      # every path (the tiny smoke cache gates most molecules); CPU unless GPU=1; needs the k15 (s0), k20, k21, k25,
      # k26, k28, k29 and k31 smoke outputs
      py=$D/k32.py
      nb "$ROOT/kaggle/k32_submit6/build/casmi26-k32-submit6.ipynb" "$py"
      sed -i 's/^CHOSEN = None /CHOSEN = dict(kind="pool+gen", keep_k24=False, fcols="both_fprich_g2", variant="hybm:0.7:2", columns=search.FEATURES + frag.FRAG_FEATURES + search.FWD_FEATURES + search.FPRICH_FEATURES + search.G2_FEATURES, slots=list(range(3, 25, 3)), gate=0.7, score="rk", gen_columns=list(gen.GEN_FEATURES)) /; s/^REQUIRE_CUDA = True /REQUIRE_CUDA = False /; s/^GRAPH_WORKERS = 4/GRAPH_WORKERS = 2/' "$py"
      grep -q '^CHOSEN = dict(kind="pool+gen"' "$py" && grep -q "^REQUIRE_CUDA = False " "$py" || { echo "smoke patch did not apply to k32"; exit 1; }
      TAIL=${TAIL:-8} run_kernel k32 "$py" "$D/k32" --memory 6g "${IN_DATA[@]}" -v "$ROOT/out/ds_wheels":/kaggle/input/casmi26-wheels:ro \
        -v "$D/k15_s0":/kaggle/input/k15:ro -v "$D/k20":/kaggle/input/k20:ro -v "$D/k21":/kaggle/input/k21:ro \
        -v "$D/k25":/kaggle/input/k25:ro -v "$D/k26":/kaggle/input/k26:ro -v "$D/k28":/kaggle/input/k28:ro \
        -v "$D/k29":/kaggle/input/k29:ro -v "$D/k31":/kaggle/input/k31:ro
      check_sub "$D/k32/submission.csv" ;;
    k23)
      # M5 P1 evaluation on the k22 smoke output (v2a FPNet); k07 smoke output for the check; k13 smoke ranks as M2.2
      nb "$ROOT/kaggle/k23_eval5/build/casmi26-k23-eval5.ipynb" "$D/k23.py"
      sed -i 's/^N_BOOT = 2000/N_BOOT = 200/; s/^EXPECT_VARIANT = "v2b"/EXPECT_VARIANT = "v2a"/; s/^REF_TOL = 0.005 /REF_TOL = None /' "$D/k23.py"
      grep -q '^EXPECT_VARIANT = "v2a"' "$D/k23.py" || { echo "smoke patch did not apply to k23"; exit 1; }
      TAIL=${TAIL:-60} run_kernel k23 "$D/k23.py" "$D/k23" -v "$D/k22_s0":/kaggle/input/k22s0:ro \
        -v "$D/k22_s1":/kaggle/input/k22s1:ro -v "$D/k22_s2":/kaggle/input/k22s2:ro -v "$D/k22_s3":/kaggle/input/k22s3:ro \
        -v "$D/k07_s0":/kaggle/input/s0:ro -v "$D/k07_s1":/kaggle/input/s1:ro -v "$D/k07_s2":/kaggle/input/s2:ro \
        -v "$D/data/pool":/kaggle/input/k05:ro -v "$D/k13":/kaggle/input/k13:ro ;;
    k24)
      # M5 P1 submission: the k23 smoke choice (M2.2, no generation), then forced generation on the M2.2 base
      # (gen ranker) and on a new pool base (comb); rdkit from out/smoke/wheels as the casmi26-wheels dataset
      nb "$ROOT/kaggle/k24_submit5/build/casmi26-k24-submit5.ipynb" "$D/k24.py"
      for v in none m22gen newgen; do
        py=$D/k24_$v.py
        cp "$D/k24.py" "$py"
        case $v in
          m22gen) sed -i 's/^CHOSEN = None /CHOSEN = dict(kind="gen", base="M2.2", slots=[3, 6, 9, 12, 15, 18, 21, 24], gate=0.7, score="rk") /' "$py" ;;
          newgen) sed -i 's/^CHOSEN = None /CHOSEN = dict(kind="gen", base="both|hybm:0.7:2", fcols="both", variant="hybm:0.7:2", slots=[5, 10, 15, 20, 25], gate=0.7, score="comb") /' "$py" ;;
        esac
        [ $v = none ] || grep -q '^CHOSEN = dict(' "$py" || { echo "smoke patch did not apply to k24"; exit 1; }
        TAIL=4 run_kernel "k24 $v" "$py" "$D/k24_$v" "${IN_DATA[@]}" -v "$ROOT/out/ds_wheels":/kaggle/input/casmi26-wheels:ro \
          -v "$D/k08":/kaggle/input/k08:ro -v "$D/k15_s0":/kaggle/input/k15:ro -v "$D/k20":/kaggle/input/k20:ro \
          -v "$D/k21":/kaggle/input/k21:ro -v "$D/k23":/kaggle/input/k23:ro -v "$D/k25":/kaggle/input/k25:ro
        check_sub "$D/k24_$v/submission.csv"
      done ;;
    k25)
      # M5 P1 fwd v2: CPU, 20 steps x 16 spectra from the k20 smoke fwd.pt; wide window as in k20 (agreement off)
      py=$D/k25.py
      nb "$ROOT/kaggle/k25_fwd2/build/casmi26-k25-fwd2.ipynb" "$py"
      sed -i 's/^STEPS, BATCH, K_NEG, TEMP, LR = 30000, 128,/STEPS, BATCH, K_NEG, TEMP, LR = 20, 16,/; s/^PPM = 10.0/PPM = 4000.0/; s/^N_BOOT = 2000/N_BOOT = 200/; s/^FPNET_VARIANT = "v2b"/FPNET_VARIANT = "v2a"/; s/^N_VAL = 300/N_VAL = 20/; s/^MIN_AGREE = 0.98 /MIN_AGREE = 0.0 /' "$py"
      grep -q "^STEPS, BATCH, K_NEG, TEMP, LR = 20, 16," "$py" && grep -q "^MIN_AGREE = 0.0 " "$py" \
        || { echo "smoke patch did not apply to k25 (source changed?)"; exit 1; }
      TAIL=${TAIL:-40} run_kernel k25 "$py" "$D/k25" -v "$D/data/work":/kaggle/input/k01:ro \
        -v "$D/data/pool":/kaggle/input/k05:ro -v "$D/data/k05b":/kaggle/input/k05b:ro -v "$D/k14":/kaggle/input/k14:ro \
        -v "$D/k15_s0":/kaggle/input/k15:ro -v "$D/k20":/kaggle/input/k20:ro ;;
    subform)
      # M6: numba subformula annotation == the naive enumeration on 300 smoke spectra
      docker run --rm --network none -v "$ROOT/src":/src:ro -v "$D/data":/d:ro -v "$ROOT/tools/smoke/check_subform.py":/c.py:ro \
        -e NUMBA_CACHE_DIR=/tmp/numba "$IMG" bash -c "python /c.py 2>&1 | grep -v -i warn | tail -5" ;;
    resume)
      # training state / resume (ckpt): fpnet and enc3 resumed after epoch 0 == straight, decoder keeps its schedule;
      # CPU, needs the k26 smoke output; rdkit (decoder SMILES) from out/ds_wheels
      mkdir -p "$D/resume"
      docker run --rm --network none --memory 6g -v "$ROOT/src":/src:ro -v "$D/data":/d:ro -v "$D/k26":/k26:ro \
        -v "$D/resume":/w -v "$ROOT/tools/smoke/check_resume.py":/c.py:ro -v "$ROOT/out/ds_wheels":/wheels:ro \
        -e NUMBA_CACHE_DIR=/tmp/numba "$IMG" bash -c "pip install -q --no-index --find-links /wheels rdkit > /dev/null 2>&1; \
        python /c.py > /w/run.log 2>&1; rc=\$?; chown -R $U /w; exit \$rc" || { tail -30 "$D/resume/run.log"; exit 1; }
      grep -v -i warn "$D/resume/run.log" | tail -8 ;;
    g2)
      # M6 P0-3 G2: fragment masses == frag.py, cosine unit checks, one batch, resume == straight (CPU, needs k26)
      mkdir -p "$D/g2"
      docker run --rm --network none --memory 6g -v "$ROOT/src":/src:ro -v "$D/data":/d:ro -v "$D/k26":/k26:ro \
        -v "$D/g2":/w -v "$ROOT/tools/smoke/check_g2.py":/c.py:ro -v "$ROOT/out/ds_wheels":/wheels:ro \
        -e NUMBA_CACHE_DIR=/tmp/numba "$IMG" bash -c "pip install -q --no-index --find-links /wheels rdkit > /dev/null 2>&1; \
        python /c.py > /w/run.log 2>&1; rc=\$?; chown -R $U /w; exit \$rc" || { tail -30 "$D/g2/run.log"; exit 1; }
      grep -v -i warn "$D/g2/run.log" | tail -12 ;;
    k29)
      # M6 P0-3 G2 kernel: 30 steps of 4 spectra x (1 + 4), wide window, 40 queries (agreement off); needs the k14,
      # k15 (s1 = v2b), k20 and k26 smoke outputs; rdkit from out/ds_wheels
      py=$D/k29.py
      nb "$ROOT/kaggle/k29_g2/build/casmi26-k29-g2.ipynb" "$py"
      sed -i 's/^TOTAL_STEPS = None /TOTAL_STEPS = 30 /; s/^BATCH, K_NEG, WORKERS = None, None, None /BATCH, K_NEG, WORKERS = 4, 4, 2 /; s/^FIT = None /FIT = (5, 10) /; s/^PPM = 10.0/PPM = 4000.0/; s/^N_BOOT = 2000/N_BOOT = 200/; s/^MIN_AGREE = 0.98 /MIN_AGREE = 0.0 /; s/^MAX_QUERIES = None /MAX_QUERIES = 40 /; s/^EVAL_PROBE = 30 /EVAL_PROBE = 3 /; s/^GRAPH_WORKERS = 4/GRAPH_WORKERS = 2/' "$py"
      grep -q "^TOTAL_STEPS = 30 " "$py" && grep -q "^MIN_AGREE = 0.0 " "$py" && grep -q "^MAX_QUERIES = 40 " "$py" \
        && grep -q "^BATCH, K_NEG, WORKERS = 4, 4, 2 " "$py" || { echo "smoke patch did not apply to k29"; exit 1; }
      TAIL=${TAIL:-30} run_kernel k29 "$py" "$D/k29" --memory 6g -e PYTHONUNBUFFERED=1 \
        -v "$ROOT/out/ds_wheels":/kaggle/input/casmi26-wheels:ro -v "$D/data/work":/kaggle/input/k01:ro \
        -v "$D/data/pool":/kaggle/input/k05:ro -v "$D/data/k05b":/kaggle/input/k05b:ro -v "$D/k14":/kaggle/input/k14:ro \
        -v "$D/k15_s1":/kaggle/input/k15:ro -v "$D/k20":/kaggle/input/k20:ro -v "$D/k26":/kaggle/input/k26:ro ;;
    k37)
      # M7 P1 G2 v2: 30 steps of 4 spectra x (1 + 4), monitor every 10, wide window, 40 queries (agreement off for
      # k15 / k20); the k29 smoke g2.pt as v1 (comb3(v1) must reproduce the k29 smoke ranks); needs the k14, k15 (s1),
      # k20, k26 and k29 smoke outputs; rdkit from out/ds_wheels
      py=$D/k37.py
      nb "$ROOT/kaggle/k37_g2v2/build/casmi26-k37-g2v2.ipynb" "$py"
      sed -i 's/^TOTAL_STEPS = 180000 /TOTAL_STEPS = 30 /; s/^BATCH, K_NEG, WORKERS = None, None, None /BATCH, K_NEG, WORKERS = 4, 4, 2 /; s/^FIT = None /FIT = (5, 10) /; s/^PPM = 10.0/PPM = 4000.0/; s/^N_BOOT = 2000/N_BOOT = 200/; s/^MIN_AGREE = 0.98 /MIN_AGREE = 0.0 /; s/^MAX_QUERIES = None /MAX_QUERIES = 40 /; s/^MONITOR_EVERY = 20000 /MONITOR_EVERY = 10 /; s/^GRAPH_WORKERS = 4/GRAPH_WORKERS = 2/' "$py"
      grep -q "^TOTAL_STEPS = 30 " "$py" && grep -q "^MIN_AGREE = 0.0 " "$py" && grep -q "^MAX_QUERIES = 40 " "$py" \
        && grep -q "^MONITOR_EVERY = 10 " "$py" || { echo "smoke patch did not apply to k37"; exit 1; }
      TAIL=${TAIL:-30} run_kernel k37 "$py" "$D/k37" --memory 6g -e PYTHONUNBUFFERED=1 \
        -v "$ROOT/out/ds_wheels":/kaggle/input/casmi26-wheels:ro -v "$D/data/work":/kaggle/input/k01:ro \
        -v "$D/data/pool":/kaggle/input/k05:ro -v "$D/data/k05b":/kaggle/input/k05b:ro -v "$D/k14":/kaggle/input/k14:ro \
        -v "$D/k15_s1":/kaggle/input/k15:ro -v "$D/k20":/kaggle/input/k20:ro -v "$D/k26":/kaggle/input/k26:ro \
        -v "$D/k29":/kaggle/input/k29:ro ;;
    k35)
      # M7 evaluation on the k30 smoke output (v2a FPNet) with the k26 smoke formulas; CPU
      nb "$ROOT/kaggle/k35_eval7/build/casmi26-k35-eval7.ipynb" "$D/k35.py"
      sed -i 's/^N_BOOT = 2000/N_BOOT = 200/; s/^EXPECT_VARIANT = "v2b"/EXPECT_VARIANT = "v2a"/' "$D/k35.py"
      grep -q '^EXPECT_VARIANT = "v2a"' "$D/k35.py" || { echo "smoke patch did not apply to k35"; exit 1; }
      M=(); for i in 0 1 2 3; do M+=(-v "$D/k30_s$i":/kaggle/input/k30s$i:ro); done
      TAIL=${TAIL:-30} run_kernel k35 "$D/k35.py" "$D/k35" --memory 6g "${M[@]}" -v "$D/data/pool":/kaggle/input/k05:ro \
        -v "$D/k26":/kaggle/input/k26:ro ;;
    k36)
      # M7 submission by variant name (k35 smoke rankers, k31's for the "both" columns): (1) the k32 smoke variant -> submission_raw.csv must equal the
      # k32 smoke submission (existing path unchanged), (2) hybb without generation, (3) hybi with S2; CPU unless GPU=1;
      # needs the k15 (s0), k20, k21, k25, k26, k28, k29, k31, k32 and k35 smoke outputs
      nb "$ROOT/kaggle/k36_submit7/build/casmi26-k36-submit7.ipynb" "$D/k36.py"
      n=0
      for v in "both_fprich_g2|hybm:0.7:2+gen:S3:0.7:rk" "both|hybb:0.7:0.8:5:10+nogen" "both_fprich_g2|hybi:0.7:0.8:5+gen:S2:0.7:rk"; do
        n=$((n + 1)); py=$D/k36_$n.py
        cp "$D/k36.py" "$py"
        sed -i "s/^VARIANT = .*/VARIANT = \"$v\"/; s/^REQUIRE_CUDA = True /REQUIRE_CUDA = False /; s/^GRAPH_WORKERS = 4/GRAPH_WORKERS = 2/" "$py"
        grep -qF "VARIANT = \"$v\"" "$py" && grep -q "^REQUIRE_CUDA = False " "$py" || { echo "smoke patch did not apply to k36"; exit 1; }
        TAIL=${TAIL:-4} run_kernel "k36 $v" "$py" "$D/k36_$n" --memory 6g "${IN_DATA[@]}" -v "$ROOT/out/ds_wheels":/kaggle/input/casmi26-wheels:ro \
          -v "$D/k15_s0":/kaggle/input/k15:ro -v "$D/k20":/kaggle/input/k20:ro -v "$D/k21":/kaggle/input/k21:ro \
          -v "$D/k25":/kaggle/input/k25:ro -v "$D/k26":/kaggle/input/k26:ro -v "$D/k28":/kaggle/input/k28:ro \
          -v "$D/k29":/kaggle/input/k29:ro -v "$D/k31":/kaggle/input/k31:ro -v "$D/k35":/kaggle/input/k35:ro
        check_sub "$D/k36_$n/submission.csv"
      done
      cmp "$D/k36_1/submission_raw.csv" "$D/k32/submission.csv" && echo "k36 (k32 variant) raw submission == k32 smoke submission"
      # (4) the k35 smoke rankers: same-formula columns, the lt08 ranker, interleave, GEN_EXTRA generation ranker
      py=$D/k36_4.py; v="both_fprich_g2_grp.lt08|hybi:0.7:0.8:5+gen:S3:0.7:rkx"
      cp "$D/k36.py" "$py"
      sed -i "s/^VARIANT = .*/VARIANT = \"$v\"/; s/^REQUIRE_CUDA = True /REQUIRE_CUDA = False /; s/^GRAPH_WORKERS = 4/GRAPH_WORKERS = 2/" "$py"
      TAIL=${TAIL:-4} run_kernel "k36 $v" "$py" "$D/k36_4" --memory 6g "${IN_DATA[@]}" -v "$ROOT/out/ds_wheels":/kaggle/input/casmi26-wheels:ro \
        -v "$D/k15_s0":/kaggle/input/k15:ro -v "$D/k20":/kaggle/input/k20:ro -v "$D/k21":/kaggle/input/k21:ro \
        -v "$D/k25":/kaggle/input/k25:ro -v "$D/k26":/kaggle/input/k26:ro -v "$D/k28":/kaggle/input/k28:ro \
        -v "$D/k29":/kaggle/input/k29:ro -v "$D/k31":/kaggle/input/k31:ro -v "$D/k35":/kaggle/input/k35:ro
      check_sub "$D/k36_4/submission.csv"
      python3 -c "import json; [print(n, (lambda j: (j['eval'], j['stats'], j['local']))(json.load(open(f'$D/k36_{n}/variant.json')))) for n in (1, 2, 3, 4)]" ;;
    k39)
      # M7 P1 fprich warm restart: CPU, batch 32, 40 steps from the k28 smoke fprich.pt (its ranks must reproduce the k28
      # smoke ranks), wide window; needs the k14, k15 (s1), k26 and k28 smoke outputs
      py=$D/k39.py
      nb "$ROOT/kaggle/k39_fprich2/build/casmi26-k39-fprich2.ipynb" "$py"
      sed -i 's/^TOTAL_STEPS = None /TOTAL_STEPS = 40 /; s/^PPM = 10.0/PPM = 4000.0/; s/^N_BOOT = 2000/N_BOOT = 200/; s/^MIN_AGREE = 0.98 /MIN_AGREE = 0.0 /; s/^BATCH = None /BATCH = 32 /' "$py"
      grep -q "^TOTAL_STEPS = 40 " "$py" && grep -q "^MIN_AGREE = 0.0 " "$py" && grep -q "^BATCH = 32 " "$py" || { echo "smoke patch did not apply to k39"; exit 1; }
      TAIL=${TAIL:-25} run_kernel k39 "$py" "$D/k39" --memory 6g -e PYTHONUNBUFFERED=1 -v "$D/data/work":/kaggle/input/k01:ro \
        -v "$D/data/pool":/kaggle/input/k05:ro -v "$D/data/k05b":/kaggle/input/k05b:ro -v "$D/k14":/kaggle/input/k14:ro \
        -v "$D/k15_s1":/kaggle/input/k15:ro -v "$D/k26":/kaggle/input/k26:ro -v "$D/k28":/kaggle/input/k28:ro
      grep -E "agreement|warm start|gate E2" "$D/k39/run.log" ;;
    k26)
      # M6 pool formulas + rich fingerprints of the smoke pool (rdkit from out/ds_wheels as the casmi26-wheels dataset)
      nb "$ROOT/kaggle/k26_fp2/build/casmi26-k26-fp2.ipynb" "$D/k26.py"
      TAIL=${TAIL:-25} run_kernel k26 "$D/k26.py" "$D/k26" -v "$ROOT/out/ds_wheels":/kaggle/input/casmi26-wheels:ro \
        -v "$D/data/work":/kaggle/input/k01:ro -v "$D/data/pool":/kaggle/input/k05:ro -v "$D/data/k05b":/kaggle/input/k05b:ro ;;
    k27)
      # M6 P0-1 enc3: CPU, 2 epochs x 20 steps of 32 spectra; wide window (so the v2b agreement check is off); needs the
      # k14, k15 (s1 = v2b) and k26 smoke outputs
      py=$D/k27.py
      nb "$ROOT/kaggle/k27_enc3/build/casmi26-k27-enc3.ipynb" "$py"
      sed -i 's/^TOTAL_STEPS = None /TOTAL_STEPS = 1600 /; s/^BATCH = None /BATCH = 32 /; s/^MAX_STEPS_PER_EPOCH = None /MAX_STEPS_PER_EPOCH = 20 /; s/^FIT = None /FIT = (5, 10) /; s/^PPM = 10.0/PPM = 4000.0/; s/^N_BOOT = 2000/N_BOOT = 200/; s/^N_VAL = 300 /N_VAL = 20 /; s/^MIN_AGREE = 0.98 /MIN_AGREE = 0.0 /' "$py"
      grep -q "^MAX_STEPS_PER_EPOCH = 20 " "$py" && grep -q "^MIN_AGREE = 0.0 " "$py" && grep -q "^PPM = 4000.0" "$py" \
        || { echo "smoke patch did not apply to k27 (source changed?)"; exit 1; }
      TAIL=${TAIL:-40} run_kernel k27 "$py" "$D/k27" -v "$D/data/work":/kaggle/input/k01:ro \
        -v "$D/data/pool":/kaggle/input/k05:ro -v "$D/data/k05b":/kaggle/input/k05b:ro -v "$D/k14":/kaggle/input/k14:ro \
        -v "$D/k15_s1":/kaggle/input/k15:ro -v "$D/k26":/kaggle/input/k26:ro ;;
    k28)
      # M6 P0-2 fprich: CPU, batch 32 (memory), one epoch (40 steps -> ceil to whole epochs); wide window (agreement off); needs the k14,
      # k15 (s1 = v2b) and k26 smoke outputs
      py=$D/k28.py
      nb "$ROOT/kaggle/k28_fprich/build/casmi26-k28-fprich.ipynb" "$py"
      sed -i 's/^TOTAL_STEPS = None /TOTAL_STEPS = 40 /; s/^PPM = 10.0/PPM = 4000.0/; s/^N_BOOT = 2000/N_BOOT = 200/; s/^MIN_AGREE = 0.98 /MIN_AGREE = 0.0 /; s/^BATCH = None /BATCH = 32 /' "$py"
      grep -q "^TOTAL_STEPS = 40 " "$py" && grep -q "^MIN_AGREE = 0.0 " "$py" && grep -q "^BATCH = 32 " "$py" || { echo "smoke patch did not apply to k28"; exit 1; }
      TAIL=${TAIL:-30} run_kernel k28 "$py" "$D/k28" --memory 6g -e PYTHONUNBUFFERED=1 -v "$D/data/work":/kaggle/input/k01:ro \
        -v "$D/data/pool":/kaggle/input/k05:ro -v "$D/data/k05b":/kaggle/input/k05b:ro -v "$D/k14":/kaggle/input/k14:ro \
        -v "$D/k15_s1":/kaggle/input/k15:ro -v "$D/k26":/kaggle/input/k26:ro ;;
    k33)
      # M6 P0-4 decoder training: CPU, 30 steps of 16; a few report queries; needs the k15 (s1 = v2b) and k26 smoke
      # outputs; rdkit from out/ds_wheels
      py=$D/k33.py
      nb "$ROOT/kaggle/k33_decoder/build/casmi26-k33-decoder.ipynb" "$py"
      sed -i 's/^TOTAL_STEPS = None /TOTAL_STEPS = 30 /; s/^FIT = None /FIT = (5, 10) /; s/^BATCH = None/BATCH = 16/; s/^N_VAL = 2000/N_VAL = 20/; s/^N_REPORT, N_REPORT_SAMPLES = 300, 128/N_REPORT, N_REPORT_SAMPLES = 4, 8/' "$py"
      grep -q "^TOTAL_STEPS = 30 " "$py" && grep -q "^N_REPORT, N_REPORT_SAMPLES = 4, 8" "$py" || { echo "smoke patch did not apply to k33"; exit 1; }
      TAIL=${TAIL:-25} run_kernel k33 "$py" "$D/k33" -v "$ROOT/out/ds_wheels":/kaggle/input/casmi26-wheels:ro \
        -v "$D/data/work":/kaggle/input/k01:ro -v "$D/data/pool":/kaggle/input/k05:ro -v "$D/data/k05b":/kaggle/input/k05b:ro \
        -v "$D/k15_s1":/kaggle/input/k15:ro -v "$D/k26":/kaggle/input/k26:ro ;;
    k34)
      # M6 P0-4 gate D on 4 queries per group, 64 samples, wide search window, the truth injected into the raw decoder
      # output (exercises the product path; the smoke decoder is untrained); needs the k15 (s1), k26 and k33 smoke outputs
      py=$D/k34.py
      nb "$ROOT/kaggle/k34_dec_eval/build/casmi26-k34-dec-eval.ipynb" "$py"
      sed -i 's/^N_BOOT = 2000/N_BOOT = 200/; s/^N_SAMPLES, MAX_FORMULAS, MIN_PER = 512, 8, 32/N_SAMPLES, MAX_FORMULAS, MIN_PER = 64, 8, 8/; s/^MAX_QUERIES = None /MAX_QUERIES = 4 /; s/^INJECT_TRUTH = False /INJECT_TRUTH = True /; s/^SEARCH_CFG = search.SearchCfg() /SEARCH_CFG = search.SearchCfg(ppm_win=4000, ppm_fallback=8000) /' "$py"
      grep -q "^MAX_QUERIES = 4 " "$py" && grep -q "^INJECT_TRUTH = True " "$py" && grep -q "^SEARCH_CFG = search.SearchCfg(ppm_win=4000" "$py" \
        || { echo "smoke patch did not apply to k34"; exit 1; }
      TAIL=${TAIL:-25} run_kernel k34 "$py" "$D/k34" -v "$ROOT/out/ds_wheels":/kaggle/input/casmi26-wheels:ro \
        -v "$D/data/work":/kaggle/input/k01:ro -v "$D/data/pool":/kaggle/input/k05:ro -v "$D/data/k05b":/kaggle/input/k05b:ro \
        -v "$D/k15_s1":/kaggle/input/k15:ro -v "$D/k26":/kaggle/input/k26:ro -v "$D/k33":/kaggle/input/k33:ro ;;
    *) echo "unknown step $step"; exit 1 ;;
  esac
done

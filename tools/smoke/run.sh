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
#   tools/smoke/run.sh metricid              # metric-identity audit of the real M2.2 / M4 submissions (no kernel)
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
    metricid)
      # keys with the scorer's RDKit 2026.03.3 (out/metric_wheels = metric/rdkit-2026-3-3-wheel) and with our kernels'
      # 2026.09.1 (out/ds_wheels), then the duplicate / disagreement report
      O=$D/metricid
      mkdir -p "$O"
      SUBS="/data/out/m22_k09/submission.csv /data/out/m4_k17/submission.csv"
      for w in metric_wheels ds_wheels; do
        echo "== metricid keys ($w)"
        docker run --rm --network none -e PIP_NO_INDEX=1 -e PIP_FIND_LINKS=/wheels -v "$ROOT/out/$w":/wheels:ro \
          -v "$ROOT":/data:ro -v "$O":/o "$IMG" \
          bash -c "python /data/tools/smoke/metric_identity.py keys /o $SUBS; rc=\$?; chown -R $U /o; exit \$rc"
      done
      echo "== metricid report"
      docker run --rm --network none -v "$ROOT":/data:ro -v "$O":/o "$IMG" \
        python /data/tools/smoke/metric_identity.py report /o $SUBS | tee "$O/report.txt" ;;
    *) echo "unknown step $step"; exit 1 ;;
  esac
done

#!/bin/bash
# M8 v4 (2026-10-20): run own_rescorer.py (tools/build_fuse.py: k46_rescore + embedded casmi26) on the local GPU in the
# Kaggle docker image, the way the fusion notebook runs it, on the candidates of a fetched run's e004_candidate_inputs.json
# (prefusion ∪ engine2 per molecule, as the notebook's rescore cell builds them), then check the output format:
# every input SMILES string is a key, fzr finite for every parseable candidate, g2 coverage, no extra keys.
#   tools/local/run_rescore.sh [fetched run dir, default out/m8_fuse/base_v1] [out dir, default out/m8_rescore_check]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
RUN=$(cd "${1:-$ROOT/out/m8_fuse/base_v1}" && pwd); OUT=${2:-$ROOT/out/m8_rescore_check}
IMG=gcr.io/kaggle-gpu-images/python:latest
U="$(id -u):$(id -g)"
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
python3 "$ROOT/tools/build_fuse.py" > /dev/null
cp "$ROOT/kaggle/k45_fuse/build/own_rescorer.py" "$OUT/own_rescorer.py"
python3 - "$RUN/e004_candidate_inputs.json" "$OUT/own_candidates.json" <<'PY'
import json, sys
inp = json.load(open(sys.argv[1]))
cand = {}
for mid, first in inp["prefusion"].items():
    u = [s for s in list(first) + list((inp["engine2"].get(mid) or {}).get("smiles") or []) if s and s != "CCO"]
    if u:
        cand[mid] = list(dict.fromkeys(u))
json.dump(cand, open(sys.argv[2], "w"))
print(len(cand), "molecules,", sum(map(len, cand.values())), "candidates")
PY
M=(-v "$ROOT/enveda-CASMI26-molecule-id-mass-spectra":/comp:ro -v "$OUT":/out -v "$OUT/own_rescorer.py":/run/own_rescorer.py:ro)
for s in casmi26-k01-cache casmi26-k05-pool casmi26-k05b-massfix casmi26-k15-fpnet2-s1 casmi26-k20-fwd casmi26-k21-frag \
         casmi26-k25-fwd2 casmi26-k26-fp2; do
  M+=(-v "$ROOT/out/m6_in/$s":/bundle/$s:ro)
done
M+=(-v "$ROOT/out/m7_in/casmi26-fprich":/bundle/casmi26-fprich:ro -v "$ROOT/out/m7_in/casmi26-g2":/bundle/casmi26-g2:ro
    -v "$ROOT/out/p3_in/casmi26-k35-eval7":/bundle/casmi26-k35-eval7:ro -v "$ROOT/out/ds_wheels":/bundle/casmi26-wheels:ro)
docker run --rm --name "casmi26-rescore-$(date +%m%d%H%M%S)" --gpus all --network none --memory 12g --memory-swap 12g \
  --shm-size 2g -e PYTHONUNBUFFERED=1 "${M[@]}" "$IMG" \
  bash -c "cd /out && python /run/own_rescorer.py /bundle /comp /out/own_candidates.json /out/own_rescore.json /tmp/own_site > run.log 2>&1; rc=\$?; echo \"exit \$rc\" >> run.log; chown -R $U /out; exit \$rc" \
  || { tail -30 "$OUT/run.log"; exit 1; }
grep -E "^rdkit|^models|molecules,|^OWN_RESCORE|^exit" "$OUT/run.log" | cut -c1-300
python3 - "$OUT/own_candidates.json" "$OUT/own_rescore.json" <<'PY'
import json, math, sys
cand, o = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
ok = set(o["fzr"]) == set(cand) == set(o["g2"])
n = nf = ng = 0
for mid, smis in cand.items():
    ok &= list(o["fzr"][mid]) == smis and list(o["g2"][mid]) == smis
    for s in smis:
        n += 1
        nf += o["fzr"][mid][s] is not None and math.isfinite(o["fzr"][mid][s])
        ng += o["g2"][mid][s] is not None and math.isfinite(o["g2"][mid][s])
print(f"keys == input strings: {ok} | fzr finite {nf}/{n}, g2 finite {ng}/{n} | stats {o['stats']}")
sys.exit(0 if ok and nf >= 0.95 * n else 1)
PY

#!/bin/bash
# M7: one k36_submit7 version for one variant of the submission queue (plan file), then wait and fetch its outputs.
#   tools/push_k36.sh 'both_fprich_g2|hybm:0.8:5+gen:S3:0.7:rk'
# -> out/m7_k36/<MMDDHHMM>/{submission.csv, submission_raw.csv, variant.json, log}. The submission itself is made by the
# user from the kernel's version page; then log it with
#   python tools/wandb_lb.py --group M7 --variant-json out/m7_k36/<MMDDHHMM>/variant.json
# The first push of k36 needs the user's OK (repo rule); later versions are part of the agreed queue.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
v=${1:?variant name}
SLUG=yuanyuan98/casmi26-k36-submit7
python3 tools/build_kernel.py k36_submit7 --define "VARIANT=\"$v\""
grep -qF "$v" kaggle/k36_submit7/build/casmi26-k36-submit7.ipynb || { echo "variant not in the build"; exit 1; }
kaggle kernels push -p kaggle/k36_submit7/build
sleep 60
out/wait_kernel.sh "$SLUG"
dest=out/m7_k36/$(date +%m%d%H%M)
python3 tools/kaggle_out.py "$SLUG" -p "$dest" --files 'submission.*\.csv|variant\.json|.*\.log'
python3 - "$dest" "$v" <<'PY'
import json, sys
import pandas as pd
d, v = sys.argv[1], sys.argv[2]
j = json.load(open(f"{d}/variant.json"))
assert j["variant"] == v, (j["variant"], v)
s = pd.read_csv(f"{d}/submission.csv")
n = s.smiles.str.split(";").map(len)
print(f"{d}: {j['variant']} | rows {len(s)}, candidates {n.min()}-{n.max()} | stats {j['stats']} | local {j['local']}")
PY

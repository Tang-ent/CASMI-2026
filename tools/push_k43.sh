#!/bin/bash
# M7 P3: one k43_submit8 version (k42 ext_submit or v12 pair), then wait and fetch its outputs.
#   tools/push_k43.sh '<chosen8.json ext_submit model>'
# -> out/m7_k43/<MMDDHHMM>/{submission.csv, submission_raw.csv, variant.json, log}. The submission itself is made by the
# user from the kernel's version page; then log it with
#   python tools/wandb_lb.py --group M7 --variant-json out/m7_k43/<MMDDHHMM>/variant.json
# The first push of k43 needs the user's OK (repo rule); later versions are part of the agreed queue.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
v=${1:?variant name}
SLUG=yuanyuan98/casmi26-k43-submit8
python3 tools/build_kernel.py k43_submit8 --define "VARIANT=\"$v\""
grep -qF "$v" kaggle/k43_submit8/build/casmi26-k43-submit8.ipynb || { echo "variant not in the build"; exit 1; }
for i in $(seq 1 60); do  # retry: a GPU batch slot may still be held right after a cancel ("Maximum batch GPU session count")
  r=$(kaggle kernels push -p kaggle/k43_submit8/build 2>&1 || true); echo "$r"
  echo "$r" | grep -q "successfully pushed" && break
  [ "$i" -eq 60 ] && { echo "push failed 60 times"; exit 1; }
  sleep 120
done
sleep 60
out/wait_kernel.sh "$SLUG"
dest=out/m7_k43/$(date +%m%d%H%M)
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

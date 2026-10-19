#!/bin/bash
# M8 (2026-10-19): push the fusion notebook (tools/build_fuse.py) as a new version of the user's fork
# yuanyuan98/enveda-casmi-2026, wait, fetch the log and the audit outputs into out/m8_fuse/<MMDDHHMM>/ and check them
# (tools/check_fuse.py). A commit run processes 12 molecules (the base notebook's SMOKE switch); the hidden-test rerun at
# submission processes all of them. The submission itself is made by the user from the version page; then
#   python tools/wandb_lb.py --group M8 --note "fuse v12 BETA 0.6 (fork v<N>)"
#   tools/push_fuse.sh [--beta 0.6]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
SLUG=yuanyuan98/enveda-casmi-2026
python3 tools/build_fuse.py "$@"
for i in $(seq 1 30); do  # retry: a GPU batch slot may still be held ("Maximum batch GPU session count")
  r=$(kaggle kernels push -p kaggle/k45_fuse/build 2>&1 || true); echo "$r"
  echo "$r" | grep -q "successfully pushed" && break
  [ "$i" -eq 30 ] && { echo "push failed 30 times"; exit 1; }
  sleep 120
done
sleep 60
out/wait_kernel.sh "$SLUG"
dest=out/m8_fuse/$(date +%m%d%H%M)
python3 tools/kaggle_out.py "$SLUG" -p "$dest" \
  --files 'submission\.csv|submission_e004_control\.csv|.*verification\.json|experiment_audit.*|own/own_(stats|lists)\.json|own/own_submission\.csv|run_manifest\.json|e004_candidate_inputs\.json'
python3 tools/check_fuse.py "$dest"

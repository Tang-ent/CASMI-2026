"""Upload the offline wandb runs written by a finished Kaggle kernel (casmi26.track) to wandb.

    python tools/wandb_sync.py <user>/casmi26-k06-fpnet [-p out/k06]
Downloads only wandb/offline-run-*/run-*.wandb and files/* (not logs/ or the latest-run link), then runs
`wandb sync` on each run directory. Requires a local `wandb login` (~/.netrc).
"""
import argparse
import glob
import os
import re
import subprocess
import sys

from kaggle_out import download, list_output

KEEP = re.compile(r"(^|/)wandb/offline-run-[^/]+/(run-[^/]+\.wandb|files/.+)$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kernel")
    ap.add_argument("-p", "--path", default=None, help="download dir (default: out/<slug>)")
    a = ap.parse_args()
    path = a.path or os.path.join("out", a.kernel.split("/")[1])
    files = [f for f in list_output(a.kernel).files if KEEP.search(f.file_name)]
    if not files:
        sys.exit(f"no offline wandb run in the output of {a.kernel}")
    for f in files:
        download(f, path)
    runs = sorted({os.path.dirname(p) for p in glob.glob(f"{path}/**/offline-run-*/run-*.wandb", recursive=True)})
    for r in runs:
        subprocess.run([sys.executable, "-m", "wandb", "sync", "--yes", r], check=True)


if __name__ == "__main__":
    main()

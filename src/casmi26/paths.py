"""Kaggle / local input path resolution."""
import glob
import os

ROOTS = ["/kaggle/input", "/kaggle/working"]
COMP_DIR_NAME = "enveda-CASMI26-molecule-id-mass-spectra"


def find(name, roots=ROOTS):
    """Shortest path that ends with `name` under the roots (raises if none)."""
    hits = []
    for root in roots:
        hits += glob.glob(os.path.join(root, "**", name), recursive=True)
    if not hits:
        raise FileNotFoundError(f"{name} not found under {roots}")
    return sorted(hits, key=len)[0]


def comp_dir():
    return os.path.dirname(find(os.path.join(COMP_DIR_NAME, "train.parquet")))

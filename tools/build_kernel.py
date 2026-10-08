"""Build a pushable Kaggle notebook from kaggle/<kernel>/main.py with src/casmi26 embedded.

    python tools/build_kernel.py k01_cache        -> kaggle/k01_cache/build/{<slug>.ipynb, kernel-metadata.json}
    kaggle kernels push -p kaggle/k01_cache/build
    python tools/build_kernel.py k07_features2:3  -> kaggle/k07_features2/build_s{0,1,2}/  (3 shard kernels <slug>-s<i>;
                                                     a cell `SHARD, N_SHARDS = i, 3` is injected after the boot cell)
kernel-metadata.json may carry "shard_kernel_sources": {"<i>": [...]}: extra kernel sources for shard i only
(e.g. a later variant that reads an earlier shard's output); the key is removed from the pushed metadata.

main.py is split into cells at lines starting with '# %%' ('# %% [markdown]' = markdown cell, '# ' stripped).
The first code cell writes the casmi26 package to /tmp/casmi26_src and puts it on sys.path, so every
notebook runs exactly the source it was built from (sha printed at runtime).
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "casmi26"


def split_cells(text):
    cells, kind, buf = [], "code", []

    def flush():
        body = "\n".join(buf).strip("\n")
        if body:
            if kind == "markdown":
                body = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.split("\n"))
            cells.append((kind, body))

    for line in text.split("\n"):
        if line.startswith("# %%"):
            flush()
            kind, buf = ("markdown" if "[markdown]" in line else "code"), []
        else:
            buf.append(line)
    flush()
    return cells


def boot_cell(slug):
    src = {p.name: p.read_text() for p in sorted(SRC.glob("*.py"))}
    sha = hashlib.sha256("".join(src[k] for k in sorted(src)).encode()).hexdigest()[:12]
    return (
        "# casmi26 package embedded by tools/build_kernel.py\n"
        "import os, sys\n"
        f"_CASMI26_SRC = {src!r}\n"
        "_d = '/tmp/casmi26_src/casmi26'\n"
        "os.makedirs(_d, exist_ok=True)\n"
        "for _n, _c in _CASMI26_SRC.items():\n"
        "    with open(os.path.join(_d, _n), 'w') as _f:\n"
        "        _f.write(_c)\n"
        "sys.path.insert(0, '/tmp/casmi26_src')\n"
        "os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/numba_cache')\n"
        f"os.environ['CASMI26_SHA'], os.environ['CASMI26_KERNEL'] = {sha!r}, {slug!r}  # casmi26.track が使う\n"
        f"print('casmi26 source sha {sha}')\n"
    ), sha


def to_cell(kind, body):
    lines = body.split("\n")
    source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    c = {"cell_type": kind, "metadata": {}, "source": source}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def build(kernel, shard=None, n_shards=1):
    kdir = ROOT / "kaggle" / kernel
    meta = json.loads((kdir / "kernel-metadata.json").read_text())
    extra = meta.pop("shard_kernel_sources", {})
    out = kdir / "build"
    if shard is not None:
        meta["kernel_sources"] = meta["kernel_sources"] + extra.get(str(shard), [])
        meta["id"] += f"-s{shard}"
        meta["title"] += f"-s{shard}"
        out = kdir / f"build_s{shard}"
    slug = meta["id"].split("/")[1]
    boot, sha = boot_cell(slug)
    cells = [to_cell("code", boot)]
    if shard is not None:
        cells.append(to_cell("code", f"SHARD, N_SHARDS = {shard}, {n_shards}"))
    cells +=[to_cell(k, b) for k, b in split_cells((kdir / "main.py").read_text())]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                      "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out.mkdir(exist_ok=True)
    meta["code_file"] = f"{slug}.ipynb"
    (out / meta["code_file"]).write_text(json.dumps(nb, indent=1))
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print(f"{kernel}: {len(cells)} cells, src sha {sha} -> {out}")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        name, _, n = arg.partition(":")
        if n:
            for i in range(int(n)):
                build(name, i, int(n))
        else:
            build(name)

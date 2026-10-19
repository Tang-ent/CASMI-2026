"""M8 (2026-10-19, plan 1-3-deep-storm.md): build the fusion notebook = the user's fork of the public Dvorkin notebook
(kaggle/k45_fuse/base, LB 0.441) + our v12 engine as a third engine in its final metric_fusion.

    python tools/build_fuse.py [--beta 0.6]
    -> kaggle/k45_fuse/build/{enveda-casmi-2026.ipynb, kernel-metadata.json, own_runner.py}
    kaggle kernels push -p kaggle/k45_fuse/build          (a new version of yuanyuan98/enveda-casmi-2026)

The base notebook is never edited in place. Each edit is an exact string replacement inside one cell that must match
exactly once (a changed base fails loudly):
  1. cell 3 (setup): BETA, the weight of our engine in metric_fusion (pre-registered 0.6 = Engine 2's alpha).
  2. a new markdown + code cell right after cell 9 (Engine 2): unpack casmi26_engine_bundle.tar (kernel output of
     casmi26-k44-bundle) to /tmp, run own_runner.py (kaggle/k44_own + the embedded casmi26 source, its own RDKit 2026.9.1
     site) in a subprocess -> OWN = {molecule_id: {smiles, keys, n_window}}; on any failure OWN = {} (the run goes on
     and equals the base).
  3. cell 29: metric_fusion(first, second, identity, ..., third=None, beta=0.6) adds `third` with weight beta; the call
     passes OWN's list.
  4. cell 35 (E006 overlay): the same third list in rerank_row, so the same-run E004 control is still reproduced
     byte for byte (run_overlay asserts it).
  5. cell 31: OWN is saved with the other fusion inputs (e004_candidate_inputs.json; audit only).
own_runner.py = tools/build_kernel.py's boot cell (src/casmi26 embedded, sha printed) + the code cells of k44_own/main.py.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from build_kernel import boot_cell, split_cells  # noqa: E402

KDIR = ROOT / "kaggle" / "k45_fuse"
BASE = KDIR / "base" / "enveda-casmi-2026.ipynb"
BASE_SHA = "6063f68a090ad8d3cc8a2f90b2299679ceaf1b0b006f8277f36600433ed8235e"  # yuanyuan98/enveda-casmi-2026 v1 (LB 0.441)
BUNDLE_KERNEL = "yuanyuan98/casmi26-k44-bundle"


def own_runner():
    boot, sha = boot_cell("casmi26-k44-own")
    code = [b for k, b in split_cells((ROOT / "kaggle" / "k44_own" / "main.py").read_text()) if k == "code"]
    return "\n\n".join([boot] + code) + "\n", sha


def own_cell(runner):
    return f'''"""M8 (2026-10-19): our own engine (casmi26 v12 = both_fprich_g2_grp|hybi:0.7:0.8:5+nogen on the COCONUT pool, no
external weights) in a subprocess with its own RDKit 2026.9.1 site -> OWN = {{molecule_id: {{smiles: top 40, keys, n_window}}}},
the third engine of metric_fusion (weight BETA). Any failure -> OWN = {{}} and the run equals the base notebook."""
OWN_DIR, OWN_ROOT, OWN_SITE = '/kaggle/working/own', '/tmp/own_bundle', '/tmp/own_site'
OWN_RUNNER = {runner!r}
OWN = {{}}
try:
    _t0 = time.time()
    os.makedirs(OWN_DIR, exist_ok=True)
    _tar = sorted(glob.glob('/kaggle/input/**/casmi26_engine_bundle.tar', recursive=True), key=len)
    assert _tar, 'casmi26_engine_bundle.tar (kernel output of casmi26-k44-bundle) is not attached'
    if not os.path.exists(os.path.join(OWN_ROOT, 'casmi26_engine_bundle.json')):
        os.makedirs(OWN_ROOT, exist_ok=True)
        subprocess.run(['tar', '-xf', _tar[0], '-C', OWN_ROOT], check=True)
    print('own bundle', _tar[0], f'unpacked {{time.time() - _t0:.0f}}s', flush=True)
    open(os.path.join(OWN_DIR, 'own_runner.py'), 'w').write(OWN_RUNNER)
    _rr = subprocess.run([sys.executable, os.path.join(OWN_DIR, 'own_runner.py'), OWN_ROOT, COMP, OWN_DIR, OWN_SITE],
                         capture_output=True, text=True, timeout=3 * 3600, cwd=OWN_DIR)
    print(_rr.stdout[-4000:]); print(_rr.stderr[-3000:])
    _rr.check_returncode()
    OWN = json.load(open(os.path.join(OWN_DIR, 'own_lists.json')))
    print('own lists', len(OWN), f'{{time.time() - _t0:.0f}}s', 'BETA', BETA, flush=True)
except Exception as e:
    OWN = {{}}
    print('OWN ENGINE FAILED -> fusion without our engine:', repr(e), flush=True)
'''


OWN_MD = """## 4b. M8 — Our Own Engine (casmi26 v12) as a Third Fusion Engine

Runs our COCONUT-pool ranker (v12, LB 0.339 alone) in a subprocess with its own RDKit and adds its top-40 list to the
final metric-identity fusion with weight `BETA` (pre-registered 0.6, the same as Engine 2)."""


def replace_once(src, old, new, what):
    n = src.count(old)
    assert n == 1, f"{what}: expected exactly one match, found {n}"
    return src.replace(old, new)


def edit(nb, beta, runner):
    cells = nb["cells"]
    text = lambda i: "".join(cells[i]["source"])

    def find_cell(prefix):
        hits = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and text(i).lstrip().startswith(prefix)]
        assert len(hits) == 1, f"cell starting with {prefix!r}: {hits}"
        return hits[0]

    def put(i, src):
        lines = src.split("\n")
        cells[i]["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]

    i3 = find_cell("import os, sys, glob, re, subprocess, time, json, hashlib, pickle")
    i9 = find_cell("'''Our engine (two-ranker + BIO + AFIX)")
    i29 = find_cell("'''Independent experimental overlay")
    i31 = find_cell('"""Observe the frozen source output')
    i35 = find_cell("def _factorial_entry(scope):")
    # 1. BETA
    put(i3, replace_once(text(i3), "SLOTS_AGG, SLOTS_GENTLE = [2, 4, 6, 8, 10], [4, 8, 12, 16, 20]\n",
                         "SLOTS_AGG, SLOTS_GENTLE = [2, 4, 6, 8, 10], [4, 8, 12, 16, 20]\n"
                         f"BETA = {beta}  # M8: weight of our own engine (casmi26 v12) in metric_fusion (pre-registered)\n",
                         "cell 3 BETA"))
    # 3. metric_fusion with a third list
    s = text(i29)
    s = replace_once(s, "def metric_fusion(first, second, identity, alpha=0.6, offset=3.0, limit=40):",
                     "def metric_fusion(first, second, identity, alpha=0.6, offset=3.0, limit=40, third=None, beta=0.6):",
                     "cell 29 signature")
    s = replace_once(s, "    for candidates, weight in ((first, 1.0), (second, alpha)):",
                     "    for candidates, weight in ((first, 1.0), (second, alpha)) + (((third, beta),) if third else ()):",
                     "cell 29 engines")
    s = replace_once(s, "                                                       alpha=ALPHA, offset=KRR)",
                     "                                                       alpha=ALPHA, offset=KRR,\n"
                     "                                                       third=(OWN.get(_mid) or {}).get('smiles'), beta=BETA)",
                     "cell 29 call")
    put(i29, s)
    # 4. the E006 overlay rebuilds the same fusion
    put(i35, replace_once(text(i35), '            offset=scope["KRR"],\n            limit=window,\n        )',
                          '            offset=scope["KRR"],\n            limit=window,\n'
                          '            third=(scope["OWN"].get(mid) or {}).get("smiles"),\n'
                          '            beta=scope["BETA"],\n        )', "cell 35 call"))
    # 5. audit
    put(i31, replace_once(text(i31), "        'engine2': {mid: eng[mid] for mid in sorted(processed)},\n",
                          "        'engine2': {mid: eng[mid] for mid in sorted(processed)},\n"
                          "        'own': {mid: scope.get('OWN', {}).get(mid) for mid in sorted(processed)},\n",
                          "cell 31 payload"))
    # 2. our engine right after Engine 2
    md = {"cell_type": "markdown", "metadata": {}, "source": []}
    code = {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": []}
    cells[i9 + 1:i9 + 1] = [md, code]
    put(i9 + 1, OWN_MD)
    put(i9 + 2, own_cell(runner))
    return nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=0.6)
    a = ap.parse_args()
    raw = BASE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == BASE_SHA, "kaggle/k45_fuse/base changed"
    runner, sha = own_runner()
    nb = edit(json.loads(raw), a.beta, runner)
    out = KDIR / "build"
    out.mkdir(exist_ok=True)
    (out / "own_runner.py").write_text(runner)
    (out / "enveda-casmi-2026.ipynb").write_text(json.dumps(nb, indent=1))
    meta = json.loads((KDIR / "base" / "kernel-metadata.json").read_text())
    assert meta["id"] == "yuanyuan98/enveda-casmi-2026", meta["id"]
    meta.pop("id_no", None)
    meta["kernel_sources"] = sorted(set(meta.get("kernel_sources", [])) | {BUNDLE_KERNEL})
    meta["docker_image_pinning_type"] = "original"  # keep the base version's environment (Python 3.12, cell 5 asserts it)
    meta["machine_shape"] = "NvidiaTeslaT4"  # cell 5 asserts a T4
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"k45_fuse: base {BASE_SHA[:12]} + own engine (src sha {sha}, BETA {a.beta}) -> {out} | "
          f"{len(nb['cells'])} cells, sources {len(meta['dataset_sources'])} datasets + {meta['kernel_sources']}")


if __name__ == "__main__":
    main()

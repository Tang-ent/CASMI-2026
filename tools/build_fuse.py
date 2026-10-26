"""M8 (plan 1-3-deep-storm.md): build the fusion notebook = the user's fork of the public Dvorkin notebook
(kaggle/k45_fuse/base, LB 0.441) + our own engine in one of two forms.

    python tools/build_fuse.py [--mode rescore --where nogl]   (v5, 2026-10-21, default: ours only where GLACIER is absent)
    python tools/build_fuse.py --mode rescore --where all      (v4, 2026-10-20: LB 0.435 vs 0.441, not adopted)
    python tools/build_fuse.py --mode engine [--beta 0.6]   (v3, 2026-10-19: LB 0.417 vs 0.441, not adopted)
    -> kaggle/k45_fuse/build/{enveda-casmi-2026.ipynb, kernel-metadata.json, own_runner.py | own_rescorer.py}
    kaggle kernels push -p kaggle/k45_fuse/build          (a new version of yuanyuan98/enveda-casmi-2026)

The base notebook is never edited in place. Each edit is an exact string replacement inside one cell that must match
exactly once (a changed base fails loudly).
--mode engine (v3): our v12 list as a third engine of metric_fusion (cell 29 and the E006 rerank_row of cell 35) with
  weight BETA; our engine runs right after Engine 2 (cell 9) from kaggle/k44_own.
--mode rescore (v4, pre-registered 2026-10-20): no candidate is added and metric_fusion is untouched. A new cell right
  before cell 29 scores the candidates entering the fusion (PREFUSION_LISTS ∪ ENG, keyed by the exact SMILES strings)
  with our fprich and G2 models (kaggle/k46_rescore) -> OWN_FZR / OWN_G2; gl_fuse.rerank_multi gets them as two more
  re-scorers with weight LAM_OWN (0.25) each: glacier branch [{}, gl, fzr, g2], frag_gap branch [FRAG_GAP, fzr, g2],
  ice_fallback unchanged, and our scores alone where nothing else re-scores (cell 29 and rerank_row do the same, so
  the same-run E004 control is still reproduced byte for byte). Gate: base lib_max (BASE[mid][2]) < OWN_GATE, which is
  LIB_TAU (0.9) on the hidden rerun and off on the 12-molecule commit (all library hits) so that the new path runs.
Both forms: the scripts are tools/build_kernel.py's boot cell (src/casmi26 embedded, sha printed) + the code cells of the
kernel's main.py, run in a subprocess with their own RDKit 2026.9.1 site from casmi26_engine_bundle.tar (kernel output of
casmi26-k44-bundle, unpacked to /tmp); any failure leaves the run equal to the base notebook (logged as ... FAILED).
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


def script(kernel, slug):
    boot, sha = boot_cell(slug)
    code = [b for k, b in split_cells((ROOT / "kaggle" / kernel / "main.py").read_text()) if k == "code"]
    return "\n\n".join([boot] + code) + "\n", sha


UNPACK = """    _tar = sorted(glob.glob('/kaggle/input/**/casmi26_engine_bundle.tar', recursive=True), key=len)
    assert _tar, 'casmi26_engine_bundle.tar (kernel output of casmi26-k44-bundle) is not attached'
    if not os.path.exists(os.path.join(OWN_ROOT, 'casmi26_engine_bundle.json')):
        os.makedirs(OWN_ROOT, exist_ok=True)
        subprocess.run(['tar', '-xf', _tar[0], '-C', OWN_ROOT], check=True)
    print('own bundle', _tar[0], f'unpacked {time.time() - _t0:.0f}s', flush=True)
"""


def rescore_cell(rescorer, where):
    return f'''"""M8 v4 (2026-10-20, pre-registered): our isomer scores (casmi26 fprich + G2, no external weights) for the candidates
that enter the final fusion (PREFUSION_LISTS ∪ ENG; no candidate is added) -> OWN_FZR / OWN_G2 = {{molecule_id: {{smiles: score}}}}.
gl_fuse.rerank_multi uses them inside same-formula groups next to GLACIER (cells 29 / 35), weight LAM_OWN each. Only
molecules whose base lib_max < OWN_GATE get scores (LIB_TAU on the hidden rerun; off on the 12-molecule commit so that the
new path runs). Any failure -> no scores and the run equals the base notebook."""
LAM_OWN = 0.25
OWN_GATE = LIB_TAU if IS_RERUN else float('inf')
OWN_WHERE = {where!r}  # 'all' (v4) or 'nogl' (v5: not next to GLACIER)
OWN_DIR, OWN_ROOT, OWN_SITE = '/kaggle/working/own', '/tmp/own_bundle', '/tmp/own_site'
OWN_RESCORER = {rescorer!r}
OWN_FZR, OWN_G2, OWN_STATS = {{}}, {{}}, {{}}
try:
    _t0 = time.time()
    os.makedirs(OWN_DIR, exist_ok=True)
    _lib = {{str(k): float(v[2]) for k, v in BASE.items()}}
    _cand, _gated = {{}}, 0
    for _m, _first in PREFUSION_LISTS.items():
        _m = str(_m)
        if not (_lib.get(_m, float('inf')) < OWN_GATE):  # strong library match (or missing from BASE): untouched
            _gated += 1
            continue
        _u = [s for s in list(_first) + list((ENG.get(_m) or {{}}).get('smiles') or []) if s and s != 'CCO']
        if _u:
            _cand[_m] = list(dict.fromkeys(_u))
    json.dump(_cand, open(os.path.join(OWN_DIR, 'own_candidates.json'), 'w'))
{{UNPACK}}    open(os.path.join(OWN_DIR, 'own_rescorer.py'), 'w').write(OWN_RESCORER)
    _rr = subprocess.run([sys.executable, os.path.join(OWN_DIR, 'own_rescorer.py'), OWN_ROOT, COMP,
                          os.path.join(OWN_DIR, 'own_candidates.json'), os.path.join(OWN_DIR, 'own_rescore.json'), OWN_SITE],
                         capture_output=True, text=True, timeout=2 * 3600, cwd=OWN_DIR)
    print(_rr.stdout[-3000:]); print(_rr.stderr[-2000:])
    _rr.check_returncode()
    _o = json.load(open(os.path.join(OWN_DIR, 'own_rescore.json')))
    OWN_FZR, OWN_G2, OWN_STATS = _o['fzr'], _o['g2'], _o['stats']
    print('own rescore', len(OWN_FZR), 'molecules', OWN_STATS.get('candidates'), 'candidates', f'{{time.time() - _t0:.0f}}s',
          'LAM', LAM_OWN, 'GATE', OWN_GATE, 'gated_out', _gated, 'WHERE', OWN_WHERE, flush=True)
except Exception as e:
    OWN_FZR, OWN_G2 = {{}}, {{}}
    print('OWN RESCORE FAILED -> base re-scoring only:', repr(e), flush=True)
'''.replace("{UNPACK}", UNPACK)


def rescore_md(where):
    near = "GLACIER / FRAG_GAP" if where == "all" else "FRAG_GAP (not GLACIER: where GLACIER re-scores, the base is kept)"
    return f"""## 13b. M8 — Our Isomer Scores (casmi26 fprich + G2) as Same-Formula Re-Scorers ({where})

No candidate is added. The candidates that enter the final fusion get our fingerprint (fprich) and graph (G2) scores; inside
same-formula groups they join {near} as re-scorers with weight `LAM_OWN` each (pre-registered 0.25), and re-score alone
where nothing else does. Molecules with a strong library match (base `lib_max >= LIB_TAU`) are untouched on the hidden
rerun."""


def replace_once(src, old, new, what):
    n = src.count(old)
    assert n == 1, f"{what}: expected exactly one match, found {n}"
    return src.replace(old, new)


def cell_tools(nb):
    cells = nb["cells"]
    text = lambda i: "".join(cells[i]["source"])

    def find_cell(prefix):
        hits = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and text(i).lstrip().startswith(prefix)]
        assert len(hits) == 1, f"cell starting with {prefix!r}: {hits}"
        return hits[0]

    def put(i, src):
        lines = src.split("\n")
        cells[i]["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]

    def insert(i, md, code):
        cells[i:i] = [{"cell_type": "markdown", "metadata": {}, "source": []},
                      {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": []}]
        put(i, md)
        put(i + 1, code)

    return cells, text, find_cell, put, insert


def edit_rescore(nb, rescorer, where):
    cells, text, find_cell, put, insert = cell_tools(nb)
    i29 = find_cell("'''Independent experimental overlay")
    i31 = find_cell('"""Observe the frozen source output')
    i35 = find_cell("def _factorial_entry(scope):")
    # cell 29 (E004): our scores join the GLACIER re-scoring, and re-score alone where nothing else does
    gl_own = where == "all"  # v4: ours also next to GLACIER; v5 ("nogl"): GLACIER's branch stays as in the base
    put(i29, replace_once(text(i29), """        if _ice and _gl:
            _order = gl_fuse.rerank_multi(_candidates, _keys, _scores,
                [_form2(s) for s in _candidates], [{}, _gl], [ICE_LAM, GL_LAM], top_n=len(_candidates))
            _candidates = [_candidates[i] for i in _order]
""", ("""        _ofz, _og2 = OWN_FZR.get(_mid) or {}, OWN_G2.get(_mid) or {}  # M8 v4
        if _ice and _gl:
            _order = gl_fuse.rerank_multi(_candidates, _keys, _scores,
                [_form2(s) for s in _candidates], [{}, _gl, _ofz, _og2], [ICE_LAM, GL_LAM, LAM_OWN, LAM_OWN],
                top_n=len(_candidates))
            _candidates = [_candidates[i] for i in _order]
""" if gl_own else """        _ofz, _og2 = OWN_FZR.get(_mid) or {}, OWN_G2.get(_mid) or {}  # M8 v5: GLACIER's branch as in the base
        if _ice and _gl:
            _order = gl_fuse.rerank_multi(_candidates, _keys, _scores,
                [_form2(s) for s in _candidates], [{}, _gl], [ICE_LAM, GL_LAM], top_n=len(_candidates))
            _candidates = [_candidates[i] for i in _order]
""") + """        elif _ofz or _og2:  # M8 v4: our isomer scores alone (rerank_row of cell 35 does the same)
            _order = gl_fuse.rerank_multi(_candidates, _keys, _scores,
                [_form2(s) for s in _candidates], [_ofz, _og2], [LAM_OWN, LAM_OWN], top_n=len(_candidates))
            _candidates = [_candidates[i] for i in _order]
""", "cell 29 rerank"))
    # cell 35 (E006 rerank_row): the same in the glacier branch, ours in the frag_gap branch, ours alone at the end
    s = text(i35)
    s = replace_once(s, """        before = list(candidates)
""", """        before = list(candidates)
        ofz, og2 = scope["OWN_FZR"].get(mid) or {}, scope["OWN_G2"].get(mid) or {}  # M8 v4
""", "cell 35 own dicts")
    if gl_own:
        s = replace_once(s, """                [{}, gl],
                [scope["ICE_LAM"], scope["GL_LAM"]],
""", """                [{}, gl, ofz, og2],
                [scope["ICE_LAM"], scope["GL_LAM"], scope["LAM_OWN"], scope["LAM_OWN"]],
""", "cell 35 glacier")
    s = replace_once(s, """                [scope["FRAG_GAP"][mid]], [0.5], top_n=len(candidates))
""", """                [scope["FRAG_GAP"][mid], ofz, og2], [0.5, scope["LAM_OWN"], scope["LAM_OWN"]], top_n=len(candidates))
""", "cell 35 frag_gap")
    s = replace_once(s, """            audit["branch"] = "ice_fallback"
        result = (
""", """            audit["branch"] = "ice_fallback"
        elif ofz or og2:
            # M8 v4: our isomer scores alone where nothing else re-scores (cell 29 does the same for the control)
            order = scope["gl_fuse"].rerank_multi(
                candidates, keys, scores, [scope["_form2"](s) for s in candidates],
                [ofz, og2], [scope["LAM_OWN"], scope["LAM_OWN"]], top_n=len(candidates))
            candidates = [candidates[i] for i in order]
            audit["branch"] = "own"
        result = (
""", "cell 35 own branch")
    put(i35, s)
    # cell 31: keep our scores with the other fusion inputs (audit only)
    put(i31, replace_once(text(i31), "        'engine2': {mid: eng[mid] for mid in sorted(processed)},\n",
                          "        'engine2': {mid: eng[mid] for mid in sorted(processed)},\n"
                          "        'own_rescore': {mid: dict(fzr=scope.get('OWN_FZR', {}).get(mid), g2=scope.get('OWN_G2', {}).get(mid))\n"
                          "                        for mid in sorted(processed)},\n",
                          "cell 31 payload"))
    # the scoring cell right before cell 29
    insert(i29, rescore_md(where), rescore_cell(rescorer, where))
    return nb


def edit_engine(nb, beta, runner):
    cells, text, find_cell, put, insert = cell_tools(nb)
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
    insert(i9 + 1, OWN_MD, own_cell(runner))
    return nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rescore", "engine"], default="rescore")
    ap.add_argument("--beta", type=float, default=0.6)
    ap.add_argument("--where", choices=["nogl", "all"], default="nogl",
                    help="rescore: 'nogl' (v5, default) = ours only where GLACIER does not re-score; 'all' = v4")
    a = ap.parse_args()
    raw = BASE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == BASE_SHA, "kaggle/k45_fuse/base changed"
    out = KDIR / "build"
    out.mkdir(exist_ok=True)
    for f in ("own_runner.py", "own_rescorer.py"):
        (out / f).unlink(missing_ok=True)
    if a.mode == "engine":
        code, sha = script("k44_own", "casmi26-k44-own")
        nb = edit_engine(json.loads(raw), a.beta, code)
        (out / "own_runner.py").write_text(code)
        what = f"own engine (BETA {a.beta})"
    else:
        code, sha = script("k46_rescore", "casmi26-k46-rescore")
        nb = edit_rescore(json.loads(raw), code, a.where)
        (out / "own_rescorer.py").write_text(code)
        what = f"own re-scorer (LAM_OWN 0.25 x 2, gate LIB_TAU on the rerun, where {a.where})"
    (out / "enveda-casmi-2026.ipynb").write_text(json.dumps(nb, indent=1))
    meta = json.loads((KDIR / "base" / "kernel-metadata.json").read_text())
    assert meta["id"] == "yuanyuan98/enveda-casmi-2026", meta["id"]
    meta.pop("id_no", None)
    meta["kernel_sources"] = sorted(set(meta.get("kernel_sources", [])) | {BUNDLE_KERNEL})
    meta["docker_image_pinning_type"] = "original"  # keep the base version's environment (Python 3.12, cell 5 asserts it)
    meta["machine_shape"] = "NvidiaTeslaT4"  # cell 5 asserts a T4
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"k45_fuse [{a.mode}]: base {BASE_SHA[:12]} + {what}, src sha {sha} -> {out} | "
          f"{len(nb['cells'])} cells, sources {len(meta['dataset_sources'])} datasets + {meta['kernel_sources']}")


if __name__ == "__main__":
    main()

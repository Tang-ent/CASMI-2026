# %% [markdown]
# # casmi26-k46-rescore（M8 v4、2026-10-20、計画 1-3-deep-storm.md / idempotent-chasing-goblet.md の事前登録）
# 融合 Notebook（ユーザーの fork）の中で、融合の前の候補（PREFUSION_LISTS ∪ ENG）に自作の異性体の点数を付けるプロセス。
# 候補は足さない。土台の rerank_multi（同じ分子式の組の中の並べ替え）に GLACIER と並ぶ再採点として渡す。
# tools/build_fuse.py が build_kernel の boot cell（src/casmi26 の埋め込み）とこのファイルのコードセルをつないで own_rescorer.py にする。
#   python own_rescorer.py <ROOT> <COMP> <CAND_JSON> <OUT_JSON> <SITE>
# - ROOT: k44_bundle の tar を展開した場所（fprich.pt、g2.pt、chosen7.json、rdkit の wheel だけを使う）。
# - CAND_JSON: {molecule_id: [SMILES, ...]}（融合 Notebook が門を通った分子だけ書く）。
# - 出力: {"fzr": {mid: {smiles: float | None}}, "g2": {mid: {smiles: float | None}}, "stats": {...}}。鍵は入力の SMILES 文字列そのまま。
# - fzr = fprich の予測指紋（MR(specs)）と、候補の Morgan r2（互変異性体の正規形、プールと同じ）+ chem.rich_fp_tail の内積（pcside と同じ）。
#   g2 = G2Model.score_smiles（グラフはこのプロセスで作る）。読めない SMILES は None。

# %%
import glob
import json
import os
import subprocess
import sys
import time

T0 = time.time()
ROOT, COMP, CAND, OUT, SITE = sys.argv[1:6]
tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
wh = [w for w in glob.glob(os.path.join(ROOT, "**", "rdkit-2026.9.1-*.whl"), recursive=True) if f"-{tag}-" in w]
assert wh, f"no rdkit 2026.9.1 wheel for {tag} under {ROOT}"
if not os.path.isdir(os.path.join(SITE, "rdkit")):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--target", SITE, wh[0]],
                       capture_output=True, text=True)
    print(r.stdout[-300:], r.stderr[-300:])
    r.check_returncode()
sys.path.insert(0, SITE)
import rdkit  # noqa: E402

print("rdkit", rdkit.__version__, "from", os.path.dirname(rdkit.__file__), flush=True)

# %%
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from casmi26 import chem, fpnet, g2, harness, paths  # noqa: E402

paths.ROOTS[:] = [ROOT]
REQUIRE_CUDA = True
FI = json.load(open(paths.find("chosen7.json")))["features"]
assert rdkit.__version__ == FI["rdkit"], f"rdkit {rdkit.__version__} differs from k30's {FI['rdkit']}"
MR = fpnet.FPModel(paths.find("fprich.pt"))
G2M = g2.G2Model(paths.find("g2.pt"), G=None)
assert int(MR.epoch) == int(FI["fprich_epoch"]) and MR.nbits == FI["fprich_nbits"], "fprich.pt differs from k30's"
assert int(G2M.step) == int(FI["g2_step"]) and not G2M.amp, "g2.pt differs from k30's"
assert not REQUIRE_CUDA or (MR.device == "cuda" and G2M.device == "cuda"), (MR.device, G2M.device)
cands = json.load(open(CAND))
te = pd.read_parquet(os.path.join(COMP, "test.parquet"))
te = te[te.molecule_id.astype(str).isin(cands)]
print(f"models loaded ({time.time() - T0:.0f}s) | {len(cands)} molecules, {sum(len(v) for v in cands.values())} candidates",
      flush=True)

# %%
def rich_fp(s):
    c = chem.canonical(s)
    if c is None:
        return None
    return np.r_[np.unpackbits(c["fp"])[:chem.FP_BITS], chem.rich_fp_tail(s)].astype(np.float32)


out = {"fzr": {}, "g2": {}}
stats = dict(molecules=0, candidates=0, unparsed=0, g2_missing=0, no_specs=0)
for i, (mid, g) in enumerate(te.groupby("molecule_id", sort=False)):
    mid = str(mid)
    smiles = list(dict.fromkeys(cands[mid]))  # 入力の順、重なりは 1 つ
    specs = harness.test_specs(g)
    stats["molecules"] += 1
    stats["candidates"] += len(smiles)
    fz = {s: None for s in smiles}
    gz = {s: None for s in smiles}
    if specs and smiles:
        zr = MR(specs)
        F = [rich_fp(s) for s in smiles]
        ok = [k for k, f in enumerate(F) if f is not None]
        stats["unparsed"] += len(smiles) - len(ok)
        if ok:
            v = np.stack([F[k][:len(zr)] for k in ok]) @ zr
            for k, x in zip(ok, v):
                fz[smiles[k]] = float(x)
            gv = G2M.score_smiles(specs, [smiles[k] for k in ok])
            for k, x in zip(ok, gv):
                if np.isfinite(x):
                    gz[smiles[k]] = float(x)
                else:
                    stats["g2_missing"] += 1
    else:
        stats["no_specs"] += 1
    out["fzr"][mid], out["g2"][mid] = fz, gz
    if (i + 1) % 50 == 0:
        print(f"  {i + 1} molecules, {time.time() - T0:.0f}s", flush=True)
out["stats"] = dict(stats, secs=round(time.time() - T0), rdkit=rdkit.__version__, src_sha=os.environ.get("CASMI26_SHA"),
                    devices=[MR.device, G2M.device])
json.dump(out, open(OUT, "w"))
print("OWN_RESCORE", json.dumps(out["stats"]), flush=True)

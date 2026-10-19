# %% [markdown]
# # casmi26-k44-own（M8、2026-10-19、計画 1-3-deep-storm.md）
# 公開 Notebook（ユーザーの fork yuanyuan98/enveda-casmi-2026）の中で、自作エンジン v12 を別プロセスとして動かすランナー。
# tools/build_fuse.py が build_kernel の boot cell（src/casmi26 の埋め込み）とこのファイルのコードセルをつないで own_runner.py にし、
# 融合 Notebook のセルに文字列で埋め込む。手元では tools/local/run_own.sh で同じ own_runner.py を動かす。
#   python own_runner.py <ROOT> <COMP> <OUT> <SITE>
# - ROOT: 自作の入力（k44_bundle の tar を展開した場所。中は /kaggle/input と同じ形で <slug>/<files>）。paths.find はここだけを探す
#   （土台の Notebook の /kaggle/input には他人のデータが多く、名前が衝突しうるため）。
# - COMP: test.parquet と sample_submission.csv のある場所（土台の SMOKE では 12 分子の /kaggle/working/smoke_comp）。
# - SITE: RDKit 2026.9.1 を pip --target で入れる場所（土台の Notebook 本体は 2026.03.3 を使うので混ぜない）。
# - 出力（OUT）: own_lists.json = {molecule_id: {smiles: v12 の順に衛生をかけた上位 K_OWN、keys: プールの IK14、n_window}}、
#   own_stats.json、own_submission.csv（k36 v12 と同じ 25 件。見えるテストで k36 v12 の提出と一致することの確かめ用）。
# 案は v12（both_fprich_g2_grp|hybi:0.7:0.8:5+nogen、k35 の ranker7、外部の重みなし）に固定（ユーザーの決定 2026-10-19）。

# %%
import glob
import json
import os
import subprocess
import sys
import time

T0 = time.time()
ROOT, COMP, OUT, SITE = sys.argv[1:5]
os.makedirs(OUT, exist_ok=True)
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
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from casmi26 import cache, chem, fpnet, frag, fwd, g2, harness, paths, pool, rank, search, subform  # noqa: E402

paths.ROOTS[:] = [ROOT]  # find() の既定の探索先（同じリストを書き換える）
VARIANT = "both_fprich_g2_grp|hybi:0.7:0.8:5+nogen"  # v12（LB 0.339）
K_OWN = 40  # 融合に渡す一覧の長さ（Engine 2 と同じ）
REQUIRE_CUDA = True  # fwd / fprich / G2 は k30 と同じ GPU で（手元の CPU だけの確かめは False）
GRAPH_WORKERS = 4
V = search.parse_variant(VARIANT)
tag = V["cols"]
assert not V["gen"] and not V["pc"] and tag == "both_fprich_g2_grp", V
EV = os.path.dirname(paths.find("chosen7.json"))
FI = json.load(open(f"{EV}/chosen7.json"))["features"]
assert rdkit.__version__ == FI["rdkit"], f"rdkit {rdkit.__version__} differs from k30's {FI['rdkit']}"
models = [lgb.Booster(model_file=f"{EV}/ranker7_{tag}_{i}.txt") for i in range(3)]
cols = models[0].feature_name()
assert all(m.feature_name() == cols for m in models)

C = cache.Cache(cache.Cache.find_dir())
P = pool.Pool(pool.Pool.find_dir(), pubchem=False)
assert len(P) == P.n_base
GR = g2.Graphs.build(P.smiles, GRAPH_WORKERS)  # CUDA より先に（fork）
FP2 = np.load(paths.find("pool_fp2.npy"))
assert FP2.shape == (len(P), (chem.RICH_FP_BITS + 7) // 8), FP2.shape

import torch  # noqa: E402

print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), f"({time.time() - T0:.0f}s)", flush=True)
M_NEW = fpnet.load_checked("fpnet2.pt", min_epoch=1)
assert int(M_NEW.epoch) == int(FI["epoch"]), "fpnet2.pt differs from the one k30 used"
use = json.load(open(paths.find("use_fwd.json")))
assert use["file"] == FI["fwd_file"], "use_fwd.json differs from the one k30 used"
FWD = fwd.FwdModel(paths.find(use["file"]))
MR = fpnet.FPModel(paths.find("fprich.pt"))
G2M = g2.G2Model(paths.find("g2.pt"), G=GR)
assert int(MR.epoch) == int(FI["fprich_epoch"]) and MR.nbits == FI["fprich_nbits"], "fprich.pt differs from k30's"
assert int(G2M.step) == int(FI["g2_step"]) and not G2M.amp, "g2.pt differs from k30's"
devs = [FWD.device, MR.device, G2M.device]
assert not REQUIRE_CUDA or all(d == "cuda" for d in devs), f"models must run on the GPU like k30: {devs}"
FT = frag.FragTable(frag.FragTable.find_dir())
S = search.Searcher(C, P, model=M_NEW, fwd=FWD, frag=FT, fprich=(MR, FP2), g2=G2M)
NAMES = S.feature_names + search.GRP_FEATURES
FTAB = subform.FormulaTable(subform.FormulaTable.find_dir())
assert all(c in NAMES for c in cols), "ranker columns are not produced by this Searcher"
idx = [NAMES.index(c) for c in cols]


def pool_order(cand, X):
    XF = np.column_stack([X, search.formula_group_features(
        {c: X[:, S.feature_names.index(c)] for c in search.GRP_BASE}, None, FTAB.fid[cand])])
    rk = rank.predict(models, XF[:, idx].astype(np.float32))
    return cand[search.variant_order(X, rk, V["score"])]


te = pd.read_parquet(os.path.join(COMP, "test.parquet"))
sample = pd.read_csv(os.path.join(COMP, "sample_submission.csv"))
print(f"variant {VARIANT} ({len(cols)} columns) | test: {len(te):,} spectra / {te.molecule_id.nunique():,} molecules "
      f"({time.time() - T0:.0f}s)", flush=True)

# %%
DEFAULT_ADDUCT = {1: "[M+H]+", -1: "[M-H]-"}
lists, rows = {}, []
stats = dict(molecules=0, ranked=0, no_candidates=0, no_mass=0, padded=0, salt_fixed=0, salt_dropped=0, own_len=[])


def hygiene(items, k, count=False):
    """Submission hygiene of k36 / k43: parent component of salts / mixtures, repeated SMILES skipped."""
    clean, seen = [], set()
    for s in items:
        p = chem.parent_smiles(s)
        if count and p != s:
            stats["salt_fixed" if p else "salt_dropped"] += 1
        if p and p not in seen:
            seen.add(p)
            clean.append(p)
        if len(clean) == k:
            break
    return clean


for i, (mid, g) in enumerate(te.groupby("molecule_id", sort=False)):
    stats["molecules"] += 1
    specs = harness.test_specs(g)
    cand, X, info = S.features(specs)
    order = pool_order(cand, X) if len(cand) else np.empty(0, np.int64)
    stats["ranked" if len(order) else "no_candidates"] += 1
    own = hygiene([P.smiles[p] for p in order[:3 * K_OWN]], K_OWN)  # 融合へ: 窓の候補だけ（質量の近い詰め物は入れない）
    key_of = {}
    for p in order[:3 * K_OWN]:
        key_of.setdefault(chem.parent_smiles(P.smiles[p]), P.key[p])
    lists[str(mid)] = dict(smiles=own, keys=[key_of.get(s) for s in own], n_window=int(len(cand)))
    stats["own_len"].append(len(own))
    # 確かめ用の提出（k36 v12 と同じ作り: 25 件に足りなければ質量の近いプールの構造で埋める）
    items = [P.smiles[p] for p in order]
    target = info["target"]
    if not np.isfinite(target):
        stats["no_mass"] += 1
        target = chem.neutral_mass(specs[0]["prec"], DEFAULT_ADDUCT[specs[0]["mode"]])
    if len(items) < 25 and np.isfinite(target):
        seen = set(items)
        for p in S.nearest_structures(target, 60):
            if P.smiles[p] not in seen:
                items.append(P.smiles[p])
                seen.add(P.smiles[p])
            if len(items) >= 25:
                break
        stats["padded"] += 1
    clean = hygiene(items, 25, count=True)
    rows.append((mid, ";".join(clean) if clean else "CCO"))
    if (i + 1) % 50 == 0:
        print(f"  {i + 1} molecules, {time.time() - T0:.0f}s", flush=True)

# %%
n = np.array(stats.pop("own_len") or [0])
stats.update(own_len_min=int(n.min()), own_len_mean=round(float(n.mean()), 2), own_len_max=int(n.max()))
sub = sample[["molecule_id"]].merge(pd.DataFrame(rows, columns=["molecule_id", "smiles"]), on="molecule_id", how="left")
sub["smiles"] = sub["smiles"].fillna("CCO")
parts = sub.smiles.map(lambda s: s.split(";"))
assert sub.molecule_id.is_unique and parts.map(len).max() <= 25 and (parts.map(lambda x: len(x) - len(set(x))) == 0).all()
sub.to_csv(os.path.join(OUT, "own_submission.csv"), index=False)
json.dump(lists, open(os.path.join(OUT, "own_lists.json"), "w"))
info = dict(variant=VARIANT, k_own=K_OWN, stats=stats, secs=round(time.time() - T0), rdkit=rdkit.__version__,
            torch=torch.__version__, devices=devs, src_sha=os.environ.get("CASMI26_SHA"), root=ROOT, comp=COMP)
json.dump(info, open(os.path.join(OUT, "own_stats.json"), "w"), indent=1)
print("OWN_STATS", json.dumps(info), flush=True)

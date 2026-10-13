# %% [markdown]
# # casmi26-k36-submit7
# M7（2026-10-13）: 名前で指定した案の提出ファイルを作る（GPU、インターネットなし）。LB の点を貯めるための提出の列
# （計画ファイルのキュー）から 1 本ずつ、VARIANT を書き換えて version を重ねる。
# - VARIANT: '<列の組>|<点数>+gen:<枠>:<門>:<生成の点数>' または '<列の組>|<点数>+nogen'（search.parse_variant）。
#   点数は hybm:τ:w（k32 と同じ）、hybb:τlo:τhi:wlo:whi（帯で w を変える）、hybi:τlo:τhi:w（帯で規則とランカーを交互に）。
# - ランカー: 評価のカーネルの出力から列の組の名前で引く。k35 の chosen7.json があれば ranker7_* / gen_ranker7_*、
#   無ければ k31 の ranker6_* / gen_ranker6_*。特徴量は k30 と同じ（v2b は CPU、fwd / fprich / G2 は GPU の fp32）
# - 生成側: lib_grp_max < 門 の分子で casmi26.gen（k30 と同じ手順）。上位を枠に入れ、残りはプール側の順で埋める（k32）
# - 提出の衛生（全ての version に共通）: 塩・混合物は親の成分に直し（chem.parent_smiles）、同じ SMILES が重なれば
#   後ろのものを飛ばして次の候補で埋める。直す前の提出も submission_raw.csv に残す
# - 25 件に足りなければ質量の近いプールの構造で埋める（k24 と同じ）
# - variant.json に案の名前、手元の評価の数字（評価の表にあれば）、src sha を残す（tools/wandb_lb.py が読む）
# G2 のグラフ（プール全体）は CUDA を使い始める前に作る（multiprocessing の fork のため）。
# 入力: 競技データ、k01、k05、k05b、k15-s1、k20、k21、k25、k26（pool_fp2.npy、pool_formula.npz）、k31（と k35）、casmi26-fprich、
# casmi26-g2、casmi26-wheels。

# %%
import glob
import os
import subprocess
import sys

wh = sorted(glob.glob("/kaggle/input/**/rdkit-*.whl", recursive=True))
assert wh, "attach the casmi26-wheels dataset (rdkit wheels)"
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--find-links",
                    os.path.dirname(wh[0]), "rdkit"], capture_output=True, text=True)
print(r.stdout[-300:], r.stderr[-300:])
import rdkit

print("rdkit", rdkit.__version__)

# %%
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from casmi26 import cache, chem, fpnet, frag, fwd, g2, gen, harness, paths, pool, rank, search, subform

T0 = time.time()
VARIANT = "both_fprich_g2|hybm:0.8:5+gen:S3:0.7:rk"  # 提出の列の 1 番
REQUIRE_CUDA = True  # fwd / fprich / G2 は k30 と同じ GPU で（手元の動作確認は False）
GRAPH_WORKERS = 4
V = search.parse_variant(VARIANT)
print("variant:", VARIANT, V)
tag = V["cols"]
EVS = []  # 評価の出力: k35（7）を先に、無い列の組は k31（6）から
for ver in ("7", "6"):
    try:
        EVS.append((ver, os.path.dirname(paths.find(f"chosen{ver}.json", roots=["/kaggle/input"]))))
    except FileNotFoundError:
        pass
VER, EV = next((v, d) for v, d in EVS if os.path.exists(f"{d}/ranker{v}_{tag}_0.txt"))
ch = json.load(open(f"{EV}/chosen{VER}.json"))
FI = ch["features"]
assert all(json.load(open(f"{d}/chosen{v}.json"))["features"] == FI for v, d in EVS), "eval outputs from different k30"
print(f"rankers from the eval{VER} output {EV} | k30 features:", FI)
assert rdkit.__version__ == FI["rdkit"], f"rdkit {rdkit.__version__} differs from k30's"
USE_RICH, USE_G2, USE_GRP = "fprich" in tag, "g2" in tag, "_grp" in tag
models = [lgb.Booster(model_file=f"{EV}/ranker{VER}_{tag}_{i}.txt") for i in range(3)]
cols = models[0].feature_name()
assert all(m.feature_name() == cols for m in models)
GEN = V["gen"]
if GEN:
    gname = {"rk": f"gen_ranker{VER}", "rkx": f"gen_ranker{VER}x"}[GEN["score"]]
    GV, GD = next((v, d) for v, d in [(VER, EV)] + EVS if os.path.exists(f"{d}/{gname.replace(VER, v, 1)}_0.txt"))
    gname = gname.replace(VER, GV, 1)
    gen_models = [lgb.Booster(model_file=f"{GD}/{gname}_{i}.txt") for i in range(3)]
    print("generation ranker", gname, "from", GD)
    gen_cols = gen_models[0].feature_name()
    assert gen_cols == list(gen.GEN_FEATURES) or gen_cols == list(gen.GEN_FEATURES) + list(gen.GEN_EXTRA), gen_cols
    GEN_X = len(gen_cols) > len(gen.GEN_FEATURES)
    SLOTS, GATE = GEN["slots"], GEN["gate"]
    USE_RICH, USE_G2 = USE_RICH or GEN_X, USE_G2 or GEN_X

C = cache.Cache(cache.Cache.find_dir())
P = pool.Pool(pool.Pool.find_dir())
assert len(P) == P.n_base, "the PubChem part must not be attached"
GR = g2.Graphs.build(P.smiles, GRAPH_WORKERS) if USE_G2 else None  # CUDA より先に（fork）
FP2 = np.load(paths.find("pool_fp2.npy")) if USE_RICH else None
if USE_RICH:
    assert FP2.shape == (len(P), (chem.RICH_FP_BITS + 7) // 8), FP2.shape

import torch

print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), f"({time.time() - T0:.0f}s)", flush=True)
M_NEW = fpnet.load_checked("fpnet2.pt", min_epoch=1)  # k30 と同じく CPU
assert int(M_NEW.epoch) == int(FI["epoch"]), "fpnet2.pt differs from the one k30 used"
use = json.load(open(paths.find("use_fwd.json")))
assert use["file"] == FI["fwd_file"], "use_fwd.json differs from the one k30 used"
FWD = fwd.FwdModel(paths.find(use["file"]))
MR = fpnet.FPModel(paths.find("fprich.pt")) if USE_RICH else None
G2M = g2.G2Model(paths.find("g2.pt"), G=GR) if USE_G2 else None
if USE_RICH:
    assert int(MR.epoch) == int(FI["fprich_epoch"]) and MR.nbits == FI["fprich_nbits"], "fprich.pt differs from k30's"
if USE_G2:
    assert int(G2M.step) == int(FI["g2_step"]) and not G2M.amp, "g2.pt differs from k30's"
devs = [FWD.device] + ([MR.device] if USE_RICH else []) + ([G2M.device] if USE_G2 else [])
assert not REQUIRE_CUDA or all(d == "cuda" for d in devs), f"models must run on the GPU like k30: {devs}"
FT = frag.FragTable(frag.FragTable.find_dir())
S = search.Searcher(C, P, model=M_NEW, fwd=FWD, frag=FT, fprich=(MR, FP2) if USE_RICH else None,
                    g2=G2M if USE_G2 else None)
NAMES = S.feature_names + (search.GRP_FEATURES if USE_GRP else [])
FTAB = subform.FormulaTable(subform.FormulaTable.find_dir()) if USE_GRP else None
assert all(c in NAMES for c in cols), "ranker columns are not produced by this Searcher"
idx = [NAMES.index(c) for c in cols]
print(f"pool: {tag} {V['score']} ({len(cols)} columns) | generation: {GEN}")


def pool_order(cand, X):
    XF = X
    if USE_GRP:  # 同じ分子式の中の列（k35 と同じ定義。提出では窓の全候補で数える）
        XF = np.column_stack([X, search.formula_group_features(
            {c: X[:, S.feature_names.index(c)] for c in search.GRP_BASE}, None, FTAB.fid[cand])])
    rk = rank.predict(models, XF[:, idx].astype(np.float32))
    return list(cand[search.variant_order(X, rk, V["score"])])


def gen_order(specs, cand, info):
    """Product SMILES of one molecule in the generation ranker's order."""
    z_rich = S.last["z_rich"]  # gen.for_query は S.features の直後に呼ぶ（S.last を使う）
    prods, Xg, _ = gen.for_query(S, specs, cand, info, fwd=FWD)
    if not prods:
        return []
    if GEN_X:
        Xg = np.column_stack([Xg, gen.extra_features(prods, specs, z_rich, G2M)])
    sc = rank.predict(gen_models, Xg.astype(np.float32))
    return [prods[i]["smiles"] for i in np.argsort(-sc, kind="mergesort")]


COMP = paths.comp_dir()
te = pd.read_parquet(os.path.join(COMP, "test.parquet"))
sample = pd.read_csv(os.path.join(COMP, "sample_submission.csv"))
print(f"test: {len(te):,} spectra / {te.molecule_id.nunique():,} molecules")

# %%
DEFAULT_ADDUCT = {1: "[M+H]+", -1: "[M-H]-"}
rows, rows_raw = [], []
stats = dict(ranked=0, no_candidates=0, no_mass=0, padded=0, gated=0, with_gen=0, gen_slotted=0, salt_fixed=0,
             salt_dropped=0)
for i, (mid, g) in enumerate(te.groupby("molecule_id", sort=False)):
    specs = harness.test_specs(g)
    cand, X, info = S.features(specs)
    order = pool_order(cand, X) if len(cand) else []
    stats["ranked" if order else "no_candidates"] += 1
    items = [P.smiles[p] for p in order]
    if GEN and info["lib_max"] < GATE:
        stats["gated"] += 1
        g_smi = gen_order(specs, cand, info) if len(cand) else []
        stats["with_gen"] += bool(g_smi)
        m = min(len(g_smi), len(SLOTS))
        stats["gen_slotted"] += m
        slots, out, gi, pi = set(SLOTS[:m]), [], 0, 0
        for pos in range(1, 26):
            if pos in slots:
                out.append(g_smi[gi])
                gi += 1
            elif pi < len(items):
                out.append(items[pi])
                pi += 1
        items = out + items[pi:]  # 25 位より後ろはプール側の残り（衛生で抜けた分を埋める）
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
    rows_raw.append((mid, ";".join(items[:25]) if items else "CCO"))
    clean, seen = [], set()
    for s in items:
        p = chem.parent_smiles(s)
        if p != s:
            stats["salt_fixed" if p else "salt_dropped"] += 1
        if p and p not in seen:
            seen.add(p)
            clean.append(p)
        if len(clean) == 25:
            break
    rows.append((mid, ";".join(clean) if clean else "CCO"))
    if (i + 1) % 100 == 0:
        print(f"  {i + 1} molecules, {time.time() - T0:.0f}s", flush=True)
print(stats)

# %%
def write(rows, path):
    sub = sample[["molecule_id"]].merge(pd.DataFrame(rows, columns=["molecule_id", "smiles"]), on="molecule_id",
                                        how="left")
    sub["smiles"] = sub["smiles"].fillna("CCO")
    parts = sub.smiles.map(lambda s: s.split(";"))
    assert len(sub) == len(sample) and sub.molecule_id.is_unique and sub.smiles.notna().all()
    assert parts.map(len).max() <= 25 and (parts.map(lambda x: len(x) - len(set(x))) == 0).all(), "row too long / dup"
    sub.to_csv(path, index=False)
    return sub


write(rows_raw, "submission_raw.csv")
sub = write(rows, "submission.csv")
local = {}
try:
    rt = pd.read_csv(f"{EV}/rule{VER}.csv").set_index("model")
    # k35 は生成まで含む名前。k31 の表はプール側の名前だけで、生成は S3:0.7:rk に固定
    names = [VARIANT] + ([VARIANT.split("+")[0]] if VARIANT.endswith("+gen:S3:0.7:rk") else [])
    key = next((k for k in names if k in rt.index), None)
    if key:
        local = {k: float(rt.loc[key, k]) for k in ["A", "B_A", "B_N", "G_N", "dA", "dB_A", "dB_N", "dG_N",
                                                    "dA_lo", "dB_N_lo", "pred_lb_delta", "eligible", "report_only"]
                 if k in rt.columns}  # eligible = 選び方を通った案（False は較正のための提出）
        local["eval_model"] = key
except FileNotFoundError:
    pass
json.dump(dict(variant=VARIANT, parsed=V, eval=f"eval{VER}", local=local, stats=stats,
               src_sha=os.environ.get("CASMI26_SHA"), rdkit=rdkit.__version__), open("variant.json", "w"), indent=1)
print("local eval of this pool side:", local)
print(sub.head(), f"\n{time.time() - T0:.0f}s")

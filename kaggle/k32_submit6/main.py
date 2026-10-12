# %% [markdown]
# # casmi26-k32-submit6
# M6 P1 の 3（2026-10-12）: k31 が選んだもの（chosen6.json）の提出ファイルを作る（GPU、インターネットなし）。
# k31 が k24 v1 のまま（keep_k24）なら、このカーネルは使わない（k24 v1 の提出がそのまま）。
# - プール側: k30 と同じ特徴量（v2b の FPNet は CPU、fwd / fprich / G2 は GPU の fp32、断片の表）と k31 のランカー
#   （ranker6_{tag}_*）、hybm:τ:w
# - 生成側: lib_grp_max < 門 の分子で casmi26.gen（k30 と同じ手順）。生成物の一覧を k31 の生成のランカー
#   （gen_ranker6_*、GEN_FEATURES）で並べ、上位を決まった枠（S3）に入れ、残りはプール側の順で埋める
# - 25 件に足りなければ質量の近いプールの構造で埋める（k24 と同じ）。1 行の中で同じ SMILES が無いことを確かめる
# G2 のグラフ（プール全体）は CUDA を使い始める前に作る（multiprocessing の fork のため）。
# CHOSEN を書けば k31 の選択の代わりにそれを出す（手元の動作確認と、ユーザーが LB で確かめたいときだけ）。
# 入力: 競技データ、k01、k05、k05b、k15-s1、k20、k21、k25、k26（pool_fp2.npy）、k31、casmi26-fprich、casmi26-g2、
# casmi26-wheels。

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

from casmi26 import cache, chem, fpnet, frag, fwd, g2, gen, harness, paths, pool, rank, search

T0 = time.time()
CHOSEN = None  # None = k31 の chosen6.json
REQUIRE_CUDA = True  # fwd / fprich / G2 は k30 と同じ GPU で（手元の動作確認は False）
GRAPH_WORKERS = 4
K31 = os.path.dirname(paths.find("chosen6.json", roots=["/kaggle/input"]))
ch = json.load(open(f"{K31}/chosen6.json"))
CH = CHOSEN or ch["chosen"]
FI = ch["features"]
print("chosen:", {k: v for k, v in CH.items() if k not in ("columns", "gen_columns")}, "| k30 features:", FI)
assert CHOSEN or not CH.get("keep_k24"), "k31 kept k24 v1: submit k24 v1, this kernel is not needed"
assert rdkit.__version__ == FI["rdkit"], f"rdkit {rdkit.__version__} differs from k30's"
tag, var = CH["fcols"], CH["variant"].split(":")
assert var[0] == "hybm"
tau, w = float(var[1]), float(var[2])
USE_RICH, USE_G2 = "fprich" in tag, "g2" in tag

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
cols = list(CH["columns"])
assert all(c in S.feature_names for c in cols), "chosen columns are not produced by this Searcher"
idx = [S.feature_names.index(c) for c in cols]
models = [lgb.Booster(model_file=f"{K31}/ranker6_{tag}_{i}.txt") for i in range(3)]
assert models[0].feature_name() == cols, "ranker columns differ from chosen6.json"
SLOTS, GATE = list(CH["slots"]), float(CH["gate"])
assert CH["score"] == "rk" and list(CH["gen_columns"]) == list(gen.GEN_FEATURES)
gen_models = [lgb.Booster(model_file=f"{K31}/gen_ranker6_{i}.txt") for i in range(3)]
print(f"pool: {tag} hybm:{tau}:{w} ({len(cols)} columns) | generation: slots {SLOTS}, gate {GATE}")


def pool_order(cand, X):
    rk = rank.predict(models, X[:, idx].astype(np.float32))
    return list(cand[np.argsort(-search.blend_score(X, tau, rk, w), kind="mergesort")])


def gen_order(specs, cand, info):
    """Product SMILES of one molecule in the generation ranker's order."""
    prods, Xg, _ = gen.for_query(S, specs, cand, info, fwd=FWD)
    if not prods:
        return []
    sc = rank.predict(gen_models, Xg.astype(np.float32))
    return [prods[i]["smiles"] for i in np.argsort(-sc, kind="mergesort")]


COMP = paths.comp_dir()
te = pd.read_parquet(os.path.join(COMP, "test.parquet"))
sample = pd.read_csv(os.path.join(COMP, "sample_submission.csv"))
print(f"test: {len(te):,} spectra / {te.molecule_id.nunique():,} molecules")

# %%
DEFAULT_ADDUCT = {1: "[M+H]+", -1: "[M-H]-"}
rows = []
stats = dict(ranked=0, no_candidates=0, no_mass=0, padded=0, gated=0, with_gen=0, gen_slotted=0)
for i, (mid, g) in enumerate(te.groupby("molecule_id", sort=False)):
    specs = harness.test_specs(g)
    cand, X, info = S.features(specs)
    order = pool_order(cand, X) if len(cand) else []
    stats["ranked" if order else "no_candidates"] += 1
    items = [P.smiles[p] for p in order]
    if info["lib_max"] < GATE:
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
        items = out
    items = items[:25]
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
    rows.append((mid, ";".join(items[:25]) if items else "CCO"))
    if (i + 1) % 100 == 0:
        print(f"  {i + 1} molecules, {time.time() - T0:.0f}s", flush=True)
print(stats)

# %%
sub = sample[["molecule_id"]].merge(pd.DataFrame(rows, columns=["molecule_id", "smiles"]), on="molecule_id", how="left")
sub["smiles"] = sub["smiles"].fillna("CCO")
parts = sub.smiles.map(lambda s: s.split(";"))
assert len(sub) == len(sample) and sub.molecule_id.is_unique and sub.smiles.notna().all()
assert parts.map(len).max() <= 25 and (parts.map(lambda x: len(x) - len(set(x))) == 0).all(), "row too long or duplicated"
sub.to_csv("submission.csv", index=False)
print(sub.head(), f"\n{time.time() - T0:.0f}s")

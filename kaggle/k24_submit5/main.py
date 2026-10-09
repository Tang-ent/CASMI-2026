# %% [markdown]
# # casmi26-k24-submit5
# M5 P1 の 4（2026-10-10）: k23 が選んだもの（chosen.json）の提出ファイルを作る（インターネットなし）。
# - プール側の土台: M2.2 なら k09 v4 と同じ道（k06 の fpnet.pt、k08 の ranker.json の hybm:0.7:2）。
#   P* なら k22 と同じ特徴量（v2b の FPNet、use_fwd.json の fwd、断片の表）と k23 のランカー
# - 生成側が選ばれていれば: lib_grp_max < 門 の分子で casmi26.gen（k22 と同じ手順、v2b の z、fwd、断片）。
#   生成物の一覧（comb または k23 の gen ranker で並べる）の上位を決まった枠に入れ、残りはプール側の順で埋める
# - 25 件に足りなければ質量の近いプールの構造で埋める（k17 と同じ）
# rdkit はデータセット casmi26-wheels の wheel から入れる。CHOSEN を書けば k23 の選択の代わりにそれを出す
# （LB で確かめたいときだけ。ユーザーの判断）。
# 入力: 競技データ、k01、k05、k05b、casmi26-fpnet（k06）、k08、k15-s1、k20、k21、k23、k25、casmi26-wheels。

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
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from casmi26 import cache, chem, fpnet, frag, fwd, gen, harness, paths, pool, rank, search

T0 = time.time()
CHOSEN = None  # None = k23 の chosen.json
K23 = os.path.dirname(paths.find("chosen.json", roots=["/kaggle/input"]))
ch = json.load(open(f"{K23}/chosen.json"))
CH = CHOSEN or ch["chosen"]
print("chosen:", CH, "| k23 features:", ch["features"])
assert rdkit.__version__ == ch["features"]["rdkit"], f"rdkit {rdkit.__version__} differs from k22's"
if CH["kind"] == "none" and not CHOSEN:
    print("k23 chose M2.2 without generation: the current M2.2 submission (k09 v4) already is this; writing it anyway")

C = cache.Cache(cache.Cache.find_dir())
P = pool.Pool(pool.Pool.find_dir())
assert len(P) == P.n_base, "the PubChem part must not be attached"
M_NEW = fpnet.load_checked("fpnet2.pt", min_epoch=1)
assert int(M_NEW.epoch) == int(ch["features"]["epoch"]), "fpnet2.pt differs from the one k22 / k23 used"
use = json.load(open(paths.find("use_fwd.json")))
assert use["file"] == ch["features"]["fwd_file"], "use_fwd.json differs from the one k22 used"
FWD = fwd.FwdModel(paths.find(use["file"]), device="cpu")
FT = frag.FragTable(frag.FragTable.find_dir())
S_NEW = search.Searcher(C, P, model=M_NEW, fwd=FWD, frag=FT)
NEED_NEW = CH["base"] != "M2.2" or CH["kind"] == "gen"

if CH["base"] == "M2.2":
    M_OLD = fpnet.load_checked()  # k06 の fpnet.pt（エポック 14）
    S_OLD = search.Searcher(C, P, model=M_OLD)
    RK_DIR = os.path.dirname(paths.find("ranker.json"))
    v = json.load(open(f"{RK_DIR}/ranker.json"))["variants"]["hybm:0.7:2"]
    old_models = [lgb.Booster(model_file=f"{RK_DIR}/{fn}") for fn in v["files"]]
    old_idx = [search.COL[c] for c in v["features"]]

    def pool_order(specs):
        cand, X, info = S_OLD.features(specs)
        if not len(cand):
            return [], info
        rk = np.mean([m.predict(X[:, old_idx]) for m in old_models], axis=0)
        return list(cand[np.argsort(-search.blend_score(X, v["tau"], rk, v["w"]), kind="mergesort")]), info
else:
    tag, var = CH["fcols"], CH["variant"].split(":")
    assert var[0] == "hybm"
    tau, w = float(var[1]), float(var[2])
    new_models = [lgb.Booster(model_file=f"{K23}/ranker_{tag}_{i}.txt") for i in range(3)]
    FCOLS = {"base": search.FEATURES, "frag": search.FEATURES + frag.FRAG_FEATURES,
             "fwd": search.FEATURES + search.FWD_FEATURES,
             "both": search.FEATURES + frag.FRAG_FEATURES + search.FWD_FEATURES}  # k23 と同じ
    assert list(CH.get("columns", FCOLS[tag])) == FCOLS[tag], "feature columns differ from k23"
    new_idx = [S_NEW.feature_names.index(c) for c in FCOLS[tag]]

    def pool_order(specs, res=None):
        cand, X, info = res or S_NEW.features(specs)
        if not len(cand):
            return [], info
        rk = rank.predict(new_models, X[:, new_idx].astype(np.float32))
        return list(cand[np.argsort(-search.blend_score(X, tau, rk, w), kind="mergesort")]), info

if CH["kind"] == "gen":
    SLOTS, GATE, GSCORE = list(CH["slots"]), float(CH["gate"]), CH["score"]
    gen_models = [lgb.Booster(model_file=f"{K23}/gen_ranker_{i}.txt") for i in range(3)] if GSCORE == "rk" else None
    print("generation:", dict(slots=SLOTS, gate=GATE, score=GSCORE))


def gen_order(specs, cand, info):
    """Product SMILES of one molecule in the chosen generated-list order."""
    prods, Xg, _ = gen.for_query(S_NEW, specs, cand, info, fwd=FWD)
    if not prods:
        return []
    if GSCORE == "rk":
        sc = rank.predict(gen_models, Xg.astype(np.float32))
    else:
        G = gen.GCOL
        ap = Xg[:, G["ap4"]]
        sd = ap.std()
        sc = Xg[:, G["fz_z"]] + ((ap - ap.mean()) / sd if sd > 1e-9 else 0.0) + Xg[:, G["fwd_z"]] + Xg[:, G["frag_z"]]
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
    res_new = S_NEW.features(specs) if NEED_NEW else None
    order, info = pool_order(specs, res_new) if CH["base"] != "M2.2" else pool_order(specs)
    if order:
        stats["ranked"] += 1
    else:
        stats["no_candidates"] += 1
    items = [P.smiles[p] for p in order]
    if CH["kind"] == "gen" and info["lib_max"] < GATE:
        stats["gated"] += 1
        cand_n, _, info_n = res_new
        g_smi = gen_order(specs, cand_n, info_n) if len(cand_n) else []
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
        for p in S_NEW.nearest_structures(target, 60):
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
assert len(sub) == len(sample) and sub.molecule_id.is_unique and sub.smiles.notna().all()
assert sub.smiles.map(lambda s: len(s.split(";"))).max() <= 25
sub.to_csv("submission.csv", index=False)
print(sub.head(), f"\n{time.time() - T0:.0f}s")

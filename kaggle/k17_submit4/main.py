# %% [markdown]
# # casmi26-k17-submit4
# M4 の提出用ノートブック（インターネット無効、2026-10-07）。k09 v4 を土台に、k16_eval4 が固定した選び方で選んだ
# 候補（chosen.json）の提出ファイルを作る。
# - FPNet: k15 の fpnet2.pt（k07 v8 の特徴量と同じもの。k16 の chosen.json に書かれたエポックと一致することを確かめる）
# - ランカー: k16 の ranker_lgb_R_{0,1,2}.txt。振り分け器: k16 の router_{0..4}.txt（route:* のときだけ使う）
# - 候補の名前: hyb:τ / hybm:τ:w / hybmT:τ:w / route:hard:t:w / route:soft:w、末尾の |ppmX は |mass_ppm| ≤ X の
#   候補だけで並べる。hybmT はランカーに q_tims（クエリが timsTOF か）を足したもの（k16 v3。テストはすべて timsTOF なので 1）
# - プールは train ∪ COCONUT（PubChem はつながないことを確かめる）

# %%
import glob
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from casmi26 import cache, chem, fpnet, harness, paths, pool, rank, search

T0 = time.time()
# ユーザーの決定（2026-10-09）: 決めた選び方では合格した候補が無いが、LB で一度確かめる。B_A（timsTOF、250 件）と
# B_N（timsTOF 以外、1,096 件）のどちらがテストのクラス2に近いかを知る手がかりに、予測の分かれる振り分け器を出す
# （k16 v3: A −0.044、B_A +0.149、B_N +0.037。予測 LB は B_N で当てはめると 0.286、B_A で当てはめると 0.313）。
VARIANT = "route:hard:0.7:5"  # None = k16 の chosen.json の選択

K16 = os.path.dirname(paths.find("chosen.json", roots=["/kaggle/input"]))
ch = json.load(open(f"{K16}/chosen.json"))
VARIANT = VARIANT or (ch["chosen"] or {}).get("model")
assert VARIANT, "k16 chose nothing (keep M2.2); set VARIANT to submit a candidate anyway"
MODEL = fpnet.load_checked("fpnet2.pt", min_epoch=1)
res15 = [json.load(open(p)) for p in glob.glob("/kaggle/input/**/result_v2*.json", recursive=True)]
assert len(res15) == 1, f"attach exactly one k15 variant (found {len(res15)})"
assert ch["features"].get("fpnet") == "fpnet2.pt" and int(ch["features"]["epoch"]) == int(MODEL.epoch) \
    and ch["features"]["variant"] == res15[0]["variant"], \
    f"FPNet {res15[0]['variant']} epoch {MODEL.epoch} differs from the one k16 evaluated: {ch['features']}"
C = cache.Cache(cache.Cache.find_dir())
P = pool.Pool(pool.Pool.find_dir())
assert len(P) == P.n_base, "the PubChem part must not be attached"
S = search.Searcher(C, P, model=MODEL)
rankers = [lgb.Booster(model_file=f"{K16}/ranker_lgb_R_{i}.txt") for i in range(3)]
rankers_t = ([lgb.Booster(model_file=f"{K16}/ranker_lgb_RT_{i}.txt") for i in range(3)]
             if VARIANT.startswith("hybmT:") else None)
routers = [lgb.Booster(model_file=f"{K16}/router_{k}.txt") for k in range(5)]
name, _, cut = VARIANT.partition("|ppm")
cut = float(cut) if cut else None
parts = name.split(":")


def score_fn(X):
    """Score of one molecule's candidates for the chosen variant (rows of X)."""
    if parts[0] == "hybmT":
        rk_t = rank.predict(rankers_t, np.c_[X, np.ones(len(X))].astype(np.float32))  # q_tims = 1
        return search.blend_score(X, float(parts[1]), rk_t, float(parts[2]))
    rk = rank.predict(rankers, X.astype(np.float32))
    if parts[0] == "hyb":
        return search.hybrid_score(X, float(parts[1]), rk)
    if parts[0] == "hybm":
        return search.blend_score(X, float(parts[1]), rk, float(parts[2]))
    if parts[0] == "route":
        _, R = search.route_features(X, rk)
        p = float(np.mean([m.predict(R) for m in routers]))
        if parts[1] == "hard":
            return search.route_score(X, rk, np.full(len(X), p), float(parts[3]), "hard", float(parts[2]))
        return search.route_score(X, rk, np.full(len(X), p), float(parts[2]), "soft")
    raise ValueError(VARIANT)


print("variant:", VARIANT, "| mass cut:", cut, "| FPNet:", ch["features"])
COMP = paths.comp_dir()
te = pd.read_parquet(os.path.join(COMP, "test.parquet"))
sample = pd.read_csv(os.path.join(COMP, "sample_submission.csv"))
print(f"test: {len(te):,} spectra / {te.molecule_id.nunique():,} molecules")

# %%
DEFAULT_ADDUCT = {1: "[M+H]+", -1: "[M-H]-"}
rows, stats = [], dict(ranked=0, no_candidates=0, no_mass=0, padded=0, top1_in_train=0, cut_rows=0)
for i, (mid, g) in enumerate(te.groupby("molecule_id", sort=False)):
    specs = harness.test_specs(g)
    cand, X, info = S.features(specs)
    order = []
    if len(cand):
        sc = score_fn(X)
        if cut is not None:
            keep = X[:, search.COL["mass_ppm"]] <= cut
            stats["cut_rows"] += int((~keep).sum())
            cand, sc = cand[keep], sc[keep]
        order = list(cand[np.argsort(-sc, kind="mergesort")][:25])
    if order:
        stats["ranked"] += 1
        stats["top1_in_train"] += int(P.in_train[order[0]])
    else:
        stats["no_candidates"] += 1
    target = info["target"]
    if not np.isfinite(target):
        stats["no_mass"] += 1
        target = chem.neutral_mass(specs[0]["prec"], DEFAULT_ADDUCT[specs[0]["mode"]])
    if len(order) < 25 and np.isfinite(target):
        seen = set(order)
        for p in S.nearest_structures(target, 60):
            if p not in seen:
                order.append(p)
                seen.add(p)
            if len(order) >= 25:
                break
        stats["padded"] += 1
    rows.append((mid, ";".join(P.smiles[p] for p in order[:25]) if order else "CCO"))
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

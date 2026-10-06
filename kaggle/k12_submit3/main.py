# %% [markdown]
# # casmi26-k12-submit3
# M3 の提出用ノートブック（インターネット無効）。k09_submit2 v4（M2.2）を写して変えたもの。M2.2 の提出に使った
# k09 / k08 を上書きしないために、別のカーネルにした。
# 候補プール = k05（train ∪ COCONUT）+ PubChem 部分（段階 S′ = 文献数 ≥ 1。データセット casmi26-pubchem-part = k05p v1 の出力）。つながっていなければ止める
# （Pool は pc_meta.json が無いと黙って PubChem なしで動くため）。スコアは k11_eval3 が選んだ variant
# （1 段目の勝者。S′ が A の基準を満たさなかった場合は default が null なので止める）。

# %%
import json
import os
import time

import numpy as np
import pandas as pd

from casmi26 import cache, chem, fpnet, harness, paths, pool, search

T0 = time.time()
VARIANT = None  # None = ranker.json の既定

MODEL = fpnet.load_checked()  # k06 の重み（データセット casmi26-fpnet。エポック 10 未満なら止める）
C = cache.Cache(cache.Cache.find_dir())
P = pool.Pool(pool.Pool.find_dir())
assert P.level.max() == 1 and len(P) > P.n_base, "PubChem part (dataset casmi26-pubchem-part) is not attached"
print(f"PubChem part: {len(P) - P.n_base:,} structures, levels {P.pc_meta['levels']}")
S = search.Searcher(C, P, model=MODEL)
RK_DIR = os.path.dirname(paths.find("ranker.json"))
rk = json.load(open(f"{RK_DIR}/ranker.json"))
VARIANT = VARIANT or rk["default"]
assert VARIANT, "k11 found no S' variant that passes the A bar (ranker.json default is null)"
print("k11 stage-2 recommendation:", rk.get("recommend"))
if VARIANT.startswith("rule:"):
    score_fn = lambda X: search.rule_score(X, VARIANT.split(":", 1)[1])
else:
    import lightgbm as lgb

    v = rk["variants"][VARIANT]
    models = [lgb.Booster(model_file=f"{RK_DIR}/{fn}") for fn in v["files"]]
    fidx = [search.COL[c] for c in v["features"]]
    rk_fn = lambda X: np.mean([m.predict(X[:, fidx]) for m in models], axis=0)
    if "w" in v:
        score_fn = lambda X: search.blend_score(X, v["tau"], rk_fn(X), v["w"])  # X = one molecule
    elif "tau" in v:
        score_fn = lambda X: search.hybrid_score(X, v["tau"], rk_fn(X))
    else:
        score_fn = rk_fn
    print("variant:", {k: x for k, x in v.items() if k != "features"})
print("score:", VARIANT)

COMP = paths.comp_dir()
te = pd.read_parquet(os.path.join(COMP, "test.parquet"))
sample = pd.read_csv(os.path.join(COMP, "sample_submission.csv"))
print(f"test: {len(te):,} spectra / {te.molecule_id.nunique():,} molecules")

# %%
DEFAULT_ADDUCT = {1: "[M+H]+", -1: "[M-H]-"}
rows, stats = [], dict(ranked=0, no_candidates=0, no_mass=0, padded=0, top1_in_train=0, top1_pubchem=0,
                      pubchem_in_top25=0)
for i, (mid, g) in enumerate(te.groupby("molecule_id", sort=False)):
    specs = harness.test_specs(g)
    cand, X, info = S.features(specs)
    order = []
    if len(cand):
        order = list(cand[np.argsort(-score_fn(X), kind="mergesort")][:25])
        stats["ranked"] += 1
        stats["top1_in_train"] += int(P.in_train[order[0]])
        stats["top1_pubchem"] += int(order[0] >= P.n_base)
        stats["pubchem_in_top25"] += int(sum(p >= P.n_base for p in order))
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

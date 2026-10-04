# %% [markdown]
# # casmi26-k04-submit
# クラス1ベースラインの提出用ノートブック（インターネット無効）。
# k01 のキャッシュと k03 のランカーを使い、分子ごとに train の構造を最大25件ランキングして submission.csv を書く。

# %%
import json
import os
import time

import numpy as np
import pandas as pd

from casmi26 import cache, chem, paths, search, spectra

T0 = time.time()
VARIANT = None  # None = ranker.json の既定。"rule:lib_analog" ならルールベースのスコア

C = cache.Cache(cache.Cache.find_dir())
S = search.Searcher(C)
RK_DIR = os.path.dirname(paths.find("ranker.json"))
rk = json.load(open(f"{RK_DIR}/ranker.json"))
VARIANT = VARIANT or rk["default"]
if VARIANT.startswith("rule:"):
    score_fn = lambda X: search.rule_score(X, VARIANT.split(":", 1)[1])
    print("score:", VARIANT)
else:
    import lightgbm as lgb

    v = rk["variants"][VARIANT]
    models = [lgb.Booster(model_file=f"{RK_DIR}/{fn}") for fn in v["files"]]
    fidx = [search.COL[c] for c in v["features"]]
    score_fn = lambda X: np.mean([m.predict(X[:, fidx]) for m in models], axis=0)
    print("ranker", VARIANT, len(models), "boosters,", len(fidx), "features")

COMP = paths.comp_dir()
te = pd.read_parquet(os.path.join(COMP, "test.parquet"))
sample = pd.read_csv(os.path.join(COMP, "sample_submission.csv"))
print(f"test: {len(te):,} spectra / {te.molecule_id.nunique():,} molecules")

# %%
DEFAULT_ADDUCT = {1: "[M+H]+", -1: "[M-H]-"}
rows, stats = [], dict(ranked=0, no_candidates=0, no_mass=0, padded=0)
smiles = C.struct["smiles"].values
for i, (mid, g) in enumerate(te.groupby("molecule_id", sort=False)):
    specs = []
    for r in g.itertuples():
        mz, it = spectra.clean_one(r.ms2_mzs, r.ms2_normalized_intensities, r.precursor_mz)
        specs.append(dict(mz=mz, it=it, prec=float(r.precursor_mz), adduct=str(r.adduct),
                          mode=1 if r.ionization_mode == "positive" else -1))
    cand, X, info = S.features(specs)
    order = []
    if len(cand):
        order = list(cand[np.argsort(-score_fn(X), kind="mergesort")][:25])
        stats["ranked"] += 1
    else:
        stats["no_candidates"] += 1
    target = info["target"]
    if not np.isfinite(target):
        # 質量表に無いアダクトしか無い場合: 極性から [M+H]+ / [M-H]- を仮定
        stats["no_mass"] += 1
        target = chem.neutral_mass(specs[0]["prec"], DEFAULT_ADDUCT[specs[0]["mode"]])
    if len(order) < 25 and np.isfinite(target):
        seen = set(order)
        for s in S.nearest_structures(target, 60):
            if s not in seen:
                order.append(s)
                seen.add(s)
            if len(order) >= 25:
                break
        stats["padded"] += 1
    rows.append((mid, ";".join(smiles[s] for s in order[:25]) if order else "CCO"))
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

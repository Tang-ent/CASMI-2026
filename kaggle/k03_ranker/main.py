# %% [markdown]
# # casmi26-k03-ranker
# k02 の特徴量で LightGBM lambdarank を学習する（学習は T のみ、A / C は評価専用）。
#
# v1 では C / T は改善したが A（テストに近い天然物）でルールより悪化した。原因の候補は件数系の特徴量
# （lib_rows / cand_nspec / cand_nlib）である。非 enveda-180 の T クエリでは、正解構造のスペクトルだけを隠すため、
# 正解の件数がデコイより少なく見える偏りがある。これを切り分けるため、特徴量と学習クエリを変えた変種を比較する。
#
# 採用基準（結果を見る前に固定）: A >= ルール - 1SE かつ C > ルール。満たす変種が無ければルールを提出に使う。

# %%
import json
import os
import time

import numpy as np
import pandas as pd

from casmi26 import harness, paths, rank, search

T0 = time.time()
OUT = "/kaggle/working"
RULE = "lib_analog"
D = os.path.dirname(paths.find("feats_T.parquet"))
feats = {s: pd.read_parquet(f"{D}/feats_{s}.parquet") for s in ["A", "C", "T"]}
infos = pd.read_csv(f"{D}/query_info.csv")
prev = pd.read_csv(f"{D}/results.csv")
qlib = infos.set_index("qid").qlib
for s, f in feats.items():
    print(s, f.qid.nunique(), "queries", f"{len(f):,} rows", f"positives {f.label.sum()}")


def evaluate(set_name, score, qids=None):
    f = feats[set_name]
    r = harness.ranks_of_truth(f.qid.values, f.label.values, score, f.sid.values)
    allq = infos[infos.set == set_name].qid.values if qids is None else qids
    return harness.metrics(r.reindex(allq).fillna(np.inf).values)


# %%
COUNT = ["lib_rows", "cand_nspec", "cand_nlib"]
ALL = list(search.FEATURES)
NO_COUNT = [c for c in ALL if c not in COUNT]
T_all = feats["T"]
T_e180 = T_all[T_all.qid.map(qlib).eq("enveda-180")]
VARIANTS = {  # name -> (training rows, features)
    "all": (T_all, ALL),
    "no_prior": (T_all, [c for c in ALL if c not in search.PRIOR_FEATURES]),
    "no_count": (T_all, NO_COUNT),
    "e180_train": (T_e180, ALL),
    "e180_no_count": (T_e180, NO_COUNT),
}
SEEDS = (0, 1, 2)
res, imp, summary = [], [], {}
for name, (T, fl) in VARIANTS.items():
    # OOF は学習に使ったクエリだけで評価する（T 全体の行に揃え、対象外の行は最下位扱い）
    oof = pd.Series(rank.cv_oof(T, fl, n_folds=5), index=T.index).reindex(T_all.index, fill_value=-1e9).values
    tq = infos[infos.qid.isin(T.qid.unique())].qid.values
    res.append(dict(set="T", model=f"lgb:{name} (oof)", **evaluate("T", oof, tq)))
    if name == "all":
        for lib in sorted(qlib[tq].unique()):
            res.append(dict(set=f"T[{lib}]", model="lgb:all (oof)", **evaluate("T", oof, tq[qlib[tq].values == lib])))
    models = [rank.fit(T, fl, seed=s) for s in SEEDS]
    for s in ["A", "C"]:
        sc = rank.predict(models, feats[s][fl].values.astype(np.float32))
        m = evaluate(s, sc)
        res.append(dict(set=s, model=f"lgb:{name}", **m))
        summary.setdefault(name, {})[s] = m
    for i, m in enumerate(models):
        m.save_model(f"{OUT}/ranker_{name}_{i}.txt")
    g = np.mean([m.feature_importance("gain") for m in models], axis=0)
    imp.append(pd.Series(g / g.sum(), index=fl, name=name))
    print(name, {s: round(summary[name][s]["mrr25"], 4) for s in ["A", "C"]}, f"{time.time() - T0:.0f}s")

res = pd.concat([prev, pd.DataFrame(res)], ignore_index=True).sort_values(["set", "mrr25"], ascending=[True, False])
res.to_csv(f"{OUT}/results.csv", index=False)
print(res.round(4).to_string(index=False))
imp = pd.concat(imp, axis=1).sort_values("all", ascending=False)
imp.to_csv(f"{OUT}/importance.csv")
print(imp.round(4).to_string())

# %% [markdown]
# ## 提出に使うモデルの選択（採用基準は冒頭のとおり）

# %%
rule_A = evaluate("A", search.rule_score(feats["A"][ALL].values, RULE))
rule_C = evaluate("C", search.rule_score(feats["C"][ALL].values, RULE))
passing = {n: m for n, m in summary.items()
           if m["A"]["mrr25"] >= rule_A["mrr25"] - rule_A["mrr_se"] and m["C"]["mrr25"] > rule_C["mrr25"]}
print(f"rule:{RULE}  A {rule_A['mrr25']:.4f} (se {rule_A['mrr_se']:.4f})  C {rule_C['mrr25']:.4f}")
print("passing variants:", {n: (round(m["A"]["mrr25"], 4), round(m["C"]["mrr25"], 4)) for n, m in passing.items()})
default = max(passing, key=lambda n: passing[n]["C"]["mrr25"]) if passing else f"rule:{RULE}"
print("default for k04:", default)
json.dump(dict(default=default,
               variants={k: dict(features=fl, files=[f"ranker_{k}_{i}.txt" for i in range(len(SEEDS))])
                         for k, (_, fl) in VARIANTS.items()}),
          open(f"{OUT}/ranker.json", "w"), indent=1)
print(f"done {time.time() - T0:.0f}s")

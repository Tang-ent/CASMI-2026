# %% [markdown]
# # casmi26-k08-eval2
# k07 の分割出力を結合し、隠し方のチェック・ルールベースの評価・提出に使うスコアの選択を行う。
#
# 採用基準
# - A ≥ クラス1ベースラインの A − 1SE（0.9338 − 0.0127）を満たすゲートの中で、R だけで計算した
#   0.2·MRR(R1) + 0.8·MRR(R2) が最大のものを選ぶ。さらに B_A がクラス1ルール（lib_analog）より改善すること
# - 満たすものが無ければ lib_analog を提出に使う
# 変更履歴: v1 は「R で最良の τ を1つ選び、A の条件を満たさなければ却下」だった（M2 の計画と食い違い）。
#   v1 の結果（gate:0.9 が A 0.912 で却下）を見たあと、ユーザーの判断で M2 と同じ「条件付きの最大」に揃えた。
# 結果は wandb にオフラインで記録し、完了後に手元で tools/wandb_sync.py で送る。

# %%
import glob
import json
import os

import numpy as np
import pandas as pd

from casmi26 import harness, search, track

OUT = "/kaggle/working"
EXP = "M1"  # wandb の group
A_REF, A_REF_SE = 0.9338, 0.0127  # クラス1ベースライン（k02, train のみのプール）
SETS = ["A", "B_A", "C", "B_C", "R1", "R2"]

infos = pd.concat([pd.read_csv(p) for p in sorted(glob.glob("/kaggle/input/**/query_info_s*.csv", recursive=True))],
                  ignore_index=True).drop_duplicates("qid")
feats = {}
for s in SETS:
    paths_ = sorted(glob.glob(f"/kaggle/input/**/feats_{s}_s*.parquet", recursive=True))
    if paths_:
        feats[s] = pd.concat([pd.read_parquet(p) for p in paths_], ignore_index=True).drop_duplicates(["qid", "pid"])
        feats[s].to_parquet(f"{OUT}/feats_{s}.parquet", index=False)
infos.to_csv(f"{OUT}/query_info.csv", index=False)
print("shards found:", len(glob.glob("/kaggle/input/**/query_info_s*.csv", recursive=True)))
print(infos.groupby("set").agg(n=("qid", "size"), recall=("recall", "mean"), cand=("n_cand", "median"),
                               secs=("secs", "mean")).round(3).to_string())

# %% [markdown]
# ## 隠し方のチェック: B 系・R2 では正解にライブラリ由来の情報が残っていないこと

# %%
for name in ["B_A", "B_C", "R2"]:
    if name in feats:
        t = feats[name][feats[name].label == 1]
        print(name, "truth max of", {c: round(float(t[c].max()), 4) for c in
                                     ["lib_max", "lib_rows", "cand_nspec", "cand_nlib", "tan_self"]})

# %% [markdown]
# ## ルールベースの評価と τ の選択

# %%
RULES = search.RULES + search.GATE_RULES
res = []
for name, f in feats.items():
    X = f[search.FEATURES].values
    allq = infos[infos.set == name].qid.values
    for rule in RULES:
        r = harness.ranks_of_truth(f.qid.values, f.label.values, search.rule_score(X, rule), f.pid.values)
        res.append(dict(set=name, model=f"rule:{rule}", **harness.metrics(r.reindex(allq).fillna(np.inf).values)))
res = pd.DataFrame(res)
res.to_csv(f"{OUT}/results.csv", index=False)
piv = res.pivot(index="model", columns="set", values="mrr25")
print(piv[[s for s in SETS if s in piv.columns]].round(4).to_string())

m = lambda s, rule: float(piv.loc[f"rule:{rule}", s])
r_obj = {g: 0.2 * m("R1", g) + 0.8 * m("R2", g) for g in search.GATE_RULES}
ok = [g for g in search.GATE_RULES
      if m("A", g) >= A_REF - A_REF_SE and m("B_A", g) > m("B_A", "lib_analog")]
choice = f"rule:{max(ok, key=r_obj.get)}" if ok else "rule:lib_analog"
print("R objective:", {k: round(v, 4) for k, v in r_obj.items()})
print(f"passing (A >= {A_REF - A_REF_SE:.4f} and B_A > lib_analog):", ok, "->", choice)
json.dump(dict(default=choice, variants={}), open(f"{OUT}/ranker.json", "w"), indent=1)

# %%
run = track.init("eval", group=EXP, config=dict(A_REF=A_REF, A_REF_SE=A_REF_SE, rules=RULES))
run.table("results", res)
run.table("query_info", infos.groupby("set").agg(n=("qid", "size"), recall=("recall", "mean"),
                                                   cand=("n_cand", "median")).reset_index())
run.summary({f"mrr25/{r.set}/{r.model}": r.mrr25 for r in res.itertuples()}
            | {f"recall/{k}": v for k, v in infos.groupby("set").recall.mean().items()}
            | {f"r_obj/{k}": v for k, v in r_obj.items()} | {"choice": choice, "passing": ok})
run.finish()

# %% [markdown]
# # casmi26-k08-eval2
# k07 の分割出力を結合し、隠し方のチェック・ルールとランカーの評価・提出に使うスコアの選択を行う。
#
# v4（M2.1）: クラス1と判定した分子（lib_grp_max ≥ τ）は lib_analog の順位を保ち、それ以外だけランカーで順位を付ける。
#   hyb:τ  = クラス2側に lgb_R（R1+R2 全体で学習）
#   hybB:τ = クラス2側に、R1+R2 のうち lib_grp_max < τ のクエリだけで学習したランカー
#   M2 のランカー（lgb_R）は B_A 0.71 だが A 0.60 で不採用だった。この案は A の結果を見てから考えたので、
#   候補は上の 2 系統 × τ 5 通りに絞り、選び方を先に固定する。
#
# 採用基準（結果を見る前に固定）
# 1. hyb / hybB の 10 個のうち、A ≥ クラス1ベースラインの A − 1SE（0.9338 − 0.0127）を満たし、
#    R だけで計算した 0.2·MRR(R1) + 0.8·MRR(R2) が最大のものを選ぶ
# 2. それと今の提出（gfz:0.7:5、LB 0.220）のうち、A の条件を満たし 0.2·A + 0.8·B_A が大きい方を提出に使う。
#    A の条件を満たす hybrid が無ければ gfz:0.7:5 のまま（基準は緩めない）
# 変更履歴: v1 は「R で最良の τ を1つ選び、A の条件を満たさなければ却下」だった（M2 の計画と食い違い）。
#   v1 の結果（gate:0.9 が A 0.912 で却下）を見たあと、ユーザーの判断で M2 と同じ「条件付きの最大」に揃えた（v2, M1 提出）。
#   v3 は M2（gfz と lgb_R、gfz:0.7:5 を提出して LB 0.220）。
# 結果は wandb にオフラインで記録し、完了後に手元で tools/wandb_sync.py で送る。

# %%
import glob
import json
import time

import numpy as np
import pandas as pd

from casmi26 import harness, pool, rank, search, track

T0 = time.time()
OUT = "/kaggle/working"
EXP = "M2.1"  # wandb の group
A_REF, A_REF_SE = 0.9338, 0.0127  # クラス1ベースライン（k02, train のみのプール）
A_MIN = A_REF - A_REF_SE
M2_CHOICE = "rule:gfz:0.7:5"  # M2 で提出したルール（LB 0.220）
SETS = ["A", "B_A", "C", "B_C", "R1", "R2"]
SEEDS = (0, 1, 2)
TAUS = (0.5, 0.6, 0.7, 0.8, 0.9)

infos = pd.concat([pd.read_csv(p) for p in sorted(glob.glob("/kaggle/input/**/query_info_s*.csv", recursive=True))],
                  ignore_index=True).drop_duplicates("qid")
feats = {}
for s in SETS:
    paths_ = sorted(glob.glob(f"/kaggle/input/**/feats_{s}_s*.parquet", recursive=True))
    if paths_:
        feats[s] = pd.concat([pd.read_parquet(p) for p in paths_], ignore_index=True).drop_duplicates(["qid", "pid"])
infos.to_csv(f"{OUT}/query_info.csv", index=False)
truth = pd.read_csv(f"{pool.Pool.find_dir()}/queries.csv", usecols=["qid", "pid"]).set_index("qid").pid
print("shards found:", len(glob.glob("/kaggle/input/**/query_info_s*.csv", recursive=True)))
print(infos.groupby("set").agg(n=("qid", "size"), recall=("recall", "mean"), cand=("n_cand", "median"),
                               secs=("secs", "mean")).round(3).to_string())
for name, f in feats.items():
    nz = f.groupby("qid").fz.std().gt(0).mean()
    print(f"{name}: {len(f):,} rows, queries with non-constant fz {nz:.3f}")
    if nz < 0.5:
        raise RuntimeError(f"fz features look empty in {name}: was the model attached in k07?")

# %% [markdown]
# ## 隠し方のチェック: B 系・R2 では正解にライブラリ由来の情報が残っていないこと

# %%
for name in ["B_A", "B_C", "R2"]:
    if name in feats:
        t = feats[name][feats[name].label == 1]
        print(name, "truth max of", {c: round(float(t[c].max()), 4) for c in
                                     ["lib_max", "lib_rows", "cand_nspec", "cand_nlib", "tan_self"]})

# %% [markdown]
# ## ルールの評価（M2 と同じ）

# %%
def evaluate(name, score, f=None, qids=None):
    """MRR@25 etc. over the queries of set `name` (or the given qids); queries without the truth count as misses."""
    f = feats[name] if f is None else f
    r = harness.ranks_of_truth(f.qid.values, f.label.values, score, f.pid.values)
    q = infos[infos.set == name].qid.values if qids is None else qids
    return harness.metrics(r.reindex(q).fillna(np.inf).values)


RULES = search.RULES + search.GATE_RULES + search.FZ_RULES
res = []
for name, f in feats.items():
    X = f[search.FEATURES].values
    for rule in RULES:
        res.append(dict(set=name, model=f"rule:{rule}", **evaluate(name, search.rule_score(X, rule))))
print(f"rules done ({time.time() - T0:.0f}s)")

# %% [markdown]
# ## ランカー（R の値は正解の構造単位で分けた 5 分割の OOF、他のセットは 3 シードの平均）
# - lgb_R: R1 + R2 全体で学習（M2 と同じ）
# - hybB の τ ごとのランカー: R1 + R2 のうち lib_grp_max < τ のクエリだけで学習

# %%
R = pd.concat([feats["R1"], feats["R2"]], ignore_index=True)
FL = list(search.FEATURES)
in_set = {n: R.qid.isin(feats[n].qid.unique()).values for n in ["R1", "R2"]}


def set_scores(models, oof_full):
    """Score arrays aligned with feats[set]: OOF for R1/R2, seed-averaged predictions for the other sets."""
    sc = {n: oof_full[in_set[n]] for n in ["R1", "R2"]}
    for n in ["A", "B_A", "C", "B_C"]:
        sc[n] = rank.predict(models, feats[n][FL].values.astype(np.float32))
    return sc


def train_ranker(mask, tag):
    rows = R[mask]
    oof_full = np.zeros(len(R))
    oof_full[mask] = rank.cv_oof(rows, FL, n_folds=5, groups=truth)
    models = [rank.fit(rows, FL, seed=s) for s in SEEDS]
    for i, mdl in enumerate(models):
        mdl.save_model(f"{OUT}/ranker_{tag}_{i}.txt")
    print(f"  ranker {tag}: {rows.qid.nunique():,} queries, {len(rows):,} rows ({time.time() - T0:.0f}s)", flush=True)
    return models, set_scores(models, oof_full)


models_lgb, S_lgb = train_ranker(np.ones(len(R), bool), "lgb_R")
for s in SETS:
    res.append(dict(set=s, model="lgb_R", **evaluate(s, S_lgb[s])))
g = np.mean([mdl.feature_importance("gain") for mdl in models_lgb], axis=0)
imp = pd.Series(g / g.sum(), index=FL, name="gain").sort_values(ascending=False)
S_hybB = {tau: train_ranker(R.lib_grp_max.values < tau, f"hybB_{tau}")[1] for tau in TAUS}

# %% [markdown]
# ## 組み合わせ（hyb / hybB）の評価

# %%
HYB = {}
for tau in TAUS:
    HYB[f"hyb:{tau}"] = (tau, S_lgb, [f"ranker_lgb_R_{i}.txt" for i in range(len(SEEDS))])
    HYB[f"hybB:{tau}"] = (tau, S_hybB[tau], [f"ranker_hybB_{tau}_{i}.txt" for i in range(len(SEEDS))])
for hname, (tau, S, _) in HYB.items():
    for s in SETS:
        res.append(dict(set=s, model=hname, **evaluate(s, search.hybrid_score(feats[s][FL].values, tau, S[s]))))

res = pd.DataFrame(res)
res.to_csv(f"{OUT}/results.csv", index=False)
piv = res.pivot(index="model", columns="set", values="mrr25")[SETS]
m = lambda s, model: float(piv.loc[model, s])
r_obj = lambda model: 0.2 * m("R1", model) + 0.8 * m("R2", model)
show = ["rule:lib_analog", "rule:gate:0.9", M2_CHOICE, "rule:fz", "lgb_R"] + list(HYB)
print(piv.loc[show].round(4).to_string())

# %% [markdown]
# ## 見立ての確認（選択には使わない）
# 各 τ でクラス2側に回る A の分子だけを取り出し、クラス2側の順位付けを比べる。
# hybB が「弱い一致のクラス1分子」をきちんと扱えるようになったのか、ゲートの陰に隠れただけなのかを見る。

# %%
gmax = {s: feats[s].groupby("qid").lib_grp_max.first() for s in ["A", "B_A"]}
XA = feats["A"][FL].values
diag = []
for tau in TAUS:
    routed = gmax["A"].index[gmax["A"] < tau].values
    row = dict(tau=tau, A_routed=len(routed) / len(gmax["A"]), B_A_routed=float((gmax["B_A"] < tau).mean()))
    if len(routed):
        for label, sc in [("lib_analog", search.rule_score(XA, "lib_analog")),
                          ("M1 branch", search.rule_score(XA, f"gate:{tau}")),
                          ("gfz w5 branch", search.rule_score(XA, f"gfz:{tau}:5")),
                          ("lgb_R", S_lgb["A"]), ("hybB ranker", S_hybB[tau]["A"])]:
            row[label] = evaluate("A", sc, qids=routed)["mrr25"]
    diag.append(row)
diag = pd.DataFrame(diag)
print("MRR@25 on the A molecules routed to the class-2 branch (lib_grp_max < tau)")
print(diag.round(3).to_string(index=False))
hyb_tab = pd.DataFrame([dict(model=h, tau=tau, A=m("A", h), B_A=m("B_A", h), B_C=m("B_C", h), R_obj=r_obj(h))
                        for h, (tau, _, _) in HYB.items()])
print(hyb_tab.round(4).to_string(index=False))
print("note: B_A (NP windows with many spectrum-less COCONUT candidates) is the test-like proxy; "
      "B_C runs high because the hidden truth is often the only spectrum-less train structure in its window")

# %% [markdown]
# ## 選択（採用基準は冒頭のとおり）

# %%
ok1 = [h for h in HYB if m("A", h) >= A_MIN]
stage1 = max(ok1, key=r_obj) if ok1 else None
print(f"stage 1 (A >= {A_MIN:.4f}, max R objective): {stage1} | passing: {ok1}")
cands = [M2_CHOICE] + ([stage1] if stage1 else [])
obj2 = {c: 0.2 * m("A", c) + 0.8 * m("B_A", c) for c in cands}
ok2 = [c for c in cands if m("A", c) >= A_MIN]
choice = max(ok2, key=obj2.get) if ok2 else "rule:lib_analog"
for c in cands:
    print(f"  {c:20s} A {m('A', c):.4f}  B_A {m('B_A', c):.4f}  0.2A+0.8B_A {obj2[c]:.4f}  "
          f"{'pass' if c in ok2 else 'FAIL A'}")
print("choice:", choice)
variants = {"lgb_R": dict(features=FL, files=[f"ranker_lgb_R_{i}.txt" for i in range(len(SEEDS))])}
variants |= {h: dict(tau=tau, features=FL, files=files) for h, (tau, _, files) in HYB.items()}
json.dump(dict(default=choice, variants=variants), open(f"{OUT}/ranker.json", "w"), indent=1)

# %%
run = track.init("eval", group=EXP, config=dict(A_REF=A_REF, A_REF_SE=A_REF_SE, M2_CHOICE=M2_CHOICE, rules=RULES,
                                                 taus=list(TAUS), seeds=list(SEEDS)))
run.table("results", res)
run.table("hybrid", hyb_tab)
run.table("diag_routed_A", diag)
run.table("importance_lgb_R", imp.reset_index().rename(columns={"index": "feature"}))
run.summary({f"mrr25/{r.set}/{r.model}": r.mrr25 for r in res.itertuples()}
            | {f"recall/{k}": v for k, v in infos.groupby("set").recall.mean().items()}
            | {f"obj2/{k}": v for k, v in obj2.items()} | {"stage1": stage1, "choice": choice})
run.finish()
print(f"done {time.time() - T0:.0f}s")

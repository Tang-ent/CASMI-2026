# %% [markdown]
# # casmi26-k08-eval2
# k07 の分割出力を結合し、隠し方のチェック・ルールとランカーの評価・提出に使うスコアの選択を行う。
#
# v5（M2.2）: クラス1と判定した分子（lib_grp_max ≥ τ）は lib_analog の順位を保ち、それ以外は
#   hybm:τ:w = z(lgb_R) + w·(ap4 + 0.2·lib_max)（z は分子ごとの z-score。gfz の fz_z をランカーに置き換えた形）
#   で順位を付ける。M2.1 の hyb:0.7 は B_A 0.520 だが A 0.918 で基準に届かなかった。クラス2側に回る A の分子では
#   M1 の順位付け（ap4 + 0.2·lib_max）がランカーより良かったので、両者を混ぜて A を保ったまま τ を上げる狙い。
#   この案も A の診断を見てから考えたので、候補は τ ∈ {0.6, 0.7, 0.8} × w ∈ {0.5, 1, 2, 5} の 12 個に絞り、
#   選び方を先に固定する。hybB（M2.1 で hyb とほぼ同じ）は学習しない。
#
# 採用基準（結果を見る前に固定）
# 1. hybm の 12 個のうち、A ≥ クラス1ベースラインの A − 1SE（0.9338 − 0.0127）を満たし、
#    R だけで計算した 0.2·MRR(R1) + 0.8·MRR(R2) が最大のものを選ぶ
# 2. それと今の提出（hyb:0.6、LB 0.272。この実行で計算し直した値）のうち、A の条件を満たし
#    0.2·A + 0.8·B_A が大きい方を提出に使う。A の条件を満たす hybm が無ければ hyb:0.6 のまま（基準は緩めない）
# 変更履歴: v1 は「R で最良の τ を1つ選び、A の条件を満たさなければ却下」だった（M2 の計画と食い違い）。
#   v1 の結果（gate:0.9 が A 0.912 で却下）を見たあと、ユーザーの判断で M2 と同じ「条件付きの最大」に揃えた（v2, M1 提出）。
#   v3 は M2（gfz と lgb_R、gfz:0.7:5 を提出して LB 0.220）。
#   v4 は M2.1（hyb / hybB、hyb:0.6 を提出して LB 0.272）。
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
EXP = "M2.2"  # wandb の group
A_REF, A_REF_SE = 0.9338, 0.0127  # クラス1ベースライン（k02, train のみのプール）
A_MIN = A_REF - A_REF_SE
M2_CHOICE = "rule:gfz:0.7:5"  # M2 で提出したルール（LB 0.220、表示のみ）
CUR_CHOICE = "hyb:0.6"  # 今の提出（M2.1、LB 0.272）
SETS = ["A", "B_A", "C", "B_C", "R1", "R2"]
SEEDS = (0, 1, 2)
TAUS = (0.5, 0.6, 0.7, 0.8, 0.9)  # hyb:τ（参照用）
BLEND_TAUS = (0.6, 0.7, 0.8)  # hybm:τ:w（選択の候補）
BLEND_WS = (0.5, 1, 2, 5)

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
# lgb_R: R1 + R2 全体で学習（M2 と同じ）

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

# %% [markdown]
# ## 組み合わせの評価
# - hyb:τ（参照用、M2.1 と同じ）: クラス2側は lgb_R
# - hybm:τ:w（選択の候補）: クラス2側は z(lgb_R) + w·(ap4 + 0.2·lib_max)。z は分子ごと（qid ごと）の z-score

# %%
LGB_FILES = [f"ranker_lgb_R_{i}.txt" for i in range(len(SEEDS))]
HYB = {f"hyb:{tau}": dict(tau=tau) for tau in TAUS}
HYBM = {f"hybm:{tau}:{w}": dict(tau=tau, w=w) for tau in BLEND_TAUS for w in BLEND_WS}


def hyb_score(s, v):
    """Score of the hybrid variant v (dict with tau, and w for hybm) on set s, aligned with feats[s]."""
    X = feats[s][FL].values
    if "w" in v:
        return search.blend_score(X, v["tau"], S_lgb[s], v["w"], qid=feats[s].qid.values)
    return search.hybrid_score(X, v["tau"], S_lgb[s])


for hname, v in (HYB | HYBM).items():
    for s in SETS:
        res.append(dict(set=s, model=hname, **evaluate(s, hyb_score(s, v))))

# k09 scores one molecule at a time (qid=None); it must reproduce the grouped scores used here.
# The check uses the A molecules with the weakest library match, i.e. the ones routed to the class-2 branch.
fa, XA = feats["A"], feats["A"][FL].values
weak = fa.groupby("qid").lib_grp_max.first().nsmallest(20).index
for hname, v in HYBM.items():
    full = hyb_score("A", v)
    for q in weak:
        i = (fa.qid == q).values
        one = search.blend_score(XA[i], v["tau"], S_lgb["A"][i], v["w"])
        assert np.allclose(one, full[i], rtol=1e-9, atol=1e-9), (hname, q)
print(f"blend check: per-molecule scores match the grouped ones ({len(weak)} A molecules x {len(HYBM)} variants)")

res = pd.DataFrame(res)
res.to_csv(f"{OUT}/results.csv", index=False)
piv = res.pivot(index="model", columns="set", values="mrr25")[SETS]
m = lambda s, model: float(piv.loc[model, s])
r_obj = lambda model: 0.2 * m("R1", model) + 0.8 * m("R2", model)
show = ["rule:lib_analog", "rule:gate:0.9", M2_CHOICE, "rule:fz", "lgb_R"] + list(HYB) + list(HYBM)
print(piv.loc[show].round(4).to_string())

# %% [markdown]
# ## 見立ての確認（選択には使わない）
# 各 τ でクラス2側に回る A の分子だけを取り出し、クラス2側の順位付けを比べる。
# hybm の列が M1 branch（ap4 + 0.2·lib_max）にどこまで近づくかを見る。

# %%
gmax = {s: feats[s].groupby("qid").lib_grp_max.first() for s in ["A", "B_A"]}
diag = []
for tau in TAUS:
    routed = gmax["A"].index[gmax["A"] < tau].values
    row = dict(tau=tau, A_routed=len(routed) / len(gmax["A"]), B_A_routed=float((gmax["B_A"] < tau).mean()))
    if len(routed):
        scs = [("lib_analog", search.rule_score(XA, "lib_analog")),
               ("M1 branch", search.rule_score(XA, f"gate:{tau}")),
               ("gfz w5 branch", search.rule_score(XA, f"gfz:{tau}:5")),
               ("lgb_R", S_lgb["A"])]
        if tau in BLEND_TAUS:
            scs += [(f"hybm w{w}", hyb_score("A", HYBM[f"hybm:{tau}:{w}"])) for w in BLEND_WS]
        for label, sc in scs:
            row[label] = evaluate("A", sc, qids=routed)["mrr25"]
    diag.append(row)
diag = pd.DataFrame(diag)
print("MRR@25 on the A molecules routed to the class-2 branch (lib_grp_max < tau)")
print(diag.round(3).to_string(index=False))
hyb_tab = pd.DataFrame([dict(model=h, tau=v["tau"], w=v.get("w", 0), A=m("A", h), B_A=m("B_A", h), B_C=m("B_C", h),
                             R1=m("R1", h), R2=m("R2", h), R_obj=r_obj(h)) for h, v in (HYB | HYBM).items()])
print(hyb_tab.round(4).to_string(index=False))
print("note: B_A (NP windows with many spectrum-less COCONUT candidates) is the test-like proxy; "
      "B_C runs high because the hidden truth is often the only spectrum-less train structure in its window")

# %% [markdown]
# ## 選択（採用基準は冒頭のとおり）

# %%
ok1 = [h for h in HYBM if m("A", h) >= A_MIN]
stage1 = max(ok1, key=r_obj) if ok1 else None
print(f"stage 1 (hybm, A >= {A_MIN:.4f}, max R objective): {stage1} | passing: {ok1}")
cands = [CUR_CHOICE] + ([stage1] if stage1 else [])
obj2 = {c: 0.2 * m("A", c) + 0.8 * m("B_A", c) for c in cands}
ok2 = [c for c in cands if m("A", c) >= A_MIN]
for c in cands:
    print(f"  {c:20s} A {m('A', c):.4f}  B_A {m('B_A', c):.4f}  0.2A+0.8B_A {obj2[c]:.4f}  "
          f"{'pass' if c in ok2 else 'FAIL A'}")
if ok2:
    choice = max(ok2, key=obj2.get)
else:
    choice = CUR_CHOICE
    print(f"WARNING: no candidate passes A; keeping the current submission {CUR_CHOICE}")
print("choice:", choice)
variants = {"lgb_R": dict(features=FL, files=LGB_FILES)}
variants |= {h: dict(v, features=FL, files=LGB_FILES) for h, v in (HYB | HYBM).items()}
json.dump(dict(default=choice, variants=variants), open(f"{OUT}/ranker.json", "w"), indent=1)

# %%
run = track.init("eval", group=EXP, config=dict(A_REF=A_REF, A_REF_SE=A_REF_SE, CUR_CHOICE=CUR_CHOICE, rules=RULES,
                                                 taus=list(TAUS), blend_taus=list(BLEND_TAUS),
                                                 blend_ws=list(BLEND_WS), seeds=list(SEEDS)))
run.table("results", res)
run.table("hybrid", hyb_tab)
run.table("diag_routed_A", diag)
run.table("importance_lgb_R", imp.reset_index().rename(columns={"index": "feature"}))
run.summary({f"mrr25/{r.set}/{r.model}": r.mrr25 for r in res.itertuples()}
            | {f"recall/{k}": v for k, v in infos.groupby("set").recall.mean().items()}
            | {f"obj2/{k}": v for k, v in obj2.items()} | {"stage1": stage1, "choice": choice})
run.finish()
print(f"done {time.time() - T0:.0f}s")

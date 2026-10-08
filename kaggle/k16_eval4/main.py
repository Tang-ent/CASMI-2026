# %% [markdown]
# # casmi26-k16-eval4
# M4（2026-10-07）: ゲートの振り分け器と hybm の格子を、固定した選び方で M2.2 と比べる。
# - v1: k07 v7 の特徴量（k06 の FPNet）。振り分け器だけの効果を測る
# - v2: k07 v8 の特徴量（FPNet v2）。最終の選択（k17_submit4 がこの出力を使う）
# 振り分け器 = 分子ごとに「クラス1か」を当てる LightGBM の 2 値分類（search.route_features の分子単位の特徴量）。
#   R1（1）と R2（0）、C（1）と B_C（0）は同じ構造の対なので、その対で学習する（正解の構造ごとの 5 分割 OOF。
#   A / B_A / V には 5 つの分割のモデルの平均）。C / B_C は選び方のセットに入らないので、A と B_N の判定は汚れない。
# 候補: hyb:τ、hybm:τ:w（M2.x と同じ）、route:hard:t:w（p ≥ t なら lib_analog、それ以外は hybm の class-2 側）、
#   route:soft:w（p·z(lib_analog) + (1−p)·z(class-2 側)）。それぞれに質量窓の後絞り |mass_ppm| ≤ 8.5 / 7 も付ける
#   （1位の ±8.5 ppm の追試）。
# 選び方（2026-10-07 にユーザーと固定）: M2.2（k13 v1 の L0|hybm:0.7:2 のクエリごとの順位）と対で比べ、
#   A の対の差の 95% 下限 > −0.02 かつ B_N の下限 > 0 を満たすもののうち、予測 LB = 0.163·A + 0.21·B_N が最大のもの。
#   満たすものが無ければ M2.2 のまま。
# 追加の道（2026-10-07、k16 v2 の結果を見る前にユーザーと決定）: B_N は 85% が timsTOF 以外の装置なので、timsTOF の
#   天然物だけに効く伸び（k14: tims=0 で A の fz +0.08）が見えない。A の非劣性に加えて、B_A の対の差の下限 > 0 かつ
#   B_N の下限 > −0.02 のものも合格にする（予測 LB の式はそのまま）。

# %%
import glob
import json
import os
import time

import numpy as np
import pandas as pd

from casmi26 import harness, paths, pool, rank, search, track

T0 = time.time()
OUT = "/kaggle/working"
EXP = "M4"
SETS = ["A", "B_A", "C", "B_C", "R1", "R2", "V"]
SEEDS = (0, 1, 2)
N_BOOT = 2000
NI_MARGIN = 0.02
S1, S2 = 0.163, 0.21  # 予測 LB の重み（k13 の当てはめ）
TAUS = (0.5, 0.6, 0.7, 0.8, 0.9)
BLEND_TAUS = (0.5, 0.6, 0.7, 0.8)
BLEND_WS = (0.5, 1, 2, 5, 10)
ROUTE_TS = (0.3, 0.5, 0.7)
ROUTE_WS = (1, 2, 5)
PPM_CUTS = (None, 8.5, 7.0)
REF = "L0|hybm:0.7:2"  # k13 v1 のモデル名（M2.2、LB 0.285）
ROUTER_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=40, feature_fraction=0.9,
                     bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=4)
ROUTER_ROUNDS = 300

PD = pool.Pool.find_dir()
truth = pd.read_csv(f"{PD}/queries.csv", usecols=["qid", "pid"]).set_index("qid").pid
in_co = pd.read_parquet(f"{PD}/pool.parquet", columns=["in_coconut"]).in_coconut.values
info_path = sorted(glob.glob("/kaggle/input/**/fpnet_info.json", recursive=True))
n_shards = len(glob.glob("/kaggle/input/**/query_info_L0_s*.csv", recursive=True))
if info_path:
    # 取り消されたカーネルは最後に成功した版の出力がつながる（v2a の k07 v8 が混ざりうる）ので、全分割の FPNet が一致するか確かめる
    tags = [json.load(open(p)) for p in info_path]
    assert len(tags) == n_shards and all({k: t[k] for k in ("variant", "epoch")} == {k: tags[0][k] for k in ("variant", "epoch")}
                                         for t in tags), f"k07 shards disagree on the FPNet: {tags}"
    FEATS_TAG = tags[0]
else:
    FEATS_TAG = {"fpnet": "fpnet.pt (k06, epoch 14)", "k07": "v7"}
print("features:", FEATS_TAG)
infos = pd.concat([pd.read_csv(p) for p in sorted(glob.glob("/kaggle/input/**/query_info_L0_s*.csv", recursive=True))],
                  ignore_index=True).drop_duplicates("qid")
feats = {}
for s in SETS:
    ps = sorted(glob.glob(f"/kaggle/input/**/feats_{s}_L0_s*.parquet", recursive=True))
    feats[s] = pd.concat([pd.read_parquet(p) for p in ps], ignore_index=True).drop_duplicates(["qid", "pid"])
    feats[s] = feats[s].sort_values(["qid", "pid"], kind="mergesort").reset_index(drop=True)
print("rows", {s: len(f) for s, f in feats.items()}, f"({time.time() - T0:.0f}s)")
qids_of = {s: np.sort(infos[infos.set == s].qid.values) for s in SETS}
FL = list(search.FEATURES)
X = {s: feats[s][FL].values for s in SETS}

# %% [markdown]
# ## ランカー lgb_R（R1 + R2 で学習。R1 / R2 は正解の構造ごとの 5 分割 OOF、他のセットは 3 シードの平均）

# %%
R = pd.concat([feats["R1"], feats["R2"]], ignore_index=True)
oof = rank.cv_oof(R, FL, n_folds=5, groups=truth)
models = [rank.fit(R, FL, seed=sd) for sd in SEEDS]
for i, mdl in enumerate(models):
    mdl.save_model(f"{OUT}/ranker_lgb_R_{i}.txt")
in_r1 = R.qid.isin(feats["R1"].qid.unique()).values
RK = {"R1": oof[in_r1], "R2": oof[~in_r1]}
for s in ["A", "B_A", "C", "B_C", "V"]:
    RK[s] = rank.predict(models, X[s].astype(np.float32))
del R, oof
print(f"lgb_R done ({time.time() - T0:.0f}s)")

# %% [markdown]
# ## 振り分け器（クラス1か）

# %%
import lightgbm as lgb

RF, RQ = {}, {}  # 分子単位の特徴量と、その qid（候補の無いクエリは含まない）
for s in SETS:
    RQ[s], RF[s] = search.route_features(X[s], RK[s], feats[s].qid.values)
TRAIN_PAIRS = [("R1", 1), ("R2", 0), ("C", 1), ("B_C", 0)]
Xr = np.concatenate([RF[s] for s, _ in TRAIN_PAIRS])
yr = np.concatenate([np.full(len(RF[s]), y) for s, y in TRAIN_PAIRS])
gr = np.concatenate([truth.reindex(RQ[s]).values for s, _ in TRAIN_PAIRS])
ug = np.unique(gr)
fold_of = dict(zip(ug, np.random.default_rng(0).permutation(len(ug)) % 5))
fr = np.array([fold_of[g] for g in gr])
oof_p = np.zeros(len(yr))
routers = []
for k in range(5):
    m = lgb.train(ROUTER_PARAMS, lgb.Dataset(Xr[fr != k], yr[fr != k], feature_name=search.ROUTE_FEATURES),
                  num_boost_round=ROUTER_ROUNDS)
    oof_p[fr == k] = m.predict(Xr[fr == k])
    m.save_model(f"{OUT}/router_{k}.txt")
    routers.append(m)
PQ = {}
o = 0
for s, _ in TRAIN_PAIRS:
    PQ[s] = oof_p[o:o + len(RF[s])]
    o += len(RF[s])
for s in ["A", "B_A", "V"]:
    PQ[s] = np.mean([m.predict(RF[s]) for m in routers], axis=0)


def auc(pos, neg):
    x = np.r_[pos, neg]
    r = pd.Series(x).rank().values
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


print("router AUC (class-1 vs its hidden twin): R1/R2 %.3f, C/B_C %.3f, A/B_A %.3f" % (
    auc(PQ["R1"], PQ["R2"]), auc(PQ["C"], PQ["B_C"]), auc(PQ["A"], PQ["B_A"])))
print("mean p:", {s: round(float(v.mean()), 3) for s, v in PQ.items()})
gain = pd.Series(np.mean([m.feature_importance("gain") for m in routers], 0), index=search.ROUTE_FEATURES)
print("router gain:", gain.sort_values(ascending=False).round(0).to_dict())
P_ROW = {s: pd.Series(PQ[s], index=RQ[s]).reindex(feats[s].qid.values).values for s in SETS}
pd.concat([pd.DataFrame({"set": s, "qid": RQ[s], "p_class1": PQ[s]}) for s in SETS]).to_parquet(
    f"{OUT}/router_p.parquet", index=False)

# %% [markdown]
# ## 候補のスコアと、クエリごとの順位

# %%
scores = {}
for tau in TAUS:
    scores[f"hyb:{tau}"] = {s: search.hybrid_score(X[s], tau, RK[s]) for s in SETS}
for tau in BLEND_TAUS:
    for w in BLEND_WS:
        scores[f"hybm:{tau}:{w}"] = {s: search.blend_score(X[s], tau, RK[s], w, qid=feats[s].qid.values) for s in SETS}
for w in ROUTE_WS:
    for t in ROUTE_TS:
        scores[f"route:hard:{t}:{w}"] = {s: search.route_score(X[s], RK[s], P_ROW[s], w, "hard", t,
                                                               qid=feats[s].qid.values) for s in SETS}
    scores[f"route:soft:{w}"] = {s: search.route_score(X[s], RK[s], P_ROW[s], w, "soft", qid=feats[s].qid.values)
                                 for s in SETS}
rank_rows = []
for name, sc in scores.items():
    for cut in PPM_CUTS:
        mname = name if cut is None else f"{name}|ppm{cut:g}"
        for s in SETS:
            f = feats[s]
            keep = np.ones(len(f), bool) if cut is None else f.mass_ppm.values <= cut
            r = harness.ranks_of_truth(f.qid.values[keep], f.label.values[keep], sc[s][keep], f.pid.values[keep])
            r = r.reindex(qids_of[s]).fillna(np.inf).values.astype(np.float32)
            rank_rows.append(pd.DataFrame({"model": mname, "set": s, "qid": qids_of[s].astype(np.int32), "rank": r}))
ranks = pd.concat(rank_rows, ignore_index=True)
del rank_rows
ranks.to_parquet(f"{OUT}/ranks_eval4.parquet", index=False)
print(f"{ranks.model.nunique()} models ranked ({time.time() - T0:.0f}s)")

# %% [markdown]
# ## M2.2（k13 v1）と対で比べる

# %%
ref = pd.read_parquet(paths.find("ranks.parquet", roots=["/kaggle/input"]))  # k13 v1
ref = ref[ref.model == REF]
assert len(ref), "k13 ranks.parquet (M2.2) not attached"
tq = {s: truth.reindex(qids_of[s]).values for s in SETS}
B_N = [(s, in_co[tq[s]]) for s in ["R2", "V"]]
ALL = SETS + ["B_N"]


def rr_of_model(g):
    rr = {}
    for s in SETS:
        x = g[g.set == s].set_index("qid")["rank"].reindex(qids_of[s]).values
        assert not np.isnan(x).any(), s
        rr[s] = harness.rr_of(x)
    rr["B_N"] = np.concatenate([rr[s][m] for s, m in B_N])
    return rr


RR = {m: rr_of_model(g) for m, g in ranks.groupby("model", sort=False)}
RR_REF = rr_of_model(ref)
print("B_N queries:", {s: int(m.sum()) for s, m in B_N})
IDX = {s: harness.boot_index(len(RR_REF[s]), N_BOOT, seed=i) for i, s in enumerate(ALL)}
REPS_REF = {s: harness.boot_reps(RR_REF[s], IDX[s]) for s in ALL}
rows = []
for m, rr in RR.items():
    for s in ALL:
        reps = harness.boot_reps(rr[s], IDX[s])
        c = harness.ci(rr[s].mean(), reps)
        d = harness.ci(rr[s].mean() - RR_REF[s].mean(), reps - REPS_REF[s])
        rows.append(dict(model=m, set=s, n=len(rr[s]), mrr=c["est"], lo=c["lo"], hi=c["hi"], d_ref=d["est"],
                         d_lo=d["lo"], d_hi=d["hi"]))
summ = pd.DataFrame(rows)
summ.to_csv(f"{OUT}/summary.csv", index=False)
v = summ.pivot(index="model", columns="set", values="mrr")
print("reference (k13 M2.2):", {s: round(float(RR_REF[s].mean()), 4) for s in ["A", "B_A", "B_N"]})
print("this kernel's hybm:0.7:2 (same rule, retrained ranker):",
      {s: round(float(v.loc["hybm:0.7:2", s]), 4) for s in ["A", "B_A", "B_N"]})

# %% [markdown]
# ## 選び方（固定）

# %%
dd = summ.set_index(["model", "set"])
rule = pd.DataFrame([dict(model=m, A=dd.loc[(m, "A"), "mrr"], dA=dd.loc[(m, "A"), "d_ref"], dA_lo=dd.loc[(m, "A"), "d_lo"],
                          B_A=dd.loc[(m, "B_A"), "mrr"], dB_A=dd.loc[(m, "B_A"), "d_ref"],
                          dB_A_lo=dd.loc[(m, "B_A"), "d_lo"], B_N=dd.loc[(m, "B_N"), "mrr"], dB_N=dd.loc[(m, "B_N"), "d_ref"],
                          dB_N_lo=dd.loc[(m, "B_N"), "d_lo"], dB_N_hi=dd.loc[(m, "B_N"), "d_hi"]) for m in RR])
rule["A_noninferior"] = rule.dA_lo > -NI_MARGIN
rule["B_N_better"] = rule.dB_N_lo > 0
rule["B_A_path"] = (rule.dB_A_lo > 0) & (rule.dB_N_lo > -NI_MARGIN)  # timsTOF の天然物側の伸び
rule["eligible"] = rule.A_noninferior & (rule.B_N_better | rule.B_A_path)
rule["pred_lb_delta"] = S1 * rule.dA + S2 * rule.dB_N
rule["C"] = [dd.loc[(m, "C"), "mrr"] for m in rule.model]  # 表示だけ（選び方には使わない）
rule["V"] = [dd.loc[(m, "V"), "mrr"] for m in rule.model]
rule = rule.sort_values("pred_lb_delta", ascending=False)
rule.to_csv(f"{OUT}/rule.csv", index=False)
with pd.option_context("display.width", 250):
    print(rule.head(40).round(4).to_string(index=False))
    print("\nrouter variants and their mass cuts:")
    print(rule[rule.model.str.startswith("route") | (rule.model.str.split("|").str[0] == "hybm:0.7:2")]
          .round(4).to_string(index=False))
el = rule[rule.eligible]
chosen = el.iloc[0].to_dict() if len(el) else None
res = dict(features=FEATS_TAG, n_eligible=int(len(el)), chosen=chosen, ref=REF,
           rule=dict(ni_margin=NI_MARGIN, s1=S1, s2=S2))
json.dump(res, open(f"{OUT}/chosen.json", "w"), indent=1, default=float)
print("\nCHOSEN:", chosen["model"] if chosen else "none — keep M2.2", f"({len(el)} eligible)")

# %% [markdown]
# ## 参考: fz だけの窓内 MRR を、FPNet v2 と「今の重みで tims=0 にしただけ」で比べる（表示だけ）
# k15 の判定は今の重みのそのまま（tims=1）との比較なので、A の伸びには推論時の印の効果（k14 で +0.08）が混ざる。
# 学習し直しの効果だけを見るため、k14 の tims0 の順位とも対で比べる。

# %%
pr = glob.glob("/kaggle/input/**/probe_ranks.parquet", recursive=True)
nr = glob.glob("/kaggle/input/**/fz_ranks_v2*.parquet", recursive=True)
if pr and len(nr) == 1:
    old = pd.read_parquet(pr[0])
    new = pd.read_parquet(nr[0]).set_index("qid")
    rows = []
    for var in ["base", "tims0"]:
        o = old[old.variant == var].set_index("qid")
        q = new.index.intersection(o.index)
        npq = in_co[new.loc[q, "pid"].values]
        st = new.loc[q, "set"].values
        for g, m in [("A", st == "A"), ("B_N", np.isin(st, ["R2", "V"]) & npq), ("C", st == "C"), ("V", st == "V")]:
            x, y = harness.rr_of(new.loc[q[m], "r_pid"]), harness.rr_of(o.loc[q[m], "r_pid"])
            rows.append(dict(old=var, group=g, new_mrr=x.mean(), old_mrr=y.mean(), **harness.paired_ci(x, y, N_BOOT, seed=1)))
    print(f"fz-only window MRR, {os.path.basename(nr[0])} vs the k06 model (as is / tims=0):")
    print(pd.DataFrame(rows).round(4).to_string(index=False))
else:
    print("k14 / k15 outputs not attached: fz-only comparison skipped")

# %%
run = track.init("eval", group=EXP, config=dict(features=FEATS_TAG, n_boot=N_BOOT, ni_margin=NI_MARGIN, ref=REF,
                                                 s1=S1, s2=S2, router_params=ROUTER_PARAMS, ppm_cuts=list(PPM_CUTS)))
run.table("rule", rule)
run.table("summary", summ)
run.summary(dict(n_eligible=len(el), chosen=chosen["model"] if chosen else "none",
                 router_auc_A=float(auc(PQ["A"], PQ["B_A"]))))
run.finish()
print(f"done {time.time() - T0:.0f}s")

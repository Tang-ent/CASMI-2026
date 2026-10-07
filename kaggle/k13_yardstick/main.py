# %% [markdown]
# # casmi26-k13-yardstick
# 評価の物差しの見直し（2026-10-07）。k07 v7 の特徴量（プールの段階 0 = PubChem なし、段階 1 = S′）を使う。
# 1. クエリごとの順位を、全部のモデルと全部のセットについて残す（ranks.parquet）
# 2. ブートストラップの信頼区間と、今の提出（M2.2 = 段階 0 の hybm:0.7:2）との対応のある差
#    （同じクエリの再標本を全モデルで共有するので、差の複製はそのまま対応のある差になる）
# 3. 大きいクラス2の代理 B_N = R2 と V（正解のスペクトルをすべて隠したセット）のうち、正解が COCONUT にあるクエリ。
#    R2 と V の構造は FPNet の学習に使われていない（holdout）。R2 のランカーのスコアは 5 分割の OOF
# あわせて、これまでの提出をこの物差しで測り直し、public LB と照らし合わせる。public LB は非公開テストの約 33%
# （約 130 分子）で、SE は約 0.04、同じ分子での 2 提出の差でも 0.015〜0.03 ある（2026-10-07 の見積もり）。
# 新しい選び方の案（このカーネルでは表示のみ。使う前にユーザーと決める）:
#   今の提出と比べて、A は非劣性（差の 95% 区間の下限 > −NI_MARGIN）、B_N は改善（下限 > 0）を満たすものだけを候補にする。
# 結果は wandb にオフラインで記録し、完了後に手元で tools/wandb_sync.py で送る。

# %%
import glob
import json
import time

import numpy as np
import pandas as pd

from casmi26 import harness, paths, pool, rank, search, track

T0 = time.time()
OUT = "/kaggle/working"
EXP = "yardstick"  # wandb の group
LEVELS = (0, 1)
SETS = ["A", "B_A", "C", "B_C", "R1", "R2", "V"]
SEEDS = (0, 1, 2)
N_BOOT = 2000
NI_MARGIN = 0.02  # A の非劣性の幅（案）。クラス1は約16%なので、A の 0.02 は LB で約 0.003
TAUS = (0.5, 0.6, 0.7, 0.8, 0.9)
BLEND_TAUS = (0.5, 0.6, 0.7, 0.8)
BLEND_WS = (0.5, 1, 2, 5, 10)
REF = "L0|hybm:0.7:2"  # 今の提出（M2.2、LB 0.285）
# これまでの提出（public LB、約 130 分子）。class-1 は train だけのプール（k04）なので、クラス1の割合の目安にだけ使う
SUBS = [dict(name="M1", ref=56821350, lb=0.205, model="L0|rule:gate:0.9"),
        dict(name="M2", ref=56836607, lb=0.220, model="L0|rule:gfz:0.7:5"),
        dict(name="M2.1", ref=56853542, lb=0.272, model="L0|hyb:0.6"),
        dict(name="M2.2", ref=56855228, lb=0.285, model="L0|hybm:0.7:2"),
        dict(name="S'", ref=56882680, lb=0.251, model="L1|hybm:0.8:10")]
CLASS1 = dict(name="class-1 only", ref=56807512, lb=0.152, A=0.9338)  # A は k02 の値
S1 = CLASS1["lb"] / CLASS1["A"]  # テストのうちクラス1の割合の目安
PROXIES = ["B_A", "B_N", "R2", "V", "B_C"]

PD = pool.Pool.find_dir()
truth = pd.read_csv(f"{PD}/queries.csv", usecols=["qid", "pid"]).set_index("qid").pid
N_BASE = json.load(open(f"{PD}/pool_meta.json"))["n_pool"]
in_co = pd.read_parquet(f"{PD}/pool.parquet", columns=["in_coconut"]).in_coconut.values
pt = pd.read_parquet(paths.find("pool_pubchem.parquet"), columns=["pid", "base_pass", "n_pmid"]).set_index("pid")
pass_sp = (pt.base_pass & (pt.n_pmid >= 1)).reindex(np.arange(N_BASE), fill_value=False).values


def load(level):
    infos = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(f"/kaggle/input/**/query_info_L{level}_s*.csv",
                                                                     recursive=True))], ignore_index=True)
    infos = infos.drop_duplicates("qid")
    feats = {}
    for s in SETS:
        ps = sorted(glob.glob(f"/kaggle/input/**/feats_{s}_L{level}_s*.parquet", recursive=True))
        feats[s] = pd.concat([pd.read_parquet(p) for p in ps], ignore_index=True).drop_duplicates(["qid", "pid"])
    return infos, feats


# %% [markdown]
# ## 段階ごとに、全部のモデルのクエリごとの順位

# %%
FL = list(search.FEATURES)
RULES = ["lib_analog", "gate:0.9", "gfz:0.7:5", "fz"]
rank_rows, qids_of, n_cand = [], {}, []
for level in LEVELS:
    infos, feats = load(level)
    pc = {s: float((f.pid >= N_BASE).mean()) for s, f in feats.items()}
    print(f"L{level}: shards {len(glob.glob(f'/kaggle/input/**/query_info_L{level}_s*.csv', recursive=True))}, "
          f"rows {({s: len(f) for s, f in feats.items()})}, PubChem share of rows {({s: round(v, 3) for s, v in pc.items()})}")
    assert (max(pc.values()) > 0) == (level > 0), f"L{level}: PubChem candidates do not match the level"
    n_cand.append(infos.groupby("set").n_cand.describe(percentiles=[0.5, 0.9])[["count", "50%", "90%"]]
                  .assign(level=level))
    for s in SETS:
        q = np.sort(infos[infos.set == s].qid.values)
        assert s not in qids_of or np.array_equal(qids_of[s], q), f"{s}: queries differ between levels"
        qids_of[s] = q
    X = {s: feats[s][FL].values for s in SETS}
    scores = {f"rule:{r}": {s: search.rule_score(X[s], r) for s in SETS} for r in RULES}
    # lgb_R: R1 + R2 で学習（R1/R2 は正解の構造単位の 5 分割 OOF、他のセットは 3 シードの平均）
    R = pd.concat([feats["R1"], feats["R2"]], ignore_index=True)
    oof = rank.cv_oof(R, FL, n_folds=5, groups=truth)
    models = [rank.fit(R, FL, seed=sd) for sd in SEEDS]
    for i, mdl in enumerate(models):
        mdl.save_model(f"{OUT}/ranker_L{level}_lgb_R_{i}.txt")
    in_r1 = R.qid.isin(feats["R1"].qid.unique()).values
    S_lgb = {"R1": oof[in_r1], "R2": oof[~in_r1]}
    for s in ["A", "B_A", "C", "B_C", "V"]:
        S_lgb[s] = rank.predict(models, X[s].astype(np.float32))
    print(f"  L{level} lgb_R: {R.qid.nunique():,} queries, {len(R):,} rows ({time.time() - T0:.0f}s)", flush=True)
    del R, oof
    scores["lgb_R"] = S_lgb
    for tau in TAUS:
        scores[f"hyb:{tau}"] = {s: search.hybrid_score(X[s], tau, S_lgb[s]) for s in SETS}
    for tau in BLEND_TAUS:
        for w in BLEND_WS:
            scores[f"hybm:{tau}:{w}"] = {s: search.blend_score(X[s], tau, S_lgb[s], w, qid=feats[s].qid.values)
                                         for s in SETS}
    for mname, sc in scores.items():
        for s in SETS:
            f = feats[s]
            r = harness.ranks_of_truth(f.qid.values, f.label.values, sc[s], f.pid.values)
            r = r.reindex(qids_of[s]).fillna(np.inf).values.astype(np.float32)
            rank_rows.append(pd.DataFrame({"level": np.int8(level), "model": f"L{level}|{mname}", "set": s,
                                           "qid": qids_of[s].astype(np.int32), "rank": r}))
    print(f"  L{level}: {len(scores)} models ranked ({time.time() - T0:.0f}s)", flush=True)
    del feats, scores, S_lgb, X
ranks = pd.concat(rank_rows, ignore_index=True)
del rank_rows
ranks.to_parquet(f"{OUT}/ranks.parquet", index=False)
n_cand = pd.concat(n_cand)
print(n_cand.round(1).to_string())

# %% [markdown]
# ## 派生のセット
# - B_N: R2 と V のうち正解が COCONUT にあるクエリ（大きいクラス2の代理）
# - B_PR: R2 のうち正解が COCONUT に無いクエリで、正解が S′ を通るときだけ正解として数える（段階 0 では全部取りこぼし）

# %%
tq = {s: truth.reindex(qids_of[s]).values for s in SETS}
SUB_SETS = {"B_N": [(s, in_co[tq[s]]) for s in ["R2", "V"]], "B_PR": [("R2", ~in_co[tq["R2"]])]}
for name, parts in SUB_SETS.items():
    print(name, "queries:", {s: int(m.sum()) for s, m in parts}, "(", int(sum(m.sum() for _, m in parts)), ")")
ALL_SETS = SETS + list(SUB_SETS)


def rr_table(model):
    """{set: reciprocal-rank array in the fixed query order} for one model (incl. the derived sets)."""
    g = ranks[ranks.model == model].groupby("set")
    rr = {s: harness.rr_of(g.get_group(s)["rank"].values) for s in SETS}
    for name, parts in SUB_SETS.items():
        x = np.concatenate([rr[s][m] for s, m in parts])
        if name == "B_PR":
            ok = pass_sp[tq["R2"][parts[0][1]]] if model.startswith("L1|") else np.zeros(len(x), bool)
            x = np.where(ok, x, 0.0)
        rr[name] = x
    return rr


MODELS = list(dict.fromkeys(ranks.model))
RR = {m: rr_table(m) for m in MODELS}
IDX = {s: harness.boot_index(len(RR[REF][s]), N_BOOT, seed=i) for i, s in enumerate(ALL_SETS)}
REPS = {m: {s: harness.boot_reps(RR[m][s], IDX[s]) for s in ALL_SETS} for m in MODELS}
summ = []
for m in MODELS:
    for s in ALL_SETS:
        c = harness.ci(RR[m][s].mean(), REPS[m][s])
        d = harness.ci(RR[m][s].mean() - RR[REF][s].mean(), REPS[m][s] - REPS[REF][s])
        summ.append(dict(model=m, set=s, n=len(RR[m][s]), mrr=c["est"], se=c["se"], lo=c["lo"], hi=c["hi"],
                         d_ref=d["est"], d_se=d["se"], d_lo=d["lo"], d_hi=d["hi"]))
summ = pd.DataFrame(summ)
summ.to_csv(f"{OUT}/summary.csv", index=False)
print(f"bootstrap done ({time.time() - T0:.0f}s)")

# %% [markdown]
# ## これまでの提出を新しい物差しで測る（MRR と 95% 区間、M2.2 との差）

# %%
sub_models = [x["model"] for x in SUBS] + ["L0|rule:lib_analog"]
piv = summ[summ.model.isin(sub_models)].pivot(index="model", columns="set", values="mrr").loc[sub_models, ALL_SETS]
print("MRR@25 (rows: M1, M2, M2.1, M2.2, S', and lib_analog = the class-1 rule)")
print(piv.round(4).to_string())
fmt = lambda r: f"{r.mrr:.3f} [{r.lo:.3f},{r.hi:.3f}]"
print("\n95% intervals")
print(summ[summ.model.isin(sub_models)].assign(v=lambda d: d.apply(fmt, axis=1))
      .pivot(index="model", columns="set", values="v").loc[sub_models, ALL_SETS].to_string())
fmt_d = lambda r: f"{r.d_ref:+.3f} [{r.d_lo:+.3f},{r.d_hi:+.3f}]"
print("\npaired difference vs M2.2 (95% interval)")
print(summ[summ.model.isin(sub_models)].assign(v=lambda d: d.apply(fmt_d, axis=1))
      .pivot(index="model", columns="set", values="v").loc[sub_models, ALL_SETS].to_string())

# %% [markdown]
# ## LB と照らし合わせる
# LB ≈ S1·A + s2·P（S1 = クラス1だけの提出の LB / A）とおき、代理 P ごとに s2 を逆算する。
# 物差しが忠実なら、s2 は提出によらずほぼ一定になる（LB の誤差は差で 0.015〜0.03、絶対値で約 0.04）。

# %%
val = summ.set_index(["model", "set"]).mrr
calib = []
for P_ in PROXIES:
    xs = np.array([val[(x["model"], P_)] for x in SUBS])
    a = np.array([val[(x["model"], "A")] for x in SUBS])
    y = np.array([x["lb"] for x in SUBS]) - S1 * a
    for fit_on, use in [("M1-M2.2", np.arange(4)), ("all 5", np.arange(5))]:
        s2 = float((xs[use] * y[use]).sum() / (xs[use] ** 2).sum())
        pred = S1 * a + s2 * xs
        for i, x in enumerate(SUBS):
            calib.append(dict(proxy=P_, fit_on=fit_on, sub=x["name"], lb=x["lb"], proxy_mrr=xs[i], A=a[i],
                              implied_s2=y[i] / xs[i], s2=s2, pred=pred[i], resid=x["lb"] - pred[i]))
calib = pd.DataFrame(calib)
calib.to_csv(f"{OUT}/calib.csv", index=False)
print(f"class-1 share S1 = {S1:.3f}")
print(calib[calib.fit_on == "M1-M2.2"].pivot(index="sub", columns="proxy", values="implied_s2")
      .loc[[x["name"] for x in SUBS], PROXIES].round(3).to_string())
rm = calib.groupby(["proxy", "fit_on"]).agg(s2=("s2", "first"), rmse=("resid", lambda r: float(np.sqrt((r ** 2).mean()))),
                                            resid_S=("resid", lambda r: float(r.iloc[-1])))
print("\nfit LB = S1*A + s2*P: s2, RMSE of the residuals, residual of S' (fit_on M1-M2.2 predicts S' out of sample)")
print(rm.round(4).to_string())
steps = []
for i in range(1, len(SUBS)):
    p, c = SUBS[i - 1], SUBS[i]
    row = dict(step=f"{p['name']} -> {c['name']}", d_lb=c["lb"] - p["lb"])
    for P_ in PROXIES:
        row[f"d_{P_}"] = val[(c["model"], P_)] - val[(p["model"], P_)]
    steps.append(row)
steps = pd.DataFrame(steps)
print("\nconsecutive submissions: change in LB and in each proxy")
print(steps.round(4).to_string(index=False))

# %% [markdown]
# ## 新しい選び方の案を、段階 0 の候補に当てはめると（表示のみ）

# %%
cand = [m for m in MODELS if m.startswith("L0|hyb")]
rows = []
for m in cand:
    dA = summ[(summ.model == m) & (summ.set == "A")].iloc[0]
    dB = summ[(summ.model == m) & (summ.set == "B_N")].iloc[0]
    dBA = summ[(summ.model == m) & (summ.set == "B_A")].iloc[0]
    rows.append(dict(model=m, A=dA.mrr, dA=dA.d_ref, dA_lo=dA.d_lo, B_A=dBA.mrr, dB_A=dBA.d_ref, B_N=dB.mrr,
                     dB_N=dB.d_ref, dB_N_lo=dB.d_lo, A_noninferior=dA.d_lo > -NI_MARGIN, B_N_better=dB.d_lo > 0))
rule = pd.DataFrame(rows)
rule["eligible"] = rule.A_noninferior & rule.B_N_better
print(rule.round(4).to_string(index=False))

# %%
run = track.init("eval", group=EXP, config=dict(levels=list(LEVELS), n_boot=N_BOOT, ni_margin=NI_MARGIN, ref=REF,
                                                 subs=SUBS, class1=CLASS1, s1=S1, proxies=PROXIES))
run.table("summary", summ)
run.table("calib", calib)
run.table("steps", steps)
run.table("rule_L0", rule)
run.summary({f"mrr/{r.set}/{r.model}": r.mrr for r in summ[summ.model.isin(sub_models)].itertuples()})
run.finish()
print(f"done {time.time() - T0:.0f}s")

# %% [markdown]
# # casmi26-k23-eval5
# M5 P1 の 3（2026-10-10）: k22 の特徴量で、プール側のランカーと生成側の差し込みを、結果を見る前に固定した選び方で選ぶ。
# 0. 照合: 両方にある (qid, pid) で、k22 の今の 34 列が k07 v9 と一致する（誤差 1e-5 以内）、両方で処理されたクエリの
#    候補の集合が同じ。一致しなければ止める
# 1. プール側: ランカーの特徴量 base（今の 34）/ +frag / +fwd / +frag+fwd × hybm:τ:w（τ ∈ {0.6, 0.7, 0.8}、
#    w ∈ {1, 2, 5}）= 36 候補。学習は k16 と同じ（R1 + R2、正解の構造ごとの 5 分割 OOF、他は 3 シードの平均）。
#    選び方（今のまま）: M2.2（k13 の L0|hybm:0.7:2）と対の差で、A の下限 > −0.02 かつ「B_N の下限 > 0」または
#    「B_A の下限 > 0 かつ B_N の下限 > −0.02」。複数あれば予測 LB = 0.163·A + 0.21·B_N が最大のもの = P*
# 2. 生成側: 生成物の一覧の上位を決まった枠（S5 / S3 / S2 / S10）に差し込み、残りはプール側の順で埋める。
#    門: lib_grp_max < τ_g（0.7 / 0.5）のクエリだけ。生成物の並べ方: comb = z(fz) + z(ap4) + z(fwd) + z(frag_expl)、
#    または gen ranker（R2 の生成物の一覧で学習した LightGBM。R2 は 5 分割 OOF、他は 3 シードの平均）。
#    土台は M2.2 と P*（あれば）。B の順位 = プール側の正解の順位を、その前に入った枠の分だけずらしたもの。
#    G の順位 = 生成物の一覧での正解の順位 k に対応する k 番目の枠の位置（無いか枠より下なら 0 点）。
#    B の評価では一覧から正解の行を除くだけで、一覧の中の z は除く前のまま（1 行の違いは小さい）。
#    選び方（2026-10-09 に固定）: G_N か G_A が ≥ 0.03 かつその下限 > 0、M2.2 と比べて A の下限 > −0.02、
#    B_N の下限 > −0.01
# 3. 最後の選択: {M2.2、P*、M2.2 + 生成、P* + 生成} のうち条件を満たすものの中で
#    予測 LB = 0.163·A + 0.21·B_N + 0.30·G_N（生成なしは G_N = 0）が最大のもの
# 同点: 予測 LB が同じなら単純な方（生成なし < プール側 < 生成あり）、その中では格子の順（S5 → S3 → S2 → S10、
# 門 0.7 → 0.5、comb → rk、特徴量 base → frag → fwd → both）を先にする（並べ替えはすべて安定）。
# 照合の厳しさ: 候補の集合は R1 / R2 以外で完全一致、R1 / R2 は 99.9% 以上（fz の切り詰めの境目のわずかな差を許す）。
# 値は fz 以外の列で 1e-5 以内（外れたら止める）、fz の列は 1e-3 を超えたら警告だけ（CPU の違いで FPNet の出力が
# わずかに変わりうるため）。列ごとの最大の差は check_k07.csv に書く。
# 表示だけ: 特徴量の重要度、B_N のライブラリ別、付加の方向だけの G（並べ方には使わない）。
# 入力: k22-s0〜s3、k07-features2-s0〜s2（v9、照合用）、k05（プール、queries.csv）、k13（M2.2 の順位）。

# %%
import glob
import json
import time

import numpy as np
import pandas as pd

from casmi26 import gen, harness, paths, pool, rank, search, track
from casmi26.frag import FRAG_FEATURES

T0 = time.time()
OUT = "/kaggle/working"
EXP = "M5"
SETS = ["A", "B_A", "C", "B_C", "R1", "R2", "V"]
SEEDS = (0, 1, 2)
N_BOOT = 2000
NI_MARGIN = 0.02
GEN_BN_MARGIN = 0.01
GEN_MIN = 0.03
S1, S2, S3 = 0.163, 0.21, 0.30
TAUS, WS = (0.6, 0.7, 0.8), (1, 2, 5)
GATES = (0.7, 0.5)
SLOTS = {"S5": [5, 10, 15, 20, 25], "S3": list(range(3, 25, 3)), "S2": list(range(2, 25, 2)), "S10": [10, 20]}
GEN_SCORES = ("comb", "rk")
REF = "L0|hybm:0.7:2"  # k13 v1 の M2.2
EXPECT_VARIANT = "v2b"
REF_K16 = {"A": 0.926, "B_N": 0.518}  # k16 v2 の hybm:0.7:2（v2b の特徴量）
REF_TOL = 0.005  # 表示の警告だけ（None = 照合しない）
CHECK_TOL, CHECK_TOL_FZ, CHECK_OVERLAP = 1e-5, 1e-3, 0.999
REMOVAL = ("de_", "dehydroxy", "H2_remove", "H2_alcohol")

PD = pool.Pool.find_dir()
truth = pd.read_csv(f"{PD}/queries.csv", usecols=["qid", "pid"]).set_index("qid").pid
in_co = pd.read_parquet(f"{PD}/pool.parquet", columns=["in_coconut"]).in_coconut.values
tags = [json.load(open(p)) for p in sorted(glob.glob("/kaggle/input/**/features_info.json", recursive=True))]
KEYS = ("variant", "epoch", "fwd", "fwd_file", "fwd_epoch", "frag_total")
assert tags and all({k: t[k] for k in KEYS} == {k: tags[0][k] for k in KEYS} for t in tags), \
    f"k22 shards disagree: {tags}"
assert tags[0]["variant"] == EXPECT_VARIANT, f"k22 features are from {tags[0]['variant']}"
print("features:", tags[0], "|", len(tags), "shards")


def load(pattern, dedupe):
    ps = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    d = pd.concat([pd.read_parquet(p) for p in ps], ignore_index=True) if ps else pd.DataFrame()
    return d.drop_duplicates(dedupe) if len(d) else d


infos = pd.concat([pd.read_csv(p) for p in sorted(glob.glob("/kaggle/input/**/query_info_s*.csv", recursive=True))],
                  ignore_index=True).drop_duplicates("qid")
feats = {}
for s in SETS:
    feats[s] = load(f"feats_{s}_s*.parquet", ["qid", "pid"]).sort_values(["qid", "pid"], kind="mergesort") \
        .reset_index(drop=True)
qids_of = {s: np.sort(infos[infos.set == s].qid.values) for s in SETS}
print("rows", {s: len(f) for s, f in feats.items()}, "| queries", {s: len(q) for s, q in qids_of.items()},
      f"({time.time() - T0:.0f}s)")
FL = list(search.FEATURES)
FCOLS = {"base": FL, "frag": FL + FRAG_FEATURES, "fwd": FL + search.FWD_FEATURES,
         "both": FL + FRAG_FEATURES + search.FWD_FEATURES}
X = {s: feats[s][FCOLS["both"]].values for s in SETS}

# %% [markdown]
# ## 0. 照合: 今の 34 列が k07 v9 と一致するか

# %%
HARD = [c for c in FL if c not in search.FZ_FEATURES]
chk = []
for s in SETS:
    old = load(f"feats_{s}_L0_s*.parquet", ["qid", "pid"])
    both_q = np.intersect1d(old.qid.unique(), feats[s].qid.unique())
    a = old[old.qid.isin(both_q)]
    b = feats[s][feats[s].qid.isin(both_q)]
    m = a.merge(b, on=["qid", "pid"], suffixes=("_o", "_n"))
    overlap = len(m) / max(len(a), len(b), 1)
    need = CHECK_OVERLAP if s in harness.TRAIN_SETS else 1.0
    assert overlap >= need, f"{s}: candidate sets differ from k07 v9 (overlap {overlap:.5f})"
    for c in FL:
        d = float(np.nanmax(np.abs(m[c + "_o"].values.astype(np.float64) - m[c + "_n"].values.astype(np.float64)))) \
            if len(m) else 0.0
        chk.append(dict(set=s, col=c, max_diff=d, overlap=overlap, n=len(m)))
chk = pd.DataFrame(chk)
chk.to_csv(f"{OUT}/check_k07.csv", index=False)
bad = chk[chk.col.isin(HARD) & (chk.max_diff > CHECK_TOL)]
assert bad.empty, f"base features differ from k07 v9:\n{bad}"
warn = chk[~chk.col.isin(HARD) & (chk.max_diff > CHECK_TOL_FZ)]
if len(warn):
    print("WARNING: fz columns differ from k07 v9 beyond", CHECK_TOL_FZ, "\n", warn.to_string(index=False))
print(f"k07 v9 check passed: max |diff| non-fz {chk[chk.col.isin(HARD)].max_diff.max():.2e}, "
      f"fz {chk[~chk.col.isin(HARD)].max_diff.max():.2e}, overlap min {chk.overlap.min():.5f} ({time.time() - T0:.0f}s)")

# %% [markdown]
# ## 1. プール側のランカー（4 通りの特徴量）と hybm の格子

# %%
def ranker(cols, tag):
    R = pd.concat([feats["R1"], feats["R2"]], ignore_index=True)
    oof = rank.cv_oof(R, cols, n_folds=5, groups=truth)
    ms = [rank.fit(R, cols, seed=sd) for sd in SEEDS]
    for i, m in enumerate(ms):
        m.save_model(f"{OUT}/ranker_{tag}_{i}.txt")
    r1 = R.qid.isin(feats["R1"].qid.unique()).values
    out = {"R1": oof[r1], "R2": oof[~r1]}
    for s in ["A", "B_A", "C", "B_C", "V"]:
        out[s] = rank.predict(ms, feats[s][cols].values.astype(np.float32))
    imp = pd.Series(np.mean([m.feature_importance("gain") for m in ms], 0), index=cols)
    print(f"ranker {tag} done ({time.time() - T0:.0f}s); top gain:",
          (imp / imp.sum()).sort_values(ascending=False).head(8).round(3).to_dict())
    return out, imp


RK, IMP = {}, {}
for tag, cols in FCOLS.items():
    RK[tag], IMP[tag] = ranker(cols, tag)
pd.DataFrame(IMP).to_csv(f"{OUT}/importance.csv")
scores = {}
for tag in FCOLS:
    for tau in TAUS:
        for w in WS:
            scores[f"{tag}|hybm:{tau}:{w}"] = {s: search.blend_score(X[s], tau, RK[tag][s], w, qid=feats[s].qid.values)
                                               for s in SETS}
DISPLAY = {f"{tag}|lgb_R": RK[tag] for tag in FCOLS}
scores.update(DISPLAY)
rank_rows = []
for name, sc in scores.items():
    for s in SETS:
        f = feats[s]
        r = harness.ranks_of_truth(f.qid.values, f.label.values, sc[s], f.pid.values)
        r = r.reindex(qids_of[s]).fillna(np.inf).values.astype(np.float32)
        rank_rows.append(pd.DataFrame({"model": name, "set": s, "qid": qids_of[s].astype(np.int32), "rank": r}))
ranks = pd.concat(rank_rows, ignore_index=True)
ranks.to_parquet(f"{OUT}/ranks_eval5.parquet", index=False)
del rank_rows
print(f"{ranks.model.nunique()} models ranked ({time.time() - T0:.0f}s)")

# %%
ref = pd.read_parquet(paths.find("ranks.parquet", roots=["/kaggle/input"]))
ref = ref[ref.model == REF]
assert len(ref), "k13 ranks.parquet (M2.2) not attached"
tq = {s: truth.reindex(qids_of[s]).values for s in SETS}
BN = {s: in_co[tq[s]] for s in ["R2", "V"]}
ALL = SETS + ["B_N"]


def rank_of(g, s):
    x = g[g.set == s].set_index("qid")["rank"].reindex(qids_of[s]).values
    assert not np.isnan(x).any(), s
    return x


def rr_from_ranks(rk):
    rr = {s: harness.rr_of(rk[s]) for s in SETS}
    rr["B_N"] = np.concatenate([rr[s][BN[s]] for s in ["R2", "V"]])
    return rr


RANK_REF = {s: rank_of(ref, s) for s in SETS}
RR_REF = rr_from_ranks(RANK_REF)
RANK = {m: {s: rank_of(g, s) for s in SETS} for m, g in ranks.groupby("model", sort=False)}
RR = {m: rr_from_ranks(r) for m, r in RANK.items()}
IDX = {s: harness.boot_index(len(RR_REF[s]), N_BOOT, seed=i) for i, s in enumerate(ALL)}
REPS_REF = {s: harness.boot_reps(RR_REF[s], IDX[s]) for s in ALL}


def vs_ref(rr, sets=ALL):
    out = {}
    for s in sets:
        reps = harness.boot_reps(rr[s], IDX[s])
        c = harness.ci(rr[s].mean(), reps)
        d = harness.ci(rr[s].mean() - RR_REF[s].mean(), reps - REPS_REF[s])
        out[s] = dict(mrr=c["est"], lo=c["lo"], hi=c["hi"], d=d["est"], d_lo=d["lo"], d_hi=d["hi"])
    return out


summ = pd.DataFrame([dict(model=m, set=s, n=len(rr[s]), **v) for m, rr in RR.items() for s, v in vs_ref(rr).items()])
summ.to_csv(f"{OUT}/summary.csv", index=False)
v = summ.set_index(["model", "set"])
print("reference (k13 M2.2):", {s: round(float(RR_REF[s].mean()), 4) for s in ["A", "B_A", "B_N"]})
got = {s: round(float(v.loc[("base|hybm:0.7:2", s), "mrr"]), 4) for s in REF_K16}
ok = REF_TOL is None or all(abs(got[s] - x) <= REF_TOL for s, x in REF_K16.items())
print("base|hybm:0.7:2 vs k16 v2:", got, "expected", REF_K16, "" if ok else "WARNING: differs")

rule = pd.DataFrame([dict(model=m, A=v.loc[(m, "A"), "mrr"], dA=v.loc[(m, "A"), "d"], dA_lo=v.loc[(m, "A"), "d_lo"],
                          B_A=v.loc[(m, "B_A"), "mrr"], dB_A=v.loc[(m, "B_A"), "d"], dB_A_lo=v.loc[(m, "B_A"), "d_lo"],
                          B_N=v.loc[(m, "B_N"), "mrr"], dB_N=v.loc[(m, "B_N"), "d"], dB_N_lo=v.loc[(m, "B_N"), "d_lo"],
                          C=v.loc[(m, "C"), "mrr"], V=v.loc[(m, "V"), "mrr"])
                     for m in RR if m not in DISPLAY])
rule["eligible"] = (rule.dA_lo > -NI_MARGIN) & ((rule.dB_N_lo > 0) | ((rule.dB_A_lo > 0) & (rule.dB_N_lo > -NI_MARGIN)))
rule["pred_lb_delta"] = S1 * rule.dA + S2 * rule.dB_N
rule = rule.sort_values("pred_lb_delta", ascending=False, kind="mergesort")
rule.to_csv(f"{OUT}/rule_pool.csv", index=False)
with pd.option_context("display.width", 250):
    print(rule.round(4).to_string(index=False))
el = rule[rule.eligible]
PSTAR = el.iloc[0].model if len(el) else None
print("P* (pool side):", PSTAR or "none — M2.2 stays the pool side", f"({len(el)} eligible)")

bn_q = np.concatenate([qids_of[s][BN[s]] for s in ["R2", "V"]])
qlib = infos.set_index("qid").qlib.reindex(bn_q).values
bylib = pd.DataFrame([dict(model=m, lib=lb, n=int((qlib == lb).sum()), mrr=RR[m]["B_N"][qlib == lb].mean(),
                           ref=RR_REF["B_N"][qlib == lb].mean())
                      for m in ["base|hybm:0.7:2"] + ([PSTAR] if PSTAR else []) for lb in np.unique(qlib)])
bylib.to_csv(f"{OUT}/bylib.csv", index=False)

# %% [markdown]
# ## 2. 生成側: 生成物の一覧を決まった枠に差し込む

# %%
GSETS = ["A", "B_A", "R2", "V"]
GEN = {s: load(f"gen_{s}_s*.parquet", ["qid", "key"]) for s in GSETS}
ginfo = pd.concat([pd.read_csv(p) for p in sorted(glob.glob("/kaggle/input/**/gen_info_s*.csv", recursive=True))],
                  ignore_index=True).drop_duplicates(["qid", "set"])
print("generated rows", {s: len(g) for s, g in GEN.items()}, "| truth generated:",
      ginfo.groupby("set").has_truth.mean().round(3).to_dict())
for s in ["B_A", "R2", "V"]:
    assert set(ginfo[ginfo.set == s].qid) >= set(qids_of[s]), f"generation missing for some {s} queries"
ht = ginfo.set_index(["set", "qid"]).has_truth
rec_ga = float(ht.loc["B_A"].reindex(qids_of["B_A"]).mean())
rec_gn = float(np.concatenate([ht.loc[s].reindex(qids_of[s]).values[BN[s]] for s in ["R2", "V"]]).mean())
print(f"truth generated (recall after the cap): G_A {rec_ga:.3f}, G_N {rec_gn:.3f} (k19: 0.564, 0.349)")


def zs_by(v, q):
    d = pd.DataFrame({"q": q, "v": v})
    g = d.groupby("q").v
    sd = g.transform("std", ddof=0).values
    return np.where(sd > 1e-9, (v - g.transform("mean").values) / np.where(sd > 1e-9, sd, 1), 0.0)


# gen ranker: R2 の生成物の一覧（正解の鍵の行が label = 1）で学習
GF = list(gen.GEN_FEATURES)
for s in GSETS:
    g = GEN[s] = GEN[s].sort_values(["qid"], kind="mergesort").reset_index(drop=True)
    g["comb"] = g.fz_z + zs_by(g.ap4.values, g.qid.values) + g.fwd_z + g.frag_z
R2g = GEN["R2"]
oof = rank.cv_oof(R2g, GF, n_folds=5, groups=truth)
gms = [rank.fit(R2g, GF, seed=sd) for sd in SEEDS]
for i, m in enumerate(gms):
    m.save_model(f"{OUT}/gen_ranker_{i}.txt")
GEN["R2"]["rk"] = oof
for s in ["A", "B_A", "V"]:
    GEN[s]["rk"] = rank.predict(gms, GEN[s][GF].values.astype(np.float32)) if len(GEN[s]) else []
imp = pd.Series(np.mean([m.feature_importance("gain") for m in gms], 0), index=GF)
print("gen ranker top gain:", (imp / imp.sum()).sort_values(ascending=False).head(8).round(3).to_dict(),
      f"({time.time() - T0:.0f}s)")

gate_val = infos.set_index("qid").lib_max  # = lib_grp_max（ライブラリの側は v7 / v9 / k22 で同じ）


def gen_lists(s, score):
    """Per qid: (n_gen without the truth row, n_gen with it, rank of the truth row in the full list or inf,
    truth row's reaction is addition-only)."""
    g = GEN[s]
    q = qids_of[s]
    if len(g) == 0:
        z = np.zeros(len(q))
        return z, z, np.full(len(q), np.inf), np.zeros(len(q), bool)
    r = harness.ranks_of_truth(g.qid.values, g.label.values, g[score].values)
    n_all = g.groupby("qid").size()
    n_lab = g.groupby("qid").label.sum()
    add = g[g.label > 0].set_index("qid").seq.map(lambda x: not any(st.startswith(REMOVAL) for st in x.split("+")))
    return ((n_all - n_lab).reindex(q).fillna(0).values, n_all.reindex(q).fillna(0).values,
            r.reindex(q).fillna(np.inf).values, add.reindex(q).fillna(False).values.astype(bool))


def shift(r, slots, m):
    """Final position of pool rank r when the first m slots hold generated items."""
    p = np.array(r, np.float64)
    for j, s in enumerate(slots):
        p = np.where((j < m) & (s <= p), p + 1, p)
    return p


def g_pos(k, slots, m):
    """Final position of the truth at rank k of the generated list (inf when not slotted)."""
    sl = np.asarray(slots + [np.inf], np.float64)
    kk = np.where(np.isfinite(k), k, len(slots) + 1).astype(int)
    ok = np.isfinite(k) & (kk <= m)
    return np.where(ok, sl[np.minimum(kk, len(slots)) - 1], np.inf)


GL = {(s, sc): gen_lists(s, sc) for s in GSETS for sc in GEN_SCORES}
BASES = {"M2.2": RANK_REF}
if PSTAR:
    BASES[PSTAR] = RANK[PSTAR]
grows, GRR = [], {}
for bname, brank in BASES.items():
    for sname, slots in SLOTS.items():
        for tg in GATES:
            for sc in GEN_SCORES:
                name = f"{bname}+gen:{sname}:{tg}:{sc}"
                rk_new = dict(brank)
                gpos, gadd = {}, {}
                for s in GSETS:
                    n_b, n_g, k, add = GL[(s, sc)]
                    gate = gate_val.reindex(qids_of[s]).fillna(0).values < tg
                    m_b = np.where(gate, np.minimum(n_b, len(slots)), 0)
                    rk_new[s] = shift(brank[s], slots, m_b)
                    m_g = np.where(gate, np.minimum(n_g, len(slots)), 0)
                    gpos[s] = g_pos(k, slots, m_g)
                    gadd[s] = np.where(add, gpos[s], np.inf)
                rr = rr_from_ranks(rk_new)
                rr["G_A"] = harness.rr_of(gpos["B_A"])
                rr["G_N"] = np.concatenate([harness.rr_of(gpos[s])[BN[s]] for s in ["R2", "V"]])
                rr["G_A_add"] = harness.rr_of(gadd["B_A"])
                rr["G_N_add"] = np.concatenate([harness.rr_of(gadd[s])[BN[s]] for s in ["R2", "V"]])
                GRR[name] = rr
                d = vs_ref(rr, ["A", "B_A", "B_N", "V"])
                rec = dict(model=name, base=bname, slots=sname, gate=tg, score=sc,
                           A=d["A"]["mrr"], dA=d["A"]["d"], dA_lo=d["A"]["d_lo"], B_A=d["B_A"]["mrr"],
                           B_N=d["B_N"]["mrr"], dB_N=d["B_N"]["d"], dB_N_lo=d["B_N"]["d_lo"])
                for gs in ["G_A", "G_N", "G_A_add", "G_N_add"]:
                    c = harness.ci(rr[gs].mean(), harness.boot_reps(rr[gs], harness.boot_index(len(rr[gs]), N_BOOT, 7)))
                    rec.update({gs: c["est"], f"{gs}_lo": c["lo"]})
                grows.append(rec)
grule = pd.DataFrame(grows)
grule["G_ok"] = ((grule.G_N >= GEN_MIN) & (grule.G_N_lo > 0)) | ((grule.G_A >= GEN_MIN) & (grule.G_A_lo > 0))
grule["eligible"] = grule.G_ok & (grule.dA_lo > -NI_MARGIN) & (grule.dB_N_lo > -GEN_BN_MARGIN)
grule["pred_lb_delta"] = S1 * grule.dA + S2 * grule.dB_N + S3 * grule.G_N
grule = grule.sort_values("pred_lb_delta", ascending=False, kind="mergesort")
grule.to_csv(f"{OUT}/rule_gen.csv", index=False)
with pd.option_context("display.width", 250):
    print(grule.round(4).to_string(index=False))

# %% [markdown]
# ## 3. 最後の選択（固定）

# %%
final = [dict(model="M2.2", kind="none", pred_lb_delta=0.0)]
if PSTAR:
    r_ = rule[rule.model == PSTAR].iloc[0]
    final.append(dict(model=PSTAR, kind="pool", pred_lb_delta=float(r_.pred_lb_delta)))
for r_ in grule[grule.eligible].itertuples():
    final.append(dict(model=r_.model, kind="gen", pred_lb_delta=float(r_.pred_lb_delta)))
final = pd.DataFrame(final)
final["simple"] = final.kind.map({"none": 0, "pool": 1, "gen": 2})
final = final.sort_values(["pred_lb_delta", "simple"], ascending=[False, True], kind="mergesort")
best = final.iloc[0].to_dict()
chosen = dict(model=best["model"], kind=best["kind"], pred_lb_delta=best["pred_lb_delta"])
if best["kind"] == "gen":
    gr = grule[grule.model == best["model"]].iloc[0]
    chosen.update(base=gr.base, slots=SLOTS[gr.slots], slots_name=gr.slots, gate=float(gr.gate), score=gr.score)
elif best["kind"] == "pool":
    chosen.update(base=PSTAR)
else:
    chosen.update(base="M2.2")
if chosen["base"] != "M2.2":
    tag, _, var = chosen["base"].partition("|")
    chosen.update(fcols=tag, columns=FCOLS[tag], variant=var)
res = dict(features=tags[0], pstar=PSTAR, chosen=chosen, n_pool_eligible=int(len(el)),
           n_gen_eligible=int(grule.eligible.sum()), final=final.to_dict("records"),
           rule=dict(ni_margin=NI_MARGIN, gen_bn_margin=GEN_BN_MARGIN, gen_min=GEN_MIN, s1=S1, s2=S2, s3=S3))
json.dump(res, open(f"{OUT}/chosen.json", "w"), indent=1, default=float)
print(final.round(4).to_string(index=False))
print("\nCHOSEN:", chosen)
run = track.init("eval", group=EXP, config=dict(features=tags[0], ref=REF), online=False)
run.summary(dict(pstar=str(PSTAR), chosen=chosen["model"], pred_lb_delta=chosen["pred_lb_delta"],
                 n_gen_eligible=int(grule.eligible.sum())))
run.table("rule_pool", rule)
run.table("rule_gen", grule)
run.finish()
print(f"total {time.time() - T0:.0f}s")

# %% [markdown]
# # casmi26-k35-eval7
# M7 P0（2026-10-13）: k30 の特徴量（再計算なし）で、k32 v2 の構成を参照にして次の案を比べる（CPU だけ）。
# 1. 帯 [0.7, 0.8) の扱い（τ 0.8 は B_A +0.074 / B_N +0.058 だが A −0.021 で非劣性に落ちた。k31）:
#    hybm:τ:w（τ ∈ {0.7, 0.75, 0.8}、w ∈ {5, 10, 20}）、hybb:0.7:0.8:5:{10, 20}（帯だけ w を上げる）、
#    hybi:0.7:0.8:5（帯で規則とランカーの順を交互に。A の正解が規則の 1 位なら 2 位以内に残る）。search.variant_order
# 2. 同じ分子式の中の列（search.formula_group_features: fz_z、fzr_z、g2_z、fwd_z、frag_z の (分子, 分子式) の組の中の
#    順位と 1 位との差、組の大きさ。k26 の pool_formula.npz）。R1 / R2 は k30 で上位 300 に切ってあるので、組の中の順位は
#    切った後の候補で数える（提出では窓の全候補。切られるのは R の 5〜10% のクエリ）
# 3. ランカーの学習集合: R1 ∪ R2 の全部 / lib_grp_max < 0.8 のクエリだけ（ランカーが実際に使われる領域。C / B_C は
#    入れない、ユーザーの決定 2026-10-13）。学習に入らない R のクエリは全体のモデルで採点する（OOF と同じく未学習）
# 4. 生成のランカー: GEN_FEATURES（rk）/ + GEN_EXTRA（rkx、k31 の報告 G_N +0.0034 [+0.001]）。枠は S3、門 0.7
# 報告だけ（選び方の候補にしない）: 生成なし（"+nogen"、提出の列の対の比較用）、fz の値の列（fz、fz_z、fz_soft、fz_gap、fz_grp_gap2）を外して順位の列だけにした
# 組（k16 v4 の案、"_rko"）。帯ごとのクエリ数と、帯の中で規則とランカーのどちらが正しいか（band7.csv）。B_N を R2 側と
# V 側に分けた差（多重比較の確認、約 120 案で下限 > 0 の偽の通過は期待 1〜2 件）。
# 参照 = k32 v2（both_fprich_g2|hybm:0.7:5+gen:S3:0.7:rk）をここで同じ手順（3 つの種）で計算し直したもの。k31 の数字
# との差は報告するだけ。
# 選び方（M6 の 1 本、変えない）: 参照と対のブートストラップの差（2,000 回、95%）で
#   非劣性（すべて）: A の下限 > −0.02、G_N の下限 > −0.01
#   改善（どれか 1 つ）: B_N の下限 > 0 / B_A の下限 > 0 かつ B_N の下限 > −0.02 / G_N の下限 > 0 かつ B_N の下限 > −0.01
#   予測 LB の差 = 0.163·dA + 0.21·dB_N + 0.30·dG_N > 0。満たす案のうち最大のもの。同点は単純な方（足したものが少ない
#   方、次に格子の順）。無ければ k32 v2 のまま
# 出力: chosen7.json、rule7.csv、summary7.csv、bylib7.csv、band7.csv、importance7.csv、ranker7_{列の組}_{i}.txt、
# gen_ranker7_{i}.txt、gen_ranker7x_{i}.txt（k36 がどの案でも提出できるように、全部のランカーを残す）。
# 入力: k30-s0〜s3、k26（pool_formula.npz）、k05（プール、queries.csv）。

# %%
import glob
import json
import time

import numpy as np
import pandas as pd

from casmi26 import gen, harness, pool, rank, search, subform, track
from casmi26.frag import FRAG_FEATURES

T0 = time.time()
OUT = "/kaggle/working"
EXP = "M7"
SETS = ["A", "B_A", "C", "B_C", "R1", "R2", "V"]
SEEDS = (0, 1, 2)
N_BOOT = 2000
NI_MARGIN, GN_MARGIN, BN_GEN_MARGIN = 0.02, 0.01, 0.01
S1, S2, S3 = 0.163, 0.21, 0.30
SLOTS = list(range(3, 25, 3))  # S3
GATE = 0.7
SUB_TAU = 0.8  # 学習集合 ".lt08": lib_grp_max < 0.8 の R のクエリ
REF = "both_fprich_g2|hybm:0.7:5+gen:S3:0.7:rk"  # k32 v2
K31_REF = {"A": 0.9391, "B_A": 0.5391, "B_N": 0.5618, "G_N": 0.0495}  # k31 の both_fprich_g2|hybm:0.7:5（表示だけ）
EXPECT_VARIANT = "v2b"
REMOVAL = ("de_", "dehydroxy", "H2_remove", "H2_alcohol")
SCORES = [("hybm", (t, w)) for t in (0.7, 0.75, 0.8) for w in (5, 10, 20)] + \
         [("hybb", (0.7, 0.8, 5, 10)), ("hybb", (0.7, 0.8, 5, 20)), ("hybi", (0.7, 0.8, 5))]
SCORE_NAME = lambda sc: sc[0] + ":" + ":".join(f"{x:g}" for x in sc[1])
MAX_QUERIES = None  # 手元の動作確認だけ

PD = pool.Pool.find_dir()
truth = pd.read_csv(f"{PD}/queries.csv", usecols=["qid", "pid"]).set_index("qid").pid
in_co = pd.read_parquet(f"{PD}/pool.parquet", columns=["in_coconut"]).in_coconut.values
FT = subform.FormulaTable(subform.FormulaTable.find_dir())
assert len(FT.fid) == len(in_co)
tags = [json.load(open(p)) for p in sorted(glob.glob("/kaggle/input/**/features_info4.json", recursive=True))]
KEYS = ("variant", "epoch", "fwd", "fwd_file", "fwd_epoch", "frag_total", "fprich_epoch", "fprich_nbits", "g2_step",
        "g2_precision")
assert tags and all({k: t[k] for k in KEYS} == {k: tags[0][k] for k in KEYS} for t in tags), f"k30 shards disagree: {tags}"
assert tags[0]["variant"] == EXPECT_VARIANT, f"k30 features are from {tags[0]['variant']}"
print("features:", tags[0], "|", len(tags), "shards")


def load(pattern, dedupe):
    ps = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    d = pd.concat([pd.read_parquet(p) for p in ps], ignore_index=True) if ps else pd.DataFrame()
    return d.drop_duplicates(dedupe) if len(d) else d


infos = pd.concat([pd.read_csv(p) for p in sorted(glob.glob("/kaggle/input/**/query_info4_s*.csv", recursive=True))],
                  ignore_index=True).drop_duplicates("qid")
feats = {s: load(f"feats4_{s}_s*.parquet", ["qid", "pid"]).sort_values(["qid", "pid"], kind="mergesort")
         .reset_index(drop=True) for s in SETS}
if MAX_QUERIES:
    keepq = {s: np.sort(f.qid.unique())[:MAX_QUERIES] for s, f in feats.items()}
    feats = {s: f[f.qid.isin(keepq[s])].reset_index(drop=True) for s, f in feats.items()}
    infos = infos[infos.qid.isin(np.concatenate(list(keepq.values())))]
qids_of = {s: np.sort(infos[infos.set == s].qid.values) for s in SETS}
for s in SETS:
    G = search.formula_group_features({c: feats[s][c].values for c in search.GRP_BASE}, feats[s].qid.values,
                                      FT.fid[feats[s].pid.values])
    feats[s] = pd.concat([feats[s], pd.DataFrame(G, columns=search.GRP_FEATURES)], axis=1)
print("rows", {s: len(f) for s, f in feats.items()}, "| queries", {s: len(q) for s, q in qids_of.items()},
      f"| same-formula group size median (A) {np.median(feats['A'].n_same_formula):.0f} ({time.time() - T0:.0f}s)")
FL = list(search.FEATURES)
FPG = FL + FRAG_FEATURES + search.FWD_FEATURES + search.FPRICH_FEATURES + search.G2_FEATURES
FZ_VAL = ["fz", "fz_z", "fz_soft", "fz_gap", "fz_grp_gap2"]
FCOLS = {"both_fprich_g2": FPG, "both_fprich_g2_grp": FPG + search.GRP_FEATURES,
         "both_fprich_g2_rko": [c for c in FPG if c not in FZ_VAL]}
REPORT_ONLY = {"both_fprich_g2_rko"}
RANKERS = [("both_fprich_g2", ""), ("both_fprich_g2", ".lt08"), ("both_fprich_g2_grp", ""),
           ("both_fprich_g2_grp", ".lt08"), ("both_fprich_g2_rko", "")]
X = {s: feats[s][FL].values for s in SETS}  # search.COL の順（blend_score / rule_score が読む）
LAB = {s: feats[s].label.values for s in SETS}
QID = {s: feats[s].qid.values for s in SETS}
GMAX = {s: X[s][:, search.COL["lib_grp_max"]] for s in SETS}

# %% [markdown]
# ## 1. プール側のランカー（列の組 × 学習集合）

# %%
def ranker(cols, name, sub):
    R = pd.concat([feats["R1"], feats["R2"]], ignore_index=True)
    keep = (R.lib_grp_max.values < SUB_TAU) if sub else np.ones(len(R), bool)
    Rk = R[keep].reset_index(drop=True)
    oof = rank.cv_oof(Rk, cols, n_folds=5, seed=0, groups=truth, model_seeds=SEEDS)
    ms = [rank.fit(Rk, cols, seed=sd) for sd in SEEDS]
    for i, m in enumerate(ms):
        m.save_model(f"{OUT}/ranker7_{name}_{i}.txt")
    pr = np.empty(len(R))
    pr[keep] = oof
    if (~keep).any():
        pr[~keep] = rank.predict(ms, R.loc[~keep, cols].values.astype(np.float32))
    r1 = R.qid.isin(feats["R1"].qid.unique()).values
    out = {"R1": pr[r1], "R2": pr[~r1]}
    for s in ["A", "B_A", "C", "B_C", "V"]:
        out[s] = rank.predict(ms, feats[s][cols].values.astype(np.float32))
    imp = pd.Series(np.mean([m.feature_importance("gain") for m in ms], 0), index=cols)
    print(f"ranker {name} ({Rk.qid.nunique()} training queries) done ({time.time() - T0:.0f}s); top gain:",
          (imp / imp.sum()).sort_values(ascending=False).head(8).round(3).to_dict(), flush=True)
    return out, imp


def truth_rank(s, rk, sc):
    """Truth rank per query of set s under score variant sc (pid ties as harness.ranks_of_truth)."""
    kind, a = sc
    f = lambda n: X[s][:, search.COL[n]]
    tie = 1e-3 * f("ap4") - 1e-6 * f("mass_ppm")
    if kind == "hybm":
        score = search.blend_score(X[s], a[0], rk, a[1], qid=QID[s])
    else:
        w = np.where(GMAX[s] >= a[0], a[3], a[2]) if kind == "hybb" else a[2]
        score = search.hybrid_score(X[s], a[1], search.zscore_by(rk, QID[s]) + w * (f("ap4") + 0.2 * f("lib_max")) + tie)
    r = harness.ranks_of_truth(QID[s], LAB[s], score, feats[s].pid.values).reindex(qids_of[s]).fillna(np.inf)
    if kind == "hybi":
        band = (GMAX[s] >= a[0]) & (GMAX[s] < a[1])
        rows = pd.Series(np.flatnonzero(band)).groupby(QID[s][band]).apply(np.asarray)
        for q, ix in rows.items():
            o = search.variant_order(X[s][ix], rk[ix], sc)
            hit = np.flatnonzero(LAB[s][ix][o] > 0)
            r[q] = hit[0] + 1 if len(hit) else np.inf
    return r.values.astype(np.float64)


RK, IMP = {}, {}
for tag, sub in RANKERS:
    RK[tag + sub], IMP[tag + sub] = ranker(FCOLS[tag], tag + sub, bool(sub))
pd.DataFrame(IMP).to_csv(f"{OUT}/importance7.csv")
# 自己照合: variant_order（提出の路）の順位が、ここでの評価の順位と一致する（hybm と hybb、各組 100 クエリ）
for s in ["A", "V"]:
    for sc in [("hybm", (0.7, 5)), ("hybb", (0.7, 0.8, 5, 20))]:
        r = truth_rank(s, RK["both_fprich_g2"][s], sc)
        for j, q in enumerate(qids_of[s][:100]):
            ix = np.flatnonzero(QID[s] == q)
            if not len(ix):
                continue
            o = search.variant_order(X[s][ix], RK["both_fprich_g2"][s][ix], sc)
            hit = np.flatnonzero(LAB[s][ix][o] > 0)
            assert (hit[0] + 1 if len(hit) else np.inf) == r[j], f"variant_order differs from the eval ranks: {s} {sc} {q}"
print("self-check: variant_order == eval ranks (hybm, hybb)")
POOL = {}
for name in RK:
    for sc in SCORES:
        POOL[f"{name}|{SCORE_NAME(sc)}"] = {s: truth_rank(s, RK[name][s], sc) for s in SETS}
print(f"{len(POOL)} pool models ranked ({time.time() - T0:.0f}s)", flush=True)

# %% [markdown]
# ## 2. 生成側（S3、門 0.7、生成のランカー rk / rkx）

# %%
GSETS = ["A", "B_A", "R2", "V"]
GEN = {s: load(f"gen4_{s}_s*.parquet", ["qid", "key"]).sort_values(["qid"], kind="mergesort").reset_index(drop=True)
       for s in GSETS}
if MAX_QUERIES:
    GEN = {s: g[g.qid.isin(qids_of[s])].reset_index(drop=True) for s, g in GEN.items()}
print("generated rows", {s: len(g) for s, g in GEN.items()})


def gen_ranker(cols, name):
    R2g = GEN["R2"]
    oof = rank.cv_oof(R2g, cols, n_folds=5, seed=0, groups=truth, model_seeds=SEEDS)
    ms = [rank.fit(R2g, cols, seed=sd) for sd in SEEDS]
    for i, m in enumerate(ms):
        m.save_model(f"{OUT}/{name}_{i}.txt")
    out = {"R2": oof}
    for s in ["A", "B_A", "V"]:
        out[s] = rank.predict(ms, GEN[s][cols].values.astype(np.float32)) if len(GEN[s]) else np.zeros(0)
    return out


GF = list(gen.GEN_FEATURES)
GRK = {"rk": gen_ranker(GF, "gen_ranker7"), "rkx": gen_ranker(GF + list(gen.GEN_EXTRA), "gen_ranker7x")}
gate_val = infos.set_index("qid").lib_max


def gen_lists(s, score):
    """Per qid: (n_gen without the truth row, n_gen with it, rank of the truth row or inf, truth is addition-only)."""
    g, q = GEN[s], qids_of[s]
    if len(g) == 0:
        z = np.zeros(len(q))
        return z, z, np.full(len(q), np.inf), np.zeros(len(q), bool)
    r = harness.ranks_of_truth(g.qid.values, g.label.values, score)
    n_all, n_lab = g.groupby("qid").size(), g.groupby("qid").label.sum()
    add = g[g.label > 0].set_index("qid").seq.map(lambda x: not any(st.startswith(REMOVAL) for st in x.split("+")))
    return ((n_all - n_lab).reindex(q).fillna(0).values, n_all.reindex(q).fillna(0).values,
            r.reindex(q).fillna(np.inf).values, add.reindex(q).fillna(False).values.astype(bool))


def shift(r, m):
    p = np.array(r, np.float64)
    for j, sl in enumerate(SLOTS):
        p = np.where((j < m) & (sl <= p), p + 1, p)
    return p


def g_pos(k, m):
    sl = np.asarray(SLOTS + [np.inf], np.float64)
    kk = np.where(np.isfinite(k), k, len(SLOTS) + 1).astype(int)
    return np.where(np.isfinite(k) & (kk <= m), sl[np.minimum(kk, len(SLOTS)) - 1], np.inf)


tq = {s: truth.reindex(qids_of[s]).values for s in SETS}
BN = {s: in_co[tq[s]] for s in ["R2", "V"]}
ALL = SETS + ["B_N"]
GL = {k: {s: gen_lists(s, g[s]) for s in GSETS} for k, g in GRK.items()}


def with_gen(prank, gk):
    rk, gpos, gadd = dict(prank), {}, {}
    for s in GSETS:
        n_b, n_g, k, add = GL[gk][s]
        gate = gate_val.reindex(qids_of[s]).fillna(0).values < GATE
        rk[s] = shift(prank[s], np.where(gate, np.minimum(n_b, len(SLOTS)), 0))
        gpos[s] = g_pos(k, np.where(gate, np.minimum(n_g, len(SLOTS)), 0))
        gadd[s] = np.where(add, gpos[s], np.inf)
    rr = {s: harness.rr_of(rk[s]) for s in SETS}
    rr["B_N"] = np.concatenate([rr[s][BN[s]] for s in ["R2", "V"]])
    rr["G_A"] = harness.rr_of(gpos["B_A"])
    rr["G_N"] = np.concatenate([harness.rr_of(gpos[s])[BN[s]] for s in ["R2", "V"]])
    rr["G_A_add"] = harness.rr_of(gadd["B_A"])
    rr["G_N_add"] = np.concatenate([harness.rr_of(gadd[s])[BN[s]] for s in ["R2", "V"]])
    return rr


def no_gen(prank):
    """Report only (M6 rule: no candidate without generation): the pool order alone, G = 0."""
    rr = {s: harness.rr_of(prank[s]) for s in SETS}
    rr["B_N"] = np.concatenate([rr[s][BN[s]] for s in ["R2", "V"]])
    z_a, z_n = np.zeros(len(rr["B_A"])), np.zeros(len(rr["B_N"]))
    rr.update(G_A=z_a, G_A_add=z_a, G_N=z_n, G_N_add=z_n)
    return rr


RR = {f"{m}+gen:S3:0.7:{gk}": with_gen(pr, gk) for m, pr in POOL.items() for gk in GRK}
RR.update({f"{m}+nogen": no_gen(pr) for m, pr in POOL.items()})
assert REF in RR

# %% [markdown]
# ## 3. 参照（k32 v2）と対で比べ、決めたとおりに選ぶ

# %%
GS = ["G_A", "G_N", "G_A_add", "G_N_add"]
IDX = {s: harness.boot_index(len(RR[REF][s]), N_BOOT, seed=i) for i, s in enumerate(ALL)}
IDX.update(G_A=IDX["B_A"], G_A_add=IDX["B_A"], G_N=IDX["B_N"], G_N_add=IDX["B_N"])
REPS_REF = {s: harness.boot_reps(RR[REF][s], IDX[s]) for s in ALL + GS}


def vs_ref(rr, sets=ALL + GS):
    out = {}
    for s in sets:
        reps = harness.boot_reps(rr[s], IDX[s])
        c = harness.ci(rr[s].mean(), reps)
        d = harness.ci(rr[s].mean() - RR[REF][s].mean(), reps - REPS_REF[s])
        out[s] = dict(mrr=c["est"], lo=c["lo"], hi=c["hi"], d=d["est"], d_lo=d["lo"], d_hi=d["hi"])
    return out


summ = pd.DataFrame([dict(model=m, set=s, n=len(rr[s]), **v) for m, rr in RR.items() for s, v in vs_ref(rr).items()])
summ.to_csv(f"{OUT}/summary7.csv", index=False)
v = summ.set_index(["model", "set"])
print("reference (k32 v2 recomputed here):", {s: round(float(v.loc[(REF, s), "mrr"]), 4) for s in K31_REF},
      "| k31:", K31_REF, "| difference:", {s: round(float(v.loc[(REF, s), "mrr"]) - K31_REF[s], 4) for s in K31_REF})
n_bn_r2 = int(BN["R2"].sum())
rows = []
for i, (m, rr) in enumerate(RR.items()):
    tag, _, rest = m.partition("|")
    sc, _, g = rest.partition("+")
    simple = int("_grp" in tag) + int(".lt08" in tag) + int(g.endswith(":rkx")) + int(not sc.startswith("hybm"))
    r = dict(model=m, cols=tag, score=sc, gen=g, simple=simple,
             report_only=tag.split(".")[0] in REPORT_ONLY or g == "nogen", grid=i)
    for s in ["A", "B_A", "B_N", "G_N", "C", "V"]:
        r.update({s: v.loc[(m, s), "mrr"], f"d{s}": v.loc[(m, s), "d"], f"d{s}_lo": v.loc[(m, s), "d_lo"]})
    r["dBN_R2"] = rr["B_N"][:n_bn_r2].mean() - RR[REF]["B_N"][:n_bn_r2].mean()
    r["dBN_V"] = rr["B_N"][n_bn_r2:].mean() - RR[REF]["B_N"][n_bn_r2:].mean()
    rows.append(r)
rule = pd.DataFrame(rows)
rule["non_inferior"] = (rule.dA_lo > -NI_MARGIN) & (rule.dG_N_lo > -GN_MARGIN)
rule["path_BN"] = rule.dB_N_lo > 0
rule["path_BA"] = (rule.dB_A_lo > 0) & (rule.dB_N_lo > -NI_MARGIN)
rule["path_G"] = (rule.dG_N_lo > 0) & (rule.dB_N_lo > -BN_GEN_MARGIN)
rule["pred_lb_delta"] = S1 * rule.dA + S2 * rule.dB_N + S3 * rule.dG_N
rule["split_ok"] = (rule.dBN_R2 > 0) & (rule.dBN_V > 0)  # 報告だけ
rule["eligible"] = (rule.model != REF) & ~rule.report_only & rule.non_inferior & \
    (rule.path_BN | rule.path_BA | rule.path_G) & (rule.pred_lb_delta > 0)
rule = rule.sort_values(["pred_lb_delta", "simple", "grid"], ascending=[False, True, True], kind="mergesort")
rule.to_csv(f"{OUT}/rule7.csv", index=False)
with pd.option_context("display.width", 260, "display.max_rows", 300):
    print(rule.drop(columns=["grid", "cols", "score", "gen"]).head(60).round(4).to_string(index=False))
el = rule[rule.eligible]
best = el.iloc[0].to_dict() if len(el) else None

# 報告だけ: 帯ごとのクエリ数と、規則だけ / ランカーだけ（w=5、門なし）/ 参照の MRR
BANDS = [0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.01]
rk_ref = RK["both_fprich_g2"]
alt = {"rule": {s: harness.ranks_of_truth(QID[s], LAB[s], search.rule_score(X[s], "lib_analog"), feats[s].pid.values)
                .reindex(qids_of[s]).fillna(np.inf).values for s in SETS},
       "ranker": {s: truth_rank(s, rk_ref[s], ("hybm", (9.0, 5))) for s in SETS}}
brow = []
for grp, parts in [("A", [("A", None)]), ("B_A", [("B_A", None)]), ("B_N", [("R2", BN["R2"]), ("V", BN["V"])])]:
    gm = np.concatenate([gate_val.reindex(qids_of[s][m if m is not None else slice(None)]).values for s, m in parts])
    rrs = {k: np.concatenate([harness.rr_of(d[s])[m if m is not None else slice(None)] for s, m in parts]) for k, d in alt.items()}
    rrs["ref"] = RR[REF][grp] if grp != "B_N" else RR[REF]["B_N"]
    for lo, hi in zip(BANDS[:-1], BANDS[1:]):
        b = (gm >= lo) & (gm < hi)
        brow.append(dict(group=grp, band=f"[{lo}, {hi})", n=int(b.sum()),
                         **{f"mrr_{k}": float(x[b].mean()) if b.any() else np.nan for k, x in rrs.items()}))
band = pd.DataFrame(brow)
band.to_csv(f"{OUT}/band7.csv", index=False)
print("bands of lib_grp_max (report only):\n", band.round(4).to_string(index=False))
bn_q = np.concatenate([qids_of[s][BN[s]] for s in ["R2", "V"]])
qlib = infos.set_index("qid").qlib.reindex(bn_q).values
show = [REF] + ([best["model"]] if best else [])
pd.DataFrame([dict(model=m, lib=lb, n=int((qlib == lb).sum()), mrr=RR[m]["B_N"][qlib == lb].mean(),
                   ref=RR[REF]["B_N"][qlib == lb].mean()) for m in show for lb in np.unique(qlib)]) \
    .to_csv(f"{OUT}/bylib7.csv", index=False)

# %% [markdown]
# ## 4. 結果

# %%
chosen = dict(model=best["model"], keep_ref=False, pred_lb_delta=float(best["pred_lb_delta"])) if best else \
    dict(model=REF, keep_ref=True, pred_lb_delta=0.0)
res = dict(features=tags[0], chosen=chosen, n_eligible=int(rule.eligible.sum()), reference=REF,
           reference_recomputed={s: float(v.loc[(REF, s), "mrr"]) for s in ["A", "B_A", "B_N", "G_A", "G_N"]},
           k31_reference=K31_REF, rankers={n: FCOLS[n.split(".")[0]] for n in RK},
           rule=dict(ni_margin=NI_MARGIN, gn_margin=GN_MARGIN, bn_gen_margin=BN_GEN_MARGIN, s1=S1, s2=S2, s3=S3,
                     pred_positive=True))
json.dump(res, open(f"{OUT}/chosen7.json", "w"), indent=1, default=float)
print("\nCHOSEN:", chosen)
run = track.init("eval", group=EXP, config=dict(features=tags[0], ref=REF), online=False)
run.summary(dict(chosen=chosen["model"], keep_ref=chosen["keep_ref"], pred_lb_delta=chosen["pred_lb_delta"],
                 n_eligible=int(rule.eligible.sum())))
run.table("rule7", rule)
run.finish()
print(f"total {time.time() - T0:.0f}s")

# %% [markdown]
# # casmi26-k31-eval6
# M6 P1 の 2（2026-10-12）: k30 の特徴量で、fprich と G2 をプール側のランカーに足した案を、結果を見る前に固定した
# 選び方で k24 v1 と比べる（CPU だけ）。
# 0. 照合: k30 の 41 列が k22 と一致する（k23 v2 と同じ決まり。候補の集合は R1 / R2 以外で完全一致、R1 / R2 は 99.9%
#    以上。fz と同点の列以外は 1e-5、同点の 8 列は 1e-5 を超えてずれたクエリが 1% 以下、fz は警告だけ）。
#    新しい列（fzr、fzr_z、fzr_rank、g2_z、g2_rank）が有限。一致しなければ止める
# 1. プール側: ランカーの列 both（P* の 41 列）/ +fprich / +g2 / +fprich+g2 × hybm:τ:w（τ ∈ {0.6, 0.7, 0.8}、
#    w ∈ {1, 2, 5}）= 36 案。ランカーは R1 + R2 で学習し、R1 / R2 は正解の構造ごとの 5 分割 OOF（分割の種 0、各分割で
#    3 つの種のモデルの平均）、他の組は 3 つの種のモデルの平均
# 2. 生成側（どの案でも同じ）: k24 v1 と同じく、lib_grp_max < 0.7 のクエリで生成物の一覧の上位を S3（3, 6, …, 24 位）
#    に差し込む。並べ方は生成のランカー（GEN_FEATURES だけ。R2 は 5 分割 OOF、各分割で 3 つの種の平均）。
#    案どうしで変わるのはプール側の列だけなので、生成の順位（G_A、G_N）はどの案でも参照と同じになる
# 3. 参照 = k24 v1 の構成（both|hybm:0.7:2 + gen:S3:0.7:rk）を、ここで同じ手順で計算し直したもの。k23 の数字とは
#    LightGBM の種の違いの分だけずれる（報告するだけで、照合はしない）
# 選び方（固定。計画 2026-10-11、M6 の P1）: 参照と対のブートストラップの差（2,000 回、95%）で
#   非劣性（すべて）: A の下限 > −0.02、G_N の下限 > −0.01
#   改善（どれか 1 つ）: B_N の下限 > 0 / B_A の下限 > 0 かつ B_N の下限 > −0.02 / G_N の下限 > 0 かつ B_N の下限 > −0.01
#   予測 LB の差 = 0.163·dA + 0.21·dB_N + 0.30·dG_N > 0（2026-10-12 のユーザーの決定、P1 の結果を見る前）
#   満たす案のうち予測 LB の差が最大のもの。同点は単純な方（足す列の塊が少ない方、次に格子の順）。無ければ k24 v1 のまま
# 報告だけ: 生成のランカーに GEN_EXTRA（生成物の fzr、g2）を足した場合の G_A / G_N（P1.5 の材料）、特徴量の重要度、
# B_N のライブラリ別、計算し直した参照と k23 の数字の差。
# 出力（k23 と名前を分ける）: chosen6.json、rule6.csv、summary6.csv、bylib6.csv、check_k22.csv、importance6.csv、
# ranker6_{tag}_{i}.txt、gen_ranker6_{i}.txt。
# 入力: k30-s0〜s3、k22-s0〜s3（照合用）、k05（プール、queries.csv）。

# %%
import glob
import json
import time

import numpy as np
import pandas as pd

from casmi26 import gen, harness, pool, rank, search, track
from casmi26.frag import FRAG_FEATURES

T0 = time.time()
OUT = "/kaggle/working"
EXP = "M6"
SETS = ["A", "B_A", "C", "B_C", "R1", "R2", "V"]
SEEDS = (0, 1, 2)
N_BOOT = 2000
NI_MARGIN, GN_MARGIN, BN_GEN_MARGIN = 0.02, 0.01, 0.01
S1, S2, S3 = 0.163, 0.21, 0.30
TAUS, WS = (0.6, 0.7, 0.8), (1, 2, 5)
SLOTS = list(range(3, 25, 3))  # S3
GATE = 0.7
REF = "both|hybm:0.7:2"  # k24 v1 のプール側（生成側はどの案でも同じ）
EXPECT_VARIANT = "v2b"
K23_REF = {"A": 0.934, "B_N": 0.539, "G_N": 0.050}  # k23 v2 の k24 v1（表示だけ）
CHECK_TOL, CHECK_TOL_FZ, CHECK_OVERLAP = 1e-5, 1e-3, 0.999
TIE_COLS = ["top_tan", "best_tan", "ap1", "tan_self", "ap_rank", "ap_gap", "top_sim", "n_analog_hi"]
TIE_QUERY_SHARE = 0.01
REMOVAL = ("de_", "dehydroxy", "H2_remove", "H2_alcohol")

PD = pool.Pool.find_dir()
truth = pd.read_csv(f"{PD}/queries.csv", usecols=["qid", "pid"]).set_index("qid").pid
in_co = pd.read_parquet(f"{PD}/pool.parquet", columns=["in_coconut"]).in_coconut.values
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
qids_of = {s: np.sort(infos[infos.set == s].qid.values) for s in SETS}
print("rows", {s: len(f) for s, f in feats.items()}, "| queries", {s: len(q) for s, q in qids_of.items()},
      f"({time.time() - T0:.0f}s)")
FL = list(search.FEATURES)
BOTH = FL + FRAG_FEATURES + search.FWD_FEATURES
FCOLS = {"both": BOTH, "both_fprich": BOTH + search.FPRICH_FEATURES, "both_g2": BOTH + search.G2_FEATURES,
         "both_fprich_g2": BOTH + search.FPRICH_FEATURES + search.G2_FEATURES}
SIMPLE = {"both": 0, "both_fprich": 1, "both_g2": 1, "both_fprich_g2": 2}
X = {s: feats[s][FCOLS["both_fprich_g2"]].values for s in SETS}  # 先頭の 34 列は search.COL の順（blend_score が読む）

# %% [markdown]
# ## 0. 照合: k30 の 41 列が k22 と一致するか、新しい列が有限か

# %%
HARD = [c for c in BOTH if c not in search.FZ_FEATURES and c not in TIE_COLS]
chk = []
for s in SETS:
    old = load(f"feats_{s}_s*.parquet", ["qid", "pid"])
    both_q = np.intersect1d(old.qid.unique(), feats[s].qid.unique())
    a = old[old.qid.isin(both_q)]
    b = feats[s][feats[s].qid.isin(both_q)]
    m = a.merge(b, on=["qid", "pid"], suffixes=("_o", "_n"))
    overlap = len(m) / max(len(a), len(b), 1)
    need = CHECK_OVERLAP if s in harness.TRAIN_SETS else 1.0
    assert overlap >= need, f"{s}: candidate sets differ from k22 (overlap {overlap:.5f})"
    for c in BOTH:
        dd = np.abs(m[c + "_o"].values.astype(np.float64) - m[c + "_n"].values.astype(np.float64))
        chk.append(dict(set=s, col=c, max_diff=float(np.nanmax(dd)) if len(m) else 0.0,
                        query_share=m.loc[dd > CHECK_TOL, "qid"].nunique() / max(m.qid.nunique(), 1),
                        overlap=overlap, n=len(m)))
    for c in search.FPRICH_FEATURES + ["g2_z", "g2_rank"]:
        assert np.isfinite(feats[s][c].values).all(), f"{s}: {c} is not finite"
chk = pd.DataFrame(chk)
chk.to_csv(f"{OUT}/check_k22.csv", index=False)
bad = chk[chk.col.isin(HARD) & (chk.max_diff > CHECK_TOL)]
assert bad.empty, f"features differ from k22:\n{bad}"
tie = chk[chk.col.isin(TIE_COLS)]
assert (tie.query_share <= TIE_QUERY_SHARE).all(), f"tie-sensitive analog columns differ too often:\n{tie}"
warn = chk[chk.col.isin(search.FZ_FEATURES) & (chk.max_diff > CHECK_TOL_FZ)]
if len(warn):
    print("WARNING: fz columns differ from k22 beyond", CHECK_TOL_FZ, "\n", warn.to_string(index=False))
g2_nan = {s: round(float(feats[s].g2.isna().mean()), 5) for s in SETS}
print(f"k22 check passed: max |diff| hard columns {chk[chk.col.isin(HARD)].max_diff.max():.2e}, tie columns query "
      f"share max {tie.query_share.max():.4f}, fz {chk[chk.col.isin(search.FZ_FEATURES)].max_diff.max():.2e}, overlap "
      f"min {chk.overlap.min():.5f}; g2 NaN share {g2_nan} ({time.time() - T0:.0f}s)")

# %% [markdown]
# ## 1. プール側のランカー（4 通りの列）と hybm の格子

# %%
def ranker(cols, tag):
    R = pd.concat([feats["R1"], feats["R2"]], ignore_index=True)
    oof = rank.cv_oof(R, cols, n_folds=5, seed=0, groups=truth, model_seeds=SEEDS)
    ms = [rank.fit(R, cols, seed=sd) for sd in SEEDS]
    for i, m in enumerate(ms):
        m.save_model(f"{OUT}/ranker6_{tag}_{i}.txt")
    r1 = R.qid.isin(feats["R1"].qid.unique()).values
    out = {"R1": oof[r1], "R2": oof[~r1]}
    for s in ["A", "B_A", "C", "B_C", "V"]:
        out[s] = rank.predict(ms, feats[s][cols].values.astype(np.float32))
    imp = pd.Series(np.mean([m.feature_importance("gain") for m in ms], 0), index=cols)
    print(f"ranker {tag} done ({time.time() - T0:.0f}s); top gain:",
          (imp / imp.sum()).sort_values(ascending=False).head(8).round(3).to_dict(), flush=True)
    return out, imp


RK, IMP = {}, {}
for tag, cols in FCOLS.items():
    RK[tag], IMP[tag] = ranker(cols, tag)
pd.DataFrame(IMP).to_csv(f"{OUT}/importance6.csv")
POOL = {}
for tag in FCOLS:
    for tau in TAUS:
        for w in WS:
            sc = {s: search.blend_score(X[s], tau, RK[tag][s], w, qid=feats[s].qid.values) for s in SETS}
            POOL[f"{tag}|hybm:{tau}:{w}"] = {
                s: harness.ranks_of_truth(feats[s].qid.values, feats[s].label.values, sc[s], feats[s].pid.values)
                .reindex(qids_of[s]).fillna(np.inf).values.astype(np.float64) for s in SETS}
assert REF in POOL
print(f"{len(POOL)} pool models ranked ({time.time() - T0:.0f}s)")

# %% [markdown]
# ## 2. 生成側（どの案でも同じ: S3、門 0.7、GEN_FEATURES の生成のランカー）

# %%
GSETS = ["A", "B_A", "R2", "V"]
GEN = {s: load(f"gen4_{s}_s*.parquet", ["qid", "key"]).sort_values(["qid"], kind="mergesort").reset_index(drop=True)
       for s in GSETS}
ginfo = pd.concat([pd.read_csv(p) for p in sorted(glob.glob("/kaggle/input/**/gen_info4_s*.csv", recursive=True))],
                  ignore_index=True).drop_duplicates(["qid", "set"])
for s in ["B_A", "R2", "V"]:
    assert set(ginfo[ginfo.set == s].qid) >= set(qids_of[s]), f"generation missing for some {s} queries"
print("generated rows", {s: len(g) for s, g in GEN.items()}, "| truth generated:",
      ginfo.groupby("set").has_truth.mean().round(3).to_dict())


def gen_ranker(cols, tag):
    R2g = GEN["R2"]
    oof = rank.cv_oof(R2g, cols, n_folds=5, seed=0, groups=truth, model_seeds=SEEDS)
    ms = [rank.fit(R2g, cols, seed=sd) for sd in SEEDS]
    out = {"R2": oof}
    for s in ["A", "B_A", "V"]:
        out[s] = rank.predict(ms, GEN[s][cols].values.astype(np.float32)) if len(GEN[s]) else np.zeros(0)
    imp = pd.Series(np.mean([m.feature_importance("gain") for m in ms], 0), index=cols)
    print(f"gen ranker {tag} top gain:", (imp / imp.sum()).sort_values(ascending=False).head(8).round(3).to_dict(),
          f"({time.time() - T0:.0f}s)", flush=True)
    return out, ms


GF = list(gen.GEN_FEATURES)
GRK, gms = gen_ranker(GF, "gen")
for i, m in enumerate(gms):
    m.save_model(f"{OUT}/gen_ranker6_{i}.txt")
GRK_X, _ = gen_ranker(GF + gen.GEN_EXTRA, "gen+extra")  # 報告だけ（P1.5 の材料）
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
    """Final position of pool rank r when the first m slots hold generated items."""
    p = np.array(r, np.float64)
    for j, sl in enumerate(SLOTS):
        p = np.where((j < m) & (sl <= p), p + 1, p)
    return p


def g_pos(k, m):
    """Final position of the truth at rank k of the generated list (inf when not slotted)."""
    sl = np.asarray(SLOTS + [np.inf], np.float64)
    kk = np.where(np.isfinite(k), k, len(SLOTS) + 1).astype(int)
    return np.where(np.isfinite(k) & (kk <= m), sl[np.minimum(kk, len(SLOTS)) - 1], np.inf)


tq = {s: truth.reindex(qids_of[s]).values for s in SETS}
BN = {s: in_co[tq[s]] for s in ["R2", "V"]}
ALL = SETS + ["B_N"]


def with_gen(prank, grk):
    """Ranks of a pool model with the fixed generation slots -> rr dict incl. G_A / G_N (and addition-only views)."""
    rk, gpos, gadd = dict(prank), {}, {}
    for s in GSETS:
        n_b, n_g, k, add = gen_lists(s, grk[s])
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


RR = {m: with_gen(pr, GRK) for m, pr in POOL.items()}

# %% [markdown]
# ## 3. 参照（k24 v1 の構成）と対で比べ、決めたとおりに選ぶ

# %%
GS = ["G_A", "G_N", "G_A_add", "G_N_add"]
IDX = {s: harness.boot_index(len(RR[REF][s]), N_BOOT, seed=i) for i, s in enumerate(ALL)}
IDX.update(G_A=IDX["B_A"], G_A_add=IDX["B_A"], G_N=IDX["B_N"], G_N_add=IDX["B_N"])  # 同じクエリの並び
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
summ.to_csv(f"{OUT}/summary6.csv", index=False)
v = summ.set_index(["model", "set"])
got = {s: round(float(v.loc[(REF, s), "mrr"]), 4) for s in K23_REF}
print("reference (k24 v1 recomputed here):", {s: round(float(v.loc[(REF, s), "mrr"]), 4) for s in ["A", "B_A", "B_N", "G_A", "G_N"]},
      "| k23:", K23_REF, "| difference:", {s: round(got[s] - K23_REF[s], 4) for s in K23_REF})
rows = []
for m in RR:
    r = dict(model=m, tag=m.split("|")[0], simple=SIMPLE[m.split("|")[0]])
    for s in ["A", "B_A", "B_N", "G_N", "C", "V"]:
        r.update({s: v.loc[(m, s), "mrr"], f"d{s}": v.loc[(m, s), "d"], f"d{s}_lo": v.loc[(m, s), "d_lo"]})
    rows.append(r)
rule = pd.DataFrame(rows)
rule["non_inferior"] = (rule.dA_lo > -NI_MARGIN) & (rule.dG_N_lo > -GN_MARGIN)
rule["path_BN"] = rule.dB_N_lo > 0
rule["path_BA"] = (rule.dB_A_lo > 0) & (rule.dB_N_lo > -NI_MARGIN)
rule["path_G"] = (rule.dG_N_lo > 0) & (rule.dB_N_lo > -BN_GEN_MARGIN)
rule["pred_lb_delta"] = S1 * rule.dA + S2 * rule.dB_N + S3 * rule.dG_N
rule["eligible"] = (rule.model != REF) & rule.non_inferior & (rule.path_BN | rule.path_BA | rule.path_G) & \
    (rule.pred_lb_delta > 0)
rule["grid"] = np.arange(len(rule))
rule = rule.sort_values(["pred_lb_delta", "simple", "grid"], ascending=[False, True, True], kind="mergesort")
rule.to_csv(f"{OUT}/rule6.csv", index=False)
with pd.option_context("display.width", 250):
    print(rule.drop(columns=["grid"]).round(4).to_string(index=False))
el = rule[rule.eligible]
best = el.iloc[0].to_dict() if len(el) else None

# 報告だけ: 生成のランカーに GEN_EXTRA を足したとき（参照のプール側で）
rr_x = with_gen(POOL[REF], GRK_X)
dx = vs_ref(rr_x, ["G_A", "G_N", "B_N", "A"])
print("report only — gen ranker with GEN_EXTRA on the reference pool side:",
      {s: dict(mrr=round(dx[s]["mrr"], 4), d=round(dx[s]["d"], 4), d_lo=round(dx[s]["d_lo"], 4)) for s in dx})
bn_q = np.concatenate([qids_of[s][BN[s]] for s in ["R2", "V"]])
qlib = infos.set_index("qid").qlib.reindex(bn_q).values
show = [REF] + ([best["model"]] if best else [])
pd.DataFrame([dict(model=m, lib=lb, n=int((qlib == lb).sum()), mrr=RR[m]["B_N"][qlib == lb].mean(),
                   ref=RR[REF]["B_N"][qlib == lb].mean()) for m in show for lb in np.unique(qlib)]) \
    .to_csv(f"{OUT}/bylib6.csv", index=False)

# %% [markdown]
# ## 4. 結果

# %%
if best:
    tag, _, var = best["model"].partition("|")
    chosen = dict(model=best["model"] + "+gen:S3:0.7:rk", kind="pool+gen", keep_k24=False,
                  pred_lb_delta=float(best["pred_lb_delta"]), base=best["model"], fcols=tag, columns=FCOLS[tag],
                  variant=var, slots=SLOTS, slots_name="S3", gate=GATE, score="rk", gen_columns=GF)
else:
    chosen = dict(model=REF + "+gen:S3:0.7:rk", kind="reference", keep_k24=True, pred_lb_delta=0.0)
res = dict(features=tags[0], chosen=chosen, n_eligible=int(rule.eligible.sum()), reference=REF,
           reference_recomputed={s: float(v.loc[(REF, s), "mrr"]) for s in ["A", "B_A", "B_N", "G_A", "G_N"]},
           k23_reference=K23_REF, gen_extra_report={s: dx[s] for s in dx},
           rule=dict(ni_margin=NI_MARGIN, gn_margin=GN_MARGIN, bn_gen_margin=BN_GEN_MARGIN, s1=S1, s2=S2, s3=S3,
                     pred_positive=True))
json.dump(res, open(f"{OUT}/chosen6.json", "w"), indent=1, default=float)
print("\nCHOSEN:", {k: x for k, x in chosen.items() if k not in ("columns", "gen_columns")})
run = track.init("eval", group=EXP, config=dict(features=tags[0], ref=REF), online=False)
run.summary(dict(chosen=chosen["model"], keep_k24=chosen["keep_k24"], pred_lb_delta=chosen["pred_lb_delta"],
                 n_eligible=int(rule.eligible.sum())))
run.table("rule6", rule)
run.finish()
print(f"total {time.time() - T0:.0f}s")

# %% [markdown]
# # casmi26-k42-eval8
# M7 P3-2（2026-10-17、結果を見る前に固定。計画 1-3-deep-storm.md）: k30 の特徴量に k41 の外部モデルの列（gl = GLACIER、
# ice = ICEBERG。ext.FEATURES）を足して、参照 v12 と比べる（CPU だけ）。
# - 参照 = v12（both_fprich_g2_grp|hybi:0.7:0.8:5+nogen、公開 LB 0.339）をここで同じ手順（3 つの種）で計算し直したもの。
#   k35 の rule7 と ±0.001 で一致することを確かめる。
# - 格子（ext）: 列の組 {both_fprich_g2_grp_gl, _ice, _gl_ice}（k41 の出力があるものだけ）× 学習集合 {R1 ∪ R2 全部,
#   漏れなしの R だけ ".clean"} × 点数 {hybi:0.7:0.8:5, hybm:0.7:5, hybm:0.8:5} × 生成 {nogen, gen:S3:0.7:rkx}。
#   ext 無し: 参照の列の組（全部 / .clean は報告だけ）× 同じ点数 × 同じ生成。
# - 選び方（ユーザーの決定 2026-10-16 / 2026-10-17）:
#   ext 無し: A の下限 > −0.02、B_N の下限 > 0、予測 LB の差 0.163·dA + 0.50·dB_N > 0 の最大（同点は単純な方）。
#   ext あり: 手元の門は無い（採否は v12 との対の LB）。LB に出す 1 本 = 全件の B_N（1,096、漏れあり）での 0.50·dB_N が
#   最大の案（A は見ない、同点は単純な方）→ chosen8.json の ext_submit。
# - 報告だけ: 漏れなし B_N 72 件（p3_clean_qids.csv）、漏れなし・正イオン・CE ありの R2 ∪ V 全部（正解が COCONUT に
#   無いものも含む）、漏れた B_N だけの差。A / B_A は漏れありの数字。ext の列の gain。B_N の R2 側と V 側。
# 出力: chosen8.json、rule8.csv、summary8.csv、rr8.parquet（qid、set、案ごとの RR）、bylib8.csv、importance8.csv、
# ranker8_{列の組}{.clean}_{i}.txt、gen_ranker8x_{i}.txt。
# 入力: k30-s0〜s3、k26（pool_formula.npz）、k05（プール、queries.csv）、k35（rule7.csv）、casmi26-p3-ext（k41 の出力）、
# casmi26-mspred（audit/）。

# %%
import glob
import json
import os
import time

import numpy as np
import pandas as pd

from casmi26 import ext, gen, harness, paths, pool, rank, search, subform, track
from casmi26.frag import FRAG_FEATURES

T0 = time.time()
OUT = "/kaggle/working"
EXP = "M7"
SETS = ["A", "B_A", "R1", "R2", "V"]
SEEDS = (0, 1, 2)
N_BOOT = 2000
NI_MARGIN = 0.02
S1, S2 = 0.163, 0.50
SLOTS = list(range(3, 25, 3))  # S3
GATE = 0.7
REF_TAG = "both_fprich_g2_grp"
REF = "both_fprich_g2_grp|hybi:0.7:0.8:5+nogen"  # k36 v12
REF_TOL = 0.001
EXPECT_VARIANT = "v2b"
REMOVAL = ("de_", "dehydroxy", "H2_remove", "H2_alcohol")
SCORES = [("hybi", (0.7, 0.8, 5)), ("hybm", (0.7, 5)), ("hybm", (0.8, 5))]
GENS = ["nogen", "gen:S3:0.7:rkx"]
SCORE_NAME = lambda sc: sc[0] + ":" + ":".join(f"{x:g}" for x in sc[1])
MAX_QUERIES = None  # 手元の動作確認だけ

PD = pool.Pool.find_dir()
truth = pd.read_csv(f"{PD}/queries.csv", usecols=["qid", "pid"]).set_index("qid").pid
in_co = pd.read_parquet(f"{PD}/pool.parquet", columns=["in_coconut"]).in_coconut.values
FT = subform.FormulaTable(subform.FormulaTable.find_dir())
tags = [json.load(open(p)) for p in sorted(glob.glob("/kaggle/input/**/features_info4.json", recursive=True))]
assert tags and all(t["variant"] == EXPECT_VARIANT for t in tags), tags
leak = pd.read_csv(paths.find("query_leak.csv")).set_index("qid")
clean72 = set(pd.read_csv(paths.find("p3_clean_qids.csv")).qid)
# k41 の出力: 1 本（ext_info_gl.json）または分割（ext_info_ice_s0.json、ext_info_ice_s1.json …）
EXT_PARTS = {}
for p in sorted(glob.glob("/kaggle/input/**/ext_info_*.json", recursive=True)):
    EXT_PARTS.setdefault(os.path.basename(p)[len("ext_info_"):-5].split("_s")[0], []).append(json.load(open(p)))
EXT_INFO = {}
for k, parts in EXT_PARTS.items():
    n = parts[0].get("n_shards", 1)
    assert sorted(x.get("shard", 0) for x in parts) == list(range(n)), f"{k}: shards {[x.get('shard') for x in parts]} of {n}"
    assert len({json.dumps(x["sha"], sort_keys=True) for x in parts}) == 1, f"{k}: shards from different weights"
    EXT_INFO[k] = dict(parts[0], entries=sum(x["entries"] for x in parts), failed=sum(x["failed"] for x in parts),
                       n_shards=n)
KINDS = [k for k in ext.KINDS if k in EXT_INFO]
assert KINDS, "attach the k41 output (casmi26-p3-ext)"
for k in KINDS:
    assert EXT_INFO[k]["ref"] == REF and EXT_INFO[k]["k_top"] == ext.K_TOP, EXT_INFO[k]
    assert MAX_QUERIES or not EXT_INFO[k]["max_queries"], f"{k}: k41 output is a smoke run"
print("ext kinds:", {k: dict(sha=EXT_INFO[k]["sha"], entries=EXT_INFO[k]["entries"], failed=EXT_INFO[k]["failed"])
                     for k in KINDS})


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
    for k in KINDS:
        ps = sorted(glob.glob(f"/kaggle/input/**/ext_{k}_{s}.parquet", recursive=True) +
                    glob.glob(f"/kaggle/input/**/ext_{k}_{s}_s[0-9]*.parquet", recursive=True))
        e = pd.concat([pd.read_parquet(p) for p in ps], ignore_index=True).drop_duplicates(["qid", "pid"])
        m = feats[s][["qid", "pid"]].merge(e, on=["qid", "pid"], how="left")
        if not MAX_QUERIES:
            assert m[ext.FEATURES[k][1]].notna().all(), f"{s}: k30 rows without {k} columns"
        feats[s] = pd.concat([feats[s], m[ext.FEATURES[k]]], axis=1)
print("rows", {s: len(f) for s, f in feats.items()}, "| queries", {s: len(q) for s, q in qids_of.items()},
      f"({time.time() - T0:.0f}s)", flush=True)
FL = list(search.FEATURES)
BASE = FL + FRAG_FEATURES + search.FWD_FEATURES + search.FPRICH_FEATURES + search.G2_FEATURES + search.GRP_FEATURES
FCOLS = {REF_TAG: BASE}
for k in KINDS:
    FCOLS[f"{REF_TAG}_{k}"] = BASE + ext.FEATURES[k]
if len(KINDS) == 2:
    FCOLS[f"{REF_TAG}_gl_ice"] = BASE + ext.FEATURES["gl"] + ext.FEATURES["ice"]
RANKERS = [(t, sub) for t in FCOLS for sub in ("", ".clean")]
X = {s: feats[s][FL].values for s in SETS}
LAB = {s: feats[s].label.values for s in SETS}
QID = {s: feats[s].qid.values for s in SETS}
GMAX = {s: X[s][:, search.COL["lib_grp_max"]] for s in SETS}

# %% [markdown]
# ## 1. プール側のランカー（列の組 × 学習集合）

# %%
def ranker(cols, name, sub):
    R = pd.concat([feats["R1"], feats["R2"]], ignore_index=True)
    keep = ~leak.leak.reindex(R.qid.values).fillna(True).values.astype(bool) if sub else np.ones(len(R), bool)
    Rk = R[keep].reset_index(drop=True)
    oof = rank.cv_oof(Rk, cols, n_folds=5, seed=0, groups=truth, model_seeds=SEEDS)
    ms = [rank.fit(Rk, cols, seed=sd) for sd in SEEDS]
    for i, m in enumerate(ms):
        m.save_model(f"{OUT}/ranker8_{name}_{i}.txt")
    pr = np.empty(len(R))
    pr[keep] = oof
    if (~keep).any():
        pr[~keep] = rank.predict(ms, R.loc[~keep, cols].values.astype(np.float32))
    r1 = R.qid.isin(feats["R1"].qid.unique()).values
    out = {"R1": pr[r1], "R2": pr[~r1]}
    for s in ["A", "B_A", "V"]:
        out[s] = rank.predict(ms, feats[s][cols].values.astype(np.float32))
    imp = pd.Series(np.mean([m.feature_importance("gain") for m in ms], 0), index=cols)
    print(f"ranker {name} ({Rk.qid.nunique()} training queries) done ({time.time() - T0:.0f}s); top gain:",
          (imp / imp.sum()).sort_values(ascending=False).head(8).round(3).to_dict(), flush=True)
    return out, imp


def truth_rank(s, rk, sc):
    """Truth rank per query of set s under score variant sc (as k35)."""
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
pd.DataFrame(IMP).to_csv(f"{OUT}/importance8.csv")
POOL = {f"{name}|{SCORE_NAME(sc)}": {s: truth_rank(s, RK[name][s], sc) for s in SETS} for name in RK for sc in SCORES}
print(f"{len(POOL)} pool models ranked ({time.time() - T0:.0f}s)", flush=True)

# %% [markdown]
# ## 2. 生成側（S3、門 0.7、生成のランカー rkx。k35 と同じ）

# %%
GSETS = ["A", "B_A", "R2", "V"]
GEN = {s: load(f"gen4_{s}_s*.parquet", ["qid", "key"]).sort_values(["qid"], kind="mergesort").reset_index(drop=True)
       for s in GSETS}
if MAX_QUERIES:
    GEN = {s: g[g.qid.isin(qids_of[s])].reset_index(drop=True) for s, g in GEN.items()}


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


GRKX = gen_ranker(list(gen.GEN_FEATURES) + list(gen.GEN_EXTRA), "gen_ranker8x")
gate_val = infos.set_index("qid").lib_max


def gen_lists(s, score):
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
BNQ = np.concatenate([qids_of[s][BN[s]] for s in ["R2", "V"]])
GLX = {s: gen_lists(s, GRKX[s]) for s in GSETS}
# 報告用の部分集合（B_N の中の位置、または R2 ∪ V の中の位置）
RV = np.concatenate([qids_of["R2"], qids_of["V"]])
lk_bn = leak.leak.reindex(BNQ).values.astype(bool)
pce = leak.n_pos_ce.reindex(RV).values > 0
SUBS = {"B_N_clean72": np.isin(BNQ, list(clean72)), "B_N_leaked": lk_bn,
        "RV_clean_pos": (~leak.leak.reindex(RV).values.astype(bool)) & pce}
print("report subsets:", {k: int(v.sum()) for k, v in SUBS.items()}, "| B_N", len(BNQ))


def finish(rk):
    rr = {s: harness.rr_of(rk[s]) for s in SETS}
    rr["B_N"] = np.concatenate([rr[s][BN[s]] for s in ["R2", "V"]])
    rv = np.concatenate([rr["R2"], rr["V"]])
    rr["B_N_clean72"] = rr["B_N"][SUBS["B_N_clean72"]]
    rr["B_N_leaked"] = rr["B_N"][SUBS["B_N_leaked"]]
    rr["RV_clean_pos"] = rv[SUBS["RV_clean_pos"]]
    return rr


def with_gen(prank):
    rk, gpos = dict(prank), {}
    for s in GSETS:
        n_b, n_g, k, _ = GLX[s]
        gate = gate_val.reindex(qids_of[s]).fillna(0).values < GATE
        rk[s] = shift(prank[s], np.where(gate, np.minimum(n_b, len(SLOTS)), 0))
        gpos[s] = g_pos(k, np.where(gate, np.minimum(n_g, len(SLOTS)), 0))
    rr = finish(rk)
    rr["G_N"] = np.concatenate([harness.rr_of(gpos[s])[BN[s]] for s in ["R2", "V"]])
    return rr


def no_gen(prank):
    rr = finish(prank)
    rr["G_N"] = np.zeros(len(rr["B_N"]))
    return rr


RR = {}
for m, pr in POOL.items():
    RR[f"{m}+nogen"] = no_gen(pr)
    RR[f"{m}+gen:S3:0.7:rkx"] = with_gen(pr)
assert REF in RR

# %% [markdown]
# ## 3. 参照 v12 と対で比べ、決めたとおりに選ぶ

# %%
ALL = ["A", "B_A", "B_N", "G_N", "V", "B_N_clean72", "B_N_leaked", "RV_clean_pos"]
IDX = {s: harness.boot_index(len(RR[REF][s]), N_BOOT, seed=i) for i, s in enumerate(ALL)}
REPS_REF = {s: harness.boot_reps(RR[REF][s], IDX[s]) for s in ALL}


def vs_ref(rr):
    out = {}
    for s in ALL:
        reps = harness.boot_reps(rr[s], IDX[s])
        c = harness.ci(rr[s].mean(), reps)
        d = harness.ci(rr[s].mean() - RR[REF][s].mean(), reps - REPS_REF[s])
        out[s] = dict(mrr=c["est"], lo=c["lo"], hi=c["hi"], d=d["est"], d_lo=d["lo"], d_hi=d["hi"])
    return out


summ = pd.DataFrame([dict(model=m, set=s, n=len(rr[s]), **v) for m, rr in RR.items() for s, v in vs_ref(rr).items()])
summ.to_csv(f"{OUT}/summary8.csv", index=False)
v = summ.set_index(["model", "set"])
rule7 = pd.read_csv(paths.find("rule7.csv")).set_index("model")
ref_here = {s: float(v.loc[(REF, s), "mrr"]) for s in ["A", "B_A", "B_N"]}
ref_k35 = {s: float(rule7.loc[REF, s]) for s in ["A", "B_A", "B_N"]}
print("reference v12 here:", {k: round(x, 4) for k, x in ref_here.items()}, "| k35:", {k: round(x, 4) for k, x in ref_k35.items()})
if not MAX_QUERIES:
    assert all(abs(ref_here[s] - ref_k35[s]) <= REF_TOL for s in ref_here), "v12 is not reproduced"
n_bn_r2 = int(BN["R2"].sum())
rows = []
for i, (m, rr) in enumerate(RR.items()):
    tag, _, rest = m.partition("|")
    sc, _, g = rest.partition("+")
    base_tag = tag.split(".")[0]
    kinds = [k for k in ext.KINDS if base_tag.endswith(f"_{k}") or (k in ("gl", "ice") and base_tag.endswith("_gl_ice"))]
    simple = len(kinds) + int(".clean" in tag) + int(g != "nogen") + int(not sc.startswith("hybi"))
    r = dict(model=m, cols=tag, score=sc, gen=g, ext=",".join(kinds), simple=simple, grid=i,
             report_only=(not kinds and ".clean" in tag))
    for s in ALL:
        r.update({s: v.loc[(m, s), "mrr"], f"d{s}": v.loc[(m, s), "d"], f"d{s}_lo": v.loc[(m, s), "d_lo"]})
    r["dBN_R2"] = rr["B_N"][:n_bn_r2].mean() - RR[REF]["B_N"][:n_bn_r2].mean()
    r["dBN_V"] = rr["B_N"][n_bn_r2:].mean() - RR[REF]["B_N"][n_bn_r2:].mean()
    rows.append(r)
rule = pd.DataFrame(rows)
rule["pred_lb_delta"] = np.where(rule.ext != "", S2 * rule.dB_N, S1 * rule.dA + S2 * rule.dB_N)
rule["eligible"] = (rule.ext == "") & (rule.model != REF) & ~rule.report_only & (rule.dA_lo > -NI_MARGIN) & \
    (rule.dB_N_lo > 0) & (rule.pred_lb_delta > 0)
rule = rule.sort_values(["pred_lb_delta", "simple", "grid"], ascending=[False, True, True], kind="mergesort")
xs = rule[rule.ext != ""]
rule["ext_submit"] = rule.model == (xs.iloc[0].model if len(xs) else None)
rule.to_csv(f"{OUT}/rule8.csv", index=False)
show = ["model", "A", "dA", "dA_lo", "B_N", "dB_N", "dB_N_lo", "dB_N_clean72", "dB_N_clean72_lo", "dB_N_leaked",
        "dRV_clean_pos", "dRV_clean_pos_lo", "dG_N", "pred_lb_delta", "eligible", "ext_submit"]
with pd.option_context("display.width", 300, "display.max_rows", 200):
    print(rule[show].round(4).to_string(index=False))
el = rule[rule.eligible]
best = el.iloc[0].to_dict() if len(el) else None
sub_ = xs.iloc[0].to_dict() if len(xs) else None
# 1 クエリごとの RR（参照と全案。漏れなしの SE をあとで測れるように）
rrq = []
for m, rr in RR.items():
    for s in SETS:
        rrq.append(pd.DataFrame({"model": m, "set": s, "qid": qids_of[s], "rr": rr[s]}))
pd.concat(rrq, ignore_index=True).to_parquet(f"{OUT}/rr8.parquet", index=False)
qlib = infos.set_index("qid").qlib.reindex(BNQ).values
models_show = [REF] + ([sub_["model"]] if sub_ else [])
pd.DataFrame([dict(model=m, lib=lb, n=int((qlib == lb).sum()), clean=int(((qlib == lb) & SUBS["B_N_clean72"]).sum()),
                   mrr=RR[m]["B_N"][qlib == lb].mean(), ref=RR[REF]["B_N"][qlib == lb].mean())
              for m in models_show for lb in np.unique(qlib)]).to_csv(f"{OUT}/bylib8.csv", index=False)

# %% [markdown]
# ## 4. 結果

# %%
chosen = dict(model=best["model"], keep_ref=False, pred_lb_delta=float(best["pred_lb_delta"])) if best else \
    dict(model=REF, keep_ref=True, pred_lb_delta=0.0)
ext_submit = dict(model=sub_["model"], pred_lb_delta=float(sub_["pred_lb_delta"]), dB_N=float(sub_["dB_N"]),
                  dB_N_lo=float(sub_["dB_N_lo"]), dB_N_clean72=float(sub_["dB_N_clean72"]),
                  dB_N_clean72_lo=float(sub_["dB_N_clean72_lo"]), dA=float(sub_["dA"])) if sub_ else None
res = dict(features=tags[0], ext=EXT_INFO, chosen=chosen, ext_submit=ext_submit, n_eligible=int(rule.eligible.sum()),
           reference=REF, reference_recomputed=ref_here, k35_reference=ref_k35, rankers={n: FCOLS[n.split(".")[0]] for n in RK},
           subsets={k: int(x.sum()) for k, x in SUBS.items()},
           rule=dict(ni_margin=NI_MARGIN, s1=S1, s2=S2, ext="no local gate; submit = max 0.50*dB_N (full B_N)"))
json.dump(res, open(f"{OUT}/chosen8.json", "w"), indent=1, default=float)
print("\nCHOSEN (no ext):", chosen, "\nEXT SUBMIT:", ext_submit)
run = track.init("eval", group=EXP, config=dict(features=tags[0], ref=REF, ext=list(KINDS)), online=False)
run.summary(dict(chosen=chosen["model"], keep_ref=chosen["keep_ref"], ext_submit=ext_submit["model"] if ext_submit else "",
                 n_eligible=int(rule.eligible.sum())))
run.table("rule8", rule)
run.finish()
print(f"total {time.time() - T0:.0f}s")

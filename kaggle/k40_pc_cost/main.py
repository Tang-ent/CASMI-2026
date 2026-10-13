# %% [markdown]
# # casmi26-k40-pc-cost
# M7 P2（2026-10-14）: PubChem S' の側の一覧の「枠の費用」を手元で測る（CPU だけ）。
# - 参照 = k35 の選んだ案 both_fprich_g2_grp|hybi:0.7:0.8:5+gen:S3:0.7:rkx を、k35 と同じ手順（R1 ∪ R2、3 つの種、
#   R1 / R2 は正解の構造ごとの 5 分割 OOF）で計算し直したもの（k35 の数字との差は表示だけ）
# - 案: 参照 + pc:P5:0.7（5, 10, 14, 20, 25 位）、参照 + pc:P2:0.7（10, 20 位）。lib_grp_max < 0.7 の分子で、窓
#   （query_info4 の target と ppm）にある S' の数 n の分だけ、枠の先頭 min(n, 枠の数) が埋まる（casmi26.slots と同じ定義）
# - S' はプールの鍵を除いてあるので、評価の組の正解は S' に無い。測れるのは費用（プールと生成の項目が後ろにずれる分）
#   だけで、得は測れない（LB で測る）。生成物と同じ鍵の S' は提出では飛ばすが、ここでは数えない（多めの費用になる側）
# 提出の決まり（計画 2026-10-14 の提案、ユーザーの確認の後に固定）: P5 → P2 の順に、参照との差で A の下限 > −0.02 かつ
# B_N の下限 > −0.02 を満たす最初の案を LB で試す（pc_choice.json）。どちらも満たさなければ出さない。
# 出力: pc7.csv（各案の A、B_A、B_N、G_N、C、V と参照との差）、pc_choice.json、pc_counts.csv。
# 入力: k30-s0〜s3、k26（pool_formula.npz）、k05（プール、queries.csv）、casmi26-pubchem-part（pc_mass.npy）。

# %%
import glob
import json
import os
import time

import numpy as np
import pandas as pd

from casmi26 import gen, harness, paths, pool, rank, search, slots, subform

T0 = time.time()
OUT = "/kaggle/working"
SETS = ["A", "B_A", "C", "B_C", "R1", "R2", "V"]
SEEDS = (0, 1, 2)
N_BOOT = 2000
NI_MARGIN, BN_MARGIN = 0.02, 0.02
TAG, SCORE = "both_fprich_g2_grp", ("hybi", (0.7, 0.8, 5))
GEN_SLOTS, GATE = slots.SLOT_SETS["S3"], 0.7
PCS = [("P5", 0.7), ("P2", 0.7)]  # 試す順（枠の多い方から）
K35_REF = {"A": 0.93895, "B_A": 0.56106, "B_N": 0.58554, "G_N": 0.05287}  # k35 の選んだ案（表示だけ）
EXPECT_VARIANT = "v2b"
REMOVAL = ("de_", "dehydroxy", "H2_remove", "H2_alcohol")

PD = pool.Pool.find_dir()
truth = pd.read_csv(f"{PD}/queries.csv", usecols=["qid", "pid"]).set_index("qid").pid
in_co = pd.read_parquet(f"{PD}/pool.parquet", columns=["in_coconut"]).in_coconut.values
FT = subform.FormulaTable(subform.FormulaTable.find_dir())
tags = [json.load(open(p)) for p in sorted(glob.glob("/kaggle/input/**/features_info4.json", recursive=True))]
assert tags and all(t == tags[0] for t in tags) and tags[0]["variant"] == EXPECT_VARIANT, "k30 shards"
pc_mass = np.sort(np.load(paths.find("pc_mass.npy")))
print("features:", tags[0], f"| S' masses {len(pc_mass):,}")


def load(pattern, dedupe):
    ps = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    d = pd.concat([pd.read_parquet(p) for p in ps], ignore_index=True) if ps else pd.DataFrame()
    return d.drop_duplicates(dedupe) if len(d) else d


infos = pd.concat([pd.read_csv(p) for p in sorted(glob.glob("/kaggle/input/**/query_info4_s*.csv", recursive=True))],
                  ignore_index=True).drop_duplicates("qid")
feats = {s: load(f"feats4_{s}_s*.parquet", ["qid", "pid"]).sort_values(["qid", "pid"], kind="mergesort")
         .reset_index(drop=True) for s in SETS}
qids_of = {s: np.sort(infos[infos.set == s].qid.values) for s in SETS}
for s in SETS:
    G = search.formula_group_features({c: feats[s][c].values for c in search.GRP_BASE}, feats[s].qid.values,
                                      FT.fid[feats[s].pid.values])
    feats[s] = pd.concat([feats[s], pd.DataFrame(G, columns=search.GRP_FEATURES)], axis=1)
FL = list(search.FEATURES)
from casmi26.frag import FRAG_FEATURES  # noqa: E402

COLS = FL + FRAG_FEATURES + search.FWD_FEATURES + search.FPRICH_FEATURES + search.G2_FEATURES + search.GRP_FEATURES
X = {s: feats[s][FL].values for s in SETS}
LAB, QID = {s: feats[s].label.values for s in SETS}, {s: feats[s].qid.values for s in SETS}
GMAX = {s: X[s][:, search.COL["lib_grp_max"]] for s in SETS}
info = infos.set_index("qid")
# S' の窓の数（クエリごと）
tgt, ppm = info.target.values, info.ppm.values
ok = np.isfinite(tgt) & np.isfinite(ppm)
tol = np.where(ok, tgt * np.nan_to_num(ppm) * 1e-6, 0)
lo = np.searchsorted(pc_mass, np.nan_to_num(tgt) - tol)
hi = np.searchsorted(pc_mass, np.nan_to_num(tgt) + tol)
n_pc = pd.Series(np.where(ok, hi - lo, 0), index=info.index)
pd.DataFrame({"qid": info.index, "set": info.set.values, "lib_max": info.lib_max.values, "n_pc": n_pc.values}) \
    .to_csv(f"{OUT}/pc_counts.csv", index=False)
print("S' per window (median by set):", pd.DataFrame({"set": info.set, "n": n_pc}).groupby("set").n.median().to_dict(),
      f"({time.time() - T0:.0f}s)")

# %% [markdown]
# ## 1. 参照のランカー（k35 と同じ手順）と並び

# %%
R = pd.concat([feats["R1"], feats["R2"]], ignore_index=True)
oof = rank.cv_oof(R, COLS, n_folds=5, seed=0, groups=truth, model_seeds=SEEDS)
ms = [rank.fit(R, COLS, seed=sd) for sd in SEEDS]
r1 = R.qid.isin(feats["R1"].qid.unique()).values
RK = {"R1": oof[r1], "R2": oof[~r1]}
for s in ["A", "B_A", "C", "B_C", "V"]:
    RK[s] = rank.predict(ms, feats[s][COLS].values.astype(np.float32))
print(f"ranker {TAG} done ({time.time() - T0:.0f}s)", flush=True)


def truth_rank(s, rk, sc):
    """k35 の truth_rank と同じ（hybi は帯の分子を search.variant_order で並べる）。"""
    kind, a = sc
    f = lambda n: X[s][:, search.COL[n]]
    tie = 1e-3 * f("ap4") - 1e-6 * f("mass_ppm")
    score = search.hybrid_score(X[s], a[1], search.zscore_by(rk, QID[s]) + a[2] * (f("ap4") + 0.2 * f("lib_max")) + tie)
    r = harness.ranks_of_truth(QID[s], LAB[s], score, feats[s].pid.values).reindex(qids_of[s]).fillna(np.inf)
    band = (GMAX[s] >= a[0]) & (GMAX[s] < a[1])
    rows = pd.Series(np.flatnonzero(band)).groupby(QID[s][band]).apply(np.asarray)
    for q, ix in rows.items():
        o = search.variant_order(X[s][ix], rk[ix], sc)
        hit = np.flatnonzero(LAB[s][ix][o] > 0)
        r[q] = hit[0] + 1 if len(hit) else np.inf
    return r.values.astype(np.float64)


PR = {s: truth_rank(s, RK[s], SCORE) for s in SETS}

# %% [markdown]
# ## 2. 生成（rkx）と S' の枠

# %%
GSETS = ["A", "B_A", "R2", "V"]
GEN = {s: load(f"gen4_{s}_s*.parquet", ["qid", "key"]).sort_values(["qid"], kind="mergesort").reset_index(drop=True)
       for s in GSETS}
GF = list(gen.GEN_FEATURES) + list(gen.GEN_EXTRA)
R2g = GEN["R2"]
goof = rank.cv_oof(R2g, GF, n_folds=5, seed=0, groups=truth, model_seeds=SEEDS)
gms = [rank.fit(R2g, GF, seed=sd) for sd in SEEDS]
GRK = {"R2": goof, **{s: rank.predict(gms, GEN[s][GF].values.astype(np.float32)) if len(GEN[s]) else np.zeros(0)
                      for s in ["A", "B_A", "V"]}}


def gen_lists(s):
    g, q = GEN[s], qids_of[s]
    r = harness.ranks_of_truth(g.qid.values, g.label.values, GRK[s])
    n_all, n_lab = g.groupby("qid").size(), g.groupby("qid").label.sum()
    add = g[g.label > 0].set_index("qid").seq.map(lambda x: not any(st.startswith(REMOVAL) for st in x.split("+")))
    return ((n_all - n_lab).reindex(q).fillna(0).values, n_all.reindex(q).fillna(0).values,
            r.reindex(q).fillna(np.inf).values, add.reindex(q).fillna(False).values.astype(bool))


GL = {s: gen_lists(s) for s in GSETS}
tq = {s: truth.reindex(qids_of[s]).values for s in SETS}
BN = {s: in_co[tq[s]] for s in ["R2", "V"]}


def final(pc=None):
    """rr per set (and B_N, G_A, G_N) for the reference with an optional S' side list pc = (slots name, gate)."""
    rr, gpos = {}, {}
    for s in SETS:
        gm = info.lib_max.reindex(qids_of[s]).fillna(0).values
        n_b = n_g = k = None
        if s in GSETS:
            n_b, n_g, k, _ = GL[s]
        occs = []
        for i in range(len(qids_of[s])):
            ins = []
            if s in GSETS and gm[i] < GATE:
                ins.append((GEN_SLOTS, n_b[i]))
            if pc and gm[i] < pc[1]:
                ins.append((slots.SLOT_SETS[pc[0]], n_pc.get(qids_of[s][i], 0)))
            occs.append(slots.occupied(ins))
        pos = slots.shifted_ranks(PR[s], occs)
        if s in GSETS:
            gpos[s] = np.array([slots.slot_rank(k[i], GEN_SLOTS, n_g[i] if gm[i] < GATE else 0)
                                for i in range(len(k))])
        rr[s] = harness.rr_of(pos)  # k35 と同じく、組の MRR はプール側の正解だけ、G_A / G_N は生成の一覧の正解だけで見る
    rr["B_N"] = np.concatenate([rr[s][BN[s]] for s in ["R2", "V"]])
    rr["G_A"] = harness.rr_of(gpos["B_A"])
    rr["G_N"] = np.concatenate([harness.rr_of(gpos[s])[BN[s]] for s in ["R2", "V"]])
    return rr


RR = {"ref": final(), **{f"pc:{n}:{g}": final((n, g)) for n, g in PCS}}
print("reference recomputed:", {s: round(float(RR["ref"][s].mean()), 4) for s in K35_REF}, "| k35:", K35_REF,
      f"({time.time() - T0:.0f}s)")

# %% [markdown]
# ## 3. 参照との差と、LB で試す案

# %%
ALL = SETS + ["B_N", "G_A", "G_N"]
IDX = {s: harness.boot_index(len(RR["ref"][s]), N_BOOT, seed=i) for i, s in enumerate(ALL)}
IDX.update(G_A=IDX["B_A"], G_N=IDX["B_N"])
rows = []
for m, rr in RR.items():
    r = dict(model=m)
    for s in ["A", "B_A", "B_N", "G_N", "C", "V", "R1", "R2"]:
        d = harness.ci(rr[s].mean() - RR["ref"][s].mean(),
                       harness.boot_reps(rr[s], IDX[s]) - harness.boot_reps(RR["ref"][s], IDX[s]))
        r.update({s: rr[s].mean(), f"d{s}": d["est"], f"d{s}_lo": d["lo"]})
    r["pred_lb_delta"] = 0.163 * r["dA"] + 0.21 * r["dB_N"] + 0.30 * r["dG_N"]
    r["ok"] = m != "ref" and r["dA_lo"] > -NI_MARGIN and r["dB_N_lo"] > -BN_MARGIN
    rows.append(r)
t = pd.DataFrame(rows)
t.to_csv(f"{OUT}/pc7.csv", index=False)
with pd.option_context("display.width", 250):
    print(t.round(4).to_string(index=False))
gated = {s: float((info.lib_max.reindex(qids_of[s]).fillna(0) < GATE).mean()) for s in ["A", "B_A", "R2", "V"]}
pick = next((r["model"] for r in rows if r["ok"]), None)
json.dump(dict(choice=pick, rule=dict(order=[f"pc:{n}:{g}" for n, g in PCS], ni_margin=NI_MARGIN, bn_margin=BN_MARGIN),
               reference=f"{TAG}|hybi:0.7:0.8:5+gen:S3:0.7:rkx", gated_share=gated,
               table=t.to_dict("records")), open(f"{OUT}/pc_choice.json", "w"), indent=1, default=float)
print(f"gated share {gated}; LB test: {pick or 'none (no variant meets the cost bounds)'}")
print(f"total {time.time() - T0:.0f}s")

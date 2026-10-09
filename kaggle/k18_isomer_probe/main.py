# %% [markdown]
# # casmi26-k18-isomer-probe
# M5 の P0-1（2026-10-09）: 正解より上に来ている候補が、正解と同じ分子式（異性体）か違う分子式かを分け、
# 断片の照合（異性体の区別）と部分分子式の照合（分子式の区別）で届く上限を測る。CPU、インターネットなし。
# - 基準は k07 v9（FPNet v2b）の特徴量での hybm:0.7:2（k16 v2 で A 0.926、B_N 0.518）。提出中の M2.2
#   （古い特徴量、B_N 0.536）ではない（k07 v7 の出力はもう見えない）。上限は M2.2 からの差として読まない
# - 分子式のまとまり = 修正後の質量 P.mass（分子式から計算した値）を 1e-6 Da で丸めたもの（電荷つきの書き方の違いを避ける）
# - 異性体の上限 r_iso = 1 + n_diff（正解が自分の異性体の中で 1 位なら）、分子式の上限 r_form = 1 + n_same
#   （分子式が分かっていれば）。n_same / n_diff = 正解より上の候補のうち同じ / 違う分子式の数
# - R1 / R2 の表は上位 300 に切り詰めてある（正解は残る）。正解より上の候補はその中で数える（今の評価と同じ条件）。
#   窓の中の正解の異性体の総数 n_iso だけはプールから直接数える
# 見込み（先に書いておく）: 350 Da で ±10 ppm は ±3.5 mDa。C・H・N・O だけの天然物では別の分子式はほとんど入らないので、
# 異性体の上限は 1 に近いはず。だからこの段は大きさを測るだけ。
# 決め方（固定）: P1-1a（結合を切った断片）は既定で作る。B_N の「分子式の上限 − 今」の推定値 ≥ +0.05 なら、
# 部分分子式の特徴量（P1-1b）も作る。
# 外したクエリについて、正解と「正解より上で一番上の異性体」の Tanimoto も出す（基準は付けず、P1-1a の前に見せる）。
# 入力: k07-features2-s0/s1/s2（v9）、k05（プール、queries.csv）、k05b（質量の修正）。

# %%
import glob
import json
import time

import numpy as np
import pandas as pd

from casmi26 import harness, pool, rank, search, track

T0 = time.time()
OUT = "/kaggle/working"
EXP = "M5"
SETS = ["A", "B_A", "C", "B_C", "R1", "R2", "V"]
SHOW = ["A", "B_A", "B_N", "C", "R2", "V"]
SEEDS = (0, 1, 2)
N_BOOT = 2000
PPM = 10.0  # n_iso を数える窓（正解の質量の ±PPM）
EXPECT_VARIANT = "v2b"  # k07 v9 の FPNet
REF_K16 = {"A": 0.926, "B_N": 0.518}  # k16 v2 の hybm:0.7:2（v2b の特徴量）
REF_TOL = 0.005  # None = 照合しない（手元の動作確認）
FORMULA_GAP_MIN = 0.05
MASS_DEC = 6

PD = pool.Pool.find_dir()
P = pool.Pool(PD, pubchem=False)
truth = pd.read_csv(f"{PD}/queries.csv", usecols=["qid", "pid"]).set_index("qid").pid
in_co = P.in_coconut
fgrp = np.round(P.mass, MASS_DEC)  # 分子式の代わり
info_path = sorted(glob.glob("/kaggle/input/**/fpnet_info.json", recursive=True))
n_shards = len(glob.glob("/kaggle/input/**/query_info_L0_s*.csv", recursive=True))
tags = [json.load(open(p)) for p in info_path]
# 取り消されたカーネルは最後に成功した版の出力がつながるので、全分割の FPNet が一致するか確かめる
assert len(tags) == n_shards and all({k: t[k] for k in ("variant", "epoch")} == {k: tags[0][k] for k in ("variant", "epoch")}
                                     for t in tags), f"k07 shards disagree on the FPNet: {tags}"
assert tags[0]["variant"] == EXPECT_VARIANT, f"k07 features are from {tags[0]['variant']}, expected {EXPECT_VARIANT}"
print("features:", tags[0])
infos = pd.concat([pd.read_csv(p) for p in sorted(glob.glob("/kaggle/input/**/query_info_L0_s*.csv", recursive=True))],
                  ignore_index=True).drop_duplicates("qid").set_index("qid")
feats = {}
for s in SETS:
    ps = sorted(glob.glob(f"/kaggle/input/**/feats_{s}_L0_s*.parquet", recursive=True))
    feats[s] = pd.concat([pd.read_parquet(p) for p in ps], ignore_index=True).drop_duplicates(["qid", "pid"])
    feats[s] = feats[s].sort_values(["qid", "pid"], kind="mergesort").reset_index(drop=True)
print("rows", {s: len(f) for s, f in feats.items()}, f"({time.time() - T0:.0f}s)")
qids_of = {s: np.sort(infos[infos.set == s].index.values) for s in SETS}
FL = list(search.FEATURES)
X = {s: feats[s][FL].values for s in SETS}

# %% [markdown]
# ## ランカー lgb_R と hybm:0.7:2（k16 と同じ手順）

# %%
R = pd.concat([feats["R1"], feats["R2"]], ignore_index=True)
oof = rank.cv_oof(R, FL, n_folds=5, groups=truth)
models = [rank.fit(R, FL, seed=sd) for sd in SEEDS]
in_r1 = R.qid.isin(feats["R1"].qid.unique()).values
RK = {"R1": oof[in_r1], "R2": oof[~in_r1]}
for s in ["A", "B_A", "C", "B_C", "V"]:
    RK[s] = rank.predict(models, X[s].astype(np.float32))
del R, oof
print(f"lgb_R done ({time.time() - T0:.0f}s)")
scores = {"hybm:0.7:2": {s: search.blend_score(X[s], 0.7, RK[s], 2, qid=feats[s].qid.values) for s in SETS},
          "fz": {s: search.rule_score(X[s], "fz") for s in SETS},
          "lgb_R": RK}

# %% [markdown]
# ## 正解より上の候補を、同じ分子式と違う分子式に分ける

# %%
def split_ranks(f, sc):
    """Per qid: rank r of the truth (pid ties, as harness.ranks_of_truth), n_same / n_diff above it, and the pid of
    the best-ranked same-formula candidate above it (-1 if none). Queries without the truth: r = inf."""
    d = pd.DataFrame({"qid": f.qid.values, "pid": f.pid.values, "label": f.label.values, "score": sc,
                      "g": fgrp[f.pid.values]})
    d = d.sort_values(["qid", "score", "pid"], ascending=[True, False, True], kind="mergesort")
    d["pos"] = d.groupby("qid").cumcount()
    t = d[d.label > 0].groupby("qid").agg(pos_t=("pos", "min"), g_t=("g", "first"))
    d = d.join(t, on="qid", how="inner").sort_values(["qid", "pos"], kind="mergesort")
    above = d.pos.values < d.pos_t.values
    same = d.g.values == d.g_t.values
    d["above_same"], d["above_diff"] = above & same, above & ~same
    out = d.groupby("qid").agg(n_same=("above_same", "sum"), n_diff=("above_diff", "sum"))
    top_iso = d[d.above_same].groupby("qid").pid.first()  # 並べた順の最初 = 一番上の異性体
    out["r"] = t.pos_t + 1
    out["top_iso"] = top_iso.reindex(out.index).fillna(-1).astype(np.int64)
    return out


def tanimoto(a, b):
    fa, fb = np.unpackbits(P.fp[a], axis=1), np.unpackbits(P.fp[b], axis=1)
    inter = (fa & fb).sum(1)
    return inter / np.maximum(fa.sum(1) + fb.sum(1) - inter, 1)


rows, rank_rows = [], []
RR = {}
for name, sc in scores.items():
    per = {}
    for s in SETS:
        o = split_ranks(feats[s], sc[s]).reindex(pd.Index(qids_of[s], name="qid"))
        o["r"] = o.r.fillna(np.inf)
        o["r_iso"] = np.where(np.isfinite(o.r), 1 + o.n_diff.fillna(0), np.inf)
        o["r_form"] = np.where(np.isfinite(o.r), 1 + o.n_same.fillna(0), np.inf)
        o["set"], o["model"] = s, name
        per[s] = o
        rank_rows.append(o.reset_index())
    tq = {s: truth.reindex(qids_of[s]).values for s in SETS}
    bn = [(s, in_co[tq[s]]) for s in ["R2", "V"]]
    per["B_N"] = pd.concat([per[s][m] for s, m in bn])
    for s in SHOW:
        o = per[s]
        rr = {k: harness.rr_of(o[c].values) for k, c in [("base", "r"), ("iso", "r_iso"), ("form", "r_form")]}
        RR[(name, s)] = rr
        rec = dict(model=name, set=s, n=len(o), base=rr["base"].mean(), iso=rr["iso"].mean(), form=rr["form"].mean())
        for k in ["iso", "form"]:
            c = harness.paired_ci(rr[k], rr["base"], N_BOOT, seed=1)
            rec.update({f"{k}_gap": c["est"], f"{k}_lo": c["lo"], f"{k}_hi": c["hi"]})
        rows.append(rec)
probe = pd.DataFrame(rows)
probe.to_csv(f"{OUT}/isomer_probe.csv", index=False)
print("MRR@25: now / if the truth were first among its isomers (iso) / if the formula were known (form)")
print(probe.round(4).to_string(index=False))
v = probe[probe.model == "hybm:0.7:2"].set_index("set")
print("check vs k16 v2 (hybm:0.7:2, v2b features):", {s: round(float(v.loc[s, "base"]), 4) for s in REF_K16},
      "expected", REF_K16)
ref_d = {s: float(v.loc[s, "base"] - x) for s, x in REF_K16.items()}
ref_ok = REF_TOL is None or all(abs(d) <= REF_TOL for d in ref_d.values())
if not ref_ok:  # 止めずに記録する（判定 I はこの照合に依らない）
    print(f"WARNING: hybm:0.7:2 differs from k16 v2 by {ref_d} (tolerance {REF_TOL})")

# %% [markdown]
# ## 外したクエリの中身（hybm:0.7:2）、ライブラリ別

# %%
ranks = pd.concat(rank_rows, ignore_index=True)
base = ranks[ranks.model == "hybm:0.7:2"].set_index(["set", "qid"])
n_iso_rows = []
for s in SETS:
    for q in qids_of[s]:
        p = int(truth[q])
        cand = P.window(P.mass[p], PPM)
        n_iso_rows.append((s, q, int((fgrp[cand] == fgrp[p]).sum()) - 1, len(cand)))
n_iso = pd.DataFrame(n_iso_rows, columns=["set", "qid", "n_iso", "n_win"]).set_index(["set", "qid"])
base = base.join(n_iso)
pid_of = truth.reindex(base.index.get_level_values("qid")).values
top_iso = base.top_iso.fillna(-1).values.astype(np.int64)
has_iso = top_iso >= 0
tan = np.full(len(base), np.nan)
tan[has_iso] = tanimoto(pid_of[has_iso].astype(np.int64), top_iso[has_iso])
base["tan_top_iso"] = tan
base.reset_index().to_parquet(f"{OUT}/isomer_ranks.parquet", index=False)


def miss_stats(o):
    m = o[(o.r > 1) & np.isfinite(o.r)]
    ns = m.n_same.fillna(0)
    t = m.tan_top_iso.dropna()
    return dict(n=len(o), n_miss=len(m), iso_win_med=float(o.n_iso.median()), win_med=float(o.n_win.median()),
                share_iso_window=float((o.n_iso / np.maximum(o.n_win - 1, 1)).median()),
                miss_top_is_iso=float((m.n_diff.fillna(0) == 0).mean()) if len(m) else np.nan,
                above_iso_med=float(ns.median()) if len(m) else np.nan,
                above_0=float((ns == 0).mean()), above_1_2=float(ns.between(1, 2).mean()),
                above_3_10=float(ns.between(3, 10).mean()), above_11=float((ns >= 11).mean()),
                tan_med=float(t.median()) if len(t) else np.nan, tan_ge_08=float((t >= 0.8).mean()) if len(t) else np.nan)


bn_idx = []
for s in ["R2", "V"]:
    q = qids_of[s]
    bn_idx += [(s, x) for x in q[in_co[truth.reindex(q).values]]]
groups = {s: base.loc[s] for s in ["A", "B_A", "C", "R2", "V"]}
groups["B_N"] = base.loc[bn_idx]
miss = pd.DataFrame([dict(set=s, **miss_stats(o)) for s, o in groups.items()])
miss.to_csv(f"{OUT}/isomer_misses.csv", index=False)
print("misses of hybm:0.7:2 (share_iso_window = isomers / other window candidates; miss_top_is_iso = every candidate "
      "above the truth is an isomer; above_* = isomers above the truth; tan_* = Tanimoto(truth, best isomer above))")
print(miss.round(3).T.to_string())

bn = groups["B_N"].copy()
bn["qlib"] = infos.qlib.reindex(bn.index.get_level_values("qid")).values
lib_rows = []
for lb, o in bn.groupby("qlib"):
    rr = {k: harness.rr_of(o[c].values) for k, c in [("base", "r"), ("iso", "r_iso"), ("form", "r_form")]}
    lib_rows.append(dict(lib=lb, n=len(o), base=rr["base"].mean(), iso=rr["iso"].mean(), form=rr["form"].mean(),
                         **{f"miss_{k}": v for k, v in miss_stats(o).items() if k in ("tan_med", "tan_ge_08",
                                                                                     "above_iso_med")}))
by_lib = pd.DataFrame(lib_rows)
by_lib.to_csv(f"{OUT}/isomer_by_lib.csv", index=False)
print("B_N by query library (hybm:0.7:2):")
print(by_lib.round(4).to_string(index=False))

# %% [markdown]
# ## 判定 I（固定）

# %%
g = probe[(probe.model == "hybm:0.7:2") & (probe.set == "B_N")].iloc[0]
build_subformula = bool(g.form_gap >= FORMULA_GAP_MIN)
res = dict(features=tags[0], build_fragments=True, build_subformula=build_subformula, ref_ok=ref_ok, ref_d=ref_d,
           B_N=dict(base=g.base, iso=g.iso, form=g.form, iso_gap=g.iso_gap, form_gap=g.form_gap,
                    form_lo=g.form_lo, form_hi=g.form_hi),
           misses=miss.set_index("set").to_dict("index"))
json.dump(res, open(f"{OUT}/probe1.json", "w"), indent=1, default=float)
print(f"gate I: P1-1a (bond-cleavage fragments) by default; P1-1b (subformula) "
      f"{'YES' if build_subformula else 'NO'} — B_N formula ceiling - now = {g.form_gap:+.4f} "
      f"(threshold +{FORMULA_GAP_MIN})")
run = track.init("probe", group=EXP, config=dict(features=tags[0], ppm=PPM), online=False)
run.summary(dict(build_subformula=build_subformula, **{f"{r.model}_{r.set}_{k}": float(getattr(r, k))
                                                       for r in probe.itertuples() for k in ["base", "iso", "form"]}))
run.table("isomer_probe", probe)
run.table("isomer_misses", miss)
run.finish()
print(f"total {time.time() - T0:.0f}s")

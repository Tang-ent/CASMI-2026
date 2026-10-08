# %% [markdown]
# # casmi26-k14-fpnet-probe
# M4 の P0（2026-10-07）: 今の FPNet（k06、エポック 14）の弱点を測り、FPNet v2（k15_fpnet2）で何を変えるかを決める。
# 決め方はこのカーネルを書いた時点（結果を見る前）に固定し、決めた設定を v2_config.json に書く。
# 1. fz だけの窓内 MRR@25（正解の質量の ±10 ppm にあるプールの構造の中での順位。A / V / R2 / C を、正解が COCONUT に
#    ある（天然物）かどうかで分ける。B_N = R2 と V の天然物）。
#    - timsTOF の印: そのまま / 0 に固定 / 1 に固定
#    - 前処理: 強度の床（キャッシュの 0.1% の上に 0 / 0.5% / 1% / 2%）× 上位本数（32 / 64 / 128）。A / B_N / C だけ
#    - 同点の扱い: 正解に有利（fpnet.window_mrr と同じ）/ pid 順（harness.ranks_of_truth と同じ）/ 正解に不利
#    今のモデルのクエリごとの順位は fz_ranks_old.parquet に書く（k15 で新しいモデルと対で比べる）。
# 2. 指紋の同点による天井: 正解と同じ指紋の候補が窓の中に何件あるか（Morgan r=2 2048 = 今 / r=3 4096 /
#    Morgan r=2 + MACCS + 原子対）。fz が完璧でも届かない MRR の上限。
# 3. 由来による偏り: 正解以外の候補の fz_z（窓の中で標準化した f·z）を、COCONUT にしか無い構造（学習では負例にしか
#    ならない）とホールドアウトの train の構造（A / C / R / V。学習で正解にならない）で、正解との Tanimoto の区間ごとに比べる。
# 4. 学習データの内訳と、v2 の抽出（構造ごとに最大 4 件、天然物 3 倍）でのエポックの大きさ。
# v2 の設定の決め方（固定）:
#   - 常に: 構造ごとに最大 CAP 件、COCONUT にもある train の構造を NP_W 回分、総ステップ TOTAL_STEPS
#   - (a) A で tims=0 の MRR がそのままより上（対の差の 95% 下限 > 0）なら tims_fixed = 0
#   - 前処理: 基準以外で、A の対の差の下限 > 0 かつ B_N の下限 > −NI_MARGIN のもののうち A の推定値が最大のもの。無ければ基準
#   - (b) 3. の係数（COCONUT にしか無い − ホールドアウト）が COEF_MAX 以下で、95% 区間の上限 < 0 なら、
#         負例を train の構造からだけ引く変種 v2b も学習する（train_v2b = true）
#   - 2. は報告だけ（天然物の天井が 0.8 未満なら、指紋の変更を後の段階の候補にする）
# 入力: k01（キャッシュ）、k05（プール、queries.csv、holdout.csv）、k05b（質量の修正）、データセット casmi26-fpnet。
# インターネット: rdkit の pip install（2. の指紋）に使う。

# %%
import subprocess
import sys

r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rdkit"], capture_output=True, text=True)
print(r.stdout[-300:], r.stderr[-300:])
import rdkit

print("rdkit", rdkit.__version__)

# %%
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch

from casmi26 import cache, chem, fpnet, harness, pool

T0 = time.time()
OUT = "/kaggle/working"
PPM = 10.0
N_BOOT = 2000
SETS = ["A", "V", "R2", "C"]
BASE = (0.0, 128)  # 前処理の基準 = k06 と同じ（キャッシュの 0.1% の床の上に床なし、上位 128 本）
FLOORS = [0.0, 0.005, 0.01, 0.02]
TOPKS = [32, 64, 128]
TAN_BINS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.01]
CAP, NP_W, TOTAL_STEPS = 4, 3, 60_000
NI_MARGIN = 0.02
COEF_MAX = -0.10
W = os.cpu_count() or 4


def log(*a):
    print(*a, f"[{time.time() - T0:.0f}s]", flush=True)


C = cache.Cache(cache.Cache.find_dir())
PD = pool.Pool.find_dir()
P = pool.Pool(PD, pubchem=False)
pq_ = pd.read_parquet(f"{PD}/pool.parquet", columns=["in_train", "in_coconut"])
in_co, in_tr = pq_.in_coconut.values, pq_.in_train.values
hold = np.sort(pd.read_csv(f"{PD}/holdout.csv").pid.values)
MODEL = fpnet.load_checked()
torch.set_num_threads(W)
Q = pd.read_csv(f"{PD}/queries.csv")
Q = Q[Q.set.isin(SETS)].reset_index(drop=True)
Q["rows"] = Q.rows.map(lambda s: [int(x) for x in str(s).split(",")])
Q["np"] = in_co[Q.pid.values]
specs = [harness.query_specs(C, r) for r in Q.rows]
pids = Q.pid.values
G = {s: (Q.set == s).values for s in SETS}
G["B_N"] = Q.set.isin(["R2", "V"]).values & Q.np.values
for s in ["V", "R2"]:
    G[f"{s}_np"], G[f"{s}_non"] = G[s] & Q.np.values, G[s] & ~Q.np.values
PREP_MASK = G["A"] | G["B_N"] | G["C"]
log("queries", {k: int(v.sum()) for k, v in G.items()}, "| holdout", len(hold))

# %% [markdown]
# ## 1. fz だけの窓内 MRR（timsTOF の印と前処理）

# %%
def run_variant(name, floor, topk, tims, mask):
    sel = np.flatnonzero(mask)
    MODEL.prep = dict(floor=floor, max_peaks=topk, tims_fixed=tims)
    Z = MODEL.zs([specs[i] for i in sel])
    r = fpnet.window_ranks(Z, P, pids[sel], PPM)
    r.insert(0, "qi", sel)
    r.insert(0, "variant", name)
    log(f"variant {name}: {len(sel)} queries, MRR (pid ties) {harness.rr_of(r.r_pid).mean():.4f}")
    return r, Z


variants = [("base", *BASE, None, np.ones(len(Q), bool)), ("tims0", *BASE, 0.0, np.ones(len(Q), bool)),
            ("tims1", *BASE, 1.0, np.ones(len(Q), bool))]
variants += [(f"floor{f:g}_top{k}", f, k, None, PREP_MASK) for f in FLOORS for k in TOPKS if (f, k) != BASE]
ranks, Z_base = [], None
for v in variants:
    r, Z = run_variant(*v)
    ranks.append(r)
    if v[0] == "base":
        Z_base = Z
MODEL.prep = dict(floor=BASE[0], max_peaks=BASE[1], tims_fixed=None)
R = pd.concat(ranks, ignore_index=True)
R["qid"], R["set"], R["pid"] = Q.qid.values[R.qi], Q.set.values[R.qi], pids[R.qi]
for t in ["opt", "pid", "pess"]:
    R[f"rr_{t}"] = harness.rr_of(R[f"r_{t}"])
R.drop(columns="qi").to_parquet(f"{OUT}/probe_ranks.parquet", index=False)
base = R[R.variant == "base"].drop(columns=["variant"])
base[["qid", "set", "pid", "n_cand", "r_opt", "r_pid", "r_pess"]].to_parquet(f"{OUT}/fz_ranks_old.parquet", index=False)

rows = []
for v, g in R.groupby("variant", sort=False):
    rr = g.set_index("qi")
    for name, m in G.items():
        q = np.intersect1d(np.flatnonzero(m), rr.index.values)
        if len(q):
            rows.append(dict(variant=v, group=name, n=len(q), mrr_opt=rr.loc[q, "rr_opt"].mean(),
                             mrr_pid=rr.loc[q, "rr_pid"].mean(), mrr_pess=rr.loc[q, "rr_pess"].mean()))
summ = pd.DataFrame(rows)
summ.to_csv(f"{OUT}/probe_summary.csv", index=False)
print(summ[summ.variant.isin(["base", "tims0", "tims1"])].pivot(index="group", columns="variant", values="mrr_pid")
      .round(4).to_string())
print("\nbase, tie rules:")
print(summ[summ.variant == "base"].set_index("group")[["n", "mrr_opt", "mrr_pid", "mrr_pess"]].round(4).to_string())
print("\npreprocessing (MRR, pid ties):")
print(summ[summ.group.isin(["A", "B_N", "C"]) & ~summ.variant.isin(["tims0", "tims1"])]
      .pivot(index="variant", columns="group", values="mrr_pid").round(4).to_string())

b_rr = base.set_index("qi").rr_pid
paired = []
for v in R.variant.unique():
    if v == "base":
        continue
    x = R[R.variant == v].set_index("qi").rr_pid
    for name in ["A", "B_N", "C", "V", "R2"]:
        q = np.intersect1d(np.flatnonzero(G[name]), x.index.values)
        if len(q):
            paired.append(dict(variant=v, group=name, **harness.paired_ci(x.loc[q].values, b_rr.loc[q].values,
                                                                            N_BOOT, seed=1)))
paired = pd.DataFrame(paired)
paired.to_csv(f"{OUT}/probe_paired.csv", index=False)
print("\npaired difference vs base (pid ties):")
print(paired.assign(ci=paired.apply(lambda r: f"{r.est:+.4f} [{r.lo:+.4f}, {r.hi:+.4f}]", axis=1))
      .pivot(index="variant", columns="group", values="ci").to_string())

# %% [markdown]
# ## 2. 指紋の同点による天井

# %%
win = [P.window(P.mass[p], PPM) for p in pids]
u = np.unique(np.concatenate(win + [pids]))
log(f"alternative fingerprints for {len(u):,} structures")
alt = pool.map_many(chem.alt_fps, list(P.smiles[u]), W, chunksize=2000)
FPS = {"morgan2_2048": P.fp}
for j, name in enumerate(chem.ALT_FPS):
    a = np.zeros((len(P), len(alt[0][j])), np.uint8)
    a[u] = np.stack([x[j] for x in alt])
    FPS[name] = a
del alt
log("alternative fingerprints done")
ceil_rows = []
for name, F in FPS.items():
    r_opt, r_pid, r_pess, empty = [], [], [], 0
    for c, p in zip(win, pids):
        ft = F[p]
        if not ft.any():
            empty += 1
        tie = (F[c] == ft).all(1) & (c != p)
        r_opt.append(1)
        r_pid.append(1 + int((tie & (c < p)).sum()))
        r_pess.append(1 + int(tie.sum()))
    rr = {t: harness.rr_of(x) for t, x in [("opt", r_opt), ("pid", r_pid), ("pess", r_pess)]}
    for g, m in G.items():
        ceil_rows.append(dict(fp=name, group=g, n=int(m.sum()), ceil_pid=rr["pid"][m].mean(),
                              ceil_pess=rr["pess"][m].mean(), share_tied=(np.asarray(r_pess)[m] > 1).mean()))
    print(f"{name}: truth fingerprints empty {empty}")
ceil = pd.DataFrame(ceil_rows)
ceil.to_csv(f"{OUT}/fp_ceiling.csv", index=False)
print(ceil.pivot(index="group", columns="fp", values="ceil_pess").round(4).to_string())
print("share of queries with a tied candidate:")
print(ceil.pivot(index="group", columns="fp", values="share_tied").round(4).to_string())
del FPS

# %% [markdown]
# ## 3. 由来による偏り（COCONUT にしか無い構造 vs ホールドアウトの train の構造）

# %%
POP = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1)
nb = len(TAN_BINS) - 1
GROUPS = ["co_only", "ho", "ho_np", "ho_non"]
S = np.zeros((len(Q), len(GROUPS), nb))
N = np.zeros((len(Q), len(GROUPS), nb))
nbits = Z_base.shape[1]
for i, (c, p) in enumerate(zip(win, pids)):
    F = P.fp[c]
    s = np.unpackbits(F, axis=1)[:, :nbits].astype(np.float32) @ Z_base[i]
    if len(s) < 3 or s.std() == 0:
        continue
    z = (s - s.mean()) / s.std()
    ft = P.fp[p]
    tan = POP[F & ft].sum(1) / np.maximum(POP[F | ft].sum(1), 1)
    b = np.clip(np.digitize(tan, TAN_BINS) - 1, 0, nb - 1)
    ho = np.isin(c, hold) & (c != p)
    co = in_co[c] & ~in_tr[c]
    for k, m in enumerate([co, ho, ho & in_co[c], ho & ~in_co[c]]):
        np.add.at(S[i, k], b[m], z[m])
        np.add.at(N[i, k], b[m], 1)
idx = harness.boot_index(len(Q), N_BOOT, seed=2)
M = np.stack([np.bincount(row, minlength=len(Q)) for row in idx]).astype(np.float32)


def coef(Ssum, Nsum, a, b):
    """Weighted mean over Tanimoto bins of mean(z | group a) - mean(z | group b), weights = count of group b."""
    ok = (Nsum[..., a, :] > 0) & (Nsum[..., b, :] > 0)
    d = np.where(ok, Ssum[..., a, :] / np.maximum(Nsum[..., a, :], 1) - Ssum[..., b, :] / np.maximum(Nsum[..., b, :], 1), 0)
    w = np.where(ok, Nsum[..., b, :], 0)
    return (d * w).sum(-1) / np.maximum(w.sum(-1), 1)


prov = []
for qname, qm in [("all", np.ones(len(Q), bool)), ("np_truth", Q.np.values), ("non_np_truth", ~Q.np.values)]:
    Sq, Nq = S * qm[:, None, None], N * qm[:, None, None]
    St, Nt = Sq.sum(0), Nq.sum(0)
    Sb = np.tensordot(M, Sq, axes=1)
    Nb = np.tensordot(M, Nq, axes=1)
    for a, b in [(0, 1), (0, 3), (0, 2), (2, 3)]:
        reps = coef(Sb, Nb, a, b)
        prov.append(dict(queries=qname, contrast=f"{GROUPS[a]} - {GROUPS[b]}",
                         **harness.ci(coef(St, Nt, a, b), reps), n_a=int(Nt[a].sum()), n_b=int(Nt[b].sum())))
    for k, g in enumerate(GROUPS):
        print(qname, g, "count by Tanimoto bin:", Nt[k].astype(int).tolist(),
              "| mean fz_z:", np.round(St[k] / np.maximum(Nt[k], 1), 3).tolist())
prov = pd.DataFrame(prov)
prov.to_csv(f"{OUT}/provenance.csv", index=False)
print(prov.round(4).to_string(index=False))

# %% [markdown]
# ## 4. 学習データの内訳と v2 の抽出

# %%
pid_all = P.sid2pid[C.sid]
ok = C.keep & (pid_all >= 0) & ~np.isin(pid_all, hold)
d = pd.DataFrame({"lib": np.asarray(C.libs)[C.lib[ok]], "pid": pid_all[ok]})
d["np"] = in_co[d.pid.values]
cnt = d.groupby("pid").size()
np_s = in_co[cnt.index.values]
take = np.minimum(cnt.values, CAP) * np.where(np_s, NP_W, 1)
# 1 スペクトルあたりの v2 の期待抽出回数 = 構造の抽出数 / 構造のスペクトル数
d["v2_w"] = pd.Series(take / cnt.values, index=cnt.index).reindex(d.pid.values).values
mix = d.groupby("lib").agg(spectra=("pid", "size"), structures=("pid", "nunique"), np_share=("np", "mean"),
                           v2_samples=("v2_w", "sum"))
mix["share_now"] = mix.spectra / mix.spectra.sum()
mix["share_v2"] = mix.v2_samples / mix.v2_samples.sum()
mix.sort_values("spectra", ascending=False).to_csv(f"{OUT}/train_mix.csv")
print(mix.sort_values("spectra", ascending=False).round(3).to_string())
steps_v2 = int(take.sum()) // fpnet.Cfg.batch
epochs_v2 = math.ceil(TOTAL_STEPS / max(steps_v2, 1))
print(f"training spectra {len(d):,} of {cnt.size:,} structures (NP {np_s.mean():.3f} of structures, "
      f"{d.np.mean():.3f} of spectra); spectra per structure q50/q90/q99/max "
      f"{np.quantile(cnt.values, [0.5, 0.9, 0.99]).tolist()} / {cnt.max()}")
print(f"v2 epoch: {int(take.sum()):,} samples ({steps_v2:,} steps of {fpnet.Cfg.batch}), NP share of samples "
      f"{(take * np_s).sum() / take.sum():.3f}; {epochs_v2} epochs for {TOTAL_STEPS:,} steps")

# %% [markdown]
# ## v2 の設定（固定した決め方を当てる）

# %%
pc = paired.set_index(["variant", "group"])
a_tims = pc.loc[("tims0", "A")]
tims_fixed = 0.0 if a_tims.lo > 0 else None
prep_ok = [(v, pc.loc[(v, "A")].est) for v in R.variant.unique() if v.startswith("floor")
           and pc.loc[(v, "A")].lo > 0 and pc.loc[(v, "B_N")].lo > -NI_MARGIN]
best_prep = max(prep_ok, key=lambda x: x[1])[0] if prep_ok else None
floor, topk = (BASE if best_prep is None else
               next((f, k) for f in FLOORS for k in TOPKS if f"floor{f:g}_top{k}" == best_prep))
pv = prov[(prov["queries"] == "all") & (prov["contrast"] == "co_only - ho")].iloc[0]
train_v2b = bool(pv.est <= COEF_MAX and pv.hi < 0)
np_ceil = ceil[(ceil.fp == "morgan2_2048") & (ceil.group == "B_N")].ceil_pess.iloc[0]
cfg = dict(max_per_struct=CAP, np_weight=NP_W, total_steps=TOTAL_STEPS, tims_fixed=tims_fixed, peak_floor=floor,
           max_peaks=topk, train_v2b=train_v2b,
           evidence=dict(tims0_A=a_tims[["est", "lo", "hi"]].to_dict(),
                         tims0_B_N=pc.loc[("tims0", "B_N")][["est", "lo", "hi"]].to_dict(),
                         prep_candidates=[v for v, _ in prep_ok], prep_chosen=best_prep,
                         provenance=pv[["est", "lo", "hi"]].to_dict(), fp_ceiling_B_N_morgan2=float(np_ceil),
                         v2_epochs=epochs_v2, v2_steps_per_epoch=steps_v2))
json.dump(cfg, open(f"{OUT}/v2_config.json", "w"), indent=1)
print(json.dumps(cfg, indent=1))
print("GPU-free option (old weights, tims = 0) vs as is:",
      {g: f"{pc.loc[('tims0', g)].est:+.4f} [{pc.loc[('tims0', g)].lo:+.4f}, {pc.loc[('tims0', g)].hi:+.4f}]"
       for g in ["A", "B_N", "C", "V"]})
if np_ceil < 0.8:
    print(f"B_N ceiling with Morgan r=2 2048 is {np_ceil:.3f} < 0.8: list a fingerprint change for a later step")
log("done")

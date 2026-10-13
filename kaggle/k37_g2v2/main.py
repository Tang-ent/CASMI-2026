# %% [markdown]
# # casmi26-k37-g2v2
# M7 P1（2026-10-13、手元の GPU）: G2（k29）の追加学習。k29 と同じ recipe（データ、64 スペクトル × 17 構造、損失、
# warmup + cosine、種 0）で、総ステップだけ 60,000 → 180,000（約 3.5 エポック）にして最初から学習する。
# 理由: k29 は約 1.16 エポックで、rank の損失が最後まで下がっていた（直近 15k ステップで −0.015）。fprich は 10 エポック。
# - 順番: (1) fz（v2b）、z(fz) + z(fwd v1) の順位が k15 / k20 と、z(fz) + z(fwd1) + z(g2 v1) の順位が k29 と一致すること
#   を確かめる（g2 v1 は k29 の g2.pt を fp32 で採点。k29 の門は fp16 だったので、v1 の一致の下限だけ 0.95）
#   (2) 門の A の幅を学習の前に固定する (3) 学習。2 万ステップごとに窓の MRR を監視（報告だけ、判断には使わない）
#   (4) 採点と門 G2v2
# 門 G2v2（M7 の計画 2026-10-13 で固定、結果を見る前）: z(fz) + z(fwd1) + z(g2 v2) − (z(fz) + z(fwd1) + z(g2 v1)) の
# B_N の差の下限 > 0、かつ A の差の下限 > −A_MARGIN（窓の全候補、pid の同点、対のブートストラップ 2,000 回）。
# A_MARGIN = 3.605 × SE で、SE は学習の前に測る comb3(v1) − comb1 の A の対の差の標準誤差（k29 と同じ考え方）。
# 換算: v1 と v2 の比較は相関が高く B_N の SE は約 0.0055（k25 の comb2 − comb1）。下限 > 0 は推定値 ≥ 約 +0.011 を
# 求め、縮み 0.29 倍でパイプラインの +0.003、LB で +0.0007。門は先へ進むかだけを決め、採用は k35 の選び方で決める。
# 入力: k01、k05、k05b、k14（比べるクエリ）、k15-s1（v2b）、k20（fwd v1）、k26（分子式）、casmi26-wheels（rdkit）、
# casmi26-g2（k29 の g2.pt。手元では k29 の g2_ranks.parquet も同じ場所に置き、一致を確かめる）。

# %%
import glob
import os
import subprocess
import sys

wh = sorted(glob.glob("/kaggle/input/**/rdkit-*.whl", recursive=True))
assert wh, "attach the casmi26-wheels dataset (rdkit wheels)"
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--find-links",
                    os.path.dirname(wh[0]), "rdkit"], capture_output=True, text=True)
print(r.stdout[-300:], r.stderr[-300:])
import rdkit

print("rdkit", rdkit.__version__)

# %%
import json
import time

import numpy as np
import pandas as pd
from rdkit import RDLogger

from casmi26 import cache, enc3, fpnet, fwd, g2, harness, paths, pool, subform, track

RDLogger.DisableLog("rdApp.*")
T0 = time.time()
OUT = "/kaggle/working"
EXP = "M7"
PPM = 10.0
N_BOOT = 2000
A_SE_FACTOR = 3.605  # 1.96 + 1.645（k29 と同じ）
TOTAL_H = 12.0  # 手元の GPU（Kaggle の 9 時間の枠は無い）。足りなければ fit_horizon が一度だけ縮めて記録する
TOTAL_STEPS = 180000  # k29 は 60,000
BATCH, K_NEG, WORKERS = None, None, None  # None = g2.Cfg（k29 と同じ）
FIT = None  # 手元の動作確認だけ: enc3.FIT_FROM / FIT_AT
MIN_AGREE = 0.98  # fz と comb1 の順位が k15 / k20 と一致する割合の下限（動作確認は窓が違うので 0）
MIN_AGREE_V1 = 0.95  # comb3(v1) と k29 の一致の下限（k29 は fp16、ここは fp32 で採点するので少し緩い）
MAX_QUERIES = None  # 手元の動作確認だけ
MONITOR_EVERY = 20000  # 監視（報告だけ）の間隔
GRAPH_WORKERS = 4
RESUME = None  # 中断した学習の続き: 前の出力の g2v2_state.pt の路
EXPECT_VARIANT = "v2b"

fpnet.Cfg.max_per_struct, fpnet.Cfg.np_weight, fpnet.Cfg.neg_train_only = 0, 3, True  # v2b の抽出
g2.Cfg.total_steps = TOTAL_STEPS
for k, v in [("batch", BATCH), ("k_neg", K_NEG), ("workers", WORKERS)]:
    if v:
        setattr(g2.Cfg, k, v)
if FIT:
    enc3.FIT_FROM, enc3.FIT_AT = FIT

C = cache.Cache(cache.Cache.find_dir())
PD = pool.Pool.find_dir()
P = pool.Pool(PD, pubchem=False)
FT = subform.FormulaTable(subform.FormulaTable.find_dir())
assert len(FT.fid) == len(P)
Q = pd.read_csv(f"{PD}/queries.csv")
Q["rows"] = Q.rows.map(lambda s: [int(x) for x in str(s).split(",")])
hold = pd.read_csv(f"{PD}/holdout.csv")
old = pd.read_parquet(paths.find("fz_ranks_old.parquet"))
if MAX_QUERIES:
    old = old.groupby("set", group_keys=False).head(MAX_QUERIES // 4).reset_index(drop=True)
qi = Q.set_index("qid").loc[old.qid.values].reset_index()
G = {s: (old.set == s).values for s in ["A", "V", "R2", "C"]}
G["B_N"] = old.set.isin(["R2", "V"]).values & P.in_coconut[old.pid.values]
print("queries", {g: int(m.sum()) for g, m in G.items()})

GR = g2.Graphs.build(P.smiles, GRAPH_WORKERS)
print(f"graphs: {GR.ok.mean():.4f} of {len(GR):,} structures readable with <= {g2.MAX_ATOMS} heavy atoms "
      f"({time.time() - T0:.0f}s)", flush=True)

import torch

print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
V1_PATH = [p for p in glob.glob("/kaggle/input/**/g2.pt", recursive=True)]
assert len(V1_PATH) == 1, f"attach exactly one g2.pt (k29, casmi26-g2): {V1_PATH}"
V1_PATH = V1_PATH[0]

# %% [markdown]
# ## 1. fz（v2b）、fwd v1、g2 v1 の順位の再現と門の幅（学習の前）

# %%
res_v = [json.load(open(p)) for p in glob.glob("/kaggle/input/**/result_v2*.json", recursive=True)]
assert len(res_v) == 1 and res_v[0]["variant"] == EXPECT_VARIANT, "attach exactly the k15 v2b shard"
FP = fpnet.FPModel(paths.find("fpnet2.pt"))
M1 = fwd.FwdModel(paths.find("fwd.pt"))
assert FP.epoch == res_v[0]["epoch"]
print("v2b epoch", FP.epoch, "| fwd v1 epoch", M1.epoch)
specs_all = [harness.query_specs(C, r) for r in qi.rows]
Z = FP.zs(specs_all)


def zs(x):
    """z-score over the finite values; NaN (no G2 score) -> 0 (k29)."""
    x = np.asarray(x, np.float64)
    f = np.isfinite(x)
    out = np.zeros_like(x)
    if f.sum() > 1 and x[f].std() > 1e-9:
        out[f] = (x[f] - x[f].mean()) / x[f].std()
    return out


cands, BASE = [], []  # BASE = z(fz) + z(fwd1) per query
rk = {k: [] for k in ["fz", "comb1"]}
for i, q in enumerate(qi.itertuples()):
    cand = P.window(P.mass[q.pid], PPM)
    s_fz = np.unpackbits(P.fp[cand], axis=1)[:, :Z.shape[1]].astype(np.float32) @ Z[i]
    s_f1 = M1.score(specs_all[i], P.fp[cand])
    cands.append(cand)
    BASE.append(zs(s_fz) + zs(s_f1))
    rk["fz"].append(fpnet.truth_ranks(s_fz, cand, q.pid)[1])
    rk["comb1"].append(fpnet.truth_ranks(BASE[-1], cand, q.pid)[1])
    if (i + 1) % 1000 == 0:
        print(f"  {i + 1}/{len(qi)} queries ({time.time() - T0:.0f}s)", flush=True)
del FP, M1, Z
torch.cuda.empty_cache()
k15 = pd.read_parquet(paths.find(f"fz_ranks_{EXPECT_VARIANT}.parquet")).set_index("qid").loc[old.qid.values]
k20 = pd.read_parquet(paths.find("fwd_ranks.parquet")).set_index("qid").loc[old.qid.values]
agree_fz = float((k15.r_pid.values == np.asarray(rk["fz"])).mean())
agree_c1 = float((k20.r_comb.values == np.asarray(rk["comb1"])).mean())
n_cand = np.array([len(c) for c in cands])
print(f"fz ranks agree with k15 on {agree_fz:.4f}, comb1 with k20 on {agree_c1:.4f}; window candidates median "
      f"{np.median(n_cand):.0f}, total {n_cand.sum():,} ({time.time() - T0:.0f}s)")
assert agree_fz >= MIN_AGREE and agree_c1 >= MIN_AGREE, "inputs do not reproduce k15 / k20"


def score_all(model, tag):
    """g2 scores of every window with `model` -> (list of score arrays, g2 ranks, comb3 ranks)."""
    t, S, r_g, r_c = time.time(), [], [], []
    for i, q in enumerate(qi.itertuples()):
        s = model.score(specs_all[i], cands[i])
        S.append(s)
        r_g.append(fpnet.truth_ranks(np.nan_to_num(s, nan=-1.0), cands[i], q.pid)[1])
        r_c.append(fpnet.truth_ranks(BASE[i] + zs(s), cands[i], q.pid)[1])
    print(f"  scored {tag}: {len(qi)} queries in {time.time() - t:.0f}s", flush=True)
    return S, np.asarray(r_g, np.float64), np.asarray(r_c, np.float64)


M_V1 = g2.G2Model(V1_PATH, G=GR)
print("g2 v1 step", M_V1.step, "from", V1_PATH)
_, rk["g2v1"], rk["comb3v1"] = score_all(M_V1, "g2 v1")
del M_V1
torch.cuda.empty_cache()
agree_v1 = None
try:
    k29 = pd.read_parquet(paths.find("g2_ranks.parquet")).set_index("qid").loc[old.qid.values]
    agree_v1 = float((k29.r_comb3.values == rk["comb3v1"]).mean())
    print(f"comb3(v1) ranks agree with k29 on {agree_v1:.4f} (k29 scored with fp16)")
    assert agree_v1 >= MIN_AGREE_V1, "g2 v1 does not reproduce k29"
except FileNotFoundError:
    print("k29 g2_ranks.parquet not attached: comb3(v1) agreement not checked")
rr0 = {k: harness.rr_of(np.asarray(v, np.float64)) for k, v in rk.items()}
a_se = float(harness.paired_ci(rr0["comb3v1"][G["A"]], rr0["comb1"][G["A"]], N_BOOT, seed=1)["se"])
A_MARGIN = A_SE_FACTOR * a_se
spec = dict(a_margin=A_MARGIN, a_se_comb3v1_minus_comb1=a_se, a_se_factor=A_SE_FACTOR, n_a=int(G["A"].sum()),
            rule="B_N lo > 0 and A lo > -a_margin for z(fz) + z(fwd1) + z(g2 v2) - (z(fz) + z(fwd1) + z(g2 v1))",
            agree_fz=agree_fz, agree_comb1=agree_c1, agree_comb3_v1=agree_v1, fixed_at_secs=round(time.time() - T0),
            total_steps=TOTAL_STEPS)
json.dump(spec, open(f"{OUT}/gate_g2v2_spec.json", "w"), indent=1)
print(f"gate G2v2 fixed before training: A lower bound > -{A_MARGIN:.4f} (= {A_SE_FACTOR} x SE {a_se:.4f} of "
      f"comb3(v1) - comb1 on A), B_N lower bound > 0", flush=True)

# %% [markdown]
# ## 2. 学習（2 万ステップごとに監視）

# %%
mon_rows = []


def monitor(net, step):
    """Report only: window MRR@25 of z(fz) + z(fwd1) + z(g2 at this step) and its difference from comb3(v1)."""
    _, r_g, r_c = score_all(g2.G2Model(G=GR, net=net), f"step {step}")
    rr_c, rr_g = harness.rr_of(r_c), harness.rr_of(r_g)
    row = dict(step=step)
    for g, m in G.items():
        row[f"{g}_comb3"] = float(rr_c[m].mean())
        row[f"{g}_g2"] = float(rr_g[m].mean())
        d = harness.paired_ci(rr_c[m], rr0["comb3v1"][m], 500, seed=1)
        row[f"{g}_d"], row[f"{g}_d_lo"] = float(d["est"]), float(d["lo"])
    mon_rows.append(row)
    pd.DataFrame(mon_rows).to_csv(f"{OUT}/g2v2_monitor_{step}.csv", index=False)  # 再開しても前の分が残るように 1 ファイルずつ
    print(f"  [monitor] step {step}: " + ", ".join(f"{g} {row[f'{g}_comb3']:.4f} ({row[f'{g}_d']:+.4f})" for g in G),
          flush=True)


budget = TOTAL_H * 3600 - (time.time() - T0) - 3600
print(f"training budget {budget / 3600:.2f} h for {TOTAL_STEPS:,} steps")
D = fpnet.TrainData(C, P, set(hold.pid.values))
NEG = enc3.SameFormulaNegatives(P, FT, set(hold.pid.values))
run = track.init("train", group=EXP, config=dict(
    {f"g2/{k}": getattr(g2.Cfg, k) for k in dir(g2.Cfg) if not k.startswith("_")},
    budget_h=budget / 3600, max_frag=g2.MAX_FRAG, max_atoms=g2.MAX_ATOMS, ppm=g2.PPM, abs_da=g2.ABS_DA, model="g2v2"))
hist = g2.train(C, D, GR, NEG, f"{OUT}/g2v2.pt", budget_secs=budget, track=run, resume=RESUME, monitor=monitor,
                monitor_every=MONITOR_EVERY)
hist = pd.DataFrame(hist)
hist.to_csv(f"{OUT}/g2v2_history.csv", index=False)
print(hist.iloc[::10].round(4).to_string(index=False))
del D

# %% [markdown]
# ## 3. 採点と門 G2v2

# %%
M = g2.G2Model(f"{OUT}/g2v2.pt", G=GR)
print("g2 v2 step", M.step)
_, rk["g2v2"], rk["comb3v2"] = score_all(M, "g2 v2")
ranks = pd.DataFrame({"qid": old.qid.values, "set": old.set.values, "pid": old.pid.values,
                      **{f"r_{k}": np.asarray(v, np.float64) for k, v in rk.items()}})
ranks.to_parquet(f"{OUT}/g2v2_ranks.parquet", index=False)
rr = {k: harness.rr_of(ranks[f"r_{k}"].values) for k in rk}
rows = []
for g, m in G.items():
    for a, b in [("comb3v2", "comb3v1"), ("g2v2", "g2v1"), ("comb3v2", "comb1"), ("comb3v1", "comb1")]:
        rows.append(dict(group=g, new=a, ref=b, mrr_new=rr[a][m].mean(), mrr_ref=rr[b][m].mean(),
                         **harness.paired_ci(rr[a][m], rr[b][m], N_BOOT, seed=1)))
cmp = pd.DataFrame(rows)
cmp.to_csv(f"{OUT}/compare_g2v2.csv", index=False)
print("window MRR@25 (all candidates, pid ties):")
print(cmp.round(4).to_string(index=False))
c = cmp[(cmp.new == "comb3v2") & (cmp.ref == "comb3v1")].set_index("group")
gate = bool(c.loc["B_N", "lo"] > 0 and c.loc["A", "lo"] > -A_MARGIN)
res = dict(gate=gate, a_margin=A_MARGIN, step=int(M.step), total_steps=TOTAL_STEPS, agree_fz=agree_fz,
           agree_comb1=agree_c1, agree_comb3_v1=agree_v1, compare=cmp.to_dict("records"),
           train_last=hist.iloc[-1].to_dict() if len(hist) else None)
json.dump(res, open(f"{OUT}/result_g2v2.json", "w"), indent=1, default=float)
print(f"gate G2v2: {'PASS' if gate else 'FAIL'} — comb3(v2) - comb3(v1): B_N lower bound {c.loc['B_N', 'lo']:+.4f} > 0 "
      f"and A lower bound {c.loc['A', 'lo']:+.4f} > -{A_MARGIN:.4f}")
run.summary(dict(gate=gate, **{f"{r.new}-{r.ref}_{r.group}_{k}": float(getattr(r, k))
                               for r in cmp.itertuples() for k in ["est", "lo"]}))
run.finish()
print(f"total {time.time() - T0:.0f}s")

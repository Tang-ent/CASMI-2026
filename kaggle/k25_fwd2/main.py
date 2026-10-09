# %% [markdown]
# # casmi26-k25-fwd2
# M5 P1 の 1′（2026-10-10）: k20 の指紋→MS/MS の予測器（fwd v1）を、窓の中で比べる損失で追加学習する（fwd v2、GPU）。
# k20 は「スペクトルを当てる」損失だけで学習し、窓内 MRR は 3 エポック目で頭打ちだった。使い方（窓の中で並べる）に
# 学習を合わせる。
# - 損失 = 1 − コサイン（v1 と同じ）+ 窓の中の順位の損失（正解と、窓の中の train の構造 16 件の予測スペクトルの
#   学習スペクトルとのコサインを並べ、正解が 1 番になるように softmax、温度 0.05）
# - 初期値は k20 の fwd.pt、学習データは v1 と同じ（v2b の抽出）。128 スペクトル × 17 構造 × 3 万ステップ、
#   学習率 1e-4（余弦で下げる）、最後のステップを使う（評価セットで選ばない）。予算 3 時間
# - 評価: k20 と同じクエリ・同じ窓（k14 の fz_ranks_old.parquet）。comb1 = z(fz) + z(fwd v1)、
#   comb2 = z(fz) + z(fwd v2)。同点は pid 順
# 使う方の決め方（固定、2026-10-10 の計画）: comb2 − comb1 の対の差で、B_N の下限 > 0 かつ A の推定値 > −0.01 なら v2、
# それ以外は v1。A は 250 件で区間が ±0.04 と広いので下限ではなく推定値で見る（A は最終の選び方でも守る）。
# 決定は use_fwd.json に書き、k22_features3 がそれに従って fwd を選ぶ。
# 入力: k01、k05、k05b、k14（比べるクエリの一覧）、k15-s1（fpnet2.pt = v2b、fz_ranks_v2b.parquet）、
# k20（fwd.pt = v1、fwd_ranks.parquet = v1 の組み合わせの順位。照合に使う）。

# %%
import json
import time

import numpy as np
import pandas as pd

from casmi26 import cache, fpnet, fwd, harness, paths, pool, track

T0 = time.time()
OUT = "/kaggle/working"
EXP = "M5"
FPNET_VARIANT = "v2b"
PPM = 10.0
N_BOOT = 2000
STEPS, BATCH, K_NEG, TEMP, LR = 30000, 128, 16, 0.05, 1e-4
BUDGET_H = 3.0
N_VAL = 300
EVAL_EVERY = 5000
MIN_AGREE = 0.98  # fz と v1 の組み合わせの順位が k15 / k20 と一致する割合の下限（手元の動作確認は窓が違うので 0）

fpnet.Cfg.max_per_struct = 0
fpnet.Cfg.np_weight = 3
fpnet.Cfg.neg_train_only = True  # 負例は train の構造だけ（v2b と同じ）

C = cache.Cache(cache.Cache.find_dir())
PD = pool.Pool.find_dir()
P = pool.Pool(PD, pubchem=False)
Q = pd.read_csv(f"{PD}/queries.csv")
Q["rows"] = Q.rows.map(lambda s: [int(x) for x in str(s).split(",")])
hold = pd.read_csv(f"{PD}/holdout.csv")
old = pd.read_parquet(paths.find("fz_ranks_old.parquet"))
qi = Q.set_index("qid").loc[old.qid.values].reset_index()
is_np = P.in_coconut[old.pid.values]
G = {s: (old.set == s).values for s in ["A", "V", "R2", "C"]}
G["B_N"] = old.set.isin(["R2", "V"]).values & is_np
rng = np.random.default_rng(0)
val = {}
for g in ["V", "B_N"]:
    ix = np.flatnonzero(G[g])
    val[g] = qi.iloc[np.sort(rng.choice(ix, min(N_VAL, len(ix)), replace=False))]
V1 = paths.find("fwd.pt")
print("queries", {g: int(m.sum()) for g, m in G.items()}, "| fwd v1:", V1)

import torch

print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

# %% [markdown]
# ## 学習（v1 から追加学習）

# %%
run = track.init("train", group=EXP, config=dict(steps=STEPS, batch=BATCH, k_neg=K_NEG, temp=TEMP, lr=LR,
                                                  budget_h=BUDGET_H, init="k20 fwd.pt"))
D = fpnet.TrainData(C, P, set(hold.pid.values))
hist = fwd.train_rank(C, D, P, V1, f"{OUT}/fwd2.pt", steps=STEPS, batch=BATCH, k_neg=K_NEG, temp=TEMP, lr=LR,
                      budget_secs=BUDGET_H * 3600, val=val, eval_every=EVAL_EVERY, track=run)
hist = pd.DataFrame(hist)
hist.to_csv(f"{OUT}/fwd2_history.csv", index=False)
print(hist.round(4).to_string(index=False))
del D

# %% [markdown]
# ## v1 と v2 を、fz と組み合わせて比べる

# %%
M1 = fwd.FwdModel(V1)
M2 = fwd.FwdModel(f"{OUT}/fwd2.pt")
FP = fpnet.FPModel(paths.find("fpnet2.pt"))
print("fwd v1 epoch", M1.epoch, "| fwd v2 steps", M2.epoch, "| FPNet", FPNET_VARIANT, "epoch", FP.epoch)
Z = FP.zs([harness.query_specs(C, r) for r in qi.rows])


def zs(x):
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-9 else np.zeros_like(x)


rk = {k: [] for k in ["fz", "fwd1", "fwd2", "comb1", "comb2"]}
for i, q in enumerate(qi.itertuples()):
    cand = P.window(P.mass[q.pid], PPM)
    specs = harness.query_specs(C, q.rows)
    s_fz = np.unpackbits(P.fp[cand], axis=1)[:, :Z.shape[1]].astype(np.float32) @ Z[i]
    s1, s2 = M1.score(specs, P.fp[cand]), M2.score(specs, P.fp[cand])
    for k, s in [("fz", s_fz), ("fwd1", s1), ("fwd2", s2), ("comb1", zs(s_fz) + zs(s1)), ("comb2", zs(s_fz) + zs(s2))]:
        rk[k].append(fpnet.truth_ranks(s, cand, q.pid)[1])
    if (i + 1) % 1000 == 0:
        print(f"  {i + 1}/{len(qi)} queries ({time.time() - T0:.0f}s)")
ranks = pd.DataFrame({"qid": old.qid.values, "set": old.set.values, "pid": old.pid.values,
                      **{f"r_{k}": np.asarray(v, np.float64) for k, v in rk.items()}})
ranks.to_parquet(f"{OUT}/fwd2_ranks.parquet", index=False)
k15 = pd.read_parquet(paths.find(f"fz_ranks_{FPNET_VARIANT}.parquet")).set_index("qid").loc[old.qid.values]
k20 = pd.read_parquet(paths.find("fwd_ranks.parquet")).set_index("qid").loc[old.qid.values]
agree_fz = float((k15.r_pid.values == ranks.r_fz.values).mean())
agree_c1 = float((k20.r_comb.values == ranks.r_comb1.values).mean())
print(f"fz ranks agree with k15 on {agree_fz:.4f}, v1 combined ranks agree with k20 on {agree_c1:.4f}")
assert agree_fz >= MIN_AGREE and agree_c1 >= MIN_AGREE, "inputs do not reproduce k15 / k20"

rr = {k: harness.rr_of(ranks[f"r_{k}"].values) for k in rk}
rows = []
for g, m in G.items():
    for a, b in [("comb2", "comb1"), ("comb2", "fz"), ("comb1", "fz"), ("fwd2", "fwd1")]:
        rows.append(dict(group=g, new=a, ref=b, mrr_new=rr[a][m].mean(), mrr_ref=rr[b][m].mean(),
                         **harness.paired_ci(rr[a][m], rr[b][m], N_BOOT, seed=1)))
cmp = pd.DataFrame(rows)
cmp.to_csv(f"{OUT}/compare_fwd2.csv", index=False)
print("window MRR@25 (pid ties):")
print(cmp.round(4).to_string(index=False))

# %% [markdown]
# ## 使う方の決め方（固定）

# %%
c = cmp[(cmp.new == "comb2") & (cmp.ref == "comb1")].set_index("group")
n_lo, a_est = float(c.loc["B_N", "lo"]), float(c.loc["A", "est"])
use_v2 = bool(n_lo > 0 and a_est > -0.01)
res = dict(use="v2" if use_v2 else "v1", file="fwd2.pt" if use_v2 else "fwd.pt", fwd2_steps=int(M2.epoch),
           fwd1_epoch=int(M1.epoch), B_N_lo=n_lo, A_est=a_est, agree_fz=agree_fz, agree_comb1=agree_c1,
           compare=cmp.to_dict("records"))
json.dump(res, open(f"{OUT}/use_fwd.json", "w"), indent=1, default=float)
print(f"use fwd {res['use']} — comb2 - comb1: B_N lower bound {n_lo:+.4f} > 0 and A estimate {a_est:+.4f} > -0.01")
run.summary(dict(use_v2=use_v2, **{f"{r.new}-{r.ref}_{r.group}_{k}": float(getattr(r, k))
                                   for r in cmp.itertuples() for k in ["est", "lo"]}))
run.finish()
print(f"total {time.time() - T0:.0f}s")

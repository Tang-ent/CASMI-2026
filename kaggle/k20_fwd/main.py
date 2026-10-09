# %% [markdown]
# # casmi26-k20-fwd
# M5 の P0-3（2026-10-09）: 指紋→MS/MS の順方向の予測器（G1、casmi26.fwd）を学習し、FPNet（v2b）の fz と組み合わせて
# 窓内 MRR が上がるかを測る（GPU）。FPNet（スペクトル→指紋）と逆向きの手がかりで、プールの指紋だけで動くので
# k07 でも提出でも RDKit が要らない。
# - 模型: Morgan r=2 2048 bit + 付加イオン + 正負 + 衝突エネルギー → 0.1 Da 刻み 0〜1,200 Da。前向きと後ろ向き（中性損失）
#   の 2 つの出力を区間ごとの重みで混ぜる（NEIMS の形）。timsTOF の印は入れない（M4 の取り違えの教訓）
# - 学習データ: FPNet v2b と同じ（ホールドアウトの構造を除く、構造ごとの上限なし、天然物 3 倍）。最後のエポックを使う
#   （評価セットでエポックを選ばない）。V / B_N の窓内 MRR は数エポックごとに記録するだけ
# - 評価: k14 の fz_ranks_old.parquet のクエリ（A / V / R2 / C、B_N = R2 と V のうち正解が COCONUT にあるもの）で、
#   正解の質量の ±10 ppm のプールの構造の中の順位（同点は pid 順）を fz（v2b）、fwd だけ、z(fz) + z(fwd) で比べる
# 続けるかどうかの決め方（固定、k15 の判定と同じ形。組み合わせ − fz の対の差の 95% 区間）:
#   A の下限 > −0.02 かつ「B_N の下限 > 0」または「A の下限 > 0 かつ B_N の下限 > −0.02」→ 合格なら fwd の特徴量を
#   k07 に足してランカーで評価する（P1-3）。不合格なら G1 はここで止める。
# 入力: k01（キャッシュ）、k05（プール、queries.csv、holdout.csv）、k05b（質量の修正）、k14（比べるクエリの一覧）、
# k15-s1（fpnet2.pt = v2b、fz_ranks_v2b.parquet = 同じクエリでの fz の順位。照合に使う）。

# %%
import json
import time

import numpy as np
import pandas as pd

from casmi26 import cache, fpnet, fwd, harness, paths, pool, track

T0 = time.time()
OUT = "/kaggle/working"
EXP = "M5"
FPNET_VARIANT = "v2b"  # k15-s1 の出力（result_v2b.json と fz_ranks_v2b.parquet があること）
PPM = 10.0
N_BOOT = 2000
NI_MARGIN = 0.02
BUDGET_H = 6.0
N_VAL = 300  # 学習中に記録する V / B_N の窓内 MRR のクエリ数（表示だけ）
MAX_STEPS = None  # None = 全部（手元の動作確認で小さくする）
MIN_AGREE = 0.98  # fz の順位が k15 と一致する割合の下限（手元の動作確認は窓が違うので 0）

# 学習データの抽出は FPNet v2b と同じ（fpnet.TrainData の行と epoch_index を使う）
fpnet.Cfg.max_per_struct = 0
fpnet.Cfg.np_weight = 3
fpnet.Cfg.neg_train_only = True
if MAX_STEPS:
    fwd.Cfg.max_steps_per_epoch = MAX_STEPS
    fwd.Cfg.epochs = 2

C = cache.Cache(cache.Cache.find_dir())
PD = pool.Pool.find_dir()
P = pool.Pool(PD, pubchem=False)
Q = pd.read_csv(f"{PD}/queries.csv")
Q["rows"] = Q.rows.map(lambda s: [int(x) for x in str(s).split(",")])
hold = pd.read_csv(f"{PD}/holdout.csv")
old = pd.read_parquet(paths.find("fz_ranks_old.parquet"))  # k14: 比べるクエリの一覧
qi = Q.set_index("qid").loc[old.qid.values].reset_index()
is_np = P.in_coconut[old.pid.values]
G = {s: (old.set == s).values for s in ["A", "V", "R2", "C"]}
G["B_N"] = old.set.isin(["R2", "V"]).values & is_np
print("queries", {g: int(m.sum()) for g, m in G.items()}, "| holdout", len(hold))
rng = np.random.default_rng(0)
val = {}
for g in ["V", "B_N"]:
    ix = np.flatnonzero(G[g])
    val[g] = qi.iloc[np.sort(rng.choice(ix, min(N_VAL, len(ix)), replace=False))]

import torch

print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

# %% [markdown]
# ## 学習

# %%
run = track.init("train", group=EXP, config=dict(
    {f"fwd/{k}": getattr(fwd.Cfg, k) for k in dir(fwd.Cfg) if not k.startswith("_")},
    bin=fwd.BIN, max_mz=fwd.MAX_MZ, np_weight=fpnet.Cfg.np_weight, budget_h=BUDGET_H, n_holdout=len(hold)))
D = fpnet.TrainData(C, P, set(hold.pid.values))
probe = D.epoch_index(np.random.default_rng(1))
print(f"training spectra {len(D):,}; epoch draw {len(probe):,}, NP share {D.is_np[probe].mean():.3f} "
      f"(all spectra {D.is_np.mean():.3f})")
del probe
hist = fwd.train(C, D, P, f"{OUT}/fwd.pt", val=val, budget_secs=BUDGET_H * 3600, track=run)
hist = pd.DataFrame(hist)
hist.to_csv(f"{OUT}/fwd_history.csv", index=False)
print(hist.round(4).to_string(index=False))
run.table("history", hist)
del D

# %% [markdown]
# ## fz（v2b）と対で比べる

# %%
M = fwd.FwdModel(f"{OUT}/fwd.pt")
print("fwd epoch", M.epoch)
assert paths.find(f"result_{FPNET_VARIANT}.json"), "k15 output of the expected variant not attached"
FP = fpnet.FPModel(paths.find("fpnet2.pt"))
print("FPNet", FPNET_VARIANT, "epoch", FP.epoch, "prep", FP.prep)
Z = FP.zs([harness.query_specs(C, r) for r in qi.rows])
rk = fwd.window_ranks(M, C, P, qi, PPM, Z=Z, log=print)
ranks = pd.DataFrame({"qid": old.qid.values, "set": old.set.values, "pid": old.pid.values,
                      "r_fz": rk["fz"], "r_fwd": rk["fwd"], "r_comb": rk["comb"]})
ranks.to_parquet(f"{OUT}/fwd_ranks.parquet", index=False)
k15 = pd.read_parquet(paths.find(f"fz_ranks_{FPNET_VARIANT}.parquet")).set_index("qid").loc[old.qid.values]
agree = float((k15.r_pid.values == ranks.r_fz.values).mean())
print(f"fz ranks agree with k15 ({FPNET_VARIANT}) on {agree:.4f} of the queries")
assert agree >= MIN_AGREE, "fz here does not reproduce k15 (wrong FPNet attached?)"

rr = {k: harness.rr_of(ranks[f"r_{k}"].values) for k in ["fz", "fwd", "comb"]}
rows = []
for g, m in G.items():
    for k in ["fwd", "comb"]:
        rows.append(dict(group=g, model=k, fz=rr["fz"][m].mean(), new=rr[k][m].mean(),
                         **harness.paired_ci(rr[k][m], rr["fz"][m], N_BOOT, seed=1)))
cmp = pd.DataFrame(rows)
cmp.to_csv(f"{OUT}/compare_fwd.csv", index=False)
print(f"window MRR@25 (pid ties), vs fz ({FPNET_VARIANT}):")
print(cmp.round(4).to_string(index=False))

# B_N のライブラリ別（表示だけ）
qlib = Q.set_index("qid").qlib.reindex(old.qid.values).map(lambda i: C.libs[i]).values
by_lib = pd.DataFrame([dict(lib=lb, n=int((G["B_N"] & (qlib == lb)).sum()),
                            fz=rr["fz"][G["B_N"] & (qlib == lb)].mean(), comb=rr["comb"][G["B_N"] & (qlib == lb)].mean(),
                            fwd=rr["fwd"][G["B_N"] & (qlib == lb)].mean()) for lb in np.unique(qlib[G["B_N"]])])
by_lib.to_csv(f"{OUT}/compare_fwd_by_lib.csv", index=False)
print("B_N by query library:")
print(by_lib.round(4).to_string(index=False))

# %% [markdown]
# ## 判定 F（固定）

# %%
c = cmp[cmp.model == "comb"].set_index("group")
a_lo, n_lo = float(c.loc["A", "lo"]), float(c.loc["B_N", "lo"])
gate = bool(a_lo > -NI_MARGIN and (n_lo > 0 or (a_lo > 0 and n_lo > -NI_MARGIN)))
res = dict(fwd_epoch=int(M.epoch), fpnet=FPNET_VARIANT, fpnet_epoch=int(FP.epoch), fz_agree_k15=agree, gate=gate,
           compare=cmp.set_index(["model", "group"])[["fz", "new", "est", "lo", "hi"]].reset_index()
           .to_dict("records"))
json.dump(res, open(f"{OUT}/probe3.json", "w"), indent=1, default=float)
print(f"gate F: {'PASS' if gate else 'FAIL'} — z(fz)+z(fwd) vs fz: A lower bound {a_lo:+.4f} > -{NI_MARGIN} and "
      f"(B_N lower bound {n_lo:+.4f} > 0 or A lower bound > 0 with B_N lower bound > -{NI_MARGIN})")
run.summary(dict(gate=gate, **{f"{m}_{g}_{k}": float(cmp[(cmp.model == m) & (cmp.group == g)][k].iloc[0])
                               for m in ["fwd", "comb"] for g in G for k in ["new", "est", "lo"]}))
run.finish()
print(f"total {time.time() - T0:.0f}s")

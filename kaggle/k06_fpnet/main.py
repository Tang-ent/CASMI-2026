# %% [markdown]
# # casmi26-k06-fpnet
# スペクトル → 分子指紋（プールの Morgan r=2、2048 bit）を予測する Transformer を学習する（GPU）。
# 学習データは train の全スペクトルから、ホールドアウト（A / C / R / V の構造）を除いたもの。
# 毎エポック、V の構造で「質量窓内の候補に対する f·z の MRR@25」を測り、最良のものを保存する。
# A（np-examples）の窓内 MRR も参考として出す（選択には使わない）。
# 学習曲線は wandb にオフラインで記録し、完了後に手元で tools/wandb_sync.py で送る。

# %%
import json
import os
import time

import numpy as np
import pandas as pd

from casmi26 import cache, fpnet, pool, track

PILOT = False  # True: 10% のデータで 1 エポックだけ回して所要時間を測る
EPOCHS = 18  # パイロット実測 0.46 h/epoch -> 約 8.3 h（10.5 h の打ち切りと 12 h 制限に余裕を残す）
EXP = "M2"  # wandb の group
T0 = time.time()
OUT = "/kaggle/working"

C = cache.Cache(cache.Cache.find_dir())
P = pool.Pool(pool.Pool.find_dir())
PD = pool.Pool.find_dir()
Q = pd.read_csv(f"{PD}/queries.csv")
Q["rows"] = Q.rows.map(lambda s: [int(x) for x in str(s).split(",")])
hold = pd.read_csv(f"{PD}/holdout.csv")
val_q = Q[Q.set == "V"]
a_q = Q[Q.set == "A"]
if PILOT:
    val_q = val_q.sample(300, random_state=0)
print("holdout", len(hold), "| V queries", len(val_q), "| A queries", len(a_q))

import torch

print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

# %%
FRAC, BUDGET_H = (0.1, 1.5) if PILOT else (1.0, 10.5)
run = track.init("train", group=EXP, config=dict(
    {k: getattr(fpnet.Cfg, k) for k in dir(fpnet.Cfg) if not k.startswith("_")},
    pilot=PILOT, epochs=1 if PILOT else EPOCHS, frac=FRAC, budget_h=BUDGET_H, n_holdout=len(hold),
    n_val=len(val_q), n_A=len(a_q)))
hist = fpnet.train(C, P, set(hold.pid.values), val_q, f"{OUT}/fpnet.pt",
                   frac=FRAC, epochs=1 if PILOT else EPOCHS,
                   budget_secs=BUDGET_H * 3600, extra_val={"A": a_q}, track=run)
pd.DataFrame(hist).to_csv(f"{OUT}/fpnet_history.csv", index=False)
print(pd.DataFrame(hist).round(4).to_string(index=False))

# %%
if PILOT and hist:
    full_steps = (hist[0]["steps"] / 0.1)
    per_step = hist[0]["epoch_secs"] / max(hist[0]["steps"], 1)
    print(f"projected full epoch: {full_steps:,.0f} steps x {per_step:.3f}s = {full_steps * per_step / 3600:.2f} h")
json.dump(dict(pilot=PILOT, secs=round(time.time() - T0)), open(f"{OUT}/fpnet_run.json", "w"))
if hist:
    run.table("history", pd.DataFrame(hist))
run.summary({"total_secs": time.time() - T0})
run.finish()
print(f"total {time.time() - T0:.0f}s")

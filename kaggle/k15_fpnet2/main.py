# %% [markdown]
# # casmi26-k15-fpnet2（分割 = 変種）
# M4 の P1（2026-10-07）: FPNet v2 を学習する（GPU）。設定は k14_fpnet_probe の v2_config.json（結果を見る前に固定した
# 決め方で決めたもの）。
# - s0 = v2a: 構造ごとに最大 max_per_struct 件、COCONUT にもある train の構造を np_weight 回分、総ステップ total_steps、
#   tims_fixed / peak_floor / max_peaks は k14 の判定のとおり
# - s1 = v2b: v2a に加えて、順位の損失の負例を窓の中の train の構造（ホールドアウト以外）からだけ引く。
#   k14 で train_v2b = true のときだけ push する。入力に s0 の出力（v2a の順位と偏りの係数）を足す
# どちらも最後のエポックを使う（評価セットでエポックを選ばない）。V / A / B_N の窓内 MRR は数エポックごとに記録するだけ。
# 学習後、k14 の今のモデルの順位（fz_ranks_old.parquet）と同じクエリ・同じ窓で新しいモデルの順位を出して対で比べ、
# 続けるかどうかを判定する（固定）: B_N の対の差の 95% 下限 > 0 かつ A の下限 > −NI_MARGIN。
# B_A の道（2026-10-08、ユーザーの決定。v2a の結果を見た後）: k16 の選び方に足した B_A の道と同じ形を、この判定にも当てる。
#   fz だけでは B_A と A は同じスペクトルなので A で見る: A の下限 > 0 かつ B_N の下限 > −NI_MARGIN でも合格。
#   v2a は B_N だけの判定では不合格（B_N −0.001 [−0.014, +0.013]）、B_A の道で合格（A +0.097 [+0.061, +0.136]）。
# v2b を使う条件（固定）: v2a が判定を通り、v2b の偏りの係数が v2a より 0 に近く、B_N で v2a に対して非劣性
# （対の差の下限 > −NI_MARGIN）。
# 出力の重みの名前は fpnet2.pt（古い fpnet.pt と paths.find で取り違えないように）。
# ユーザーの決定（2026-10-07、k14 の学習データの内訳を見て。評価の結果は見ていない）: 構造ごとの上限 4 件は外す。
#   天然物の構造ほどスペクトルが多い（riken は 94% が天然物で 1 構造 16 件）ため、上限 4 件では天然物のスペクトルの
#   割合が 25.9% → 21.1%（3 倍の重みの後でも）に下がり、計画の狙いと逆になる。上限なし・天然物 3 倍で約 51%。
# 学習曲線は wandb（group M4）に送る（オフラインなら完了後に手元で tools/wandb_sync.py）。

# %%
import json
import os
import time

import numpy as np
import pandas as pd

from casmi26 import cache, fpnet, harness, paths, pool, track

SHARD, N_SHARDS = globals().get("SHARD", 0), globals().get("N_SHARDS", 1)
VARIANT = ["v2a", "v2b"][SHARD]
EXP = "M4"
BUDGET_H = 10.75
PPM = 10.0
N_BOOT = 2000
NI_MARGIN = 0.02
TOTAL_STEPS = None  # None = v2_config.json の値（手元の動作確認で小さくする）
T0 = time.time()
OUT = "/kaggle/working"

cfg = json.load(open(paths.find("v2_config.json")))
if TOTAL_STEPS:
    cfg["total_steps"] = TOTAL_STEPS
cfg["max_per_struct"] = 0  # ユーザーの決定（上の説明）
assert VARIANT == "v2a" or cfg["train_v2b"], "v2b is trained only when k14 set train_v2b"
for k in ["max_per_struct", "np_weight", "total_steps", "tims_fixed", "peak_floor", "max_peaks"]:
    setattr(fpnet.Cfg, k, cfg[k])
fpnet.Cfg.neg_train_only = VARIANT == "v2b"
print(VARIANT, {k: v for k, v in cfg.items() if k != "evidence"})

C = cache.Cache(cache.Cache.find_dir())
PD = pool.Pool.find_dir()
P = pool.Pool(PD, pubchem=False)
Q = pd.read_csv(f"{PD}/queries.csv")
Q["rows"] = Q.rows.map(lambda s: [int(x) for x in str(s).split(",")])
hold = pd.read_csv(f"{PD}/holdout.csv")
val_q = Q[Q.set == "V"]
a_q = Q[Q.set == "A"]
bn_q = Q[Q.set.isin(["R2", "V"]) & P.in_coconut[Q.pid.values]]
print("holdout", len(hold), "| V", len(val_q), "| A", len(a_q), "| B_N", len(bn_q))

import torch

print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

# %% [markdown]
# ## 学習

# %%
eval_every = 2  # 上限なし・天然物 3 倍で 1 エポック約 6,500 ステップ、6 万ステップで約 10 エポック
run = track.init("train", group=EXP, config=dict(
    {k: getattr(fpnet.Cfg, k) for k in dir(fpnet.Cfg) if not k.startswith("_")},
    variant=VARIANT, budget_h=BUDGET_H, eval_every=eval_every, n_holdout=len(hold), k14=cfg["evidence"]))
hist = fpnet.train(C, P, set(hold.pid.values), val_q, f"{OUT}/fpnet2.pt", budget_secs=BUDGET_H * 3600,
                   extra_val={"A": a_q, "B_N": bn_q}, track=run, select="last", eval_every=eval_every)
hist = pd.DataFrame(hist)
hist.to_csv(f"{OUT}/fpnet_history.csv", index=False)
print(hist.round(4).to_string(index=False))
run.table("history", hist)

# %% [markdown]
# ## 今のモデル（k14）と対で比べる

# %%
M = fpnet.FPModel(f"{OUT}/fpnet2.pt")
print("loaded", f"{OUT}/fpnet2.pt", "epoch", M.epoch, "prep", M.prep)
old = pd.read_parquet(paths.find("fz_ranks_old.parquet"))
qi = Q.set_index("qid").loc[old.qid.values]
Z = M.zs([harness.query_specs(C, r) for r in qi.rows])
new = fpnet.window_ranks(Z, P, old.pid.values, PPM)
new.insert(0, "pid", old.pid.values)
new.insert(0, "set", old.set.values)
new.insert(0, "qid", old.qid.values)
new.to_parquet(f"{OUT}/fz_ranks_{VARIANT}.parquet", index=False)
is_np = P.in_coconut[old.pid.values]
G = {s: (old.set == s).values for s in ["A", "V", "R2", "C"]}
G["B_N"] = old.set.isin(["R2", "V"]).values & is_np
rr_new, rr_old = harness.rr_of(new.r_pid), harness.rr_of(old.r_pid)
cmp = pd.DataFrame([dict(group=g, old=rr_old[m].mean(), new=rr_new[m].mean(),
                         **harness.paired_ci(rr_new[m], rr_old[m], N_BOOT, seed=1)) for g, m in G.items()])
cmp.to_csv(f"{OUT}/compare_old_{VARIANT}.csv", index=False)
print("fz-only window MRR (pid ties), new vs the k06 model:")
print(cmp.round(4).to_string(index=False))
c = cmp.set_index("group")
def gate_of(c):
    """Continue with this FPNet? (fz-only, paired vs the k06 model): A non-inferior and (B_N better or the B_A path)."""
    a_lo, n_lo = c["A"]["lo"], c["B_N"]["lo"]
    return bool(a_lo > -NI_MARGIN and (n_lo > 0 or (a_lo > 0 and n_lo > -NI_MARGIN)))


gate = gate_of(c[["lo"]].to_dict("index"))
prov = fpnet.provenance_coef(Z, P, old.pid.values, hold.pid.values, PPM, N_BOOT)
print("provenance coefficient (COCONUT-only - holdout):", {k: round(v, 4) for k, v in prov.items()})
res = dict(variant=VARIANT, epoch=int(M.epoch), gate=gate, compare=c[["old", "new", "est", "lo", "hi"]].to_dict("index"),
           provenance=prov)

if VARIANT == "v2b":
    a = pd.read_parquet(paths.find("fz_ranks_v2a.parquet")).set_index("qid").loc[old.qid.values]
    ra = json.load(open(paths.find("result_v2a.json")))
    rr_a = harness.rr_of(a.r_pid.values)
    vs_a = harness.paired_ci(rr_new[G["B_N"]], rr_a[G["B_N"]], N_BOOT, seed=1)
    use_v2b = bool(gate_of(ra["compare"]) and abs(prov["est"]) < abs(ra["provenance"]["est"]) and vs_a["lo"] > -NI_MARGIN)
    res.update(vs_v2a_B_N=vs_a, provenance_v2a=ra["provenance"], use_v2b=use_v2b)
    print("v2b vs v2a on B_N:", {k: round(v, 4) for k, v in vs_a.items()}, "| use v2b:", use_v2b)
json.dump(res, open(f"{OUT}/result_{VARIANT}.json", "w"), indent=1, default=float)
print(f"gate ({VARIANT}): {'PASS' if gate else 'FAIL'} — A lower bound > -{NI_MARGIN} and (B_N lower bound > 0 "
      f"or A lower bound > 0 with B_N lower bound > -{NI_MARGIN})")
run.summary(dict(gate=gate, epoch=int(M.epoch), **{f"{g}_{k}": float(c.loc[g, k]) for g in c.index
                                                  for k in ["new", "est", "lo"]}))
run.finish()
print(f"total {time.time() - T0:.0f}s")

# %% [markdown]
# # casmi26-k27-enc3
# M6 の P0-1（2026-10-12、GPU）: ピークに部分分子式をつけるスペクトルエンコーダ（casmi26.enc3、MIST 型）を学習し、
# FPNet v2b と同じクエリ・同じ候補で対にして比べる。
# - 学習: v2b と同じ設定（ホールドアウトの構造を除く、構造ごとの上限なし、天然物 3 倍、timsTOF の印 0、Morgan 2048、
#   BCE + 窓の中の順位の損失、v2b と同じ総ステップ = エポック数 × エポックのステップ数、最後のチェックポイント）。
#   変えるのは入力（ピークごとの断片の式・中性損失の式・ppm の誤差・割り当ての印、ピークどうしの式の差による注意の
#   バイアス、プリカーサーのイオンの式）と、順位の損失の負例を正解と同じ分子式の train の構造に限ること
# - 評価: k14 の fz_ranks_old.parquet のクエリ（A / V / R2 / C、B_N = R2 と V のうち正解が COCONUT にあるもの）。
#   正解の質量の ±10 ppm の窓で、正解と同じ分子式の候補だけを並べる（z_F が分子式ごとに違うので、分子式をまたぐと
#   尺度の違いが混ざる）。v2b も同じ候補で並べる。同点は pid 順。窓の全候補での順位（候補ごとに自分の分子式の z）は
#   報告だけ。v2b の全候補の順位が k15-s1 の fz_ranks_v2b.parquet と一致するかを最初に確かめる
# 門 E（固定、計画 2026-10-11）: 次の (a) か (b) を満たし、かつそれぞれの A の差の下限 > −0.02。
#   (a) 単独: enc3 − v2b の B_N の差 ≥ +0.03 かつ下限 > 0
#   (b) 組み合わせ: z(fz_v2b) + z(fz_enc3) − v2b の B_N の差 ≥ +0.05 かつ下限 > 0
# V と B_N の学習中の数字は監視だけで、チェックポイントは選ばない（最後のエポック）。
# 入力: k01、k05、k05b、k14（比べるクエリの一覧）、k15-s1（v2b の fpnet2.pt と fz_ranks_v2b.parquet）、k26（分子式）。

# %%
import json
import time

import numpy as np
import pandas as pd

from casmi26 import cache, enc3, fpnet, harness, paths, pool, subform, track

T0 = time.time()
OUT = "/kaggle/working"
EXP = "M6"
PPM = 10.0
N_BOOT = 2000
NI_MARGIN = 0.02
GATE_A, GATE_B = 0.03, 0.05
BUDGET_H = 10.5
N_VAL = 300  # 学習中に記録する V / B_N のクエリ数（表示だけ）
TOTAL_STEPS = None  # None = v2b と同じ 60,000（手元の動作確認で小さくする）
MIN_AGREE = 0.98  # v2b の全候補の順位が k15 と一致する割合の下限（手元の動作確認は窓が違うので 0）
BATCH = None  # None = 512（手元の動作確認で小さくする）
MAX_STEPS_PER_EPOCH = None  # 手元の動作確認だけ
EXPECT_VARIANT = "v2b"
FIT = None  # 手元の動作確認だけ: 1 ステップの時間を測る区間（既定は enc3.FIT_FROM, enc3.FIT_AT = 100, 600）
if FIT:
    enc3.FIT_FROM, enc3.FIT_AT = FIT
if TOTAL_STEPS:
    enc3.Cfg.total_steps = TOTAL_STEPS
if BATCH:
    enc3.Cfg.batch = BATCH
if MAX_STEPS_PER_EPOCH:
    enc3.Cfg.max_steps_per_epoch = MAX_STEPS_PER_EPOCH

C = cache.Cache(cache.Cache.find_dir())
PD = pool.Pool.find_dir()
P = pool.Pool(PD, pubchem=False)
FT = subform.FormulaTable(subform.FormulaTable.find_dir())
assert len(FT.fid) == len(P), "pool_formula.npz does not match the pool"
Q = pd.read_csv(f"{PD}/queries.csv")
Q["rows"] = Q.rows.map(lambda s: [int(x) for x in str(s).split(",")])
hold = pd.read_csv(f"{PD}/holdout.csv")
old = pd.read_parquet(paths.find("fz_ranks_old.parquet"))
qi = Q.set_index("qid").loc[old.qid.values].reset_index()
is_np = P.in_coconut[old.pid.values]
G = {s: (old.set == s).values for s in ["A", "V", "R2", "C"]}
G["B_N"] = old.set.isin(["R2", "V"]).values & is_np
print("queries", {g: int(m.sum()) for g, m in G.items()}, "| holdout", len(hold), "| formulas", len(FT.keys))
specs_all = [harness.query_specs(C, r) for r in qi.rows]
pids = old.pid.values.astype(np.int64)
rng = np.random.default_rng(0)
val = {}
for g in ["V", "B_N"]:
    ix = np.sort(rng.choice(np.flatnonzero(G[g]), min(N_VAL, int(G[g].sum())), replace=False))
    val[g] = ([specs_all[i] for i in ix], pids[ix])

import torch

print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

# %% [markdown]
# ## 学習

# %%
run = track.init("train", group=EXP, config=dict(
    {f"enc3/{k}": getattr(enc3.Cfg, k) for k in dir(enc3.Cfg) if not k.startswith("_")},
    budget_h=BUDGET_H, n_holdout=len(hold), gate=dict(a=GATE_A, b=GATE_B, ni=NI_MARGIN)))
hist = enc3.train(C, P, FT, set(hold.pid.values), f"{OUT}/enc3.pt", val=val, budget_secs=BUDGET_H * 3600,
                  track=run, eval_every=2)
hist = pd.DataFrame(hist)
hist.to_csv(f"{OUT}/enc3_history.csv", index=False)
print(hist.round(4).to_string(index=False))
run.table("history", hist)

# %% [markdown]
# ## v2b と対で比べる（正解と同じ分子式の候補）

# %%
M3 = enc3.Enc3Model(f"{OUT}/enc3.pt", FT)
print("enc3 epoch", M3.epoch, "| device", M3.device)
res_v = [json.load(open(p)) for p in __import__("glob").glob("/kaggle/input/**/result_v2*.json", recursive=True)]
assert len(res_v) == 1 and res_v[0]["variant"] == EXPECT_VARIANT, "attach exactly the k15 v2b shard"
MB = fpnet.FPModel(paths.find("fpnet2.pt"))
assert MB.epoch == res_v[0]["epoch"]
ZB = MB.zs(specs_all)
# 1) v2b の全候補の順位が k15 と一致するか
rb_all = fpnet.window_ranks(ZB, P, pids, PPM)
ref = pd.read_parquet(paths.find(f"fz_ranks_{EXPECT_VARIANT}.parquet")).set_index("qid").loc[old.qid.values]
agree = float((rb_all.r_pid.values == ref.r_pid.values).mean())
print(f"v2b all-candidate ranks vs k15: agreement {agree:.4f}")
assert agree >= MIN_AGREE, "v2b ranks do not reproduce k15"


def zs_(x):
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-9 else np.zeros_like(x)


# 2) 正解と同じ分子式の候補で、v2b / enc3 / 組み合わせの順位（同点は pid 順）。enc3 の全候補の順位も（報告だけ）
rows = []
B = 256
for a in range(0, len(pids), B):
    part = range(a, min(a + B, len(pids)))
    jobs, slots = [], []
    for i in part:
        cand = P.window(P.mass[pids[i]], PPM)
        fids = FT.fid[cand]
        u = list(dict.fromkeys(int(f) for f in fids))
        slots.append((cand, fids, {f: len(jobs) + k for k, f in enumerate(u)}))
        jobs.extend((specs_all[i], f) for f in u)
    Z3 = M3.z_many(jobs)
    for i, (cand, fids, pos) in zip(part, slots):
        p = int(pids[i])
        F = np.unpackbits(P.fp[cand], axis=1)[:, :Z3.shape[1]].astype(np.float32)
        s3 = np.einsum("nb,nb->n", F, Z3[[pos[int(f)] for f in fids]])
        sb = F @ ZB[i]
        same = fids == FT.fid[p]
        c_s = cand[same]
        rows.append(dict(n_cand=len(cand), n_same=int(same.sum()),
                         v2b=fpnet.truth_ranks(sb[same], c_s, p)[1], enc3=fpnet.truth_ranks(s3[same], c_s, p)[1],
                         comb=fpnet.truth_ranks(zs_(sb[same]) + zs_(s3[same]), c_s, p)[1],
                         v2b_all=fpnet.truth_ranks(sb, cand, p)[1], enc3_all=fpnet.truth_ranks(s3, cand, p)[1]))
    print(f"  {min(a + B, len(pids))}/{len(pids)} queries ({time.time() - T0:.0f}s)", flush=True)
R = pd.DataFrame(rows)
R.insert(0, "pid", pids)
R.insert(0, "set", old.set.values)
R.insert(0, "qid", old.qid.values)
R.to_parquet(f"{OUT}/enc3_ranks.parquet", index=False)
rr = {c: harness.rr_of(R[c].values) for c in ["v2b", "enc3", "comb", "v2b_all", "enc3_all"]}
cmp = []
for g, m in G.items():
    row = dict(group=g, n=int(m.sum()), n_same_median=float(R.n_same[m].median()), v2b=rr["v2b"][m].mean(),
               enc3=rr["enc3"][m].mean(), comb=rr["comb"][m].mean(), v2b_all=rr["v2b_all"][m].mean(),
               enc3_all=rr["enc3_all"][m].mean())
    for name in ["enc3", "comb"]:
        d = harness.paired_ci(rr[name][m], rr["v2b"][m], N_BOOT, seed=1)
        row.update({f"{name}_est": d["est"], f"{name}_lo": d["lo"], f"{name}_hi": d["hi"]})
    d = harness.paired_ci(rr["enc3_all"][m], rr["v2b_all"][m], N_BOOT, seed=1)
    row.update(all_est=d["est"], all_lo=d["lo"], all_hi=d["hi"])
    cmp.append(row)
cmp = pd.DataFrame(cmp).set_index("group")
cmp.to_csv(f"{OUT}/compare_enc3.csv")
print("same-formula window MRR@25 (pid ties); differences vs v2b with paired 95% CIs:")
print(cmp.round(4).to_string())
c = cmp
gate_a = bool(c.loc["B_N", "enc3_est"] >= GATE_A and c.loc["B_N", "enc3_lo"] > 0 and c.loc["A", "enc3_lo"] > -NI_MARGIN)
gate_b = bool(c.loc["B_N", "comb_est"] >= GATE_B and c.loc["B_N", "comb_lo"] > 0 and c.loc["A", "comb_lo"] > -NI_MARGIN)
gate = gate_a or gate_b
res = dict(epoch=int(M3.epoch), steps=int(hist.gstep.iloc[-1]), gate=gate, gate_a=gate_a, gate_b=gate_b,
           v2b_agree=agree, compare=c.to_dict("index"))
json.dump(res, open(f"{OUT}/result_enc3.json", "w"), indent=1, default=float)
print(f"gate E: {'PASS' if gate else 'FAIL'} (a: {gate_a}, b: {gate_b}) — (a) enc3 - v2b on B_N >= +{GATE_A} with lower "
      f"bound > 0, or (b) z(v2b) + z(enc3) - v2b >= +{GATE_B} with lower bound > 0; A lower bound > -{NI_MARGIN} for "
      f"the same comparison")
run.summary(dict(gate=gate, gate_a=gate_a, gate_b=gate_b, v2b_agree=agree,
                 **{f"{g}_{k}": float(c.loc[g, k]) for g in c.index for k in ["v2b", "enc3", "comb", "enc3_est",
                                                                             "enc3_lo", "comb_est", "comb_lo"]}))
run.finish()
print(f"total {time.time() - T0:.0f}s")

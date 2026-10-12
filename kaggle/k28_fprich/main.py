# %% [markdown]
# # casmi26-k28-fprich
# M6 の P0-2（2026-10-12、GPU）: FPNet を v2b と同じ設定のまま、目標の指紋だけを豊かにして学習し直す
# （k26 の pool_fp2.npy: Morgan r2 2048（先頭、P.fp と同じ）、FCFP4、AtomPair、Topological Torsion、RDKit path 各 2048、
# MACCS 167、計 10,407 ビット）。点数は全ビットの f·z（ベルヌーイ対数尤度）。
# - 学習: k15 の v2b と同じ（v2_config.json、上限なし、天然物 3 倍、負例は train の構造だけ、timsTOF の印 0、
#   総ステップ 60,000 = 10 エポック、最後のエポック）。順位の損失も全ビットの f·z
# - 評価: k14 の fz_ranks_old.parquet のクエリ、正解の質量の ±10 ppm の窓の全候補（分子式に依らないので全候補）、
#   同点は pid 順。v2b は k15-s1 の重みで同じ窓を計算し直し、k15 の fz_ranks_v2b.parquet と一致するかを確かめる
# 門 E（固定、計画 2026-10-11）: 次の (a) か (b) を満たし、かつそれぞれの A の差の下限 > −0.02。
#   (a) 単独: fprich − v2b の B_N の差 ≥ +0.03 かつ下限 > 0
#   (b) 組み合わせ: z(fz_v2b) + z(fz_rich) − v2b の B_N の差 ≥ +0.05 かつ下限 > 0
# 報告だけ: fprich の z の先頭 2048 ビット（Morgan）だけで並べた順位（豊かな目標が補助の学習として効いたのか、
# 点数のビットが増えたことが効いたのかの切り分け）。
# 入力: k01、k05、k05b、k14（v2_config.json、比べるクエリの一覧）、k15-s1（v2b）、k26（pool_fp2.npy）。

# %%
import glob
import json
import time

import numpy as np
import pandas as pd

from casmi26 import cache, chem, fpnet, harness, paths, pool, track

T0 = time.time()
OUT = "/kaggle/working"
EXP = "M6"
PPM = 10.0
N_BOOT = 2000
NI_MARGIN = 0.02
GATE_A, GATE_B = 0.03, 0.05
BUDGET_H = 10.75
TOTAL_STEPS = None  # None = v2_config.json の値（手元の動作確認で小さくする）
MIN_AGREE = 0.98  # v2b の順位が k15 と一致する割合の下限（手元の動作確認は窓が違うので 0）
EXPECT_VARIANT = "v2b"
BATCH = None  # None = fpnet.Cfg.batch = 512（手元の動作確認でメモリを抑えるために小さくする）
RESUME = None  # 中断した学習の続き: 前の出力の fprich_state.pt の路（tools/local/run_k28.sh の RESUME=1 が書き換える）

cfg = json.load(open(paths.find("v2_config.json")))
if TOTAL_STEPS:
    cfg["total_steps"] = TOTAL_STEPS
cfg["max_per_struct"] = 0  # k15 と同じ（ユーザーの決定 2026-10-07）
for k in ["max_per_struct", "np_weight", "total_steps", "tims_fixed", "peak_floor", "max_peaks"]:
    setattr(fpnet.Cfg, k, cfg[k])
fpnet.Cfg.neg_train_only = True  # v2b
if BATCH:
    fpnet.Cfg.batch = BATCH
print({k: v for k, v in cfg.items() if k != "evidence"})

C = cache.Cache(cache.Cache.find_dir())
PD = pool.Pool.find_dir()
P = pool.Pool(PD, pubchem=False)
FP2 = np.load(paths.find("pool_fp2.npy"))
NB = chem.RICH_FP_BITS
assert FP2.shape == (len(P), (NB + 7) // 8), FP2.shape
assert (np.unpackbits(FP2[:2000], axis=1)[:, :chem.FP_BITS] == np.unpackbits(P.fp[:2000], axis=1)[:, :chem.FP_BITS]).all()
Q = pd.read_csv(f"{PD}/queries.csv")
Q["rows"] = Q.rows.map(lambda s: [int(x) for x in str(s).split(",")])
hold = pd.read_csv(f"{PD}/holdout.csv")
val_q = Q[Q.set == "V"]
a_q = Q[Q.set == "A"]
bn_q = Q[Q.set.isin(["R2", "V"]) & P.in_coconut[Q.pid.values]]
print("holdout", len(hold), "| V", len(val_q), "| A", len(a_q), "| B_N", len(bn_q), "| rich bits", NB)

import torch

print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

# %% [markdown]
# ## 学習

# %%
run = track.init("train", group=EXP, config=dict(
    {k: getattr(fpnet.Cfg, k) for k in dir(fpnet.Cfg) if not k.startswith("_")},
    variant="fprich", nbits=NB, blocks=chem.RICH_FP_BLOCKS, budget_h=BUDGET_H, n_holdout=len(hold)))
hist = fpnet.train(C, P, set(hold.pid.values), val_q, f"{OUT}/fprich.pt", budget_secs=BUDGET_H * 3600,
                   extra_val={"A": a_q, "B_N": bn_q}, track=run, select="last", eval_every=2, fp=FP2, nbits=NB,
                   resume=RESUME)
hist = pd.DataFrame(hist)
hist.to_csv(f"{OUT}/fprich_history.csv", index=False)
print(hist.round(4).to_string(index=False))
run.table("history", hist)

# %% [markdown]
# ## v2b と対で比べる（窓の全候補）

# %%
MR = fpnet.FPModel(f"{OUT}/fprich.pt")
res_v = [json.load(open(p)) for p in glob.glob("/kaggle/input/**/result_v2*.json", recursive=True)]
assert len(res_v) == 1 and res_v[0]["variant"] == EXPECT_VARIANT, "attach exactly the k15 v2b shard"
MB = fpnet.FPModel(paths.find("fpnet2.pt"))
assert MB.epoch == res_v[0]["epoch"] and MB.nbits == chem.FP_BITS and MR.nbits == NB
print("fprich epoch", MR.epoch, "| v2b epoch", MB.epoch)
old = pd.read_parquet(paths.find("fz_ranks_old.parquet"))
qi = Q.set_index("qid").loc[old.qid.values]
specs_all = [harness.query_specs(C, r) for r in qi.rows]
pids = old.pid.values.astype(np.int64)
ZB, ZR = MB.zs(specs_all), MR.zs(specs_all)
ref = pd.read_parquet(paths.find(f"fz_ranks_{EXPECT_VARIANT}.parquet")).set_index("qid").loc[old.qid.values]


def zs_(x):
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-9 else np.zeros_like(x)


rows = []
for i, p in enumerate(pids):
    cand = P.window(P.mass[p], PPM)
    F2 = np.unpackbits(FP2[cand], axis=1)[:, :NB].astype(np.float32)
    sb = F2[:, :chem.FP_BITS] @ ZB[i]
    sr = F2 @ ZR[i]
    sm = F2[:, :chem.FP_BITS] @ ZR[i, :chem.FP_BITS]
    rows.append(dict(n_cand=len(cand), v2b=fpnet.truth_ranks(sb, cand, p)[1], rich=fpnet.truth_ranks(sr, cand, p)[1],
                     comb=fpnet.truth_ranks(zs_(sb) + zs_(sr), cand, p)[1],
                     rich_morgan=fpnet.truth_ranks(sm, cand, p)[1]))
R = pd.DataFrame(rows)
for c_ in ["qid", "set", "pid"][::-1]:
    R.insert(0, c_, old[c_].values)
R.to_parquet(f"{OUT}/fprich_ranks.parquet", index=False)
agree = float((R.v2b.values == ref.r_pid.values).mean())
print(f"v2b ranks vs k15: agreement {agree:.4f}")
assert agree >= MIN_AGREE, "v2b ranks do not reproduce k15"
is_np = P.in_coconut[pids]
G = {s: (old.set == s).values for s in ["A", "V", "R2", "C"]}
G["B_N"] = old.set.isin(["R2", "V"]).values & is_np
rr = {c_: harness.rr_of(R[c_].values) for c_ in ["v2b", "rich", "comb", "rich_morgan"]}
cmp = []
for g, m in G.items():
    row = dict(group=g, n=int(m.sum()), **{c_: rr[c_][m].mean() for c_ in rr})
    for name in ["rich", "comb", "rich_morgan"]:
        d = harness.paired_ci(rr[name][m], rr["v2b"][m], N_BOOT, seed=1)
        row.update({f"{name}_est": d["est"], f"{name}_lo": d["lo"], f"{name}_hi": d["hi"]})
    cmp.append(row)
c = pd.DataFrame(cmp).set_index("group")
c.to_csv(f"{OUT}/compare_fprich.csv")
print("window MRR@25 (all candidates, pid ties); differences vs v2b with paired 95% CIs:")
print(c.round(4).to_string())
gate_a = bool(c.loc["B_N", "rich_est"] >= GATE_A and c.loc["B_N", "rich_lo"] > 0 and c.loc["A", "rich_lo"] > -NI_MARGIN)
gate_b = bool(c.loc["B_N", "comb_est"] >= GATE_B and c.loc["B_N", "comb_lo"] > 0 and c.loc["A", "comb_lo"] > -NI_MARGIN)
gate = gate_a or gate_b
res = dict(epoch=int(MR.epoch), gate=gate, gate_a=gate_a, gate_b=gate_b, v2b_agree=agree, nbits=NB,
           compare=c.to_dict("index"))
json.dump(res, open(f"{OUT}/result_fprich.json", "w"), indent=1, default=float)
print(f"gate E: {'PASS' if gate else 'FAIL'} (a: {gate_a}, b: {gate_b}) — (a) fprich - v2b on B_N >= +{GATE_A} with "
      f"lower bound > 0, or (b) z(v2b) + z(fprich) - v2b >= +{GATE_B} with lower bound > 0; A lower bound > -{NI_MARGIN} "
      f"for the same comparison")
run.summary(dict(gate=gate, gate_a=gate_a, gate_b=gate_b, v2b_agree=agree,
                 **{f"{g}_{k}": float(c.loc[g, k]) for g in c.index for k in ["v2b", "rich", "comb", "rich_est",
                                                                             "rich_lo", "comb_est", "comb_lo"]}))
run.finish()
print(f"total {time.time() - T0:.0f}s")

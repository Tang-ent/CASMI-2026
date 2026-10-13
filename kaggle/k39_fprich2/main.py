# %% [markdown]
# # casmi26-k39-fprich2
# M7 P1（2026-10-13、手元の GPU）: fprich（k28）の追加学習。k28 の重みから始め、v2b と同じ recipe（v2_config.json、
# 上限なし、天然物 3 倍、負例は train だけ、timsTOF の印 0、10,407 ビット）で、さらに 10 エポック（64,690 ステップ）を
# 新しい warmup + cosine で学習する（warm restart。AdamW は新しく、抽出の種は 1 にして k28 と違う順で引く）。
# 理由: 学習の損失は 10 エポック目でも下がっていた（bce 0.211 → 0.209、rank 0.098 → 0.089）。ただし検証の窓の MRR は
# 5〜7 エポック目から横ばいなので、期待は小さい（計画ファイル M7 P1）。毎エポックの窓の V / A / B_N は報告だけ。
# - 順番: (1) v2b と fprich v1 の順位を窓の全候補で計算し、k15 / k28 と一致することを確かめる (2) 門の A の幅を学習の
#   前に固定する (3) 学習 (4) 採点と門 E2
# 門 E2（M7 の計画 2026-10-13 で固定、結果を見る前）: z(v2b) + z(fprich2) − (z(v2b) + z(fprich v1)) の B_N の差の
# 下限 > 0、かつ A の差の下限 > −A_MARGIN（窓の全候補、pid の同点、対のブートストラップ 2,000 回）。
# A_MARGIN = 3.605 × SE、SE は学習の前に測る comb(v1) − v2b の A の対の差の標準誤差（k29 / k37 と同じ考え方）。
# 換算: 相関の高い比較なので B_N の SE は約 0.0055、下限 > 0 は推定値 ≥ 約 +0.011 を求め、縮み 0.18 倍（fprich の
# 窓 → パイプライン、k31）でパイプラインの +0.002、LB で +0.0004。門は先へ進むかだけを決める（採用は k35 の選び方）。
# 入力: k01、k05、k05b、k14（v2_config.json、比べるクエリ）、k15-s1（v2b）、k26（pool_fp2.npy）、casmi26-fprich
# （k28 の fprich.pt。手元では k28 の fprich_ranks.parquet も同じ場所に置き、一致を確かめる）。

# %%
import glob
import json
import time

import numpy as np
import pandas as pd

from casmi26 import cache, chem, fpnet, harness, paths, pool, track

T0 = time.time()
OUT = "/kaggle/working"
EXP = "M7"
PPM = 10.0
N_BOOT = 2000
A_SE_FACTOR = 3.605
BUDGET_H = 10.75
TOTAL_STEPS = None  # None = v2_config.json の値（10 エポック。手元の動作確認で小さくする）
SEED = 1  # k28 は 0（エポックの抽出の順を変える）
MIN_AGREE = 0.98  # v2b / fprich v1 の順位が k15 / k28 と一致する割合の下限（手元の動作確認は窓が違うので 0）
EXPECT_VARIANT = "v2b"
BATCH = None  # None = fpnet.Cfg.batch = 512
RESUME = None  # 中断した学習の続き: 前の出力の fprich2_state.pt の路

cfg = json.load(open(paths.find("v2_config.json")))
if TOTAL_STEPS:
    cfg["total_steps"] = TOTAL_STEPS
cfg["max_per_struct"] = 0  # k15 / k28 と同じ
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
Q = pd.read_csv(f"{PD}/queries.csv")
Q["rows"] = Q.rows.map(lambda s: [int(x) for x in str(s).split(",")])
hold = pd.read_csv(f"{PD}/holdout.csv")
val_q = Q[Q.set == "V"]
a_q = Q[Q.set == "A"]
bn_q = Q[Q.set.isin(["R2", "V"]) & P.in_coconut[Q.pid.values]]
V1_PATH = glob.glob("/kaggle/input/**/fprich.pt", recursive=True)
assert len(V1_PATH) == 1, f"attach exactly one fprich.pt (k28, casmi26-fprich): {V1_PATH}"
V1_PATH = V1_PATH[0]

import torch

print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

# %% [markdown]
# ## 1. v2b と fprich v1 の再現、門の幅（学習の前）

# %%
res_v = [json.load(open(p)) for p in glob.glob("/kaggle/input/**/result_v2*.json", recursive=True)]
assert len(res_v) == 1 and res_v[0]["variant"] == EXPECT_VARIANT, "attach exactly the k15 v2b shard"
MB = fpnet.FPModel(paths.find("fpnet2.pt"))
M1 = fpnet.FPModel(V1_PATH)
assert MB.epoch == res_v[0]["epoch"] and MB.nbits == chem.FP_BITS and M1.nbits == NB
print("v2b epoch", MB.epoch, "| fprich v1 epoch", M1.epoch, "from", V1_PATH)
old = pd.read_parquet(paths.find("fz_ranks_old.parquet"))
qi = Q.set_index("qid").loc[old.qid.values]
specs_all = [harness.query_specs(C, r) for r in qi.rows]
pids = old.pid.values.astype(np.int64)
ZB, Z1 = MB.zs(specs_all), M1.zs(specs_all)
del MB, M1
torch.cuda.empty_cache()


def zs_(x):
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-9 else np.zeros_like(x)


cands, F2s, SB = [], [], []
rk = {k: [] for k in ["v2b", "rich1", "comb1"]}
for i, p in enumerate(pids):
    cand = P.window(P.mass[p], PPM)
    F2 = np.unpackbits(FP2[cand], axis=1)[:, :NB].astype(np.float32)
    sb, s1 = F2[:, :chem.FP_BITS] @ ZB[i], F2 @ Z1[i]
    cands.append(cand)
    SB.append(zs_(sb))
    rk["v2b"].append(fpnet.truth_ranks(sb, cand, p)[1])
    rk["rich1"].append(fpnet.truth_ranks(s1, cand, p)[1])
    rk["comb1"].append(fpnet.truth_ranks(SB[-1] + zs_(s1), cand, p)[1])
ref15 = pd.read_parquet(paths.find(f"fz_ranks_{EXPECT_VARIANT}.parquet")).set_index("qid").loc[old.qid.values]
agree = dict(v2b=float((ref15.r_pid.values == np.asarray(rk["v2b"])).mean()))
try:
    k28 = pd.read_parquet(paths.find("fprich_ranks.parquet")).set_index("qid").loc[old.qid.values]
    agree.update(rich1=float((k28.rich.values == np.asarray(rk["rich1"])).mean()),
                 comb1=float((k28.comb.values == np.asarray(rk["comb1"])).mean()))
except FileNotFoundError:
    print("k28 fprich_ranks.parquet not attached: fprich v1 agreement not checked")
print("agreement:", agree, f"({time.time() - T0:.0f}s)")
assert all(v >= MIN_AGREE for v in agree.values()), "inputs do not reproduce k15 / k28"
is_np = P.in_coconut[pids]
G = {s: (old.set == s).values for s in ["A", "V", "R2", "C"]}
G["B_N"] = old.set.isin(["R2", "V"]).values & is_np
rr0 = {k: harness.rr_of(np.asarray(v, np.float64)) for k, v in rk.items()}
a_se = float(harness.paired_ci(rr0["comb1"][G["A"]], rr0["v2b"][G["A"]], N_BOOT, seed=1)["se"])
A_MARGIN = A_SE_FACTOR * a_se
spec = dict(a_margin=A_MARGIN, a_se_comb1_minus_v2b=a_se, a_se_factor=A_SE_FACTOR, n_a=int(G["A"].sum()),
            rule="B_N lo > 0 and A lo > -a_margin for z(v2b) + z(fprich2) - (z(v2b) + z(fprich v1))", agree=agree,
            fixed_at_secs=round(time.time() - T0), seed=SEED, init=V1_PATH)
json.dump(spec, open(f"{OUT}/gate_e2_spec.json", "w"), indent=1)
print(f"gate E2 fixed before training: A lower bound > -{A_MARGIN:.4f} (= {A_SE_FACTOR} x SE {a_se:.4f} of comb(v1) - "
      f"v2b on A), B_N lower bound > 0", flush=True)

# %% [markdown]
# ## 2. 学習（k28 の重みから、新しい warmup + cosine）

# %%
run = track.init("train", group=EXP, config=dict(
    {k: getattr(fpnet.Cfg, k) for k in dir(fpnet.Cfg) if not k.startswith("_")},
    variant="fprich2", nbits=NB, budget_h=BUDGET_H, n_holdout=len(hold), init=V1_PATH, seed=SEED))
hist = fpnet.train(C, P, set(hold.pid.values), val_q, f"{OUT}/fprich2.pt", budget_secs=BUDGET_H * 3600,
                   extra_val={"A": a_q, "B_N": bn_q}, track=run, select="last", eval_every=1, fp=FP2, nbits=NB,
                   resume=RESUME, init=V1_PATH, seed=SEED)
hist = pd.DataFrame(hist)
hist.to_csv(f"{OUT}/fprich2_history.csv", index=False)
print(hist.round(4).to_string(index=False))
run.table("history", hist)

# %% [markdown]
# ## 3. 採点と門 E2

# %%
M2 = fpnet.FPModel(f"{OUT}/fprich2.pt")
print("fprich2 epoch", M2.epoch)
Z2 = M2.zs(specs_all)
rk.update({k: [] for k in ["rich2", "comb2"]})
for i, p in enumerate(pids):
    F2 = np.unpackbits(FP2[cands[i]], axis=1)[:, :NB].astype(np.float32)
    s2 = F2 @ Z2[i]
    rk["rich2"].append(fpnet.truth_ranks(s2, cands[i], p)[1])
    rk["comb2"].append(fpnet.truth_ranks(SB[i] + zs_(s2), cands[i], p)[1])
ranks = pd.DataFrame({"qid": old.qid.values, "set": old.set.values, "pid": pids,
                      **{k: np.asarray(v, np.float64) for k, v in rk.items()}})
ranks.to_parquet(f"{OUT}/fprich2_ranks.parquet", index=False)
rr = {k: harness.rr_of(ranks[k].values) for k in rk}
rows = []
for g, m in G.items():
    for a, b in [("comb2", "comb1"), ("rich2", "rich1"), ("comb2", "v2b"), ("comb1", "v2b")]:
        rows.append(dict(group=g, new=a, ref=b, mrr_new=rr[a][m].mean(), mrr_ref=rr[b][m].mean(),
                         **harness.paired_ci(rr[a][m], rr[b][m], N_BOOT, seed=1)))
cmp = pd.DataFrame(rows)
cmp.to_csv(f"{OUT}/compare_fprich2.csv", index=False)
print("window MRR@25 (all candidates, pid ties):")
print(cmp.round(4).to_string(index=False))
c = cmp[(cmp.new == "comb2") & (cmp.ref == "comb1")].set_index("group")
gate = bool(c.loc["B_N", "lo"] > 0 and c.loc["A", "lo"] > -A_MARGIN)
res = dict(gate=gate, a_margin=A_MARGIN, epoch=int(M2.epoch), agree=agree, compare=cmp.to_dict("records"))
json.dump(res, open(f"{OUT}/result_fprich2.json", "w"), indent=1, default=float)
print(f"gate E2: {'PASS' if gate else 'FAIL'} — comb(fprich2) - comb(fprich v1): B_N lower bound {c.loc['B_N', 'lo']:+.4f} "
      f"> 0 and A lower bound {c.loc['A', 'lo']:+.4f} > -{A_MARGIN:.4f}")
run.summary(dict(gate=gate, **{f"{r.new}-{r.ref}_{r.group}_{k}": float(getattr(r, k))
                               for r in cmp.itertuples() for k in ["est", "lo"]}))
run.finish()
print(f"total {time.time() - T0:.0f}s")

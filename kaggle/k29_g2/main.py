# %% [markdown]
# # casmi26-k29-g2
# M6 の P0-3（2026-10-12、GPU）: 断片ごとの強度を学ぶ順方向の予測器（G2、ICEBERG 型、自前の学習だけ）。
# - 断片は frag.py と同じ定義（環に入らない単結合 1〜2 本、同じ非芳香環の 2 本、30 Da 以上）で、原子の集合を持つ
#   （1 構造あたり最大 512 個、1 本切りを先に）。重原子が 160 を超える構造と RDKit で読めない構造は点数なし（z = 0）
# - モデル: 原子と結合の MPNN 4 層（d=256）→ 断片の中の原子の平均、切った結合、分子の平均、大きさ、条件（付加イオン、
#   極性、CE）から H のずれ −2〜+2 ごとの logit → 構造の（断片 × ずれ）全体の softmax が予測強度
# - コサイン: 観測はプリカーサー −1.5 Da 未満の上位 64 ピーク（sqrt 強度）。予測のイオンは「断片の質量 + k·H ± proton」、
#   同じ m/z（1e-4 Da）はまとめ、max(10 ppm, 2 mDa) で最も近い観測ピークに割り当てる（subform と同じ許容）
# - 学習: (1 − 正解のコサイン) + 正解と同じ分子式の train の構造 16 件との窓の中の softmax（温度 0.05）。
#   v2b の抽出（上限なし、天然物 3 倍、負例は train だけ）、64 スペクトル × 17 構造、総ステップは g2.Cfg（fit_horizon で
#   枠に収める）。2,000 ステップごとに状態を保存し、RESUME で続けられる
# - 順番: (1) fz（v2b）と z(fz) + z(fwd v1) の順位が k15 / k20 と一致することを学習の前に確かめる
#   (2) 学習していないモデルで G2 の採点の時間を測り、学習の枠を決める (3) 学習 (4) 採点と門
# 門 G2（計画 2026-10-11 で固定。A の幅だけ 2026-10-12 にユーザーが決め直した。k29 の結果を誰も見ていないときの決定）:
# z(fz) + z(fwd1) + z(g2) − (z(fz) + z(fwd1)) の B_N の差の下限 > 0、かつ A の差の下限 > −A_MARGIN（窓の全候補、pid の
# 同点、対のブートストラップ 2,000 回）。A_MARGIN = 3.605 × SE で、SE は学習の前に再現の段で測る comb1 − fz の A の対の
# 差の標準誤差（新しい信号を 1 つ足す比較の SE。本当は A が変わらない案が 95% 通る幅）。g2 の点数が 1 つも無いうちに
# 決めて gate_g2_spec.json に書く。窓の評価の SE（約 0.02）では、元の幅 −0.02 だと A が変わらない案が約 16% しか
# 通らないため（REPORT 2.5.1）。g2 だけの順位、V、R2、C は報告だけ。
# 入力: k01、k05、k05b、k14（比べるクエリ）、k15-s1（v2b）、k20（fwd v1）、k26（分子式）、casmi26-wheels（rdkit）。

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
EXP = "M6"
PPM = 10.0
N_BOOT = 2000
A_SE_FACTOR = 3.605  # 1.96 + 1.645: 本当は A が変わらない案が 95% 通る幅（門 G2 の A、上の説明）
TOTAL_H = 9.0  # このカーネル全体の枠
TOTAL_STEPS = None  # None = g2.Cfg.total_steps（手元の動作確認で小さくする）
BATCH, K_NEG, WORKERS = None, None, None  # None = g2.Cfg
FIT = None  # 手元の動作確認だけ: enc3.FIT_FROM / FIT_AT
MIN_AGREE = 0.98  # fz と v1 の組み合わせの順位が k15 / k20 と一致する割合の下限（手元の動作確認は窓が違うので 0）
MAX_QUERIES = None  # 手元の動作確認だけ
EVAL_PROBE = 30  # 採点の時間を測るクエリの数
GRAPH_WORKERS = 4
RESUME = None  # 中断した学習の続き: 前の出力の g2_state.pt の路
EXPECT_VARIANT = "v2b"

fpnet.Cfg.max_per_struct, fpnet.Cfg.np_weight, fpnet.Cfg.neg_train_only = 0, 3, True  # v2b の抽出
for k, v in [("total_steps", TOTAL_STEPS), ("batch", BATCH), ("k_neg", K_NEG), ("workers", WORKERS)]:
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

# %% [markdown]
# ## 1. fz（v2b）と fwd v1 の順位の再現（学習の前）

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
    """z-score over the finite values; NaN (no G2 score) -> 0."""
    x = np.asarray(x, np.float64)
    f = np.isfinite(x)
    out = np.zeros_like(x)
    if f.sum() > 1 and x[f].std() > 1e-9:
        out[f] = (x[f] - x[f].mean()) / x[f].std()
    return out


cands, S_FZ, S_F1 = [], [], []
rk = {k: [] for k in ["fz", "comb1"]}
for i, q in enumerate(qi.itertuples()):
    cand = P.window(P.mass[q.pid], PPM)
    s_fz = np.unpackbits(P.fp[cand], axis=1)[:, :Z.shape[1]].astype(np.float32) @ Z[i]
    s_f1 = M1.score(specs_all[i], P.fp[cand])
    cands.append(cand)
    S_FZ.append(s_fz)
    S_F1.append(s_f1)
    rk["fz"].append(fpnet.truth_ranks(s_fz, cand, q.pid)[1])
    rk["comb1"].append(fpnet.truth_ranks(zs(s_fz) + zs(s_f1), cand, q.pid)[1])
    if (i + 1) % 1000 == 0:
        print(f"  {i + 1}/{len(qi)} queries ({time.time() - T0:.0f}s)", flush=True)
k15 = pd.read_parquet(paths.find(f"fz_ranks_{EXPECT_VARIANT}.parquet")).set_index("qid").loc[old.qid.values]
k20 = pd.read_parquet(paths.find("fwd_ranks.parquet")).set_index("qid").loc[old.qid.values]
agree_fz = float((k15.r_pid.values == np.asarray(rk["fz"])).mean())
agree_c1 = float((k20.r_comb.values == np.asarray(rk["comb1"])).mean())
n_cand = np.array([len(c) for c in cands])
print(f"fz ranks agree with k15 on {agree_fz:.4f}, z(fz) + z(fwd v1) ranks agree with k20 on {agree_c1:.4f}; "
      f"window candidates median {np.median(n_cand):.0f}, total {n_cand.sum():,} ({time.time() - T0:.0f}s)")
assert agree_fz >= MIN_AGREE and agree_c1 >= MIN_AGREE, "inputs do not reproduce k15 / k20"
# 門 G2 の A の幅（学習の前、g2 の点数が無いうちに固定する）
rr0 = {k: harness.rr_of(np.asarray(v, np.float64)) for k, v in rk.items()}
a_se = float(harness.paired_ci(rr0["comb1"][G["A"]], rr0["fz"][G["A"]], N_BOOT, seed=1)["se"])
A_MARGIN = A_SE_FACTOR * a_se
spec = dict(a_margin=A_MARGIN, a_se_comb1_minus_fz=a_se, a_se_factor=A_SE_FACTOR, n_a=int(G["A"].sum()),
            rule="B_N lo > 0 and A lo > -a_margin for z(fz) + z(fwd1) + z(g2) - (z(fz) + z(fwd1))",
            fixed_at_secs=round(time.time() - T0))
json.dump(spec, open(f"{OUT}/gate_g2_spec.json", "w"), indent=1)
print(f"gate G2 fixed before training: A lower bound > -{A_MARGIN:.4f} (= {A_SE_FACTOR} x SE {a_se:.4f} of comb1 - fz "
      f"on A), B_N lower bound > 0", flush=True)
del FP, M1, Z
torch.cuda.empty_cache()

# %% [markdown]
# ## 2. 採点の時間を測り、学習の枠を決める

# %%
dev = "cuda" if torch.cuda.is_available() else "cpu"
probe = np.random.default_rng(0).choice(len(qi), min(EVAL_PROBE, len(qi)), replace=False)
m0 = g2.G2Model(G=GR, net=g2.build_model().to(dev).eval())
m0.score(specs_all[probe[0]], cands[probe[0]])  # numba のコンパイルと GPU の立ち上がりを測らない
t = time.time()
for i in probe:
    m0.score(specs_all[i], cands[i])
per_cand = (time.time() - t) / max(n_cand[probe].sum(), 1)
eval_secs = per_cand * n_cand.sum()
budget = TOTAL_H * 3600 - (time.time() - T0) - 1.25 * eval_secs - 900
print(f"g2 scoring: {per_cand * 1e3:.3f} ms per candidate -> {eval_secs / 60:.0f} min for all windows; "
      f"training budget {budget / 3600:.2f} h")
del m0

# %% [markdown]
# ## 3. 学習

# %%
D = fpnet.TrainData(C, P, set(hold.pid.values))
NEG = enc3.SameFormulaNegatives(P, FT, set(hold.pid.values))
run = track.init("train", group=EXP, config=dict(
    {f"g2/{k}": getattr(g2.Cfg, k) for k in dir(g2.Cfg) if not k.startswith("_")},
    budget_h=budget / 3600, max_frag=g2.MAX_FRAG, max_atoms=g2.MAX_ATOMS, ppm=g2.PPM, abs_da=g2.ABS_DA))
hist = g2.train(C, D, GR, NEG, f"{OUT}/g2.pt", budget_secs=budget, track=run, resume=RESUME)
hist = pd.DataFrame(hist)
hist.to_csv(f"{OUT}/g2_history.csv", index=False)
print(hist.round(4).to_string(index=False))
del D

# %% [markdown]
# ## 4. 採点と門 G2

# %%
M = g2.G2Model(f"{OUT}/g2.pt", G=GR)
print("g2 step", M.step)
rk.update({k: [] for k in ["g2", "comb3"]})
n_nan, n_truth_nan, t = 0, 0, time.time()
for i, q in enumerate(qi.itertuples()):
    s_g2 = M.score(specs_all[i], cands[i])
    n_nan += int(np.isnan(s_g2).sum())
    n_truth_nan += int(not GR.ok[q.pid])
    rk["g2"].append(fpnet.truth_ranks(np.nan_to_num(s_g2, nan=-1.0), cands[i], q.pid)[1])
    rk["comb3"].append(fpnet.truth_ranks(zs(S_FZ[i]) + zs(S_F1[i]) + zs(s_g2), cands[i], q.pid)[1])
    if (i + 1) % 1000 == 0:
        print(f"  {i + 1}/{len(qi)} queries ({time.time() - t:.0f}s)", flush=True)
ranks = pd.DataFrame({"qid": old.qid.values, "set": old.set.values, "pid": old.pid.values,
                      **{f"r_{k}": np.asarray(v, np.float64) for k, v in rk.items()}})
ranks.to_parquet(f"{OUT}/g2_ranks.parquet", index=False)
rr = {k: harness.rr_of(ranks[f"r_{k}"].values) for k in rk}
rows = []
for g, m in G.items():
    for a, b in [("comb3", "comb1"), ("g2", "fz"), ("comb1", "fz")]:
        rows.append(dict(group=g, new=a, ref=b, mrr_new=rr[a][m].mean(), mrr_ref=rr[b][m].mean(),
                         **harness.paired_ci(rr[a][m], rr[b][m], N_BOOT, seed=1)))
cmp = pd.DataFrame(rows)
cmp.to_csv(f"{OUT}/compare_g2.csv", index=False)
print("window MRR@25 (all candidates, pid ties):")
print(cmp.round(4).to_string(index=False))
c = cmp[(cmp.new == "comb3") & (cmp.ref == "comb1")].set_index("group")
gate = bool(c.loc["B_N", "lo"] > 0 and c.loc["A", "lo"] > -A_MARGIN)
cover = dict(graph_share=float(GR.ok.mean()), window_no_graph=n_nan / max(n_cand.sum(), 1),
             truth_no_graph=n_truth_nan / len(qi), train_capped=float(hist.capped.mean()) if len(hist) else None,
             train_ceiling=float(hist.ceiling.iloc[-1]) if len(hist) else None,
             train_zero_ceiling=float(hist.zero_ceiling.mean()) if len(hist) else None)
res = dict(gate=gate, a_margin=A_MARGIN, step=int(M.step), agree_fz=agree_fz, agree_comb1=agree_c1, coverage=cover,
           compare=cmp.to_dict("records"))
json.dump(res, open(f"{OUT}/result_g2.json", "w"), indent=1, default=float)
print("coverage:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in cover.items()})
print(f"gate G2: {'PASS' if gate else 'FAIL'} — z(fz) + z(fwd1) + z(g2) - (z(fz) + z(fwd1)): B_N lower bound "
      f"{c.loc['B_N', 'lo']:+.4f} > 0 and A lower bound {c.loc['A', 'lo']:+.4f} > -{A_MARGIN:.4f}")
run.summary(dict(gate=gate, **{f"{r.new}-{r.ref}_{r.group}_{k}": float(getattr(r, k))
                               for r in cmp.itertuples() for k in ["est", "lo"]}))
run.finish()
print(f"total {time.time() - T0:.0f}s")

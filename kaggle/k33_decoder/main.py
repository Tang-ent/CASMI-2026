# %% [markdown]
# # casmi26-k33-decoder
# M6 の P0-4 の学習（2026-10-12、GPU）: クラス 3 の構造デコーダ（casmi26.decoder）を学習する。指紋のロジット
# （Morgan r2 2048、v2b の z の形）と中性の分子式から SMILES を出す GPT 型の decoder。
# - 学習データ: プールの構造（train ∪ COCONUT）から FPNet のホールドアウト（A、C、R、V の構造）を除いたもの。
#   立体を外した正準 SMILES と、ランダムな SMILES を合わせて 1 構造 5 本
# - 条件の指紋: 構造の指紋に雑音を入れたロジット。雑音の強さは学習の前に一度だけ、R1 のクエリでの v2b の z
#   （正解のビットが立つところと立たないところの平均と標準偏差）に合わせる。R1 は FPNet のホールドアウトで、
#   B_N にも G_N にも入らないので、門 D の判定に響かない（V は G_N に入るので使わない）
# - 総ステップ 60,000（バッチ 256）、最後のチェックポイント。R1 の 2,000 構造（本物の v2b の z を条件）の
#   教師ありの損失は監視だけ
# - 最後に報告だけ: R1 の 300 クエリで、正しい分子式を条件に 128 本を引き、読める割合、分子式が合う割合、正解が
#   入る割合
# 門 D の判定は k34_dec_eval で行う。
# 入力: k01、k05、k05b、k15-s1（v2b）、k26（分子式）、casmi26-wheels（rdkit）。

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

from casmi26 import cache, decoder, derive, enc3, fpnet, harness, paths, pool, subform, track

T0 = time.time()
OUT = "/kaggle/working"
EXP = "M6"
WORKERS = 4
BUDGET_H = 8.0
N_VAL = 2000
N_REPORT, N_REPORT_SAMPLES = 300, 128
TOTAL_STEPS = None  # None = decoder.Cfg.total_steps（手元の動作確認で小さくする）
BATCH = None
EXPECT_VARIANT = "v2b"
FIT = None  # 手元の動作確認だけ: 1 ステップの時間を測る区間（既定は enc3.FIT_FROM, enc3.FIT_AT = 100, 600）
if FIT:
    enc3.FIT_FROM, enc3.FIT_AT = FIT
if TOTAL_STEPS:
    decoder.Cfg.total_steps = TOTAL_STEPS
if BATCH:
    decoder.Cfg.batch = BATCH

C = cache.Cache(cache.Cache.find_dir())
PD = pool.Pool.find_dir()
P = pool.Pool(PD, pubchem=False)
FT = subform.FormulaTable(subform.FormulaTable.find_dir())
assert len(FT.fid) == len(P)
Q = pd.read_csv(f"{PD}/queries.csv")
Q["rows"] = Q.rows.map(lambda s: [int(x) for x in str(s).split(",")])
hold = pd.read_csv(f"{PD}/holdout.csv")
hold_set = set(hold.pid.values)
res_v = [json.load(open(p)) for p in glob.glob("/kaggle/input/**/result_v2*.json", recursive=True)]
assert len(res_v) == 1 and res_v[0]["variant"] == EXPECT_VARIANT, "attach exactly the k15 v2b shard"
MB = fpnet.FPModel(paths.find("fpnet2.pt"))
assert MB.epoch == res_v[0]["epoch"]
print("v2b epoch", MB.epoch, "| device", MB.device)

# %% [markdown]
# ## 学習データ（ランダムな SMILES）と語彙

# %%
train_pids = np.array([p for p in range(len(P)) if p not in hold_set and FT.fid[p] >= 0 and FT.supported[FT.fid[p]]])
variants = decoder.random_smiles_many(P.smiles[train_pids].tolist(), decoder.Cfg.n_random, WORKERS)
vocab = decoder.build_vocab([decoder.tokenize(v[0]) for v in variants if v])
stoi = {t: i for i, t in enumerate(vocab)}
SD = decoder.build_seqdata(variants, train_pids, stoi, decoder.Cfg.max_len)
n_parsed = sum(1 for v in variants if v)
print(f"training structures {len(train_pids):,} (parsed {n_parsed:,}), vocab {len(vocab)}, "
      f"structures with a sequence {len(set(SD.pid.tolist())):,} ({time.time() - T0:.0f}s)", flush=True)
del variants

# %% [markdown]
# ## 雑音の強さ（R1 の v2b の z、学習の前に一度だけ）と監視用の集合

# %%
r1 = Q[Q.set == "R1"].drop_duplicates("pid")
specs_r1 = [harness.query_specs(C, r) for r in r1.rows]
Z1 = MB.zs(specs_r1)
F1 = np.unpackbits(P.fp[r1.pid.values], axis=1)[:, :2048].astype(np.float32)
noise = decoder.noise_stats(Z1, F1)
print("noise (R1, v2b logits):", {k: round(v, 3) for k, v in noise.items()}, "| R1 structures", len(r1))
rng = np.random.default_rng(0)
vi = np.sort(rng.choice(len(r1), min(N_VAL, len(r1)), replace=False))
val_ids, val_z, val_f = [], [], []
for i in vi:
    p = int(r1.pid.values[i])
    v = decoder.random_smiles(P.smiles[p], 1)
    e = decoder.encode(decoder.tokenize(v[0]), stoi, decoder.Cfg.max_len) if v else None
    if e is None or FT.fid[p] < 0:
        continue
    val_ids.append(np.r_[decoder.BOS, e, decoder.EOS])
    val_z.append(decoder.cond_z(Z1[i]))
    val_f.append(decoder.cond_formula(FT.counts[FT.fid[p]], FT.supported[FT.fid[p]]))
T = max(len(x) for x in val_ids)
val_arr = np.zeros((len(val_ids), T), np.int64)
for j, x in enumerate(val_ids):
    val_arr[j, :len(x)] = x
val = (np.stack(val_z), np.stack(val_f), val_arr)
print("monitoring set:", len(val_ids), "R1 structures")

import torch

print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

# %% [markdown]
# ## 学習

# %%
run = track.init("train", group=EXP, config=dict(
    {f"dec/{k}": getattr(decoder.Cfg, k) for k in dir(decoder.Cfg) if not k.startswith("_")},
    noise=noise, n_train_structures=int(len(train_pids)), n_seq=len(SD), vocab=len(vocab), budget_h=BUDGET_H))
hist = decoder.train(SD, P.fp, FT, vocab, noise, f"{OUT}/decoder.pt", val=val, budget_secs=BUDGET_H * 3600,
                     track=run)
hist = pd.DataFrame(hist)
hist.to_csv(f"{OUT}/decoder_history.csv", index=False)
print(hist.round(4).to_string(index=False))

# %% [markdown]
# ## 報告だけ: R1 で正しい分子式を条件に引いたときの中身

# %%
from rdkit import Chem, RDLogger
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

RDLogger.DisableLog("rdApp.*")
DM = decoder.DecoderModel(f"{OUT}/decoder.pt")
rows = []
for i in vi[:N_REPORT]:
    p = int(r1.pid.values[i])
    f = FT.fid[p]
    if f < 0:
        continue
    t0 = time.time()
    out = DM.sample(Z1[i], [(FT.counts[f], FT.supported[f])], [N_REPORT_SAMPLES], seed=int(i))
    tk = derive.ik14(P.smiles[p])
    n_ok = n_form = 0
    keys = set()
    for _, smi in out:
        m = Chem.MolFromSmiles(smi) if smi else None
        if m is None:
            continue
        n_ok += 1
        v, _ = subform.parse(CalcMolFormula(m))
        if (v == FT.counts[f]).all():
            n_form += 1
            k = derive.ik14(smi)
            if k:
                keys.add(k)
    rows.append(dict(pid=p, valid=n_ok / len(out), formula_ok=n_form / len(out), unique=len(keys), hit=tk in keys,
                     secs=time.time() - t0))
rep = pd.DataFrame(rows)
rep.to_csv(f"{OUT}/decoder_r1_report.csv", index=False)
summ = dict(valid=rep.valid.mean(), formula_ok=rep.formula_ok.mean(), unique_median=rep.unique.median(),
            hit=rep.hit.mean(), secs=rep.secs.mean(), n=len(rep), steps=int(DM.step))
print("R1 report (true formula, 128 samples):", {k: round(float(v), 4) for k, v in summ.items()})
json.dump(dict(summ, noise=noise, vocab=len(vocab), n_train_structures=int(len(train_pids)), n_seq=len(SD)),
          open(f"{OUT}/decoder_info.json", "w"), indent=1, default=float)
run.summary({f"r1_{k}": float(v) for k, v in summ.items()})
run.finish()
print(f"total {time.time() - T0:.0f}s")

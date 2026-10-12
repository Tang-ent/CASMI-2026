# %% [markdown]
# # casmi26-k34-dec-eval
# M6 の P0-4 の評価（2026-10-12、GPU）: k33 のデコーダの生成物を、反応による生成（casmi26.gen、k22 と同じ手順）に
# 合流させたとき、クラス 3 の代わりのセットで正解の順位が上がるかを測る。
# - G_N = B_N と同じクエリ（R2 ∪ V で正解が COCONUT にあるもの、正解のスペクトルはすべてのライブラリから隠す）、
#   G_A = B_A と同じクエリ。窓の候補から正解を除き（W）、生成物の鍵は窓の鍵と重ならないものだけ（正解の鍵は残して
#   印をつける。k22 と同じ）
# - 反応の一覧 R: gen.for_query（親は類縁体と目標 − Δ のプールの構造、最大 500 件）
# - デコーダの一覧 D: W にある分子式のうち候補の多い順に最大 8 個、512 本を候補の数に比例して割り振り（1 つに最低 32 本）、
#   v2b の z と分子式を条件に引く。RDKit で読めて分子式が合うものだけ、標準の InChIKey14 で重複を除き、窓の鍵と R の鍵を除く
# - 引き方（ユーザーの決定 2026-10-12、k33 の報告を見た後。門 D は変えない）: 分子式を守る引き方。引く途中で重原子を
#   元素ごとに数え、目標の分子式を超えるトークンと、数がそろう前の終わりを禁じる（decoder.DecoderModel.sample の
#   constrain）。k33 の報告では、制約なしで分子式が合うのは 4.6% だった。H は制約せず、生成物の分子式で確かめる
# - 点数: k19 と同じ z(fz) + z(ap4)（一覧ごとに z を計算し直す）。正解の順位は同点を不利に数え、一覧に無ければ圏外
# 門 D（固定、計画 2026-10-11）: G_N で、W ∪ R ∪ D の MRR@25 − W ∪ R の MRR@25 が +0.02 以上で、下限 > 0。
# 報告だけ: G_A、正解とデコーダの学習集合の最大 Tanimoto（0.7 未満 / 以上）で分けた G_N、デコーダだけ（W ∪ D）の MRR、
# 生成物が読める割合・分子式が合う割合・正解が入る割合。
# 入力: k01、k05、k05b、k15-s1（v2b）、k26（分子式）、k33（decoder.pt）、casmi26-wheels（rdkit）。

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
from rdkit import Chem, RDLogger
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from casmi26 import cache, chem, decoder, derive, fpnet, gen, harness, paths, pool, search, subform, track

RDLogger.DisableLog("rdApp.*")
T0 = time.time()
OUT = "/kaggle/working"
EXP = "M6"
N_BOOT = 2000
GATE_D = 0.02
N_SAMPLES, MAX_FORMULAS, MIN_PER = 512, 8, 32
BUDGET_SECS = 8 * 3600
SEARCH_CFG = search.SearchCfg()  # 手元の動作確認では窓を広げる
MAX_QUERIES = None  # 手元の動作確認だけ
CONSTRAIN = True  # 分子式を守る引き方（上の説明）
INJECT_TRUTH = False  # 手元の動作確認だけ: 正解の SMILES を生の出力に足して、生成物を並べる道を通す
EXPECT_VARIANT = "v2b"

C = cache.Cache(cache.Cache.find_dir())
PD = pool.Pool.find_dir()
P = pool.Pool(PD, pubchem=False)
FT = subform.FormulaTable(subform.FormulaTable.find_dir())
Q = pd.read_csv(f"{PD}/queries.csv")
Q["rows"] = Q.rows.map(lambda s: [int(x) for x in str(s).split(",")])
hold = pd.read_csv(f"{PD}/holdout.csv")
res_v = [json.load(open(p)) for p in glob.glob("/kaggle/input/**/result_v2*.json", recursive=True)]
assert len(res_v) == 1 and res_v[0]["variant"] == EXPECT_VARIANT, "attach exactly the k15 v2b shard"
MB = fpnet.load_checked("fpnet2.pt", min_epoch=0)  # 版は result_v2b.json のエポックで確かめる
assert MB.epoch == res_v[0]["epoch"]
DM = decoder.DecoderModel(paths.find("decoder.pt"))
print("v2b epoch", MB.epoch, "| decoder step", DM.step, "| device", DM.device)
S = search.Searcher(C, P, cfg=SEARCH_CFG, model=MB)
gq = Q[Q.set.isin(["R2", "V"]) & P.in_coconut[Q.pid.values]].assign(g="G_N")
ga = Q[Q.set == "B_A"].assign(g="G_A")
QQ = pd.concat([ga, gq.sample(frac=1.0, random_state=0)], ignore_index=True)  # G_A first, G_N in random order
if MAX_QUERIES:
    QQ = QQ.groupby("g").head(MAX_QUERIES).reset_index(drop=True)
print(QQ.g.value_counts().to_dict())
C_FZ, C_AP4 = search.COL["fz"], search.COL["ap4"]


def zs_(x):
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-9 else np.zeros_like(x)


def rank_pess(score, label):
    """1-based rank of the labelled row with ties against it (inf when absent)."""
    if not label.any():
        return np.inf
    t = score[label].max()
    return 1 + int((score[~label] >= t).sum())


def ap4_of(F, apid, asim):
    if not len(apid) or not len(F):
        return np.zeros(len(F), np.float32)
    A = np.unpackbits(P.fp[apid], axis=1)[:, :chem.FP_BITS].astype(np.float32)
    inter = F @ A.T
    tan = inter / (F.sum(1)[:, None] + A.sum(1)[None, :] - inter + 1e-9)
    return (tan * (np.clip(asim, 0, None) ** S.cfg.sim_power)[None, :]).max(1)


# %% [markdown]
# ## クエリごとに W、R、D を作って並べる

# %%
rows = []
for qn, q in enumerate(QQ.itertuples()):
    if time.time() - T0 > BUDGET_SECS:
        print(f"budget reached after {qn} queries")
        break
    t0 = time.time()
    specs = harness.query_specs(C, q.rows)
    cand, X, info = S.features(specs, dict(hide_lib=q.hide_lib, hide_pid=q.hide_pid, hide_pid_lib=q.hide_pid_lib))
    if len(cand) == 0 or S.last["z"] is None:
        continue
    z, apid, asim = S.last["z"], S.last["apid"], S.last["asim"]
    tk = derive.ik14(P.smiles[q.pid])
    prods, Xg, st = gen.for_query(S, specs, cand, info, keep_key=tk)
    t1 = time.time()
    w = cand != q.pid
    Wc = cand[w]
    fz_w, ap_w = X[w, C_FZ] * 100.0, X[w, C_AP4]
    fz_r = Xg[:, gen.GCOL["fz"]] * 100.0 if len(prods) else np.zeros(0)
    ap_r = Xg[:, gen.GCOL["ap4"]] if len(prods) else np.zeros(0)
    lab_r = np.array([p["key"] == tk for p in prods], bool)
    # デコーダ: W の分子式（候補の多い順）
    fw = FT.fid[Wc]
    uf, cnt = np.unique(fw[fw >= 0], return_counts=True)
    o = np.argsort(-cnt, kind="mergesort")
    uf, cnt = uf[o][:MAX_FORMULAS], cnt[o][:MAX_FORMULAS]
    n_per = decoder.split_samples(cnt.tolist(), N_SAMPLES, MAX_FORMULAS, MIN_PER)
    forms = [(FT.counts[f], FT.supported[f]) for f in uf]
    raw = DM.sample(z, forms, n_per, seed=int(q.qid), constrain=CONSTRAIN) if len(uf) else []
    if INJECT_TRUTH and len(uf) and (uf == FT.fid[q.pid]).any():
        raw = raw + [(int(np.flatnonzero(uf == FT.fid[q.pid])[0]), P.smiles[q.pid])]
    win_keys = {derive.ik14(P.smiles[c]) for c in cand}
    seen = {p["key"] for p in prods}
    dsmi, dkeys = [], []
    n_valid = n_form = 0
    for j, smi in raw:
        m = Chem.MolFromSmiles(smi) if smi else None
        if m is None:
            continue
        n_valid += 1
        v, _ = subform.parse(CalcMolFormula(m))
        if not (v == FT.counts[uf[j]]).all():
            continue
        n_form += 1
        k = derive.ik14(smi)
        if k is None or k in seen or (k in win_keys and k != tk):
            continue
        seen.add(k)
        dsmi.append(smi)
        dkeys.append(k)
    lab_d = np.array([k == tk for k in dkeys], bool)
    if dsmi:
        Fd = np.unpackbits(np.stack([chem.morgan_packed(s) for s in dsmi]), axis=1)[:, :chem.FP_BITS].astype(np.float32)
        fz_d, ap_d = Fd @ z, ap4_of(Fd, apid, asim)
    else:
        fz_d, ap_d = np.zeros(0), np.zeros(0)
    lab_w = np.zeros(len(Wc), bool)

    def rank_of(parts):
        fz = np.concatenate([p[0] for p in parts])
        ap = np.concatenate([p[1] for p in parts])
        lab = np.concatenate([p[2] for p in parts])
        return rank_pess(zs_(fz) + zs_(ap), lab)

    Wp, Rp, Dp = (fz_w, ap_w, lab_w), (fz_r, ap_r, lab_r), (fz_d, ap_d, lab_d)
    rows.append(dict(qid=q.qid, g=q.g, set=q.set, pid=q.pid, n_w=len(Wc), n_r=len(prods), n_d=len(dsmi),
                     n_raw=len(raw), valid=n_valid / max(len(raw), 1), formula_ok=n_form / max(len(raw), 1),
                     n_formulas=len(uf), truth_formula_in_w=bool((fw == FT.fid[q.pid]).any()),
                     in_r=bool(lab_r.any()), in_d=bool(lab_d.any()),
                     base=rank_of([Wp, Rp]), union=rank_of([Wp, Rp, Dp]), dec_only=rank_of([Wp, Dp]),
                     secs_feat_gen=t1 - t0, secs_dec=time.time() - t1))
    if (qn + 1) % 100 == 0:
        d = pd.DataFrame(rows)
        print(f"  {qn + 1}/{len(QQ)} ({time.time() - T0:.0f}s) in_r {d.in_r.mean():.3f} in_d {d.in_d.mean():.3f} "
              f"valid {d.valid.mean():.3f} formula_ok {d.formula_ok.mean():.3f} n_d {d.n_d.median():.0f} "
              f"dec {d.secs_dec.mean():.2f}s", flush=True)
R = pd.DataFrame(rows)
R.to_parquet(f"{OUT}/dec_ranks.parquet", index=False)

# %% [markdown]
# ## 正解とデコーダの学習集合の最大 Tanimoto（報告の層）

# %%
import torch

dev = "cuda" if torch.cuda.is_available() else "cpu"
hs = set(hold.pid.values)
train_pids = np.array([p for p in range(len(P)) if p not in hs and FT.fid[p] >= 0 and FT.supported[FT.fid[p]]])
truth = R.pid.values.astype(np.int64)
Tt = torch.as_tensor(np.unpackbits(P.fp[truth], axis=1)[:, :chem.FP_BITS].astype(np.float16), device=dev)
nt = Tt.float().sum(1)
best = torch.zeros(len(truth), device=dev)
for a in range(0, len(train_pids), 100_000):
    Fx = torch.as_tensor(np.unpackbits(P.fp[train_pids[a:a + 100_000]], axis=1)[:, :chem.FP_BITS].astype(np.float16),
                         device=dev)
    inter = (Tt @ Fx.T).float()
    tan = inter / (nt[:, None] + Fx.float().sum(1)[None, :] - inter + 1e-9)
    best = torch.maximum(best, tan.max(1).values)
R["max_tan_train"] = best.cpu().numpy()
R.to_parquet(f"{OUT}/dec_ranks.parquet", index=False)

# %% [markdown]
# ## 門 D

# %%
out = []
groups = {"G_N": R.g == "G_N", "G_A": R.g == "G_A",
          "G_N_tan<0.7": (R.g == "G_N") & (R.max_tan_train < 0.7), "G_N_tan>=0.7": (R.g == "G_N") & (R.max_tan_train >= 0.7)}
for name, m in groups.items():
    m = m.values
    rb, ru, rd = harness.rr_of(R.base[m]), harness.rr_of(R.union[m]), harness.rr_of(R.dec_only[m])
    d = harness.paired_ci(ru, rb, N_BOOT, seed=1)
    out.append(dict(group=name, n=int(m.sum()), base=rb.mean(), union=ru.mean(), dec_only=rd.mean(), est=d["est"],
                    lo=d["lo"], hi=d["hi"], in_r=R.in_r[m].mean(), in_d=R.in_d[m].mean(),
                    in_d_not_r=(R.in_d[m] & ~R.in_r[m]).mean(), valid=R.valid[m].mean(),
                    formula_ok=R.formula_ok[m].mean(), n_d_median=R.n_d[m].median(),
                    truth_formula_in_w=R.truth_formula_in_w[m].mean()))
c = pd.DataFrame(out).set_index("group")
c.to_csv(f"{OUT}/compare_dec.csv")
print("MRR@25 (ties against the truth): base = W ∪ R, union = W ∪ R ∪ D, dec_only = W ∪ D; est = union − base")
print(c.round(4).to_string())
gate = bool(c.loc["G_N", "est"] >= GATE_D and c.loc["G_N", "lo"] > 0)
res = dict(gate=gate, decoder_step=int(DM.step), constrain=CONSTRAIN, n_done=len(R), compare=c.to_dict("index"))
json.dump(res, open(f"{OUT}/result_dec.json", "w"), indent=1, default=float)
print(f"gate D: {'PASS' if gate else 'FAIL'} — G_N: (W ∪ R ∪ D) − (W ∪ R) MRR@25 >= +{GATE_D} with lower bound > 0")
run = track.init("eval", group=EXP, config=dict(n_samples=N_SAMPLES, max_formulas=MAX_FORMULAS, min_per=MIN_PER,
                                                 decoder_step=int(DM.step), constrain=CONSTRAIN), online=False)
run.summary(dict(gate=gate, **{f"{g}_{k}": float(c.loc[g, k]) for g in c.index for k in ["base", "union", "est", "lo",
                                                                                         "in_d", "in_d_not_r"]}))
run.finish()
print(f"total {time.time() - T0:.0f}s")

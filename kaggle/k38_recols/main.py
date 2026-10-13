# %% [markdown]
# # casmi26-k38-recols
# M7 P1（2026-10-13）: 追加学習したモデル（G2 v2 = k37 の g2v2.pt、fprich2 = k39 の fprich2.pt）で、k30 のプール側の
# 列だけを計算し直す（CPU だけ、GPU の枠を使わない。k30 の 4 分割をやり直すと GPU で約 6 時間かかるため）。
# - 窓は k30 と同じ: query_info4 の target と ppm で P.window（Searcher と同じ）。クエリのスペクトルは harness.query_specs
#   から、中性質量が有限のものを先頭 max_query_spectra 本（Searcher と同じ）。列の z と順位は窓の全候補で計算し、k30 が
#   残した行（R1 / R2 は上位 300 に切ってある）だけを書き出す
# - 先に v1 のモデル（k30 が使った g2.pt / fprich.pt）で N_CHECK クエリを計算し直し、k30 の列と一致することを確かめる
#   （k30 は GPU の fp32、ここは CPU の fp32。値の列は許容 CHECK_TOL、順位の列はずれた行の割合 ≤ 1%）
# - 生成物の GEN_EXTRA（fzr、g2）は計算し直せない（gen4 に生成物の SMILES が無い）。生成側は v1 のまま
# 出力: recols_{set}.parquet（qid、pid と、計算し直した列。名前は k30 と同じ）、recols_info.json、check_recols.csv。
# 入力: k01、k05、k05b、k30-s0〜s3、k26（pool_fp2.npy）、casmi26-g2（g2.pt と g2v2.pt）、casmi26-fprich（fprich.pt と
# fprich2.pt）、casmi26-wheels（rdkit。G2 のグラフ）。

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

from casmi26 import cache, chem, fpnet, g2, harness, paths, pool, search

T0 = time.time()
OUT = "/kaggle/working"
SETS = ["A", "B_A", "C", "B_C", "R1", "R2", "V"]
NEW = {"g2": "g2v2.pt", "fprich": None}  # 計算し直すブロック → 新しいモデルのファイル名（None = そのまま）
N_CHECK = 300  # v1 で計算し直して k30 と照合するクエリの数（全組から）
CHECK_TOL = {"g2": 1e-4, "g2_z": 1e-3, "fzr": 5e-3, "fzr_z": 1e-3}
RANK_SHARE = 0.01
GRAPH_WORKERS = 4
MAX_QUERIES = None  # 手元の動作確認だけ
COLS = {"g2": search.G2_FEATURES, "fprich": search.FPRICH_FEATURES}
assert any(NEW.values()), "nothing to recompute"

infos = pd.concat([pd.read_csv(p) for p in sorted(glob.glob("/kaggle/input/**/query_info4_s*.csv", recursive=True))],
                  ignore_index=True).drop_duplicates("qid")
fi = [json.load(open(p)) for p in sorted(glob.glob("/kaggle/input/**/features_info4.json", recursive=True))]
assert fi and all(f == fi[0] for f in fi), "k30 shards disagree"
FI = fi[0]
blocks = [b for b, f in NEW.items() if f]
need = ["qid", "pid"] + sum((COLS[b] for b in blocks), [])
feats = {s: pd.concat([pd.read_parquet(p, columns=need) for p in
                       sorted(glob.glob(f"/kaggle/input/**/feats4_{s}_s*.parquet", recursive=True))], ignore_index=True)
         .drop_duplicates(["qid", "pid"]) for s in SETS}
if MAX_QUERIES:
    feats = {s: f[f.qid.isin(np.sort(f.qid.unique())[:MAX_QUERIES])] for s, f in feats.items()}
print("k30:", FI, "| rows", {s: len(f) for s, f in feats.items()}, f"({time.time() - T0:.0f}s)")

C = cache.Cache(cache.Cache.find_dir())
PD = pool.Pool.find_dir()
P = pool.Pool(PD, pubchem=False)
Q = pd.read_csv(f"{PD}/queries.csv").set_index("qid")
CFG = search.SearchCfg()
GR = g2.Graphs.build(P.smiles, GRAPH_WORKERS) if "g2" in blocks else None
FP2 = np.load(paths.find("pool_fp2.npy")) if "fprich" in blocks else None
MODELS = {}
if "g2" in blocks:
    old = g2.G2Model(paths.find("g2.pt"), G=GR)
    assert int(old.step) == int(FI["g2_step"]), "g2.pt is not the model k30 used"
    MODELS["g2"] = (old, g2.G2Model(paths.find(NEW["g2"]), G=GR))
    print("g2: v1 step", old.step, "| new", NEW["g2"], "step", MODELS["g2"][1].step)
if "fprich" in blocks:
    old = fpnet.FPModel(paths.find("fprich.pt"))
    assert int(old.epoch) == int(FI["fprich_epoch"]) and old.nbits == FI["fprich_nbits"], "fprich.pt differs from k30's"
    MODELS["fprich"] = (old, fpnet.FPModel(paths.find(NEW["fprich"])))
    print("fprich: v1 epoch", old.epoch, "| new", NEW["fprich"], "epoch", MODELS["fprich"][1].epoch)
info = infos.set_index("qid")


def specs_of(qid):
    rows = [int(x) for x in str(Q.loc[qid, "rows"]).split(",")]
    sp = [s for s in harness.query_specs(C, rows) if np.isfinite(chem.neutral_mass(s["prec"], s["adduct"]))]
    return sp[:CFG.max_query_spectra]


def block_cols(b, m, specs, cand):
    if b == "g2":
        g = m.score(specs, cand)
        return np.c_[g, search._zs_finite(g), search.rank_norm(np.nan_to_num(g, nan=-1.0))]
    zr = m(specs)
    fr = np.unpackbits(FP2[cand], axis=1)[:, :len(zr)].astype(np.float32) @ zr
    return np.c_[fr / 100.0, search._zs(fr), search.rank_norm(fr)]


def recompute(qid, which):
    """-> DataFrame(pid, cols of the blocks) on the full k30 window of qid; which = 0 (v1) or 1 (new)."""
    r = info.loc[qid]
    cand = P.window(float(r.target), float(r.ppm))
    specs = specs_of(qid)
    out = pd.DataFrame({"pid": cand})
    for b in blocks:
        out[COLS[b]] = block_cols(b, MODELS[b][which], specs, cand)
    return out

# %% [markdown]
# ## 1. v1 で k30 の列を再現する

# %%
allq = pd.concat([f[["qid"]].drop_duplicates().assign(set=s) for s, f in feats.items()], ignore_index=True)
chk_q = allq.sample(min(N_CHECK, len(allq)), random_state=0)
rows = []
for q in chk_q.itertuples():
    a = feats[q.set][feats[q.set].qid == q.qid]
    m = a.merge(recompute(q.qid, 0), on="pid", suffixes=("_k30", "_v1"))
    assert len(m) == len(a), f"qid {q.qid}: k30 rows outside the recomputed window"
    for c in need[2:]:
        d = np.abs(np.nan_to_num(m[c + "_k30"].values, nan=-9) - np.nan_to_num(m[c + "_v1"].values, nan=-9))
        rows.append(dict(qid=q.qid, set=q.set, col=c, max_diff=float(d.max()) if len(d) else 0.0,
                         share=float((d > 1e-3).mean()) if len(d) else 0.0))
chk = pd.DataFrame(rows)
chk.to_csv(f"{OUT}/check_recols.csv", index=False)
summ = chk.groupby("col").agg(max_diff=("max_diff", "max"), share=("share", "mean"))
print(f"v1 vs k30 on {len(chk_q)} queries:\n{summ}")
for c, tol in CHECK_TOL.items():
    if c in summ.index:
        assert summ.loc[c, "max_diff"] <= tol, f"{c} does not reproduce k30 ({summ.loc[c, 'max_diff']:.2e} > {tol})"
for c in [c for c in summ.index if c.endswith("_rank")]:
    assert summ.loc[c, "share"] <= RANK_SHARE, f"{c}: {summ.loc[c, 'share']:.4f} of the rows differ"
print(f"check passed ({time.time() - T0:.0f}s)", flush=True)

# %% [markdown]
# ## 2. 新しいモデルで全クエリを計算し直す

# %%
res = {}
for s, f in feats.items():
    parts, t = [], time.time()
    for i, qid in enumerate(np.sort(f.qid.unique())):
        new = recompute(qid, 1)
        parts.append(f.loc[f.qid == qid, ["qid", "pid"]].merge(new, on="pid", how="left"))
        if (i + 1) % 1000 == 0:
            print(f"  {s}: {i + 1} queries ({time.time() - t:.0f}s)", flush=True)
    out = pd.concat(parts, ignore_index=True)
    assert len(out) == len(f) and out[need[2:]].notna().sum().min() >= 0
    out.to_parquet(f"{OUT}/recols_{s}.parquet", index=False)
    res[s] = dict(rows=len(out), queries=int(out.qid.nunique()), secs=round(time.time() - t))
    print(s, res[s], flush=True)
meta = dict(blocks={b: NEW[b] for b in blocks}, k30=FI, sets=res, check=summ.reset_index().to_dict("records"),
            g2_step=int(MODELS["g2"][1].step) if "g2" in blocks else None,
            fprich_epoch=int(MODELS["fprich"][1].epoch) if "fprich" in blocks else None,
            src_sha=os.environ.get("CASMI26_SHA"))
json.dump(meta, open(f"{OUT}/recols_info.json", "w"), indent=1, default=float)
print(f"total {time.time() - T0:.0f}s")

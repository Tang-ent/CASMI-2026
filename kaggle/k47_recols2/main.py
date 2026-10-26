# %% [markdown]
# # casmi26-k47-recols2
# 計画 9.2（2026-10-23、結果を見る前に固定）: k30 の窓で、スペクトルごとの統計（search.FRAG_STATS、G2_STATS）と、
# アダクトごとに統合したスペクトルの点数（search.CONS）の 10 列を計算する（k38 から複製。手元の GPU で動かし、push しない）。
# - 窓は k30 と同じ: query_info4 の target と ppm で P.window（Searcher と同じ）。クエリのスペクトルは harness.query_specs
#   から、中性質量が有限のものを先頭 max_query_spectra 本（Searcher と同じ）。列の z は窓の全候補で計算し、k30 が残した
#   行（R1 / R2 は上位 300 に切ってある）だけを書き出す
# - 列の作り方は Searcher(..., spec_stats=True, consensus=True)._features の「more」と同じ呼び出し（new_cols）。
#   frag は FragTable（k21）、G2 は k30 と同じ g2.pt（step 60000、fp32、GPU）
# - 計算するのは A、R1、R2、V。B_A は A と同じスペクトル・同じ窓なので、queries.csv の rows で A の qid に対応させて
#   A の値を使い回す（C / B_C は k30 の出力が手元に無いので外す）
# - 先に N_CHECK クエリで、同じ呼び出しが返す平均の列（frag_expl、frag_top10、g2）が k30 と許容 CHECK_TOL で一致する
#   ことを確かめる（照合の表 check_recols2.csv を書いてから止める。許容は緩めない）。全クエリの通しでも同じ差を測り、
#   recols2_info.json に残す（報告だけ）
# 出力: recols2_{set}.parquet（qid、pid と FRAG_STATS + G2_STATS + CONS の 10 列。set = A、B_A、R1、R2、V）、
# recols2_info.json、check_recols2.csv。
# 入力: k01、k05、k05b、k21（frag_off / frag_mass.npy）、k30-s0〜s3、casmi26-g2（g2.pt）、casmi26-wheels（rdkit。G2 のグラフ）。

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

from casmi26 import cache, chem, frag, g2, harness, paths, pool, search

T0 = time.time()
OUT = "/kaggle/working"
SETS = ["A", "R1", "R2", "V"]  # 計算する組（B_A は A の対応で書き出す）
N_CHECK = 300  # 平均の列を k30 と照合するクエリの数（計算する組の全体から）
CHECK_COLS = ["frag_expl", "frag_top10", "g2"]
CHECK_TOL = 1e-4
REQUIRE_CUDA = True  # G2 は k30 と同じく GPU の fp32
GRAPH_WORKERS = 4
MAX_QUERIES = None  # 手元の動作確認だけ（組ごとの先頭のクエリ数）
NEW_COLS = search.FRAG_STATS + search.G2_STATS + search.CONS

infos = pd.concat([pd.read_csv(p) for p in sorted(glob.glob("/kaggle/input/**/query_info4_s*.csv", recursive=True))],
                  ignore_index=True).drop_duplicates("qid")
fi = [json.load(open(p)) for p in sorted(glob.glob("/kaggle/input/**/features_info4.json", recursive=True))]
assert fi and all(f == fi[0] for f in fi), "k30 shards disagree"
FI = fi[0]
need = ["qid", "pid"] + CHECK_COLS
feats = {s: pd.concat([pd.read_parquet(p, columns=need) for p in
                       sorted(glob.glob(f"/kaggle/input/**/feats4_{s}_s*.parquet", recursive=True))], ignore_index=True)
         .drop_duplicates(["qid", "pid"]) for s in SETS + ["B_A"]}
if MAX_QUERIES:
    feats = {s: f[f.qid.isin(np.sort(f.qid.unique())[:MAX_QUERIES])] for s, f in feats.items()}
print("k30:", FI, "| rows", {s: len(f) for s, f in feats.items()}, f"({time.time() - T0:.0f}s)")

C = cache.Cache(cache.Cache.find_dir())
PD = pool.Pool.find_dir()
P = pool.Pool(PD, pubchem=False)
Q = pd.read_csv(f"{PD}/queries.csv").set_index("qid")
CFG = search.SearchCfg()
# B_A → A（同じ rows。窓の target と ppm も同じこと）
qa = Q[Q.set == "A"].reset_index().set_index("rows").qid
BA2A = Q[Q.set == "B_A"].rows.map(qa)
assert BA2A.notna().all() and BA2A.is_unique and len(BA2A) == (Q.set == "A").sum(), "B_A is not a copy of A"
BA2A = BA2A.astype(int)
info = infos.set_index("qid")
assert np.allclose(info.loc[BA2A.index, ["target", "ppm"]].values, info.loc[BA2A.values, ["target", "ppm"]].values,
                   rtol=0, atol=0), "B_A windows differ from A's"
GR = g2.Graphs.build(P.smiles, GRAPH_WORKERS)  # CUDA より先に（fork）
print(f"G2 graphs: {GR.ok.mean():.4f} of {len(GR):,} ({time.time() - T0:.0f}s)", flush=True)
G2M = g2.G2Model(paths.find("g2.pt"), G=GR)
assert int(G2M.step) == int(FI["g2_step"]) and not G2M.amp, "g2.pt is not the model k30 used (fp32)"
assert not REQUIRE_CUDA or G2M.device == "cuda", "G2 must run on the GPU (as k30)"
FT = frag.FragTable(frag.FragTable.find_dir())
assert len(FT.off) == len(P) + 1, "fragment table does not match the pool"
import torch

print("g2 step", G2M.step, "| device", G2M.device,
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", "| torch", torch.__version__, flush=True)


def specs_of(qid):
    rows = [int(x) for x in str(Q.loc[qid, "rows"]).split(",")]
    sp = [s for s in harness.query_specs(C, rows) if np.isfinite(chem.neutral_mass(s["prec"], s["adduct"]))]
    return sp[:CFG.max_query_spectra]


def new_cols(specs, cand):
    """-> (means (n, 3) = frag_expl, frag_top10, g2; new (n, 10) = FRAG_STATS + G2_STATS + CONS), the same calls as
    Searcher(..., spec_stats=True, consensus=True)._features."""
    fcsr = FT.csr(cand)
    expl, top10, E, used = frag.match_features(specs, *fcsr, per_spec=True)
    fs = np.column_stack(search.spec_stats_of(E, [specs[i].get("ce", np.nan) for i in used], expl))
    g, GM, gce, _ = G2M.score(specs, cand, per_spec=True)
    gs = np.column_stack(search.spec_stats_of(GM, gce, g))
    fc = frag.consensus_expl(specs, *fcsr)
    gc = G2M.score_consensus(specs, cand)
    cons = np.c_[fc, search._zs(fc), gc, search._zs_finite(gc)]
    return np.c_[expl, top10, g], np.hstack([fs, gs, cons]).astype(np.float32)


MEMO = {}


def recompute(qid):
    """-> DataFrame(pid, CHECK_COLS + NEW_COLS) on the full k30 window of qid (A / R1 / R2 / V qid; kept for B_A)."""
    if qid in MEMO:
        return MEMO[qid]
    r = info.loc[qid]
    cand = P.window(float(r.target), float(r.ppm))
    means, new = new_cols(specs_of(qid), cand)
    out = pd.concat([pd.DataFrame({"pid": cand}), pd.DataFrame(means, columns=CHECK_COLS),
                     pd.DataFrame(new, columns=NEW_COLS)], axis=1)
    if Q.loc[qid, "set"] == "A":
        MEMO[qid] = out
    return out


def diff_of(a, b):
    return np.abs(np.nan_to_num(a, nan=-9) - np.nan_to_num(b, nan=-9))

# %% [markdown]
# ## 1. 平均の列が k30 を再現すること

# %%
allq = pd.concat([feats[s][["qid"]].drop_duplicates().assign(set=s) for s in SETS], ignore_index=True)
chk_q = allq.sample(min(N_CHECK, len(allq)), random_state=0)
rows, t = [], time.time()
for q in chk_q.itertuples():
    a = feats[q.set][feats[q.set].qid == q.qid]
    m = a.merge(recompute(q.qid)[["pid"] + CHECK_COLS], on="pid", suffixes=("_k30", "_new"))
    assert len(m) == len(a), f"qid {q.qid}: k30 rows outside the recomputed window"
    for c in CHECK_COLS:
        d = diff_of(m[c + "_k30"].values, m[c + "_new"].values)
        rows.append(dict(qid=q.qid, set=q.set, col=c, max_diff=float(d.max()) if len(d) else 0.0,
                         share=float((d > CHECK_TOL).mean()) if len(d) else 0.0))
chk = pd.DataFrame(rows)
chk.to_csv(f"{OUT}/check_recols2.csv", index=False)
summ = chk.groupby("col").agg(max_diff=("max_diff", "max"), share=("share", "mean"))
print(f"new code vs k30 on {len(chk_q)} queries ({time.time() - t:.0f}s):\n{summ}", flush=True)
for c in CHECK_COLS:
    assert summ.loc[c, "max_diff"] <= CHECK_TOL, f"{c} does not reproduce k30 ({summ.loc[c, 'max_diff']:.2e} > {CHECK_TOL})"
print(f"check passed ({time.time() - T0:.0f}s)", flush=True)

# %% [markdown]
# ## 2. 全クエリの 10 列

# %%
res, full = {}, {}
for s in SETS + ["B_A"]:
    f = feats[s]
    parts, t, dmax = [], time.time(), {c: 0.0 for c in CHECK_COLS}
    for i, qid in enumerate(np.sort(f.qid.unique())):
        a = f[f.qid == qid]
        new = recompute(int(BA2A[qid]) if s == "B_A" else qid)
        m = a.merge(new, on="pid", how="left", suffixes=("_k30", ""))
        assert len(m) == len(a) and m.frag_expl_min.notna().all(), f"{s} qid {qid}: k30 rows outside the window"
        for c in CHECK_COLS:
            dmax[c] = max(dmax[c], float(diff_of(m[c + "_k30"].values, m[c].values).max()) if len(m) else 0.0)
        parts.append(m[["qid", "pid"] + NEW_COLS])
        if (i + 1) % 1000 == 0:
            print(f"  {s}: {i + 1} queries ({time.time() - t:.0f}s)", flush=True)
    out = pd.concat(parts, ignore_index=True)
    assert len(out) == len(f)
    out.to_parquet(f"{OUT}/recols2_{s}.parquet", index=False)
    res[s] = dict(rows=len(out), queries=int(out.qid.nunique()), secs=round(time.time() - t),
                  nan_share={c: float(out[c].isna().mean()) for c in NEW_COLS if out[c].isna().any()})
    full[s] = dmax
    print(s, res[s], "| max diff vs k30", {c: f"{x:.2e}" for c, x in dmax.items()}, flush=True)
meta = dict(columns=NEW_COLS, k30=FI, sets=res, max_queries=MAX_QUERIES, check_tol=CHECK_TOL,
            check=summ.reset_index().to_dict("records"), full_max_diff=full, g2_step=int(G2M.step),
            g2_device=G2M.device, g2_precision="fp32",
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, b_a_from="A (queries.csv rows)",
            src_sha=os.environ.get("CASMI26_SHA"))
json.dump(meta, open(f"{OUT}/recols2_info.json", "w"), indent=1, default=float)
print(f"total {time.time() - T0:.0f}s")

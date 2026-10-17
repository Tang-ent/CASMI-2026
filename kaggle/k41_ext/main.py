# %% [markdown]
# # casmi26-k41-ext
# M7 P3-1（2026-10-17、ユーザーの決定の後、結果を見る前に固定。計画 1-3-deep-storm.md）: 外部の学習済みの重み
# （ms-pred の GLACIER msg = "gl"、ICEBERG 2.1 msg_all = "ice"。論文で公開された重みだけ）で、k30 の候補に列を足す。
# - 採点する候補: 参照 v12（both_fprich_g2_grp|hybi:0.7:0.8:5+nogen）の順の上位 K_TOP = 100。R1 / R2 は k35 と同じ
#   rank.cv_oof（5 分割、seed 0、groups = 正解、種 0 / 1 / 2）の OOF の点数で並べる（保存のランカーは R で in-sample）。
#   A / B_A / V は k35 の ranker7_both_fprich_g2_grp_{0,1,2}。同じクエリの行を持つ組（A と B_A、R1 と R2）は予測を
#   共有し、列は各集合の上位 100 だけで作る（提出と同じ）。
# - スペクトル、CE、装置、類似度、列の定義は casmi26/ext.py の冒頭（事前登録）。窓の全候補で z / rank / 同じ分子式の
#   組の列を計算し（提出と同じ）、k30 の行（R は上位 300）だけを書き出す。
# - 自己照合: (1) 参照 v12 の A / B_A / B_N が k35 の rule7 と ±RULE7_TOL で一致（LightGBM のスレッドの違い）。
#   (2) probe（MassSpecGym test の 200 構造）の自分当て top-1 が casmi26-mspred の記録と ±0.02。
# - 再開: 塊ごとに parts_{KIND}/part_NNNN.parquet を書く。同じ出力先で走り直すと、済んだ組を飛ばす。
# 出力: ext_{KIND}_{set}.parquet（qid、pid、ext.FEATURES[KIND]）、topk_{set}.parquet（qid、pid、order）、
# diag_{KIND}.csv（クエリごと: スペクトル数、採点数、正解が上位 100 に入ったか、正解の ext の順位）、ext_info_{KIND}.json。
# 入力: k01、k05、k05b、k26（pool_formula.npz）、k30-s0〜s3、k35、casmi26-mspred、casmi26-wheels（rdkit）、競技データ。

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

import lightgbm as lgb
import numpy as np
import pandas as pd

from casmi26 import cache, ext, harness, paths, pool, rank, search, subform
from casmi26.frag import FRAG_FEATURES

T0 = time.time()
OUT = "/kaggle/working"
KIND = "gl"  # gl = GLACIER、ice = ICEBERG（build の --define で替える）
SETS = ["A", "B_A", "R1", "R2", "V"]
REF_TAG = "both_fprich_g2_grp"
REF = "both_fprich_g2_grp|hybi:0.7:0.8:5+nogen"  # k36 v12（公開 LB 0.339）
SEEDS = (0, 1, 2)
K_TOP = ext.K_TOP
CHUNK = 20000  # 1 塊の予測の件数
WORKERS = 0  # GLACIER の DataLoader の worker（手元は run_k41.sh で 4 と --shm-size。Kaggle は 0）
BATCH = 64
RULE7_TOL = 0.0015
EXPECT_VARIANT = "v2b"
MAX_QUERIES = None  # 手元の動作確認だけ（集合ごとの先頭のクエリ数）
SHARD, N_SHARDS = globals().get("SHARD", 0), globals().get("N_SHARDS", 1)  # build_kernel.py k41_ext:<n>（組の番号 % n）
SFX = f"_s{SHARD}" if N_SHARDS > 1 else ""
SCORE = search.parse_variant(REF)["score"]
assert KIND in ext.KINDS
EF = ext.FEATURES[KIND]

# 外部モデルを先に読み、probe を照合する（Kaggle の実行環境の違いで落ちるなら OOF の前に落ちる）
M = ext.ExtModel(KIND, workers=WORKERS, batch_size=BATCH)
print("model", KIND, M.sha, "device", M.device, f"({time.time() - T0:.0f}s)", flush=True)
root = ext.mspred_root()
man = json.load(open(os.path.join(root, ext.MANIFEST)))
lab = pd.read_csv(os.path.join(root, "labels_200.tsv"), sep="\t")
tru = pd.read_csv(os.path.join(root, "true_200.tsv"), sep="\t").set_index("identifier").loc[lab.spec]
parse = lambda x: np.array([float(v) for v in str(x).split(",")])
can = [ext.canonical_input(x, KIND) for x in lab.smiles]
pr = M.predict([(c, r.ionization, ce, r.instrument, float(r.precursor)) for c, r in zip(can, lab.itertuples())
                for ce in (20.0, 40.0, 60.0)])
S = np.array([[max((ext.entropy_sim(parse(t.mzs), parse(t.intensities), *pr[3 * j + k]) if pr[3 * j + k] else 0.0)
                    for k in range(3)) for j in range(len(lab))] for t in tru.itertuples()])
S = np.nan_to_num(S)
probe_top1 = float(np.mean([(S[i] > S[i, i]).sum() == 0 for i in range(len(lab))]))
print(f"probe identity top-1 {probe_top1:.3f} (recorded {man['probe_top1'][KIND]:.3f})", flush=True)
assert abs(probe_top1 - man["probe_top1"][KIND]) <= 0.02 + 1e-9, "probe not reproduced"

PD = pool.Pool.find_dir()
Q = pd.read_csv(f"{PD}/queries.csv").set_index("qid")
truth = Q.pid
in_co = pd.read_parquet(f"{PD}/pool.parquet", columns=["in_coconut"]).in_coconut.values
FT = subform.FormulaTable(subform.FormulaTable.find_dir())
tags = [json.load(open(p)) for p in sorted(glob.glob("/kaggle/input/**/features_info4.json", recursive=True))]
assert tags and all(t["variant"] == EXPECT_VARIANT for t in tags), tags
infos = pd.concat([pd.read_csv(p) for p in sorted(glob.glob("/kaggle/input/**/query_info4_s*.csv", recursive=True))],
                  ignore_index=True).drop_duplicates("qid")


def load(pattern):
    ps = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    return pd.concat([pd.read_parquet(p) for p in ps], ignore_index=True).drop_duplicates(["qid", "pid"])


feats = {s: load(f"feats4_{s}_s*.parquet").sort_values(["qid", "pid"], kind="mergesort").reset_index(drop=True)
         for s in SETS}
if MAX_QUERIES:
    keepq = {s: np.sort(f.qid.unique())[:MAX_QUERIES] for s, f in feats.items()}
    # 対の集合は同じクエリの行を持つ組を残す（B_A は A の、R2 は R1 の相手）
    for a, b in [("A", "B_A"), ("R1", "R2")]:
        rows_a = set(Q.rows.reindex(keepq[a]))
        keepq[b] = np.sort(feats[b].qid.unique()[Q.rows.reindex(feats[b].qid.unique()).isin(rows_a).values])
    feats = {s: f[f.qid.isin(keepq[s])].reset_index(drop=True) for s, f in feats.items()}
    infos = infos[infos.qid.isin(np.concatenate(list(keepq.values())))]
qids_of = {s: np.sort(infos[infos.set == s].qid.values) for s in SETS}
for s in SETS:
    G = search.formula_group_features({c: feats[s][c].values for c in search.GRP_BASE}, feats[s].qid.values,
                                      FT.fid[feats[s].pid.values])
    feats[s] = pd.concat([feats[s], pd.DataFrame(G, columns=search.GRP_FEATURES)], axis=1)
FL = list(search.FEATURES)
COLS = FL + FRAG_FEATURES + search.FWD_FEATURES + search.FPRICH_FEATURES + search.G2_FEATURES + search.GRP_FEATURES
print("rows", {s: len(f) for s, f in feats.items()}, "| queries", {s: len(q) for s, q in qids_of.items()},
      f"({time.time() - T0:.0f}s)", flush=True)

# %% [markdown]
# ## 1. 参照 v12 の順（R は OOF）と、k35 の rule7 との照合

# %%
models = [lgb.Booster(model_file=paths.find(f"ranker7_{REF_TAG}_{i}.txt")) for i in range(3)]
assert all(m.feature_name() == COLS for m in models), "ranker7 columns differ from the k35 column set"
RK = {}
oof_path = f"{OUT}/oof_R.npy"
R = pd.concat([feats["R1"], feats["R2"]], ignore_index=True)
if os.path.exists(oof_path) and len(np.load(oof_path)) == len(R):
    oof = np.load(oof_path)
    print("OOF of R from", oof_path)
else:
    oof = rank.cv_oof(R, COLS, n_folds=5, seed=0, groups=truth, model_seeds=SEEDS)
    np.save(oof_path, oof)
r1 = R.qid.isin(feats["R1"].qid.unique()).values
RK["R1"], RK["R2"] = oof[r1], oof[~r1]
for s in ["A", "B_A", "V"]:
    RK[s] = rank.predict(models, feats[s][COLS].values.astype(np.float32))
print(f"ranker scores done ({time.time() - T0:.0f}s)", flush=True)
X = {s: feats[s][FL].values for s in SETS}
LAB = {s: feats[s].label.values for s in SETS}
QID = {s: feats[s].qid.values for s in SETS}
PID = {s: feats[s].pid.values for s in SETS}


def per_qid(s):
    """qid -> row indices of set s (feats are sorted by qid)."""
    q, start = np.unique(QID[s], return_index=True)
    ends = np.r_[start[1:], len(QID[s])]
    return {int(a): np.arange(b, e) for a, b, e in zip(q, start, ends)}


IX = {s: per_qid(s) for s in SETS}
TOP, TRANK = {}, {}
for s in SETS:
    tops, tr = {}, {}
    for q, ix in IX[s].items():
        o = search.variant_order(X[s][ix], RK[s][ix], SCORE)
        tops[q] = PID[s][ix][o[:K_TOP]]
        hit = np.flatnonzero(LAB[s][ix][o] > 0)
        tr[q] = hit[0] + 1 if len(hit) else np.inf
    TOP[s], TRANK[s] = tops, tr
    pd.DataFrame([(q, p, j) for q, ps in tops.items() for j, p in enumerate(ps)], columns=["qid", "pid", "order"]) \
        .to_parquet(f"{OUT}/topk_{s}.parquet", index=False)
rr = {s: harness.rr_of(np.array([TRANK[s].get(int(q), np.inf) for q in qids_of[s]])) for s in SETS}
BN = {s: in_co[truth.reindex(qids_of[s]).values] for s in ["R2", "V"]}
ref = dict(A=rr["A"].mean(), B_A=rr["B_A"].mean(),
           B_N=np.concatenate([rr["R2"][BN["R2"]], rr["V"][BN["V"]]]).mean())
rule7 = pd.read_csv(paths.find("rule7.csv")).set_index("model")
k35 = {k: float(rule7.loc[REF, k]) for k in ref}
print("reference v12 here:", {k: round(v, 4) for k, v in ref.items()}, "| k35 rule7:", {k: round(v, 4) for k, v in k35.items()})
if not MAX_QUERIES:
    for k in ref:
        assert abs(ref[k] - k35[k]) <= RULE7_TOL, f"v12 {k} {ref[k]:.4f} differs from k35's {k35[k]:.4f}"
cover = {s: float(np.mean([TRANK[s][q] <= K_TOP for q in TRANK[s]])) for s in SETS}
print("truth within the top", K_TOP, cover, f"({time.time() - T0:.0f}s)", flush=True)

# %% [markdown]
# ## 2. 外部モデル（probe の照合の後、組ごとに塊で予測）

# %%

C = cache.Cache(cache.Cache.find_dir())
P = pool.Pool(PD, pubchem=False)
CET = ext.CETable(os.path.join(paths.comp_dir(), "train.parquet"), n_rows=len(C.sid))
CFG = search.SearchCfg()
info = infos.set_index("qid")
groups = {}
for s in SETS:
    for q in qids_of[s]:
        if int(q) in IX[s]:  # queries without any candidate have no k30 rows
            groups.setdefault(str(Q.loc[q, "rows"]), []).append((s, int(q)))
GL = sorted(groups.items(), key=lambda kv: min(q for _, q in kv[1]))
print(f"{len(GL)} query groups ({sum(len(v) for _, v in GL)} queries)", flush=True)
PARTS = f"{OUT}/parts_{KIND}"
os.makedirs(PARTS, exist_ok=True)
done = set()
for p in sorted(glob.glob(f"{PARTS}/part_*.parquet")):
    done |= set(pd.read_parquet(p, columns=["grp"]).grp.unique().tolist())
DIAG = f"{PARTS}/diag.csv"
if done:
    print(f"resume: {len(done)} groups already done", flush=True)
CAN = {}


def can_of(pid):
    if pid not in CAN:
        CAN[pid] = ext.canonical_input(P.smiles[pid], KIND)
    return CAN[pid]


def finish(chunk, cache_):
    """chunk: list of (group index, specs, members) -> part DataFrame and diagnostics rows."""
    out, diag = [], []
    for gi, specs, mem in chunk:
        union = np.unique(np.concatenate([TOP[s][q] for s, q in mem]))
        sc = dict(zip(union.tolist(), ext.candidate_scores([can_of(int(p)) for p in union], specs, cache_, KIND)))
        for s, q in mem:
            r_ = info.loc[q]
            cand = P.window(float(r_.target), float(r_.ppm))
            v = np.full(len(cand), np.nan)
            pos = pd.Series(np.arange(len(cand)), index=cand)
            top = TOP[s][q]
            at = pos.reindex(top).values
            assert not np.isnan(at).any(), f"{s} {q}: top candidates outside the recomputed window"
            v[at.astype(np.int64)] = [sc[int(p)] for p in top]
            cols = ext.columns(v, FT.fid[cand])
            rows = IX[s][q]
            sub = pd.DataFrame(cols, columns=EF)
            sub.insert(0, "pid", cand)
            m = pd.DataFrame({"pid": PID[s][rows]}).merge(sub, on="pid", how="left")
            assert m[EF[1]].notna().all(), f"{s} {q}: k30 rows outside the recomputed window"
            m.insert(0, "qid", q)
            m.insert(0, "set", s)
            m.insert(0, "grp", gi)
            out.append(m)
            tp = int(truth[q])
            t_in = tp in set(top.tolist())
            tv = sc.get(tp, np.nan) if t_in else np.nan
            vt = np.array([sc[int(p)] for p in top])
            diag.append(dict(set=s, qid=q, grp=gi, n_specs=len(specs), n_top=len(top),
                             n_scored=int(np.isfinite(vt).sum()), truth_in_top=t_in, truth_v12_rank=TRANK[s][q],
                             truth_ext_rank=(1 + int((vt > tv).sum())) if np.isfinite(tv) else np.inf))
    return pd.concat(out, ignore_index=True), diag


chunk, need, pc, t_last = [], {}, len(glob.glob(f"{PARTS}/part_*.parquet")), time.time()
n_entries = n_done = 0
todo = [(gi, rows_, mem) for gi, (rows_, mem) in enumerate(GL) if gi not in done and gi % N_SHARDS == SHARD]
print(f"shard {SHARD} of {N_SHARDS}: {len(todo)} groups to do", flush=True)
for j, (gi, rows_, mem) in enumerate(todo):
    specs = ext.query_specs_ce(C, [int(x) for x in rows_.split(",")], CET, CFG.max_query_spectra)
    union = np.unique(np.concatenate([TOP[s][q] for s, q in mem]))
    for p in union:
        c = can_of(int(p))
        if c is None:
            continue
        for sp in specs:
            for ce in sp["ce_list"]:
                need.setdefault(ext.entry_key(c, sp, ce, KIND), None)
    chunk.append((gi, specs, mem))
    if len(need) >= CHUNK or j == len(todo) - 1:
        keys = list(need)
        for k, p_ in zip(keys, M.predict(keys)):
            need[k] = p_
        part, diag = finish(chunk, need)
        part.to_parquet(f"{PARTS}/part_{pc:04d}.parquet", index=False)
        pd.DataFrame(diag).to_csv(DIAG, mode="a", header=not os.path.exists(DIAG), index=False)
        n_entries += len(keys)
        n_done += len(chunk)
        pc += 1
        el = time.time() - T0
        print(f"  part {pc}: {n_done}/{len(todo)} groups, {n_entries:,} entries, {len(keys) / (time.time() - t_last):.0f}"
              f" entries/s, failed {M.n_fail}, {el / 3600:.2f} h", flush=True)
        chunk, need, t_last = [], {}, time.time()

# %% [markdown]
# ## 3. 書き出しと報告（報告だけ。採否は対の LB で決める: ユーザーの決定 2026-10-17）

# %%
allp = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{PARTS}/part_*.parquet"))], ignore_index=True)
allp = allp.drop_duplicates(["set", "qid", "pid"])
res = {}
for s in SETS:
    e = allp[allp.set == s][["qid", "pid"] + EF].sort_values(["qid", "pid"], kind="mergesort").reset_index(drop=True)
    mine = {q for gi, (_, mem) in enumerate(GL) if gi % N_SHARDS == SHARD for s_, q in mem if s_ == s}
    assert len(e) == int(np.isin(QID[s], list(mine)).sum()), f"{s}: {len(e)} rows vs k30's {len(feats[s])}"
    e.to_parquet(f"{OUT}/ext_{KIND}_{s}{SFX}.parquet", index=False)
    res[s] = dict(rows=len(e), queries=int(e.qid.nunique()), scored_rows=int(np.isfinite(e[KIND]).sum()))
dg = pd.read_csv(DIAG).drop_duplicates(["set", "qid"])
dg.to_csv(f"{OUT}/diag_{KIND}{SFX}.csv", index=False)
rep = dg.groupby("set").agg(n=("qid", "size"), with_specs=("n_specs", lambda x: float((x > 0).mean())),
                            truth_in_top=("truth_in_top", "mean"),
                            ext_top1=("truth_ext_rank", lambda x: float((x == 1).mean())),
                            ext_mrr_in_top=("truth_ext_rank", lambda x: float(np.mean(1 / x))))
print("report only (ext alone among the scored top", K_TOP, "):\n", rep.round(3).to_string())
meta = dict(kind=KIND, sha=M.sha, ref=REF, k_top=K_TOP, reference_here=ref, k35_rule7=k35, truth_within_top=cover,
            probe_top1=probe_top1, sets=res, entries=n_entries, failed=M.n_fail, workers=WORKERS, shard=SHARD,
            n_shards=N_SHARDS,
            secs=round(time.time() - T0), max_queries=MAX_QUERIES, src_sha=os.environ.get("CASMI26_SHA"),
            rdkit=rdkit.__version__)
json.dump(meta, open(f"{OUT}/ext_info_{KIND}{SFX}.json", "w"), indent=1, default=float)
print(json.dumps({k: meta[k] for k in ["kind", "sets", "entries", "failed", "secs"]}, default=float))
print(f"total {time.time() - T0:.0f}s")

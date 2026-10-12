# %% [markdown]
# # casmi26-k30-features4（分割）
# M6 P1 の 1（2026-10-12）: k22_features3 の特徴量に、P0 の門を通った 2 つの点数を足す（GPU）。
# - プール側: k22 と同じクエリ・同じ切り詰め・同じ 41 列（v2b の FPNet は k22 と同じく CPU、fwd は GPU の fp32、frag）を
#   そのまま出し、その後ろに fprich の 3 列（fzr = pool_fp2 の 10,407 ビット @ fprich の z / 100、fzr_z、fzr_rank）と
#   G2 の 3 列（g2 = 断片ごとの強度のコサイン、g2_z、g2_rank。グラフの無い構造は NaN / z 0 / 最下位）を足す
# - 生成側: k22 と同じ手順の生成物に、GEN_FEATURES（今の 14 列）とは別に GEN_EXTRA（fzr、fzr_z、g2、g2_z）を足す
#   （生成物の豊かな指紋は Morgan + chem.rich_fp_tail、G2 は生成物の SMILES からグラフを作って採点）
# - fprich と G2 は GPU の fp32（G2Model の既定。k32 の提出と同じ値にするため）。G2 のグラフ（プール全体）は CUDA を
#   使い始める前に作る（multiprocessing の fork のため）
# - k31_eval6 は k22 と k30 の両方を読むので、出力の名前には 4 を付ける（feats4_*、gen4_*、query_info4_*、gen_info4_*、
#   features_info4.json、test_timing4.csv）
# 入力: k01、k05、k05b、k15-s1（fpnet2.pt = v2b）、k20（fwd.pt）、k25（use_fwd.json）、k21（断片の表）、k26（pool_fp2.npy）、
# casmi26-fprich（fprich.pt）、casmi26-g2（g2.pt）、casmi26-wheels（rdkit）。

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
import glob
import json
import os
import time

import numpy as np
import pandas as pd

from casmi26 import cache, chem, derive, fpnet, frag, fwd, g2, gen, harness, paths, pool, search

SHARD, N_SHARDS = globals().get("SHARD", 0), globals().get("N_SHARDS", 1)
T0 = time.time()
OUT = "/kaggle/working"
BUDGET_R_SECS = 6 * 3600
MAX_CAND_TRAIN = 300
SETS = ["A", "B_A", "C", "B_C", "V"]  # R1 / R2 は別に、構造を対にして処理する
GEN_SETS = {"A", "B_A", "V", "R2"}
N_TEST = 400  # shard 0 の公開テストの時間の計測（分子数）
SEARCH_CFG = search.SearchCfg()  # 手元の動作確認では窓を広げる
REQUIRE_CUDA = True  # fwd、fprich、G2 は GPU で（分割ごとに装置が違うと特徴量がずれる。手元の動作確認は False）
GRAPH_WORKERS = 4
REQUIRE_GATES = True  # fprich と G2 が門を通ったモデルであること（手元の動作確認のモデルは通っていないので False）
print("cpus", os.cpu_count(), flush=True)

C = cache.Cache(cache.Cache.find_dir())
P = pool.Pool(pool.Pool.find_dir())
assert len(P) == P.n_base, "the PubChem part must not be attached"
GR = g2.Graphs.build(P.smiles, GRAPH_WORKERS)  # CUDA より先に（fork）
print(f"G2 graphs: {GR.ok.mean():.4f} of {len(GR):,} ({time.time() - T0:.0f}s)", flush=True)
FP2 = np.load(paths.find("pool_fp2.npy"))
assert FP2.shape == (len(P), (chem.RICH_FP_BITS + 7) // 8), FP2.shape
assert (np.unpackbits(FP2[:2000], axis=1)[:, :chem.FP_BITS] == np.unpackbits(P.fp[:2000], axis=1)[:, :chem.FP_BITS]).all()

import torch

print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "| gpus", torch.cuda.device_count(), flush=True)
MODEL = fpnet.load_checked("fpnet2.pt", min_epoch=1)  # v2b は k22 と同じく CPU（41 列を k22 と同じ値にする）
res = [json.load(open(p)) for p in glob.glob("/kaggle/input/**/result_v2*.json", recursive=True)]
assert len(res) == 1 and res[0]["epoch"] == MODEL.epoch, "attach exactly the k15 variant of fpnet2.pt"
use = json.load(open(paths.find("use_fwd.json")))
FWD = fwd.FwdModel(paths.find(use["file"]))
MR = fpnet.FPModel(paths.find("fprich.pt"))
res_r = json.load(open(paths.find("result_fprich.json")))
assert (res_r["gate"] or not REQUIRE_GATES) and MR.epoch == res_r["epoch"] and MR.nbits == chem.RICH_FP_BITS, \
    "attach the k28 fprich model"
G2M = g2.G2Model(paths.find("g2.pt"), G=GR)
res_g = json.load(open(paths.find("result_g2.json")))
assert (res_g["gate"] or not REQUIRE_GATES) and G2M.step == res_g["step"] and not G2M.amp, \
    "attach the k29 G2 model (fp32 inference)"
assert not REQUIRE_CUDA or (FWD.device == MR.device == G2M.device == "cuda"), "fwd / fprich / G2 must run on the GPU"
FT = frag.FragTable(frag.FragTable.find_dir())
fmeta = json.load(open(paths.find("frag_meta.json")))
INFO = dict(fpnet="fpnet2.pt", variant=res[0]["variant"], epoch=int(MODEL.epoch), prep=MODEL.prep,
            fwd=use["use"], fwd_file=use["file"], fwd_epoch=int(FWD.epoch), fwd_device=FWD.device, fwd_precision="fp32",
            frag_n=fmeta["n"], frag_total=fmeta["n_frag"], rdkit=rdkit.__version__,
            fprich="fprich.pt", fprich_epoch=int(MR.epoch), fprich_nbits=int(MR.nbits), fprich_device=MR.device,
            g2="g2.pt", g2_step=int(G2M.step), g2_device=G2M.device, g2_precision="fp32", g2_graphs=float(GR.ok.mean()),
            k30="v1")
print("features:", INFO)
assert len(FT.off) == len(P) + 1, "fragment table does not match the pool"
S = search.Searcher(C, P, cfg=SEARCH_CFG, model=MODEL, fwd=FWD, frag=FT, fprich=(MR, FP2), g2=G2M)
print("columns:", len(S.feature_names), S.feature_names[-13:])
Q = pd.read_csv(f"{pool.Pool.find_dir()}/queries.csv")
Q["rows"] = Q.rows.map(lambda s: [int(x) for x in str(s).split(",")])
pids = np.sort(Q.pid.unique())
Q = Q[Q.pid.isin(set(pids[SHARD::N_SHARDS]))]
print(f"shard {SHARD}/{N_SHARDS}:", Q.set.value_counts().to_dict())
json.dump(INFO, open(f"{OUT}/features_info4.json", "w"), indent=1)

# %% [markdown]
# ## 生成（特徴量の直後に、Searcher.last から）

# %%
GEN, GINFO = {s: [] for s in GEN_SETS}, []
GCOLS = gen.GEN_FEATURES + gen.GEN_EXTRA


def after(q, specs, cand, X, info):
    if q.set not in GEN_SETS:
        return
    t0 = time.time()
    tk = derive.ik14(P.smiles[q.pid])
    prods, Xg, st = gen.for_query(S, specs, cand, info, keep_key=tk, fwd=FWD)
    if prods:
        Xe = gen.extra_features(prods, specs, S.last["z_rich"], G2M)
        d = pd.DataFrame(np.c_[Xg, Xe], columns=GCOLS)
        d.insert(0, "seq", [p["seq"] for p in prods])
        d.insert(0, "src", [p["src"] for p in prods])
        d.insert(0, "label", np.array([p["key"] == tk for p in prods], np.int8))
        d.insert(0, "key", [p["key"] for p in prods])
        d.insert(0, "qid", np.int32(q.qid))
        GEN[q.set].append(d)
    GINFO.append(dict(qid=q.qid, set=q.set, n_gen=len(prods), has_truth=any(p["key"] == tk for p in prods),
                      secs=time.time() - t0, **st))


# %% [markdown]
# ## 公開テストでの時間（shard 0 のみ）

# %%
if SHARD == 0 and N_TEST:
    te = pd.read_parquet(os.path.join(paths.comp_dir(), "test.parquet"))
    tim = []
    for mid, g in list(te.groupby("molecule_id", sort=False))[:N_TEST]:
        specs = harness.test_specs(g)
        t0 = time.time()
        cand, X, info = S.features(specs)
        t1 = time.time()
        prods, Xg, st = gen.for_query(S, specs, cand, info, fwd=FWD)
        Xe = gen.extra_features(prods, specs, S.last["z_rich"], G2M)
        tim.append(dict(n_cand=len(cand), feat_secs=t1 - t0, gen_secs=time.time() - t1, n_gen=len(prods),
                        lib_grp_max=info["lib_max"]))
    tim = pd.DataFrame(tim)
    tim.to_csv(f"{OUT}/test_timing4.csv", index=False)
    print(f"public test ({len(tim)} molecules): features {tim.feat_secs.mean():.2f}s, generation "
          f"{tim.gen_secs.mean():.2f}s (max {tim.gen_secs.max():.1f}) per molecule, products median "
          f"{tim.n_gen.median():.0f} ({time.time() - T0:.0f}s)", flush=True)

# %% [markdown]
# ## 特徴量と生成（A / B_A / C / B_C / V を先に、R1 / R2 は構造を対にして 100 構造ずつ）

# %%
keep_rule = lambda X: [search.rule_score(X, "lib_analog"), search.rule_score(X, "fz")]
feats, infos = {}, []
for name in SETS:
    f, info = harness.run_queries(S, Q[Q.set == name], keep_rule=keep_rule, max_cand_train=MAX_CAND_TRAIN,
                                  after=after)
    feats[name] = f
    infos.append(info)
    gi = pd.DataFrame([g for g in GINFO if g["set"] == name])
    print(f"set {name}: {len(info)} queries, recall {info.recall.mean():.3f}, median candidates "
          f"{info.n_cand.median():.0f}, {info.secs.mean():.2f}s/query"
          + (f" | generation {gi.secs.mean():.2f}s/query, products median {gi.n_gen.median():.0f}, "
             f"truth generated {gi.has_truth.mean():.3f}" if len(gi) else "") + f" ({time.time() - T0:.0f}s)",
          flush=True)
r_pids = Q[Q.set == "R1"].pid.values
parts = {"R1": [], "R2": []}
t_start = time.time()
for i in range(0, len(r_pids), 100):
    if time.time() - t_start > BUDGET_R_SECS:
        print(f"R budget reached after {i} structures")
        break
    chunk = set(r_pids[i:i + 100])
    for name in ["R1", "R2"]:
        parts[name].append(harness.run_queries(S, Q[(Q.set == name) & Q.pid.isin(chunk)], keep_rule=keep_rule,
                                               max_cand_train=MAX_CAND_TRAIN, log=lambda *a: None, after=after))
    if (i // 100) % 2 == 0:
        print(f"  R {min(i + 100, len(r_pids))}/{len(r_pids)} structures ({time.time() - T0:.0f}s)", flush=True)
for name in ["R1", "R2"]:
    if parts[name]:
        feats[name] = pd.concat([x[0] for x in parts[name]], ignore_index=True)
        infos.append(pd.concat([x[1] for x in parts[name]], ignore_index=True))
pd.concat(infos, ignore_index=True).to_csv(f"{OUT}/query_info4_s{SHARD}.csv", index=False)
for name, f in feats.items():
    f.to_parquet(f"{OUT}/feats4_{name}_s{SHARD}.parquet", index=False)
for name, parts_g in GEN.items():
    g = pd.concat(parts_g, ignore_index=True) if parts_g else pd.DataFrame(
        columns=["qid", "key", "label", "src", "seq"] + GCOLS)
    g.to_parquet(f"{OUT}/gen4_{name}_s{SHARD}.parquet", index=False)
gi = pd.DataFrame(GINFO)
gi.to_csv(f"{OUT}/gen_info4_s{SHARD}.csv", index=False)
new = [c for c in S.feature_names if c in search.FPRICH_FEATURES + search.G2_FEATURES]
nan_share = {n: float(np.mean([f[c].isna().mean() for c in new])) for n, f in feats.items() if len(f)}
print({k: len(v) for k, v in feats.items()}, "| NaN share of the new columns:", nan_share, "| generated:",
      gi.groupby("set").agg(n=("qid", "size"), truth=("has_truth", "mean"), n_gen=("n_gen", "median"),
                            secs=("secs", "mean")).round(3).to_dict("index"), f"total {time.time() - T0:.0f}s")

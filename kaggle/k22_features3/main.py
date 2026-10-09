# %% [markdown]
# # casmi26-k22-features3（分割）
# M5 P1 の 2（2026-10-10）: プールの候補の特徴量（今の 34 列 + 断片 4 列 + fwd 3 列）と、クラス3の生成物の一覧と
# その特徴量を作る（GPU は fwd の推論に使う）。
# - プール側: k07 v9 と同じクエリ・同じ切り詰め（R1 / R2 は lib_analog と fz の上位 300 ずつ、正解は残す）。
#   Searcher に fwd（k25 の use_fwd.json が選んだ v1 / v2）と断片の表（k21）を渡し、今の 34 列の後ろに列を足す
# - 生成側: A、B_A、V、R2 のクエリで casmi26.gen（k19 と同じ手順）。窓の候補の鍵と同じ生成物は除くが、正解の鍵は
#   残して label = 1 を付ける（B の評価では落とし、G の評価では使う）。生成物の特徴量はスペクトルの証拠だけ
#   （GEN_FEATURES）。反応の列（src、seq）は表示用で、並べ方には使わない
# - shard 0 は公開テスト（400 分子、train の抜粋）で特徴量と生成の時間を測る（k24 の見積もり）
# 出力: feats_{set}_s{i}.parquet、query_info_s{i}.csv、gen_{set}_s{i}.parquet、gen_info_s{i}.csv、
# features_info.json（FPNet、fwd、断片の版）、test_timing.csv（shard 0）。
# 入力: k01、k05、k05b、k15-s1（fpnet2.pt = v2b）、k20（fwd.pt）、k25（fwd2.pt、use_fwd.json）、k21（断片の表）、
# casmi26-wheels（rdkit。k24 と同じ版にするため pip のネットからは入れない）。

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

from casmi26 import cache, derive, fpnet, frag, fwd, gen, harness, paths, pool, search

SHARD, N_SHARDS = globals().get("SHARD", 0), globals().get("N_SHARDS", 1)
T0 = time.time()
OUT = "/kaggle/working"
BUDGET_R_SECS = 6 * 3600
MAX_CAND_TRAIN = 300
SETS = ["A", "B_A", "C", "B_C", "V"]  # R1 / R2 は別に、構造を対にして処理する
GEN_SETS = {"A", "B_A", "V", "R2"}
N_TEST = 400  # shard 0 の公開テストの時間の計測（分子数）
SEARCH_CFG = search.SearchCfg()  # 手元の動作確認では窓を広げる
REQUIRE_CUDA = True  # fwd は GPU で（分割ごとに装置が違うと特徴量がずれる。手元の動作確認は False）

MODEL = fpnet.load_checked("fpnet2.pt", min_epoch=1)
res = [json.load(open(p)) for p in glob.glob("/kaggle/input/**/result_v2*.json", recursive=True)]
assert len(res) == 1 and res[0]["epoch"] == MODEL.epoch, "attach exactly the k15 variant of fpnet2.pt"
use = json.load(open(paths.find("use_fwd.json")))
FWD = fwd.FwdModel(paths.find(use["file"]))
assert FWD.device == "cuda" or not REQUIRE_CUDA, "fwd must run on the GPU"
FT = frag.FragTable(frag.FragTable.find_dir())
fmeta = json.load(open(paths.find("frag_meta.json")))
INFO = dict(fpnet="fpnet2.pt", variant=res[0]["variant"], epoch=int(MODEL.epoch), prep=MODEL.prep,
            fwd=use["use"], fwd_file=use["file"], fwd_epoch=int(FWD.epoch), fwd_device=FWD.device, fwd_precision="fp32",
            frag_n=fmeta["n"], frag_total=fmeta["n_frag"], rdkit=rdkit.__version__, k22="v1")
print("features:", INFO)
C = cache.Cache(cache.Cache.find_dir())
P = pool.Pool(pool.Pool.find_dir())
assert len(P) == P.n_base, "the PubChem part must not be attached"
assert len(FT.off) == len(P) + 1, "fragment table does not match the pool"
S = search.Searcher(C, P, cfg=SEARCH_CFG, model=MODEL, fwd=FWD, frag=FT)
print("columns:", len(S.feature_names), S.feature_names[-7:])
Q = pd.read_csv(f"{pool.Pool.find_dir()}/queries.csv")
Q["rows"] = Q.rows.map(lambda s: [int(x) for x in str(s).split(",")])
pids = np.sort(Q.pid.unique())
Q = Q[Q.pid.isin(set(pids[SHARD::N_SHARDS]))]
print(f"shard {SHARD}/{N_SHARDS}:", Q.set.value_counts().to_dict())
json.dump(INFO, open(f"{OUT}/features_info.json", "w"), indent=1)

# %% [markdown]
# ## 生成（特徴量の直後に、Searcher.last から）

# %%
GEN, GINFO = {s: [] for s in GEN_SETS}, []


def after(q, specs, cand, X, info):
    if q.set not in GEN_SETS:
        return
    t0 = time.time()
    tk = derive.ik14(P.smiles[q.pid])
    prods, Xg, st = gen.for_query(S, specs, cand, info, keep_key=tk, fwd=FWD)
    if prods:
        d = pd.DataFrame(Xg, columns=gen.GEN_FEATURES)
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
        tim.append(dict(n_cand=len(cand), feat_secs=t1 - t0, gen_secs=time.time() - t1, n_gen=len(prods),
                        lib_grp_max=info["lib_max"]))
    tim = pd.DataFrame(tim)
    tim.to_csv(f"{OUT}/test_timing.csv", index=False)
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
pd.concat(infos, ignore_index=True).to_csv(f"{OUT}/query_info_s{SHARD}.csv", index=False)
for name, f in feats.items():
    f.to_parquet(f"{OUT}/feats_{name}_s{SHARD}.parquet", index=False)
for name, parts_g in GEN.items():
    g = pd.concat(parts_g, ignore_index=True) if parts_g else pd.DataFrame(
        columns=["qid", "key", "label", "src", "seq"] + gen.GEN_FEATURES)
    g.to_parquet(f"{OUT}/gen_{name}_s{SHARD}.parquet", index=False)
gi = pd.DataFrame(GINFO)
gi.to_csv(f"{OUT}/gen_info_s{SHARD}.csv", index=False)
print({k: len(v) for k, v in feats.items()}, "| generated:",
      gi.groupby("set").agg(n=("qid", "size"), truth=("has_truth", "mean"), n_gen=("n_gen", "median"),
                            secs=("secs", "mean")).round(3).to_dict("index"), f"total {time.time() - T0:.0f}s")

# %% [markdown]
# # casmi26-k07-features2（分割実行）
# 候補プール（train ∪ COCONUT）で検証セットの特徴量を計算する。構造（pid）単位で N_SHARDS 個に分け、
# このカーネルは SHARD 番目だけを処理する（A と B_A、R1 と R2 は同じ構造なので同じ分割に入る）。
# 結合・評価・ルールの選択は k08_eval2 で行う。
# SHARD / N_SHARDS は tools/build_kernel.py が先頭のセルで注入する（単独実行時は 0 / 1）。

# %%
import os
import time

import numpy as np
import pandas as pd

from casmi26 import cache, harness, paths, pool, search, spectra

SHARD, N_SHARDS = globals().get("SHARD", 0), globals().get("N_SHARDS", 1)
T0 = time.time()
OUT = "/kaggle/working"
BUDGET_R_SECS = 8 * 3600

C = cache.Cache(cache.Cache.find_dir())
P = pool.Pool(pool.Pool.find_dir())
S = search.Searcher(C, P)
Q = pd.read_csv(f"{pool.Pool.find_dir()}/queries.csv")
Q["rows"] = Q.rows.map(lambda s: [int(x) for x in str(s).split(",")])
Q = Q[Q.set != "V"]
pids = np.sort(Q.pid.unique())
Q = Q[Q.pid.isin(set(pids[SHARD::N_SHARDS]))]
print(f"shard {SHARD}/{N_SHARDS}:", Q.set.value_counts().to_dict())

# %% [markdown]
# ## 正当性チェック（shard 0 のみ）: 公開 test（train の抜粋）で lib_max ≥ 0.99 の候補がほぼ全分子にあるか

# %%
if SHARD == 0:
    te = pd.read_parquet(os.path.join(paths.comp_dir(), "test.parquet"))
    tops = []
    for mid, g in te.groupby("molecule_id", sort=False):
        specs = [dict(mz=m, it=i, prec=float(r.precursor_mz), adduct=r.adduct,
                      mode=1 if r.ionization_mode == "positive" else -1)
                 for r in g.itertuples()
                 for m, i in [spectra.clean_one(r.ms2_mzs, r.ms2_normalized_intensities, r.precursor_mz)]]
        cand, X, info = S.features(specs)
        tops.append(dict(n_cand=len(cand), lib_max=info["lib_max"], secs=info["secs"]))
    tops = pd.DataFrame(tops)
    print("public test: lib_max >= 0.99 for", f"{(tops.lib_max >= 0.99).mean():.3f}",
          f"| median candidates {tops.n_cand.median():.0f} | {tops.secs.mean():.2f}s/molecule ({time.time() - T0:.0f}s)")

# %% [markdown]
# ## 特徴量の計算
# A / B_A / C / B_C を先に、R1 / R2 は同じ構造を対にして 100 構造ずつ処理する（時間切れでも両方が揃う）。

# %%
keep_rule = lambda X: search.rule_score(X, "lib_analog")
feats, infos = {}, []
for name in ["A", "B_A", "C", "B_C"]:
    f, info = harness.run_queries(S, Q[Q.set == name], keep_rule=keep_rule)
    feats[name] = f
    infos.append(info)
    print(f"set {name}: {len(info)} queries, recall {info.recall.mean():.3f}, median candidates "
          f"{info.n_cand.median():.0f}, {info.secs.mean():.2f}s/query ({time.time() - T0:.0f}s)")
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
                                               log=lambda *a: None))
    print(f"  R {min(i + 100, len(r_pids))}/{len(r_pids)} structures ({time.time() - T0:.0f}s)", flush=True)
for name in ["R1", "R2"]:
    if parts[name]:
        feats[name] = pd.concat([x[0] for x in parts[name]], ignore_index=True)
        infos.append(pd.concat([x[1] for x in parts[name]], ignore_index=True))
pd.concat(infos, ignore_index=True).to_csv(f"{OUT}/query_info_s{SHARD}.csv", index=False)
for name, f in feats.items():
    f.to_parquet(f"{OUT}/feats_{name}_s{SHARD}.parquet", index=False)
print({k: len(v) for k, v in feats.items()}, f"total {time.time() - T0:.0f}s")

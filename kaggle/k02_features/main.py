# %% [markdown]
# # casmi26-k02-features
# 検証ハーネスのクエリセット A / C / T を作り、候補ごとの特徴量を計算して保存する。
# 最初に公開 test.parquet で正当性を確認し、最後にルールベースのスコアで MRR@25 を出す。

# %%
import os
import time

import numpy as np
import pandas as pd

from casmi26 import cache, harness, paths, search, spectra

T0 = time.time()
N_C, N_T = 1000, 4000
BUDGET_T_SECS = 7 * 3600  # T セットの計算はこの時間で打ち切る
OUT = "/kaggle/working"

C = cache.Cache(cache.Cache.find_dir())
S = search.Searcher(C)
te = pd.read_parquet(os.path.join(paths.comp_dir(), "test.parquet"))

# %% [markdown]
# ## 正当性チェック: 公開 test は train の抜粋なので、ほぼ全分子でライブラリ類似度 ≈ 1 の候補があるはず
# ここが 0.95 を大きく下回るなら、前処理・アダクト解釈・質量窓のどこかにバグがある。

# %%
tops = []
for mid, g in te.groupby("molecule_id", sort=False):
    specs = []
    for r in g.itertuples():
        mz, it = spectra.clean_one(r.ms2_mzs, r.ms2_normalized_intensities, r.precursor_mz)
        specs.append(dict(mz=mz, it=it, prec=float(r.precursor_mz), adduct=r.adduct,
                          mode=1 if r.ionization_mode == "positive" else -1))
    cand, X, info = S.features(specs)
    tops.append(dict(molecule_id=mid, n_cand=len(cand), lib_max=info["lib_max"], secs=info["secs"]))
tops = pd.DataFrame(tops)
tops.to_csv(f"{OUT}/public_test_check.csv", index=False)
print("public test: lib_max >= 0.99 for", f"{(tops.lib_max >= 0.99).mean():.3f}",
      "| >= 0.9:", f"{(tops.lib_max >= 0.9).mean():.3f}",
      "| no candidates:", int((tops.n_cand == 0).sum()),
      f"| median candidates {tops.n_cand.median():.0f} | {tops.secs.mean():.2f}s/molecule ({time.time() - T0:.0f}s)")

# %% [markdown]
# ## クエリセットの作成

# %%
k_dist = te.groupby("molecule_id").size().value_counts().to_dict()
print("spectra per molecule in public test:", dict(sorted(k_dist.items())))
Q = harness.build_sets(C, k_dist, n_c=N_C, n_t=N_T, seed=0)
Q.assign(rows=Q.rows.map(lambda r: ",".join(map(str, r)))).to_csv(f"{OUT}/queries.csv", index=False)
print(Q.groupby("set").agg(n=("qid", "size"), spectra=("rows", lambda s: s.map(len).mean())))
print("T query libraries:", Q[Q.set == "T"].qlib.map(lambda l: C.libs[l]).value_counts().to_dict())

# %% [markdown]
# ## 特徴量の計算（A → C → T の順）

# %%
keep_rule = lambda X: search.rule_score(X, "lib_analog")
feats, infos = {}, []
for name in ["A", "C", "T"]:
    q = Q[Q.set == name]
    if name == "T":
        # 打ち切りに備えて、250 クエリずつ処理する
        t_start = time.time()
        done = []
        for i in range(0, len(q), 250):
            if time.time() - t_start > BUDGET_T_SECS:
                print(f"T budget reached after {i} queries")
                break
            done.append(harness.run_queries(S, q.iloc[i:i + 250], keep_rule=keep_rule))
        if not done:
            continue
        f = pd.concat([d[0] for d in done], ignore_index=True)
        info = pd.concat([d[1] for d in done], ignore_index=True)
    else:
        f, info = harness.run_queries(S, q, keep_rule=keep_rule)
    f.to_parquet(f"{OUT}/feats_{name}.parquet", index=False)
    feats[name] = f
    infos.append(info)
    print(f"set {name}: {len(info)} queries, {len(f):,} rows, recall {info.recall.mean():.3f}, "
          f"median candidates {info.n_cand.median():.0f}, {info.secs.mean():.2f}s/query ({time.time() - T0:.0f}s)")
infos = pd.concat(infos, ignore_index=True)
infos.to_csv(f"{OUT}/query_info.csv", index=False)

# %% [markdown]
# ## ルールベースの評価
# T はクエリ元ライブラリ別にも出す。特定ライブラリだけ top1 が極端に高ければ、複製スペクトルによる見かけの高さを疑う。

# %%
res = []
for name, f in feats.items():
    X = f[search.FEATURES].values
    inf = infos[infos.set == name]
    for rule in search.RULES:
        r = harness.ranks_of_truth(f.qid.values, f.label.values, search.rule_score(X, rule), f.sid.values)
        r = r.reindex(inf.qid.values).fillna(np.inf)  # 候補が無かったクエリも分母に含める
        res.append(dict(set=name, model=f"rule:{rule}", **harness.metrics(r.values)))
        if name == "T" and rule == "lib_analog":
            for lib, qids in inf.groupby("qlib").qid:
                res.append(dict(set=f"T[{lib}]", model=f"rule:{rule}", **harness.metrics(r.reindex(qids.values).values)))
res = pd.DataFrame(res)
res.to_csv(f"{OUT}/results.csv", index=False)
print(res.round(4).to_string(index=False))
print(f"total {time.time() - T0:.0f}s")

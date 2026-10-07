# %% [markdown]
# # casmi26-k07-features2（分割実行）
# 候補プール（train ∪ COCONUT）で検証セットの特徴量を計算する。構造（pid）単位で N_SHARDS 個に分け、
# このカーネルは SHARD 番目だけを処理する（A と B_A、R1 と R2 は同じ構造なので同じ分割に入る）。
# 結合・評価・ルールの選択は k08_eval2 で行う。
# v3（M2）: k06 の指紋予測モデルで f·z の特徴量も計算する。R の候補は lib_analog と fz それぞれの上位 300 件の和集合を残す。
# v5（M3）: 候補プールに PubChem 部分（段階 S′ = 文献数 ≥ 1。k05p v1 の出力をデータセット casmi26-pubchem-part にしたもの）をつなぐ。つながっていなければ止める
#   （Pool は pc_meta.json が無いと黙って PubChem なしで動くため）。候補が増えるので、R の切り詰めの上限を
#   keep_rule ごとに 300 → 1,000 に上げる。
# v7（物差しの見直し、2026-10-07）: プールの段階 0（PubChem なし。M1〜M2.2 と同じ）と段階 1（S′）の両方で計算し、
#   出力の名前に段階を入れる（feats_<set>_L<段階>_s<分割>）。R の切り詰めの上限は当時に合わせて段階 0 は 300、
#   段階 1 は 1,000。FPNet の検証用の V（2,000 構造、正解のスペクトルをすべて隠す）も計算する。V は FPNet の学習に
#   使われていないので、R2 と合わせて大きなクラス2の代理（正解が COCONUT にあるもの）に使う。評価は k13_yardstick。
# SHARD / N_SHARDS は tools/build_kernel.py が先頭のセルで注入する（単独実行時は 0 / 1）。

# %%
import os
import time

import numpy as np
import pandas as pd

from casmi26 import cache, fpnet, harness, paths, pool, search

SHARD, N_SHARDS = globals().get("SHARD", 0), globals().get("N_SHARDS", 1)
T0 = time.time()
OUT = "/kaggle/working"
BUDGET_R_SECS = 4 * 3600  # 段階ごと
LEVELS = {0: 300, 1: 1000}  # プールの段階 -> R1/R2 の切り詰めの上限（keep_rule ごと）
SETS = ["A", "B_A", "C", "B_C", "V"]  # R1 / R2 は別に、構造を対にして処理する

MODEL = fpnet.load_checked()  # k06 の重み（データセット casmi26-fpnet。エポック 10 未満なら止める）
C = cache.Cache(cache.Cache.find_dir())
P = pool.Pool(pool.Pool.find_dir())
assert P.level.max() == 1 and len(P) > P.n_base, "PubChem part (dataset casmi26-pubchem-part) is not attached"
print(f"PubChem part: {len(P) - P.n_base:,} structures, levels {P.pc_meta['levels']}")
S = search.Searcher(C, P, model=MODEL)
Q = pd.read_csv(f"{pool.Pool.find_dir()}/queries.csv")
Q["rows"] = Q.rows.map(lambda s: [int(x) for x in str(s).split(",")])
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
        specs = harness.test_specs(g)
        cand, X, info = S.features(specs)
        agree = (float(np.argmax(X[:, search.COL["fz"]]) == np.argmax(search.rule_score(X, "lib_analog")))
                 if len(cand) else np.nan)
        tops.append(dict(n_cand=len(cand), lib_max=info["lib_max"], secs=info["secs"], fz_agree=agree,
                         ce_known=np.mean([np.isfinite(s["ce"]) for s in specs]), tims=specs[0]["tims"]))
    tops = pd.DataFrame(tops)
    print("public test: lib_max >= 0.99 for", f"{(tops.lib_max >= 0.99).mean():.3f}",
          f"| fz top-1 == library top-1 for {tops.fz_agree.mean():.3f}",
          f"| CE known {tops.ce_known.mean():.3f} | timsTOF {tops.tims.mean():.3f}",
          f"| median candidates {tops.n_cand.median():.0f} | {tops.secs.mean():.2f}s/molecule ({time.time() - T0:.0f}s)")

# %% [markdown]
# ## 特徴量の計算（段階ごと）
# A / B_A / C / B_C / V を先に、R1 / R2 は同じ構造を対にして 100 構造ずつ処理する（時間切れでも両方が揃う）。

# %%
keep_rule = lambda X: [search.rule_score(X, "lib_analog"), search.rule_score(X, "fz")]
for level, cap in LEVELS.items():
    P.set_level(level)
    feats, infos = {}, []
    for name in SETS:
        f, info = harness.run_queries(S, Q[Q.set == name], keep_rule=keep_rule, max_cand_train=cap)
        feats[name] = f
        infos.append(info)
        print(f"L{level} set {name}: {len(info)} queries, recall {info.recall.mean():.3f}, median candidates "
              f"{info.n_cand.median():.0f}, {info.secs.mean():.2f}s/query ({time.time() - T0:.0f}s)", flush=True)
    r_pids = Q[Q.set == "R1"].pid.values
    parts = {"R1": [], "R2": []}
    t_start = time.time()
    for i in range(0, len(r_pids), 100):
        if time.time() - t_start > BUDGET_R_SECS:
            print(f"L{level} R budget reached after {i} structures")
            break
        chunk = set(r_pids[i:i + 100])
        for name in ["R1", "R2"]:
            parts[name].append(harness.run_queries(S, Q[(Q.set == name) & Q.pid.isin(chunk)], keep_rule=keep_rule,
                                                   max_cand_train=cap, log=lambda *a: None))
        if (i // 100) % 5 == 0:
            print(f"  L{level} R {min(i + 100, len(r_pids))}/{len(r_pids)} structures ({time.time() - T0:.0f}s)",
                  flush=True)
    for name in ["R1", "R2"]:
        if parts[name]:
            feats[name] = pd.concat([x[0] for x in parts[name]], ignore_index=True)
            infos.append(pd.concat([x[1] for x in parts[name]], ignore_index=True))
    pd.concat(infos, ignore_index=True).assign(level=level).to_csv(f"{OUT}/query_info_L{level}_s{SHARD}.csv",
                                                                    index=False)
    for name, f in feats.items():
        f.to_parquet(f"{OUT}/feats_{name}_L{level}_s{SHARD}.parquet", index=False)
    print(f"L{level}:", {k: len(v) for k, v in feats.items()}, f"total {time.time() - T0:.0f}s", flush=True)
    del feats, infos, parts

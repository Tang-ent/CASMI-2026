# %% [markdown]
# # casmi26-k05-pool
# 候補プール（train ∪ COCONUT）を作る。両方の SMILES を同じ互変異性体正規化にかけてから指紋とキーを作り、
# キー単位で重複を除く。あわせて検証セット（A / B_A / C / B_C / R1 / R2 / V）と、指紋モデルの学習から除外する
# ホールドアウト一覧を作って保存する。

# %%
import subprocess
import sys

r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rdkit"], capture_output=True, text=True)
print(r.stdout[-500:], r.stderr[-500:])
import rdkit

print("rdkit", rdkit.__version__)

# %%
import json
import os
import time

import numpy as np
import pandas as pd

from casmi26 import cache, harness, paths, pool

T0 = time.time()
OUT = "/kaggle/working"
C = cache.Cache(cache.Cache.find_dir())
zip_path, release = pool.download_coconut("/tmp/coconut.zip")  # 出力（/kaggle/working）には残さない
print("COCONUT release", release)

# %%
meta = pool.build(C, zip_path, f"{OUT}/pool", release=release, workers=os.cpu_count() or 4)
os.remove(zip_path)
print(json.dumps(meta, indent=1))

# %% [markdown]
# ## 検証セットとホールドアウト

# %%
P = pool.Pool(f"{OUT}/pool")
te = pd.read_parquet(os.path.join(paths.comp_dir(), "test.parquet"))
k_dist = te.groupby("molecule_id").size().value_counts().to_dict()
Q, hold = harness.build_sets(C, P, k_dist, n_c=1000, n_r=3000, n_v=2000, seed=0)
Q.assign(rows=Q.rows.map(lambda r: ",".join(map(str, r)))).to_csv(f"{OUT}/pool/queries.csv", index=False)
hold.to_csv(f"{OUT}/pool/holdout.csv", index=False)
print(Q.groupby("set").agg(n=("qid", "size"), spectra=("rows", lambda s: s.map(len).mean())).to_string())
print("holdout structures:", hold.set.value_counts().to_dict(), f"({len(hold) / len(P):.2%} of the pool)")
print("R query libraries:", Q[Q.set == "R1"].qlib.map(lambda l: C.libs[l]).value_counts().to_dict())

# %% [markdown]
# ## チェック: np-examples（A）の構造のうち COCONUT にもある割合

# %%
pq_ = pd.read_parquet(f"{OUT}/pool/pool.parquet", columns=["in_train", "in_coconut"])
for s in ["A", "C", "R1", "V"]:
    p = Q[Q.set == s].pid.values
    print(f"{s}: {len(p)} structures, in COCONUT {pq_.in_coconut.values[p].mean():.2%}")
print("pool:", len(pq_), "| train only", int((pq_.in_train & ~pq_.in_coconut).sum()),
      "| COCONUT only", int((~pq_.in_train & pq_.in_coconut).sum()), "| both", int((pq_.in_train & pq_.in_coconut).sum()))
for fn in sorted(os.listdir(f"{OUT}/pool")):
    print(f"{fn:16s} {os.path.getsize(f'{OUT}/pool/{fn}') / 1e6:9.1f} MB")
print(f"total {time.time() - T0:.0f}s")

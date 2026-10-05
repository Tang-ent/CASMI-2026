# %% [markdown]
# # casmi26-k05p-pubchem
# M3 の P1: k10 で選んだ PubChem の構造を互変異性体正規化し、候補プールの末尾に足す「PubChem 部分」を作る。
# 段階 S′ = 文献数 ≥ 1（CID-PMID に 1 件以上）の新しい構造。k10 の決め方で選ばれた系統「天然物らしさ ≥ t または
#   文献数 ≥ 1」の中で最も小さい段階。上限 150 万件の S は、文献数 ≥ 1 だけで 215 万件あるため作れず、
#   ユーザーの判断で S′ から作ることにした（2026-10-06）。
# 既存の pid はそのまま（PubChem 部分は pid ≥ 724,595）。ファイル名は pc_*（pool_* と重ならない）。
# 入力: k10（pubchem_new.parquet, levels.json）、k05（今のプールの鍵）、k05b（質量の修正、確認用）。
# CID-SMILES.gz は PubChem から読み直す（k10 は SMILES を保存していない）。

# %%
import subprocess
import sys

r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rdkit"], capture_output=True, text=True)
print(r.stdout[-300:], r.stderr[-300:])
import rdkit

print("rdkit", rdkit.__version__)

# %%
import json
import os
import shutil
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from casmi26 import paths, pool, pubchem

T0 = time.time()
OUT = "/kaggle/working"
W = os.cpu_count() or 4
LEVEL, LEVEL_DEF = 1, "S': n_pmid >= 1 (CID-PMID links), new structures (not in train / COCONUT) from k10"
DISKS = [d for d in ["/kaggle/tmp", "/tmp", "/kaggle/working"] if os.path.isdir(d)]
TMP = os.path.join(max(DISKS, key=lambda d: shutil.disk_usage(d).free), "pubchem_tmp")
os.makedirs(TMP, exist_ok=True)


def log(*a):
    print(*a, f"[{time.time() - T0:.0f}s]", flush=True)


print("cpus", W)

# %% [markdown]
# ## S′ の CID と SMILES

# %%
cols = ["cid", "n_pmid", "n_syn", "n_pat", "pcl_anno", "np"]
t = pq.read_table(paths.find("pubchem_new.parquet"), columns=cols, filters=[("n_pmid", ">=", 1)]).to_pandas()
t = t.sort_values(["n_pmid", "cid"], ascending=[False, True], kind="mergesort")  # 鍵が重なったら文献数の多い方を残す
log(f"S' candidates: {len(t):,} (k10 estimate 2,154,874)")
want = np.zeros(int(t.cid.max()) + 1, bool)
want[t.cid.values] = True
p, dl = pubchem.source("CID-SMILES.gz", TMP, log=log)
cids, smi = pubchem.scan_smiles(p, want, log)
if dl:
    os.remove(p)
t = t.merge(pd.DataFrame({"cid": cids, "smiles": smi}).drop_duplicates("cid"), on="cid", how="inner")
log(f"with SMILES: {len(t):,}")

# %% [markdown]
# ## 正規化して PubChem 部分を書く

# %%
PD = pool.Pool.find_dir()
base_keys = pd.read_parquet(os.path.join(PD, "pool.parquet"), columns=["key"]).key.values
k10_levels = json.load(open(paths.find("levels.json")))
meta = pool.build_pubchem_part(t, base_keys, f"{OUT}/pubchem", LEVEL, LEVEL_DEF, workers=W, log=log,
                               extra_meta=dict(n_selected=int(len(t)), k10_family=k10_levels.get("family")))
del t

# %% [markdown]
# ## 確認: プールにつながること、A の窓の候補数（PubChem なし / S′）

# %%
P = pool.Pool(PD)
assert len(P) == P.n_base + meta["n"] and P.n_base == len(base_keys)
assert not np.isin(P.key[P.n_base:], base_keys).any(), "PubChem part overlaps the pool"
assert np.isfinite(P.mass).all()
Q = pd.read_csv(os.path.join(PD, "queries.csv"), usecols=["set", "pid"])
for s in ["A", "C", "R1"]:
    m = P.mass[Q[Q.set == s].pid.unique()]
    cnt = {}
    for k in (0, LEVEL):
        P.set_level(k)
        cnt[k] = np.array([len(P.window(x, 10.0)) for x in m])
    print(f"{s}: candidates per +-10 ppm window, no PubChem median {np.median(cnt[0]):.0f} / p90 "
          f"{np.quantile(cnt[0], 0.9):.0f}; with S' median {np.median(cnt[LEVEL]):.0f} / p90 {np.quantile(cnt[LEVEL], 0.9):.0f}")
pc = pd.read_parquet(f"{OUT}/pubchem/pc_pool.parquet", columns=["mass", "n_pmid"])
print("PubChem part mass quantiles:", np.quantile(pc.mass, [0, 0.1, 0.5, 0.9, 1]).round(1).tolist())
for fn in sorted(os.listdir(f"{OUT}/pubchem")):
    print(f"{fn:18s} {os.path.getsize(f'{OUT}/pubchem/{fn}') / 1e6:9.1f} MB")
log("done")

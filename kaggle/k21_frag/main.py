# %% [markdown]
# # casmi26-k21-frag
# M5 P1 の 1（2026-10-10）: プールの全構造（train ∪ COCONUT、72.5 万）の、結合を切った断片の質量を前もって計算する
# （casmi26.frag。定義は計画で固定: 環に入っていない単結合 1 本・その 2 本の組・同じ非芳香環の結合 2 本の組を切った成分、
# 原子ともとの H の質量の和、30 Da 以上、1e-4 Da で丸めて重複を除く、1 構造あたり最大 2,000 件）。
# 出力: frag_off.npy（int64、pid ごとの区切り、長さ = プールの構造数 + 1）、frag_mass.npy（float32、pid ごとに昇順）、
# frag_meta.json（件数の分布、読めなかった構造の数、時間）。k22_features3 と k24_submit5 が使う。
# 入力: k05（pool.parquet）。インターネット: rdkit の pip install に使う。

# %%
import subprocess
import sys

r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rdkit"], capture_output=True, text=True)
print(r.stdout[-300:], r.stderr[-300:])
import rdkit

print("rdkit", rdkit.__version__)

# %%
import json
import time

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from casmi26 import frag, pool

RDLogger.DisableLog("rdApp.*")
T0 = time.time()
OUT = "/kaggle/working"
smiles = pd.read_parquet(f"{pool.Pool.find_dir()}/pool.parquet", columns=["smiles"]).smiles.values
print(f"{len(smiles):,} pool structures")
counts = np.zeros(len(smiles), np.int64)
parts, failed = [], 0
for i, s in enumerate(smiles):
    m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
    f = frag.fragments_of_mol(m) if m is not None else np.zeros(0)
    failed += m is None
    counts[i] = len(f)
    parts.append(f.astype(np.float32))
    if (i + 1) % 100_000 == 0:
        print(f"  {i + 1:,} ({time.time() - T0:.0f}s)", flush=True)
off = np.r_[0, np.cumsum(counts)].astype(np.int64)
mass = np.concatenate(parts).astype(np.float32)
np.save(f"{OUT}/frag_off.npy", off)
np.save(f"{OUT}/frag_mass.npy", mass)
meta = dict(n=len(smiles), n_frag=int(len(mass)), failed=int(failed), empty=int((counts == 0).sum()),
            median=float(np.median(counts)), p90=float(np.percentile(counts, 90)), max=int(counts.max()),
            at_cap=int((counts >= frag.MAX_FRAG).sum()), secs=time.time() - T0,
            definition=dict(min_mass=frag.MIN_MASS, max_frag=frag.MAX_FRAG, dec=frag.DEC))
json.dump(meta, open(f"{OUT}/frag_meta.json", "w"), indent=1)
print(meta)

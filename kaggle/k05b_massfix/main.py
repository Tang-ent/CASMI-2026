# %% [markdown]
# # casmi26-k05b-massfix
# 候補プールの質量補正。COCONUT には一部の分子がプロトン化した陽イオンの形で入っており、InChIKey14 では
# 中性の train 構造と同じキーにまとまる一方、分子式の質量が H+ 1 個分ずれて候補窓から外れていた。
# train にある構造は train の分子式の質量、それ以外の荷電体は RDKit で中性化した分子式の質量に置き換える。
# プールの id・指紋・ホールドアウトは変えない（k06 の学習と整合させるため）。

# %%
import subprocess
import sys

r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rdkit"], capture_output=True, text=True)
print(r.stdout[-300:], r.stderr[-300:])

# %%
import json

import numpy as np
import pandas as pd

from casmi26 import cache, pool

OUT = "/kaggle/working"
C = cache.Cache(cache.Cache.find_dir())
d = pool.Pool.find_dir()
mass, stats = pool.neutral_masses(d, C)
np.save(f"{OUT}/pool_mass.npy", mass)
json.dump(stats, open(f"{OUT}/pool_mass.json", "w"), indent=1)

raw = pd.read_parquet(f"{d}/pool.parquet")
diff = mass - raw["mass"].values
ch = np.abs(diff) > 1e-4
print(f"changed {ch.sum():,} of {len(raw):,}; shift (Da) quantiles:",
      np.round(np.quantile(diff[ch], [0.01, 0.5, 0.99]), 4) if ch.any() else "-")
print(raw.loc[ch, ["formula", "smiles"]].assign(shift=diff[ch]).head(10).to_string())

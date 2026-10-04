# %% [markdown]
# # casmi26-k01-cache
# train.parquet からスペクトルキャッシュ・構造表・代表スペクトル・Morgan 指紋を作り、`/kaggle/working/work` に保存する。
# 後続のノートブック（k02 / k04）はこの出力を入力として使う。

# %%
import subprocess
import sys

r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rdkit"], capture_output=True, text=True)
print(r.stdout[-500:], r.stderr[-500:])
import rdkit

print("rdkit", rdkit.__version__)

# %%
import os
import time

import numpy as np

from casmi26 import cache, chem, paths

COMP = paths.comp_dir()
OUT = "/kaggle/working/work"
print("competition data:", COMP)
info = cache.build(os.path.join(COMP, "train.parquet"), OUT, with_fp=True, workers=os.cpu_count() or 4)
info

# %% [markdown]
# ## 検証: 分子式由来の質量とプリカーサー由来の中性質量が一致するか

# %%
C = cache.Cache(OUT)
rows = np.flatnonzero(C.keep)
add_names = np.array(C.adducts, dtype=object)[C.adduct[rows]]
nm = np.array([chem.neutral_mass(p, a) for p, a in zip(C.prec[rows], add_names)])
sm = C.smass[C.sid[rows]]
ok = np.isfinite(nm)
dppm = (nm[ok] - sm[ok]) / sm[ok] * 1e6
print(f"adduct in mass table: {ok.mean():.2%} of kept spectra")
print("|neutral(prec) - formula mass| ppm quantiles:",
      {q: round(float(np.quantile(np.abs(dppm), q)), 2) for q in (0.5, 0.9, 0.99)})
rep = C.ppm[rows][ok]
f = np.isfinite(rep)
print("median |ours - reported precursor_error_ppm|:", round(float(np.median(np.abs(np.abs(dppm[f]) - np.abs(rep[f])))), 3))
for f_ in ["C21H28O5", "C5H14NO+", "C12H19N2O2+"]:
    print(f_, chem.formula_mass(f_))

# %%
for fn in sorted(os.listdir(OUT)):
    print(f"{fn:20s} {os.path.getsize(os.path.join(OUT, fn)) / 1e6:9.1f} MB")

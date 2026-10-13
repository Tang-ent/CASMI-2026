"""Smoke S' part for the M7 PubChem side list: the rows of the real S' (out/m3_k05p_v1/pubchem, = the
casmi26-pubchem-part dataset) within +-30 ppm of a smoke pool mass -> out/smoke/pcside/ (same file layout).
Run inside the Kaggle image: python setup_pcside.py <real S' dir> <smoke pool_mass.npy> <out dir>"""
import json
import os
import sys

import numpy as np
import pandas as pd

src, pool_mass, out = sys.argv[1:4]
m = np.load(os.path.join(src, "pc_mass.npy"))
pm = np.sort(np.load(pool_mass))
pm = pm[np.isfinite(pm)]
i = np.clip(np.searchsorted(pm, m), 1, len(pm) - 1)
near = np.minimum(np.abs(m - pm[i - 1]), np.abs(m - pm[i])) <= m * 30e-6
rows = np.flatnonzero(near)
os.makedirs(out, exist_ok=True)
pd.read_parquet(os.path.join(src, "pc_pool.parquet")).iloc[rows].reset_index(drop=True) \
    .to_parquet(os.path.join(out, "pc_pool.parquet"), index=False)
np.save(os.path.join(out, "pc_mass.npy"), m[rows])
np.save(os.path.join(out, "pc_fp.npy"), np.load(os.path.join(src, "pc_fp.npy"), mmap_mode="r")[rows])
meta = json.load(open(os.path.join(src, "pc_meta.json")))
json.dump(dict(meta, n=int(len(rows)), smoke=True), open(os.path.join(out, "pc_meta.json"), "w"), indent=1)
print(f"smoke S': {len(rows):,} of {len(m):,} rows near {len(pm)} pool masses -> {out}")

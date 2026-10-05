"""Build the smoke dataset (run inside the Kaggle image by tools/smoke/run.sh; repo at /data, output at /smoke).

530 train structures (80 np-examples, 150 enveda-180, 300 others) + a fake COCONUT zip -> cache (k01), pool (k05),
mass fix (k05b), queries/holdout (harness sets). Needs internet for `pip install rdkit`.
"""
import subprocess, sys, os, io, zipfile
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rdkit"], check=True)
sys.path.insert(0, "/data/src")
os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba_cache"
import numpy as np, pandas as pd, pyarrow.parquet as pq
TR = "/data/enveda-CASMI26-molecule-id-mass-spectra/train.parquet"
meta = pq.read_table(TR, columns=["ingest_lib", "inchikey14", "normalized_smiles"]).to_pandas()
libs = meta.groupby("inchikey14").ingest_lib.agg(set)
rng = np.random.default_rng(0)
npx = [k for k, s in libs.items() if "enveda-np-examples" in s]
e180 = [k for k, s in libs.items() if "enveda-180" in s and len(s) > 1]
other = [k for k, s in libs.items() if "enveda-180" not in s and len(s) > 1]
keys = list(rng.choice(npx, 80, replace=False)) + list(rng.choice(e180, 150, replace=False)) + list(rng.choice(other, 300, replace=False))
pq.write_table(pq.read_table(TR, filters=[("inchikey14", "in", keys)]), "/smoke/train.parquet", row_group_size=20000)
sm = meta.drop_duplicates("inchikey14").set_index("inchikey14").normalized_smiles
extra = rng.choice([k for k in sm.index if k not in set(keys)], 300, replace=False)
co = list(sm.loc[keys[::2]].values) + list(sm.loc[extra].values)
buf = io.StringIO(); pd.DataFrame({"identifier": range(len(co)), "canonical_smiles": co}).to_csv(buf, index=False)
with zipfile.ZipFile("/smoke/coconut.zip", "w") as z: z.writestr("coconut_csv.csv", buf.getvalue())
from casmi26 import cache, pool, harness
cache.build("/smoke/train.parquet", "/smoke/work", with_fp=False, log=lambda *a: None)
C = cache.Cache("/smoke/work")
pool.build(C, "/smoke/coconut.zip", "/smoke/pool", release="fake", workers=4, log=lambda *a: None)
mass, st = pool.neutral_masses("/smoke/pool", C)
os.makedirs("/smoke/k05b", exist_ok=True); np.save("/smoke/k05b/pool_mass.npy", mass)
P = pool.Pool("/smoke/pool")
te = pd.read_parquet("/data/enveda-CASMI26-molecule-id-mass-spectra/test.parquet")
Q, hold = harness.build_sets(C, P, te.groupby("molecule_id").size().value_counts().to_dict(), n_c=40, n_r=60, n_v=30)
Q.assign(rows=Q.rows.map(lambda r: ",".join(map(str, r)))).to_csv("/smoke/pool/queries.csv", index=False)
hold.to_csv("/smoke/pool/holdout.csv", index=False)
print("setup done, queries", len(Q), Q.set.value_counts().to_dict())

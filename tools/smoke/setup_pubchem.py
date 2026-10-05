"""Build the smoke PubChem files for k10 (run inside the Kaggle image by tools/smoke/run.sh; repo at /data).

Input: out/smoke/pubchem/raw/*.txt (the first lines of the real PubChem Extras files, fetched once on the host by
run.sh) and the smoke pool. Output: out/smoke/pubchem/CID-*.gz in the real format, where the smoke pool structures
are added as CIDs 100000 + pid (every 10th pid is left out, i.e. "not in PubChem"), plus a small PubChemLite CSV.
rdkit comes from out/smoke/wheels (run.sh sets PIP_NO_INDEX / PIP_FIND_LINKS).
"""
import gzip
import os
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rdkit"], check=True)
sys.path.insert(0, "/data/src")
import pandas as pd  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem.rdMolDescriptors import CalcMolFormula  # noqa: E402

from casmi26 import chem  # noqa: E402

RDLogger.DisableLog("rdApp.*")
D = "/data/out/smoke/pubchem"
raw = {f: open(f"{D}/raw/{f}.txt").read().splitlines()
       for f in ["CID-Mass", "CID-SMILES", "CID-InChI-Key", "CID-PMID", "CID-Synonym-filtered", "CID-Patent"]}
pp = pd.read_parquet("/data/out/smoke/data/pool/pool.parquet")
add = {k: [] for k in raw}
pcl = []
for pid, smi in enumerate(pp.smiles):
    if pid % 10 == 9:
        continue
    m = Chem.MolFromSmiles(smi)
    if m is None:
        continue
    cid = 100000 + pid
    f = CalcMolFormula(m)
    ik = Chem.MolToInchiKey(m)
    mass = chem.formula_mass(f)
    add["CID-Mass"].append(f"{cid}\t{f}\t{mass:.8f}\t{mass:.8f}")
    add["CID-SMILES"].append(f"{cid}\t{smi}")
    add["CID-InChI-Key"].append(f"{cid}\tInChI=1S/x\t{ik}")
    add["CID-PMID"] += [f"{cid}\t{900000 + j}\t1" for j in range(pid % 3)]
    add["CID-Synonym-filtered"] += [f"{cid}\tname {pid} {j}" for j in range(pid % 4 + 1)]
    add["CID-Patent"] += [f"{cid}\tUS-{pid}{j}-A1" for j in range(pid % 5 == 0)]
    if pid % 6 == 0:
        pcl.append((cid, ik[:14], pid % 3, pid % 5, pid % 4))
for line in raw["CID-InChI-Key"][:300]:
    c, _, k = line.split("\t")
    pcl.append((int(c), k[:14], 1, 2, 3))
for f in raw:
    with gzip.open(f"{D}/{f}.gz", "wt") as fh:
        fh.write("\n".join(raw[f] + add[f]) + "\n")
    print(f, len(raw[f]), "+", len(add[f]))
pd.DataFrame(pcl, columns=["Identifier", "FirstBlock", "PubMed_Count", "Patent_Count", "AnnoTypeCount"]).to_csv(
    f"{D}/PubChemLite_smoke.csv", index=False)
print("PubChemLite rows", len(pcl))

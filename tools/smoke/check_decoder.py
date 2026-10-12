"""Diagnostic (M6, 2026-10-12): the trained k33 decoder on the smoke R1 queries with the real v2b z and the true
formula, sampled with and without the formula constraint (as the k33 report). Run in docker with --gpus all:
mounts /src (code), /d (smoke data), /k26 (smoke k26), /m (decoder.pt, fpnet2.pt)."""
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "/src")
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem.rdMolDescriptors import CalcMolFormula  # noqa: E402

from casmi26 import cache, decoder, derive, fpnet, harness, pool, subform  # noqa: E402

RDLogger.DisableLog("rdApp.*")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
C = cache.Cache("/d/work")
P = pool.Pool("/d/pool", pubchem=False)
FT = subform.FormulaTable("/k26")
Q = pd.read_csv("/d/pool/queries.csv")
Q["rows"] = Q.rows.map(lambda s: [int(x) for x in str(s).split(",")])
r1 = Q[Q.set == "R1"].drop_duplicates("pid")
MB = fpnet.FPModel("/m/fpnet2.pt")
DM = decoder.DecoderModel("/m/decoder.pt")
Z = MB.zs([harness.query_specs(C, r) for r in r1.rows])
heavy_ix = [i for i in range(subform.NE) if i != subform.H_IX]
for constrain in (False, True):
    rows, t0 = [], time.time()
    for i, p in enumerate(r1.pid.values):
        f = FT.fid[p]
        tgt = FT.counts[f]
        out = DM.sample(Z[i], [(tgt, FT.supported[f])], [N], seed=int(p), constrain=constrain)
        tk = derive.ik14(P.smiles[p])
        n_ok = n_heavy = n_form = 0
        keys = set()
        for _, smi in out:
            m = Chem.MolFromSmiles(smi) if smi else None
            if m is None:
                continue
            n_ok += 1
            v, _ = subform.parse(CalcMolFormula(m))
            if (v[heavy_ix] == tgt[heavy_ix]).all():
                n_heavy += 1
                if v[subform.H_IX] == tgt[subform.H_IX]:
                    n_form += 1
                    keys.add(derive.ik14(smi))
        rows.append(dict(valid=n_ok / len(out), heavy_ok=n_heavy / len(out), formula_ok=n_form / len(out),
                         unique=len(keys), hit=tk in keys))
    d = pd.DataFrame(rows)
    print(f"constrain={constrain}: n={len(d)} valid {d.valid.mean():.3f} heavy_ok {d.heavy_ok.mean():.3f} "
          f"formula_ok {d.formula_ok.mean():.3f} unique median {d.unique.median():.0f} hit {d.hit.mean():.3f} "
          f"({time.time() - t0:.0f}s)", flush=True)

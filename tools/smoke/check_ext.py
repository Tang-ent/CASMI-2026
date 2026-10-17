"""ext.py checks (M7 P3, 2026-10-17), in the Kaggle docker image (tools/smoke/run.sh ext; GPU=1 for the CUDA path):
(a) entropy_sim (numba) == the probe's Python implementation (out/p3_probe/score_probe.py) on random pairs
(b) merge_predictions: identical m/z merged, sum 1; columns(): NaN convention (z 0, rank bottom, frank / fgap bottom)
(c) probe reproduction: 200 MassSpecGym test structures x CE 20/40/60, identity top-1 within 0.02 of the probe
    (GLACIER 0.840, ICEBERG 0.780; max similarity over the three CEs, as the probe)
(d) score_candidates on one probe structure against 5 others: its own score is the largest; a bad SMILES is NaN
(e) timing per prediction entry (and GLACIER with DataLoader workers when EXT_WORKERS > 0)
Mounts /src (code), /kaggle/input/casmi26-mspred (out/ds_mspred), /wheels (rdkit), /w (scratch)."""
import math
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "/src")
from casmi26 import ext  # noqa: E402

ok = True
rng = np.random.default_rng(0)
root = ext.install_mspred()
print("ms-pred root", root)


def py_entropy(mz1, i1, mz2, i2, ppm=20.0):  # out/p3_probe/score_probe.py (sparse_entropy_sim)
    i1 = i1 / i1.sum()
    i2 = i2 / i2.sum()
    m1, m2 = [], []
    used = np.zeros(len(mz2), bool)
    o2 = np.argsort(mz2)
    mz2, i2 = mz2[o2], i2[o2]
    for a, b in zip(mz1, i1):
        tol = a * ppm * 1e-6
        k = np.searchsorted(mz2, a - tol)
        hit = -1
        while k < len(mz2) and mz2[k] <= a + tol:
            if not used[k]:
                hit = k
                break
            k += 1
        if hit >= 0:
            used[hit] = True
            m1.append(b)
            m2.append(i2[hit])
        else:
            m1.append(b)
            m2.append(0.0)
    for k in np.where(~used)[0]:
        m1.append(0.0)
        m2.append(i2[k])
    p, q = np.array(m1), np.array(m2)

    def H(x):
        x = x[x > 0]
        return -(x * np.log(x)).sum()

    return 1 - (2 * H((p + q) / 2) - H(p) - H(q)) / math.log(4)


# (a)
mx = 0.0
for _ in range(300):
    n1, n2 = rng.integers(1, 60), rng.integers(1, 100)
    base = np.sort(rng.uniform(50, 600, max(n1, n2)))
    m1 = np.sort(rng.choice(base, n1, replace=False) * (1 + rng.normal(0, 5e-6, n1)))
    m2 = rng.choice(base, n2, replace=False) * (1 + rng.normal(0, 5e-6, n2))
    a, b = rng.random(n1), rng.random(n2)
    mx = max(mx, abs(ext.entropy_sim(m1, a, m2, b) - py_entropy(m1, a, m2, b)))
print(f"(a) entropy_sim vs probe implementation: max |diff| {mx:.2e}")
ok &= mx < 1e-9
s_self = ext.entropy_sim([100.0, 200.0], [1.0, 3.0], [200.0, 100.0], [3.0, 1.0])
ok &= abs(s_self - 1.0) < 1e-9 and np.isnan(ext.entropy_sim([], [], [1.0], [1.0]))

# (b)
m = ext.merge_predictions([(np.array([100.0, 150.0]), np.array([1.0, 1.0])), None,
                           (np.array([100.00001, 200.0]), np.array([2.0, 2.0]))])
ok &= len(m[0]) == 3 and abs(m[1].sum() - 1) < 1e-12 and abs(m[1][0] - 0.5) < 1e-12
v = np.array([0.5, np.nan, 0.2, 0.9, np.nan])
fid = np.array([1, 1, 1, 2, 2])
c = ext.columns(v, fid)
print("(b) columns:\n", np.round(c, 3))
ok &= c[1, 1] == 0 and c[4, 1] == 0  # z of unscored = 0
ok &= c[1, 2] == c[4, 2] > c[[0, 2, 3], 2].max()  # rank_norm: unscored below every scored row (ties averaged)
ok &= c[1, 3] == 3 and c[0, 3] == 1 and c[2, 3] == 2 and c[4, 3] == 2 and c[3, 3] == 1  # frank, unscored last
ok &= c[1, 4] < -5 and c[0, 4] == 0 and c[3, 4] == 0

# (c)
lab = pd.read_csv(os.path.join(root, "labels_200.tsv"), sep="\t")
true = pd.read_csv(os.path.join(root, "true_200.tsv"), sep="\t").set_index("identifier").loc[lab.spec]
parse = lambda s: np.array([float(x) for x in str(s).split(",")])
T = [(parse(r.mzs), parse(r.intensities)) for r in true.itertuples()]
CES = (20.0, 40.0, 60.0)
import json  # noqa: E402

expect = json.load(open(os.path.join(root, ext.MANIFEST)))["probe_top1"]
times = {}
for kind in ext.KINDS:
    t0 = time.time()
    M = ext.ExtModel(kind, root, workers=int(os.environ.get("EXT_WORKERS", "0")))
    t_load = time.time() - t0
    can = [ext.canonical_input(s, kind) for s in lab.smiles]
    ents = [(c, r.ionization, ce, r.instrument, float(r.precursor)) for c, r in zip(can, lab.itertuples()) for ce in CES]
    t0 = time.time()
    preds = M.predict(ents)
    times[kind] = (time.time() - t0) / len(ents)
    P = [[preds[3 * i + j] for j in range(3)] for i in range(len(lab))]
    S = np.array([[max((ext.entropy_sim(tm, ti, p[0], p[1]) if p is not None else 0.0) for p in P[j])
                   for j in range(len(lab))] for tm, ti in T])
    S = np.nan_to_num(S, nan=0.0)
    rank = np.array([1 + int((S[i] > S[i, i]).sum()) for i in range(len(lab))])
    top1 = float(np.mean(rank == 1))
    print(f"(c) {kind}: load {t_load:.1f}s, {times[kind] * 1e3:.1f} ms/entry, failed {M.n_fail}, identity top-1 "
          f"{top1:.3f} (probe {expect[kind]:.3f}), MRR {np.mean(1 / rank):.3f}, sha {M.sha}")
    ok &= abs(top1 - expect[kind]) <= 0.02 + 1e-9 and M.n_fail == 0
    # (d)
    i = 0
    r0 = lab.iloc[i]
    spec = dict(mz=T[i][0], it=T[i][1], prec=float(r0.precursor), adduct=r0.ionization, mode=1,
                ce_list=(20.0, 40.0), instrument=r0.instrument)
    sc = ext.score_candidates(M, [spec], [lab.smiles[i]] + list(lab.smiles[1:6]) + ["C1CC1.[Na+]", "not_a_smiles"])
    print(f"(d) {kind}: scores {np.round(sc, 3)}")
    ok &= int(np.nanargmax(sc)) == 0 and np.isnan(sc[-1]) and np.isnan(sc[-2])
    del M
    import torch  # noqa: E402

    torch.cuda.empty_cache()
print("(e) seconds per entry:", {k: round(v, 4) for k, v in times.items()})
print("ALL OK" if ok else "FAILED")
sys.exit(0 if ok else 1)

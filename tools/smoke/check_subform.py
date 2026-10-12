"""Smoke check of casmi26.subform: the numba annotation equals the naive enumeration on 300 smoke spectra
(structures below 400 Da so that the naive loop stays fast), for both tolerance settings."""
import sys
import time

import numpy as np

sys.path.insert(0, "/src")
from casmi26 import cache, fpnet, pool, subform  # noqa: E402

C = cache.Cache("/d/work")
P = pool.Pool("/d/pool", pubchem=False)
pf = __import__("pandas").read_parquet("/d/pool/pool.parquet", columns=["formula"]).formula.values
D = fpnet.TrainData(C, P, set())
rng = np.random.default_rng(0)
n_ok = n_peaks = n_assigned = 0
t_fast = t_naive = 0.0
for j in rng.permutation(len(D)):
    if n_ok >= 300:
        break
    r, p = D.rows[j], D.pid[j]
    if P.mass[p] > 400:
        continue
    v, s = subform.parse(pf[p])
    ion, sign = subform.ion_counts(v, C.adducts[C.adduct[r]]) if s else (None, 0)
    if ion is None:
        continue
    mz, _ = fpnet.peak_tokens(*C.spectrum(r), 30)
    for abs_da in (subform.ABS_DA, 0.0):
        t = time.time()
        a = subform.annotate(mz, ion, sign, subform.PPM, abs_da)
        t_fast += time.time() - t
        t = time.time()
        b = subform.annotate_naive(mz, ion, sign, subform.PPM, abs_da)
        t_naive += time.time() - t
        assert (a[2] == b[2]).all(), (pf[p], mz[a[2] != b[2]])
        assert (a[0][a[2]] == b[0][b[2]]).all(), (pf[p], a[0][a[2]][:3], b[0][b[2]][:3])
        assert np.allclose(a[1], b[1], atol=1e-3)
    n_ok += 1
    n_peaks += len(mz)
    n_assigned += int(a[2].sum())
print(f"subform check: {n_ok} spectra, {n_peaks} peaks, assigned (10 ppm only) {n_assigned / max(n_peaks, 1):.3f}; "
      f"numba {t_fast:.2f}s vs naive {t_naive:.1f}s — identical")

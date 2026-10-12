"""Subformula annotation of MS/MS peaks (M6 P0-1, k26_fp2 / k27_enc3).

A neutral formula is a count vector over ELEM. The precursor ion formula = n_M * M + the adduct's element delta
(ION_DELTA); every fragment ion is a subformula of it (each element 0..its count in the ion, H free in 0..ion H),
charged like the precursor (positive: formula mass - electron, negative: + electron). A peak gets the subformula whose
ion mass is closest to the peak m/z within the tolerance max(ppm * m/z, abs_da) (SIRIUS convention: 10 ppm, at least
2 mDa), among those with RDBE >= -0.5; peaks without one stay unassigned. Formulas with an element outside ELEM, adducts
outside ION_DELTA, and ions with more than MAX_COMBOS heavy-atom combinations are not annotated (all peaks unassigned).

The search loops over the heavy-atom combinations (mixed radix over the non-H elements), takes the peaks within the
combination's mass band (heavy mass .. + all H, binary search on the sorted m/z) and solves the H count per peak
(numba, nogil, so the training producer threads run in parallel).
"""
import re

import numpy as np
from numba import njit

from .chem import E_MASS, MONO

ELEM = ["C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I", "Si", "B", "Se", "Na", "K"]
EIX = {e: i for i, e in enumerate(ELEM)}
NE = len(ELEM)
H_IX = EIX["H"]
EMASS = np.array([MONO[e] for e in ELEM], np.float64)
# RDBE = 1 + sum(count * RDBE_W) (C, Si +1; monovalent -1/2; trivalent +1/2; divalent 0)
RDBE_W = np.array([{"C": 1.0, "Si": 1.0, "N": 0.5, "P": 0.5, "B": 0.5, "H": -0.5, "F": -0.5, "Cl": -0.5, "Br": -0.5,
                    "I": -0.5, "Na": -0.5, "K": -0.5}.get(e, 0.0) for e in ELEM], np.float64)
PPM = 10.0
ABS_DA = 0.002
MAX_COMBOS = 300_000
_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


def _vec(d):
    v = np.zeros(NE, np.int64)
    for e, n in d.items():
        v[EIX[e]] = n
    return v


# adduct -> (n_M, charge sign, element delta of the ion formula); dimers and 2+ are left out (not annotated)
ION_DELTA = {
    "[M+H]+": (1, 1, _vec({"H": 1})),
    "[M+NH4]+": (1, 1, _vec({"N": 1, "H": 4})),
    "[M+Na]+": (1, 1, _vec({"Na": 1})),
    "[M+K]+": (1, 1, _vec({"K": 1})),
    "[M-H2O+H]+": (1, 1, _vec({"H": -1, "O": -1})),
    "[M-2H2O+H]+": (1, 1, _vec({"H": -3, "O": -2})),
    "[M]+": (1, 1, _vec({})),
    "[M-H]-": (1, -1, _vec({"H": -1})),
    "[M-H2O-H]-": (1, -1, _vec({"H": -3, "O": -1})),
    "[M+CH2O2-H]-": (1, -1, _vec({"C": 1, "H": 1, "O": 2})),
    "[M+C2H4O2-H]-": (1, -1, _vec({"C": 2, "H": 3, "O": 2})),
    "[M+Cl]-": (1, -1, _vec({"Cl": 1})),
    "[M]-": (1, -1, _vec({})),
}


def parse(formula):
    """Neutral formula string -> (counts over ELEM int64, supported). Charges ('+', '-2') are ignored here; unknown
    elements make it unsupported (counts of the known ones are still filled)."""
    v = np.zeros(NE, np.int64)
    if not isinstance(formula, str):
        return v, False
    f = re.sub(r"[+-]\d*$", "", formula.strip())
    ok = bool(f)
    for e, n in _TOKEN.findall(f):
        k = int(n) if n else 1
        if e in EIX:
            v[EIX[e]] += k
        else:
            ok = False
    return v, ok


def key_of(v):
    """Canonical Hill-order string of a count vector (C, H first, then alphabetical)."""
    order = ["C", "H"] + sorted(e for e in ELEM if e not in ("C", "H"))
    return "".join(f"{e}{v[EIX[e]] if v[EIX[e]] != 1 else ''}" for e in order if v[EIX[e]] > 0)


def ion_counts(counts, adduct):
    """(ion count vector, charge sign) of the precursor ion, or (None, 0) when the adduct is not annotated or the ion
    formula has a negative count."""
    a = ION_DELTA.get(str(adduct).replace(" ", ""))
    if a is None:
        return None, 0
    n, sign, delta = a
    ion = n * np.asarray(counts, np.int64) + delta
    if (ion < 0).any():
        return None, 0
    return ion, sign


def n_combos(ion):
    heavy = np.delete(np.asarray(ion, np.int64), H_IX)
    return int(np.prod(heavy + 1, dtype=np.float64))


@njit(nogil=True, cache=True)
def _annotate(mz, ion, sign, emass, rdbe_w, h_ix, ppm, abs_da, out_c, out_err, out_ok):
    """Fill out_c (n, NE) / out_err (n,) / out_ok (n,) for peaks mz (n,) (any order)."""
    ne = ion.shape[0]
    n = mz.shape[0]
    for i in range(n):
        out_ok[i] = False
        out_err[i] = 0.0
    best = np.full(n, 1e9)
    mh = emass[h_ix]
    nh = ion[h_ix]
    e_shift = -sign * 5.48579909e-4  # ion mass = formula mass - sign * electron
    tol = np.empty(n)
    maxtol = 0.0
    for i in range(n):
        tol[i] = max(mz[i] * ppm * 1e-6, abs_da)
        maxtol = max(maxtol, tol[i])
    order = np.argsort(mz)
    smz = mz[order]
    span = nh * mh
    cur = np.zeros(ne, np.int64)
    while True:
        m_heavy = 0.0
        rd = 1.0
        for e in range(ne):
            if e != h_ix:
                m_heavy += cur[e] * emass[e]
                rd += cur[e] * rdbe_w[e]
        if m_heavy > 0:  # at least one heavy atom
            lo = m_heavy + e_shift
            k0 = np.searchsorted(smz, lo - maxtol)
            k1 = np.searchsorted(smz, lo + span + maxtol, side="right")
            for k in range(k0, k1):
                i = order[k]
                r = mz[i] - lo
                if r < -tol[i] or r > nh * mh + tol[i]:
                    continue
                h = int(np.floor(r / mh + 0.5))
                if h < 0:
                    h = 0
                if h > nh:
                    h = nh
                err = r - h * mh
                if abs(err) > tol[i] or abs(err) >= best[i]:
                    continue
                if rd + h * rdbe_w[h_ix] < -0.5:
                    continue
                best[i] = abs(err)
                out_ok[i] = True
                out_err[i] = err / mz[i] * 1e6
                for e in range(ne):
                    out_c[i, e] = cur[e]
                out_c[i, h_ix] = h
        # next heavy combination (mixed radix, H skipped)
        e = 0
        while e < ne:
            if e == h_ix:
                e += 1
                continue
            if cur[e] < ion[e]:
                cur[e] += 1
                break
            cur[e] = 0
            e += 1
        if e >= ne:
            break


def annotate(mz, ion, sign, ppm=PPM, abs_da=ABS_DA):
    """Peaks mz -> (fragment counts (n, NE) int16, ppm error (n,) float32, assigned (n,) bool).
    ion: precursor ion counts (or None = not annotated)."""
    mz = np.asarray(mz, np.float64)
    n = len(mz)
    out_c = np.zeros((n, NE), np.int64)
    out_err = np.zeros(n, np.float64)
    out_ok = np.zeros(n, np.bool_)
    if ion is not None and n and n_combos(ion) <= MAX_COMBOS:
        _annotate(mz, np.asarray(ion, np.int64), int(sign), EMASS, RDBE_W, H_IX, float(ppm), float(abs_da),
                  out_c, out_err, out_ok)
    return out_c.astype(np.int16), out_err.astype(np.float32), out_ok


def annotate_naive(mz, ion, sign, ppm=PPM, abs_da=ABS_DA):
    """Reference implementation (enumerates every subformula incl. H; for the smoke check only)."""
    import itertools
    mz = np.asarray(mz, np.float64)
    ranges = [range(int(c) + 1) for c in ion]
    best = [(1e9, None, 0.0)] * len(mz)
    for combo in itertools.product(*ranges):
        v = np.array(combo)
        if np.delete(v, H_IX).sum() == 0:
            continue
        if 1.0 + float(v @ RDBE_W) < -0.5:
            continue
        m = float(v @ EMASS) - sign * E_MASS
        for i, x in enumerate(mz):
            err = x - m
            if abs(err) <= max(x * ppm * 1e-6, abs_da) and abs(err) < best[i][0]:
                best[i] = (abs(err), v, err / x * 1e6)
    c = np.zeros((len(mz), NE), np.int16)
    ok = np.zeros(len(mz), bool)
    e = np.zeros(len(mz), np.float32)
    for i, (a, v, er) in enumerate(best):
        if v is not None:
            c[i], ok[i], e[i] = v, True, er
    return c, e, ok


class FormulaTable:
    """Neutral formula of every pool entry (k26_fp2 output): fid per pid, counts / key / supported per fid."""

    def __init__(self, d):
        import os
        z = np.load(os.path.join(d, "pool_formula.npz"), allow_pickle=False)
        self.fid = z["fid"]
        self.counts = z["counts"].astype(np.int64)
        self.supported = z["supported"]
        self.keys = z["keys"]
        self.mass = z["mass"]

    @staticmethod
    def find_dir():
        import os
        from .paths import find
        return os.path.dirname(find("pool_formula.npz"))

"""Bond-cleavage fragments and their match to query peaks (M5 P1-1a; k21_frag precomputes the pool).

Fragments of a structure (fixed definition, M5 P1 plan):
- cut sets: one acyclic single bond, two acyclic single bonds, or two bonds of the same non-aromatic ring
- fragment mass = sum over the component's atoms of (atom mass + the H the atom carries in the intact molecule);
  no H is added at the cut, hydrogen shifts of -2..+2 are allowed when matching
- components >= 30 Da, rounded to 1e-4 Da, unique; one-cut fragments first, then two-cut ones up to 2,000 per structure
RDKit only turns a molecule into arrays; the enumeration is numba (also used at run time on generated products).

Match: for each query spectrum, the top 30 peaks by intensity below precursor - 1.5 Da; a peak is explained when
peak m/z - (k * H +- proton) is within max(0.01 Da, 20 ppm) of a fragment (k = -2..2, proton sign by polarity).
frag_expl = explained sqrt-intensity share, frag_top10 = explained share of the top-10 peaks (both averaged over
spectra). Fragment ions are matched as protonated / deprotonated only (Na / K fragment ions do not match).
match_features(..., per_spec=True) also returns each used spectrum's explained share (n_used, n_cand) and its index in
specs (for the per-spectrum statistics of search.FRAG_STATS; the averaged columns stay the same values).
consensus_expl (plan 9.2): frag_expl of the spectra merged per adduct (spectra.adduct_groups, how='sum'), averaged
over the adducts with the number of member spectra as weights.
"""
import numpy as np
from numba import njit

from .chem import MONO, PROTON
from .spectra import SimCfg

H = MONO["H"]
MIN_MASS = 30.0
MAX_FRAG = 2000
DEC = 4
TOP_PEAKS = 30
PREC_GAP = 1.5
H_SHIFTS = np.arange(-2, 3)
FRAG_FEATURES = ["frag_expl", "frag_top10", "frag_z", "frag_rank"]


def mol_arrays(mol):
    """(atom masses incl. own H (n,), bonds (m, 2) int64, cut sets (k, 2) int64 with -1 = none, depth (k,) int8),
    or None when an element has no tabulated mass."""
    from rdkit import Chem
    am = np.empty(mol.GetNumAtoms())
    for a in mol.GetAtoms():
        el = a.GetSymbol()
        if el not in MONO:
            return None
        am[a.GetIdx()] = MONO[el] + H * a.GetTotalNumHs()
    bonds = np.array([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()], np.int64).reshape(-1, 2)
    single = [b.GetIdx() for b in mol.GetBonds() if not b.IsInRing() and b.GetBondType() == Chem.BondType.SINGLE]
    cuts = [(i, -1) for i in single]
    depth = [1] * len(cuts)
    for x in range(len(single)):
        for y in range(x + 1, len(single)):
            cuts.append((single[x], single[y]))
            depth.append(2)
    for ring in mol.GetRingInfo().BondRings():
        if any(mol.GetBondWithIdx(i).GetIsAromatic() for i in ring):
            continue
        for x in range(len(ring)):
            for y in range(x + 1, len(ring)):
                cuts.append((ring[x], ring[y]))
                depth.append(2)
    return am, bonds, np.array(cuts, np.int64).reshape(-1, 2), np.array(depth, np.int8)


@njit(cache=True)
def _find(par, i):
    while par[i] != i:
        par[i] = par[par[i]]
        i = par[i]
    return i


@njit(cache=True)
def _components(am, bonds, cuts, depth):
    """Masses of the components of every cut set that splits the molecule -> (masses, depth of the cut)."""
    n, m = len(am), len(bonds)
    out_m = np.empty(len(cuts) * 3, np.float64)
    out_d = np.empty(len(cuts) * 3, np.int8)
    k = 0
    par = np.empty(n, np.int64)
    acc = np.empty(n, np.float64)
    for c in range(len(cuts)):
        for i in range(n):
            par[i] = i
            acc[i] = 0.0
        for b in range(m):
            if b == cuts[c, 0] or b == cuts[c, 1]:
                continue
            ra, rb = _find(par, bonds[b, 0]), _find(par, bonds[b, 1])
            if ra != rb:
                par[ra] = rb
        nroot = 0
        for i in range(n):
            r = _find(par, i)
            acc[r] += am[i]
        for i in range(n):
            if par[i] == i:
                nroot += 1
        if nroot < 2:
            continue
        for i in range(n):
            if par[i] == i:
                if k >= len(out_m):
                    break
                out_m[k] = acc[i]
                out_d[k] = depth[c]
                k += 1
    return out_m[:k], out_d[:k]


def fragments_of_mol(mol):
    """Sorted unique fragment masses (float64) of an RDKit molecule (empty when it cannot be handled)."""
    arr = mol_arrays(mol)
    if arr is None or len(arr[1]) == 0 or len(arr[2]) == 0:
        return np.zeros(0)
    ms, d = _components(*arr)
    ms = np.round(ms, DEC)
    keep = ms >= MIN_MASS
    one = np.unique(ms[keep & (d == 1)])
    two = np.setdiff1d(np.unique(ms[keep & (d == 2)]), one)
    if len(one) >= MAX_FRAG:
        return np.sort(one[:MAX_FRAG])
    return np.sort(np.r_[one, two[:MAX_FRAG - len(one)]])


def fragments(smiles):
    from rdkit import Chem
    m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    return fragments_of_mol(m) if m is not None else np.zeros(0)


class FragTable:
    """Precomputed fragment masses of the pool (k21_frag): frag_off.npy (n + 1) and frag_mass.npy."""

    def __init__(self, d):
        self.off = np.load(f"{d}/frag_off.npy")
        self.mass = np.load(f"{d}/frag_mass.npy", mmap_mode="r")

    @staticmethod
    def find_dir():
        import os
        from . import paths
        return os.path.dirname(paths.find("frag_off.npy"))

    def csr(self, pids):
        """Local CSR (off (n + 1), masses) of the given pool ids."""
        lo, hi = self.off[pids], self.off[np.asarray(pids) + 1]
        off = np.r_[0, np.cumsum(hi - lo)].astype(np.int64)
        ms = np.concatenate([np.asarray(self.mass[a:b], np.float64) for a, b in zip(lo, hi)]) if len(pids) else \
            np.zeros(0)
        return off, ms


def csr_of_lists(lists):
    off = np.r_[0, np.cumsum([len(x) for x in lists])].astype(np.int64)
    ms = np.concatenate(lists).astype(np.float64) if lists else np.zeros(0)
    return off, ms


@njit(cache=True)
def _match(off, ms, lo, hi):
    """matched (n_cand, n_peaks): any of a peak's [lo, hi] needed-fragment intervals (n_peaks, n_shift) hits a mass."""
    nc, npk, ns = len(off) - 1, lo.shape[0], lo.shape[1]
    out = np.zeros((nc, npk), np.bool_)
    for c in range(nc):
        a, b = off[c], off[c + 1]
        if a == b:
            continue
        for p in range(npk):
            for s in range(ns):
                # first mass >= lo
                L, R = a, b
                while L < R:
                    M = (L + R) // 2
                    if ms[M] < lo[p, s]:
                        L = M + 1
                    else:
                        R = M
                if L < b and ms[L] <= hi[p, s]:
                    out[c, p] = True
                    break
    return out


def match_features(specs, off, ms, sim=SimCfg, per_spec=False):
    """(frag_expl, frag_top10) per candidate of a local CSR, averaged over the usable query spectra.
    per_spec=True: (frag_expl, frag_top10, E (n_used, n_cand) float32 explained share of each used spectrum, used
    (n_used,) int64 index of that spectrum in specs); the first two are the same values as with per_spec=False."""
    nc = len(off) - 1
    expl, top10, n = np.zeros(nc), np.zeros(nc), 0
    rows, used = [], []
    for i, s in enumerate(specs):
        mz, it = np.asarray(s["mz"], np.float64), np.asarray(s["it"], np.float64)
        k = mz < float(s["prec"]) - PREC_GAP
        mz, it = mz[k], it[k]
        if len(mz) == 0:
            continue
        o = np.argsort(-it, kind="mergesort")[:TOP_PEAKS]
        mz, w = mz[o], np.sqrt(np.clip(it[o], 0, None))
        if w.sum() <= 0:
            continue
        ion = PROTON if s["mode"] > 0 else -PROTON
        need = mz[:, None] - (H_SHIFTS[None, :] * H + ion)
        tol = np.maximum(sim.tol_da, mz * sim.tol_ppm * 1e-6)[:, None]
        hit = _match(off, ms, need - tol, need + tol)
        e = (hit * w[None, :]).sum(1) / w.sum()
        expl += e
        top10 += hit[:, :10].mean(1)
        n += 1
        if per_spec:
            rows.append(e.astype(np.float32))
            used.append(i)
    if n:
        expl /= n
        top10 /= n
    if per_spec:
        E = np.stack(rows) if rows else np.zeros((0, nc), np.float32)
        return expl.astype(np.float32), top10.astype(np.float32), E, np.array(used, np.int64)
    return expl.astype(np.float32), top10.astype(np.float32)


def consensus_expl(specs, off, ms, sim=SimCfg):
    """(n_cand,) float32 consensus frag_expl: the usable spectra merged per adduct (spectra.adduct_groups with
    how='sum', prec = the members' median) -> match_features of each merged spectrum -> mean over the adducts weighted
    by their member count (0 when no merged spectrum is usable, as match_features)."""
    from .spectra import adduct_groups
    nc = len(off) - 1
    acc, wsum = np.zeros(nc), 0.0
    for g in adduct_groups(specs, how="sum", gap=PREC_GAP):
        e, _, _, used = match_features([g], off, ms, sim, per_spec=True)
        if len(used):
            acc += g["n"] * e.astype(np.float64)
            wsum += g["n"]
    return (acc / wsum if wsum else acc).astype(np.float32)

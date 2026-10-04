"""Peak cleaning and entropy similarity (Li et al., Nat. Methods 2021), numba-compiled.

Two levels of preprocessing:
  storage  clean_csr: peaks <= precursor + 2 Da, relative floor, top-k by intensity, m/z sorted, base peak = 1
  search   prep:      relative floor, top-k, intensity power, sum = 1, entropy weighting for low-entropy spectra
"""
import numpy as np
from numba import njit, prange

CACHE_FLOOR = 0.001
CACHE_TOPK = 256
MAX_OVER_PREC = 2.0


class SimCfg:
    floor = 0.002
    topk = 256
    power = 1.0
    ent_weight = True
    tol_da = 0.01
    tol_ppm = 20.0


@njit(cache=True)
def clean_csr(off, mz, it, prec, max_over_prec, floor, topk):
    """Storage cleaning of CSR spectra. Returns (off, mz float32, it float32) with it relative to the base peak."""
    n = len(off) - 1
    out_mz = np.empty(len(mz), np.float32)
    out_it = np.empty(len(mz), np.float32)
    new_off = np.zeros(n + 1, np.int64)
    p = 0
    for r in range(n):
        a = off[r]
        b = off[r + 1]
        lim = prec[r] + max_over_prec
        mx = 0.0
        for i in range(a, b):
            if mz[i] > 0 and mz[i] <= lim and it[i] > mx:
                mx = it[i]
        if mx <= 0:
            new_off[r + 1] = p
            continue
        thr = floor * mx
        idx = np.empty(b - a, np.int64)
        c = 0
        for i in range(a, b):
            if mz[i] > 0 and mz[i] <= lim and it[i] >= thr:
                idx[c] = i
                c += 1
        idx = idx[:c]
        if c > topk:
            v = np.empty(c)
            for j in range(c):
                v[j] = -it[idx[j]]
            idx = idx[np.argsort(v)[:topk]]
            c = topk
        m = np.empty(c)
        for j in range(c):
            m[j] = mz[idx[j]]
        o = np.argsort(m)
        for j in range(c):
            out_mz[p] = mz[idx[o[j]]]
            out_it[p] = it[idx[o[j]]] / mx
            p += 1
        new_off[r + 1] = p
    return new_off, out_mz[:p].copy(), out_it[:p].copy()


@njit(cache=True)
def prep(mz, it, floor, topk, power, ent_weight):
    """Search-time normalisation of one m/z-sorted spectrum -> (mz, p) with sum(p) = 1."""
    n = len(mz)
    mx = 0.0
    for i in range(n):
        if it[i] > mx:
            mx = it[i]
    if n == 0 or mx <= 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    thr = floor * mx
    c = 0
    for i in range(n):
        if it[i] >= thr:
            c += 1
    idx = np.empty(c, np.int64)
    j = 0
    for i in range(n):
        if it[i] >= thr:
            idx[j] = i
            j += 1
    if c > topk:
        v = np.empty(c)
        for i in range(c):
            v[i] = -it[idx[i]]
        idx = np.sort(idx[np.argsort(v)[:topk]])
        c = topk
    om = np.empty(c, np.float32)
    op = np.empty(c, np.float32)
    s = 0.0
    for i in range(c):
        om[i] = mz[idx[i]]
        v = it[idx[i]] ** power
        op[i] = v
        s += v
    for i in range(c):
        op[i] /= s
    if ent_weight:
        S = 0.0
        for i in range(c):
            if op[i] > 0:
                S -= op[i] * np.log(op[i])
        if S < 3.0:
            w = 0.25 + 0.25 * S
            s2 = 0.0
            for i in range(c):
                op[i] = op[i] ** w
                s2 += op[i]
            for i in range(c):
                op[i] /= s2
    return om, op


@njit(cache=True)
def entropy_sim(qm, qp, cm, cp, shift, tol_da, tol_ppm):
    """Unweighted entropy similarity of two prepped spectra; `shift` is added to the m/z of the second one."""
    n = len(qm)
    m = len(cm)
    if n == 0 or m == 0:
        return 0.0
    sa = 0.0
    for x in range(n):
        if qp[x] > 0:
            sa -= qp[x] * np.log(qp[x])
    sb = 0.0
    for x in range(m):
        if cp[x] > 0:
            sb -= cp[x] * np.log(cp[x])
    i = 0
    j = 0
    sab = 0.0
    while i < n and j < m:
        c = cm[j] + shift
        d = qm[i] - c
        tol = max(tol_da, qm[i] * tol_ppm * 1e-6)
        if d < -tol:
            v = qp[i] * 0.5
            i += 1
        elif d > tol:
            v = cp[j] * 0.5
            j += 1
        else:
            v = (qp[i] + cp[j]) * 0.5
            i += 1
            j += 1
        if v > 0:
            sab -= v * np.log(v)
    while i < n:
        v = qp[i] * 0.5
        if v > 0:
            sab -= v * np.log(v)
        i += 1
    while j < m:
        v = cp[j] * 0.5
        if v > 0:
            sab -= v * np.log(v)
        j += 1
    return 1.0 - (2.0 * sab - sa - sb) / np.log(4.0)


@njit(cache=True, parallel=True)
def sim_rows(qm, qp, rows, off, mz, it, shifts, floor, topk, power, ent_weight, tol_da, tol_ppm):
    """Similarity of a prepped query to stored spectra `rows`: (direct, max(direct, precursor-shifted))."""
    out = np.zeros(len(rows), np.float32)
    outs = np.zeros(len(rows), np.float32)
    for k in prange(len(rows)):
        r = rows[k]
        a = off[r]
        b = off[r + 1]
        if b <= a:
            continue
        cm, cp = prep(mz[a:b], it[a:b], floor, topk, power, ent_weight)
        if len(cm) == 0:
            continue
        s = entropy_sim(qm, qp, cm, cp, 0.0, tol_da, tol_ppm)
        out[k] = s
        s2 = s
        if abs(shifts[k]) > tol_da:
            s2 = max(s, entropy_sim(qm, qp, cm, cp, shifts[k], tol_da, tol_ppm))
        outs[k] = s2
    return out, outs


@njit(cache=True)
def prep_store(rows, off, mz, it, floor, topk, power, ent_weight):
    """Prepped copies of stored spectra `rows` as a new CSR store (roff, rmz, rp)."""
    n = len(rows)
    lens = np.zeros(n + 1, np.int64)
    for k in range(n):
        r = rows[k]
        lens[k + 1] = lens[k] + min(off[r + 1] - off[r], topk)
    rmz = np.empty(lens[n], np.float32)
    rp = np.empty(lens[n], np.float32)
    roff = np.zeros(n + 1, np.int64)
    p = 0
    for k in range(n):
        r = rows[k]
        cm, cp = prep(mz[off[r]:off[r + 1]], it[off[r]:off[r + 1]], floor, topk, power, ent_weight)
        rmz[p:p + len(cm)] = cm
        rp[p:p + len(cp)] = cp
        p += len(cm)
        roff[k + 1] = p
    return roff, rmz[:p].copy(), rp[:p].copy()


@njit(cache=True, parallel=True)
def sim_store_shift(qm, qp, idx, roff, rmz, rp, shifts, tol_da, tol_ppm):
    """max(direct, shifted) similarity of a prepped query to prepped store entries `idx`."""
    out = np.zeros(len(idx), np.float32)
    for k in prange(len(idx)):
        r = idx[k]
        a = roff[r]
        b = roff[r + 1]
        if b <= a:
            continue
        s = entropy_sim(qm, qp, rmz[a:b], rp[a:b], 0.0, tol_da, tol_ppm)
        if abs(shifts[k]) > tol_da:
            s = max(s, entropy_sim(qm, qp, rmz[a:b], rp[a:b], shifts[k], tol_da, tol_ppm))
        out[k] = s
    return out


def clean_one(mz, it, prec):
    """Storage cleaning of a single raw spectrum (for test queries)."""
    mz = np.asarray(mz, np.float64)
    it = np.asarray(it, np.float64)
    off = np.array([0, len(mz)], np.int64)
    _, m, i = clean_csr(off, mz, it, np.array([float(prec)]), MAX_OVER_PREC, CACHE_FLOOR, CACHE_TOPK)
    return m, i


def merge_spectra(specs, tol=0.01):
    """Collapse several (mz, it) spectra into one: each scaled to base peak 1, close peaks keep the larger."""
    specs = [(np.asarray(a, np.float64), np.asarray(b, np.float64)) for a, b in specs if len(a)]
    if not specs:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    mz = np.concatenate([a for a, _ in specs])
    it = np.concatenate([b / max(b.max(), 1e-12) for _, b in specs])
    o = np.argsort(mz, kind="mergesort")
    mz, it = mz[o], it[o]
    keep = np.ones(len(mz), bool)
    last = 0
    for j in range(1, len(mz)):
        if mz[j] - mz[last] < tol:
            if it[j] > it[last]:
                keep[last] = False
                last = j
            else:
                keep[j] = False
        else:
            last = j
    return mz[keep].astype(np.float32), it[keep].astype(np.float32)


def prep_query(mz, it, cfg=SimCfg):
    return prep(np.asarray(mz, np.float32), np.asarray(it, np.float32), cfg.floor, cfg.topk, cfg.power, cfg.ent_weight)

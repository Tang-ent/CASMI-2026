"""Fixed-slot merging of side lists into a ranked list (M5 generation S3; M7 PubChem side list P5 / P2).

The submission path (merge) and the evaluation path (shifted_rank, slot_rank) use the same definition: a side list
with m items takes the first m of its slot positions (1-based); the main (pool) list fills the free positions in its
own order. Several side lists must use disjoint positions."""
import numpy as np

SLOT_SETS = {"S5": [5, 10, 15, 20, 25], "S3": list(range(3, 25, 3)), "S2": list(range(2, 25, 2)), "S10": [10, 20],
             "P5": [5, 10, 14, 20, 25], "P2": [10, 20]}


def occupied(inserts):
    """inserts: [(positions, n_items)] -> sorted occupied positions (asserts disjointness)."""
    occ = []
    for pos, n in inserts:
        occ += list(pos[:max(0, min(int(n), len(pos)))])
    assert len(occ) == len(set(occ)), f"side lists overlap: {occ}"
    return sorted(occ)


def merge(main, inserts):
    """main: ranked items; inserts: [(positions, items)] -> merged list (all items kept: side items whose position lies
    beyond the end of main are appended in position order)."""
    at = {}
    for pos, items in inserts:
        for p, it in zip(pos, items):
            assert p not in at, f"side lists overlap at {p}"
            at[p] = it
    out, mi, p = [], 0, 1
    while mi < len(main) or at:
        if p in at:
            out.append(at.pop(p))
        elif mi < len(main):
            out.append(main[mi])
            mi += 1
        else:  # main exhausted: remaining side items in position order
            out += [at.pop(k) for k in sorted(at)]
        p += 1
    return out


def shifted_rank(r, occ):
    """Final position of the main-list item at rank r (1-based; inf stays inf) when positions occ are taken."""
    if not np.isfinite(r):
        return np.inf
    p = int(r)
    for o in occ:  # occ sorted: every taken position at or before the item's final position pushes it back by one
        if o <= p:
            p += 1
    return float(p)


def shifted_ranks(r, occs):
    """Vectorised shifted_rank over queries: r (n,), occs = list of n sorted position lists."""
    return np.array([shifted_rank(x, o) for x, o in zip(np.asarray(r, np.float64), occs)])


def slot_rank(k, positions, n):
    """Final position of the side-list item at rank k (1-based; inf = absent) when the list has n items."""
    if not np.isfinite(k) or k > min(n, len(positions)):
        return np.inf
    return float(positions[int(k) - 1])

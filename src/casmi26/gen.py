"""Class-3 candidates of one query (M5 P1-2; k22_features3, k24_submit5): parents, products and product features.

Same procedure as k19_gen_probe: parents = (a) the top analogs by spectral similarity (>= min_sim) and (b) pool
structures at target - delta for every 1-2 step delta of casmi26.derive, ranked by f.z (top n per delta), at most
max_parents; products of the mass-matching step sequences (derive), kept within +-ppm of the target, the first
max_ranked in parent order, deduplicated by standard InChIKey14 and against the pool window's keys (keep_key is
exempt: the evaluation keeps the truth's key so that one run serves both the truth-in-pool and the class-3 view).

Product features (GEN_FEATURES) are spectral evidence only; z / ranks are within the product list. No provenance:
reaction names, direction, steps and the parent's source are not features (the proxy sets favour removals).
parent_ap1 = max_j Tanimoto(parent, analog_j) * sim_j, the same formula for analog and pool parents.
"""
import numpy as np

from . import chem, derive, frag
from .search import _zs, rank_norm

GEN_FEATURES = ["fz", "fz_z", "fz_rank", "ap4", "ap1", "best_tan", "top_tan", "ap_rank", "fwd", "fwd_z",
                "frag_expl", "frag_top10", "frag_z", "parent_ap1"]
GCOL = {n: i for i, n in enumerate(GEN_FEATURES)}


class GenCfg:
    ppm = 10.0
    n_analog_parents = 20
    min_sim = 0.3
    n_pool_per_delta = 5
    max_parents = 50
    max_per_parent = 100
    max_ranked = 500


_DELTAS = []


def _deltas():
    if not _DELTAS:
        _DELTAS.append(np.array(sorted(derive.all_deltas())))
    return _DELTAS[0]


def _unpack(fp):
    return np.unpackbits(np.atleast_2d(fp), axis=1)[:, :chem.FP_BITS].astype(np.float32)


def _tan(F, A):
    inter = F @ A.T
    return inter / (F.sum(1)[:, None] + A.sum(1)[None, :] - inter + 1e-9)


def parents(P, target, z, apid, asim, cfg=GenCfg):
    """[(pid, src)] with src "a" (analog) / "b" (pool window at target - delta), at most cfg.max_parents."""
    tol = target * cfg.ppm * 1e-6
    out = [(int(apid[j]), "a") for j in range(min(len(apid), cfg.n_analog_parents)) if asim[j] >= cfg.min_sim]
    seen = {p for p, _ in out}
    pool_c = []
    for d in _deltas():
        pm = target - d
        if pm < 50:
            continue
        w = P.window(pm, tol / pm * 1e6)
        if len(w) == 0:
            continue
        s = _unpack(P.fp[w]) @ z
        for k in np.argsort(-s)[:cfg.n_pool_per_delta]:
            pool_c.append((float(s[k]), int(w[k])))
    for _, p in sorted(pool_c, reverse=True):
        if len(out) >= cfg.max_parents:
            break
        if p not in seen:
            out.append((p, "b"))
            seen.add(p)
    return out


def products(P, target, pars, win_keys, keep_key=None, cfg=GenCfg):
    """-> (list of dict(smiles, key, parent, src, seq), n after the mass filter (before the cap))."""
    tol = target * cfg.ppm * 1e-6
    prods = {}
    for p, src in pars:
        seqs = derive.sequences(target - P.mass[p], tol)
        if not seqs:
            continue
        for smi, seq in derive.generate(P.smiles[p], seqs, cfg.max_per_parent).items():
            if smi not in prods:
                prods[smi] = (p, src, seq)
    prods = {smi: v for smi, v in prods.items() if abs(derive.exact_mass(smi) - target) <= tol}
    out, seen = [], set()
    for smi in list(prods)[:cfg.max_ranked]:
        k = derive.ik14(smi)
        if k is None or k in seen or (k in win_keys and k != keep_key):
            continue
        seen.add(k)
        p, src, seq = prods[smi]
        out.append(dict(smiles=smi, key=k, parent=p, src=src, seq="+".join(seq)))
    return out, len(prods)


def features(prods, specs, z, apid, asim, P, fwd=None, sim_power=4.0):
    """GEN_FEATURES (n, len) float32 of the products (rows in the order of prods)."""
    n = len(prods)
    X = np.zeros((n, len(GEN_FEATURES)), np.float32)
    if n == 0:
        return X
    fps = np.stack([chem.morgan_packed(d["smiles"]) for d in prods])
    F = _unpack(fps)
    fz = F @ z if z is not None else np.zeros(n, np.float32)
    X[:, GCOL["fz"]] = fz / 100.0
    X[:, GCOL["fz_z"]] = _zs(fz)
    X[:, GCOL["fz_rank"]] = rank_norm(fz)
    if len(apid):
        A = _unpack(P.fp[apid])
        w = np.clip(asim, 0, None)
        tan = _tan(F, A)
        ap4 = (tan * (w ** sim_power)[None, :]).max(1)
        X[:, GCOL["ap4"]] = ap4
        X[:, GCOL["ap1"]] = (tan * w[None, :]).max(1)
        X[:, GCOL["best_tan"]] = tan.max(1)
        X[:, GCOL["top_tan"]] = tan[:, 0]
        X[:, GCOL["ap_rank"]] = rank_norm(ap4)
        par = np.array([d["parent"] for d in prods])
        X[:, GCOL["parent_ap1"]] = (_tan(_unpack(P.fp[par]), A) * w[None, :]).max(1)
    if fwd is not None:
        fw = fwd.score(specs, fps)
        X[:, GCOL["fwd"]] = fw
        X[:, GCOL["fwd_z"]] = _zs(fw)
    expl, top10 = frag.match_features(specs, *frag.csr_of_lists([frag.fragments(d["smiles"]) for d in prods]))
    X[:, GCOL["frag_expl"]] = expl
    X[:, GCOL["frag_top10"]] = top10
    X[:, GCOL["frag_z"]] = _zs(expl)
    return X


def for_query(S, specs, cand, info, keep_key=None, fwd=None, cfg=GenCfg):
    """Products and their features for one query right after S.features(specs, ...) (uses S.last).
    -> (prods list, X, stats dict)."""
    last = S.last
    z, target = last["z"], info["target"]
    if len(cand) == 0 or z is None or not np.isfinite(target):
        return [], np.zeros((0, len(GEN_FEATURES)), np.float32), dict(n_par=0, n_all=0)
    pars = parents(S.P, target, z, last["apid"], last["asim"], cfg)
    win_keys = {derive.ik14(S.P.smiles[c]) for c in cand}
    prods, n_all = products(S.P, target, pars, win_keys, keep_key, cfg)
    X = features(prods, specs, z, last["apid"], last["asim"], S.P, fwd, S.cfg.sim_power)
    return prods, X, dict(n_par=len(pars), n_par_a=sum(1 for _, s in pars if s == "a"), n_all=n_all)

"""Per-molecule candidate features: library search channel + analog propagation channel.

A query is one molecule = list of spectra dicts {mz, it (storage-cleaned, base peak 1), prec, adduct, mode (+1/-1)}.
Candidates are pool structures (pid; train U COCONUT, or train only) whose mass lies within +-ppm_win of the
target neutral mass. `hide` removes library spectra for simulation: a whole library (hide_lib) and/or one
structure's spectra (hide_pid) in one library (hide_pid_lib >= 0) or in every library (hide_pid_lib = -1).
"""
import time
from dataclasses import dataclass

import numpy as np

from . import chem, spectra
from .pool import Pool
from .spectra import SimCfg

FEATURES = [
    # library channel
    "lib_max", "lib_mean", "lib_top2", "lib_shift_max", "lib_same_adduct", "lib_rows", "lib_hit",
    "lib_rank", "lib_gap", "lib_grp_max", "lib_grp_gap2",
    # analog channel
    "ap4", "ap1", "best_tan", "top_tan", "tan_self", "ap_rank", "ap_gap", "top_sim", "n_analog_hi",
    # mass / group context
    "mass_ppm", "n_cand", "n_query", "n_pos", "n_neg", "target_mass",
    # structure priors (ablation group)
    "cand_nspec", "cand_nlib",
    # spectrum -> fingerprint model (zeros when no model is attached)
    "fz", "fz_z", "fz_rank", "fz_gap", "fz_soft", "fz_grp_gap2",
]
FZ_FEATURES = ["fz", "fz_z", "fz_rank", "fz_gap", "fz_soft", "fz_grp_gap2"]
PRIOR_FEATURES = ["cand_nspec", "cand_nlib"]
COL = {n: i for i, n in enumerate(FEATURES)}


@dataclass
class SearchCfg:
    ppm_win: float = 10.0
    ppm_fallback: float = 30.0
    analog_win: float = 200.0
    n_analog: int = 100
    sim_power: float = 4.0
    max_query_spectra: int = 16


def rank_norm(x):
    """0 = best (largest) .. 1 = worst, ties share the average rank."""
    n = len(x)
    if n <= 1:
        return np.zeros(n, np.float32)
    o = np.argsort(-x, kind="mergesort")
    xs = x[o]
    r = np.empty(n, np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return (r / (n - 1)).astype(np.float32)


def _unpack(fp_packed):
    return np.unpackbits(fp_packed, axis=1)[:, :chem.FP_BITS].astype(np.float32)


class Searcher:
    def __init__(self, cache, pool=None, cfg=None, sim=SimCfg, model=None):
        self.C = cache
        self.model = model  # fpnet.FPModel or None
        self.P = pool if pool is not None else Pool(cache=cache)
        self.cfg = cfg or SearchCfg()
        self.sim = sim
        self._pos = np.full(len(self.P), -1, np.int64)
        self._cand = None  # pids whose _pos entry is set (reset after each query; the pool can be large)
        self.rep_pid = self.P.sid2pid[cache.rep_sid]

    # ------------------------------------------------------------------ helpers
    def _window(self, sorted_mass, target, ppm):
        tol = target * ppm * 1e-6
        return (np.searchsorted(sorted_mass, target - tol, "left"),
                np.searchsorted(sorted_mass, target + tol, "right"))

    def _hidden(self, pid, lib, hide):
        h = np.zeros(len(pid), bool)
        if hide.get("hide_lib", -1) >= 0:
            h |= lib == hide["hide_lib"]
        if hide.get("hide_pid", -1) >= 0:
            m = pid == hide["hide_pid"]
            if hide.get("hide_pid_lib", -1) >= 0:
                m &= lib == hide["hide_pid_lib"]
            h |= m
        return h

    def _cand_index(self, pid):
        """Row index in the current candidate set for each pid (-1 = not a candidate / unmapped)."""
        return np.where(pid >= 0, self._pos[np.maximum(pid, 0)], -1)

    def target_mass(self, specs):
        nm = np.array([chem.neutral_mass(s["prec"], s["adduct"]) for s in specs])
        nm = nm[np.isfinite(nm)]
        return float(np.median(nm)) if len(nm) else np.nan

    # ------------------------------------------------------------------ main
    def features(self, specs, hide=None):
        """-> (cand pid int64 (n,), X float32 (n, len(FEATURES)), info dict)."""
        try:
            return self._features(specs, hide or {})
        finally:
            if self._cand is not None:  # 途中で例外が出ても候補マップを汚さない（使った位置だけ戻す）
                self._pos[self._cand] = -1
                self._cand = None

    def _features(self, specs, hide):
        C, P, cfg, sim = self.C, self.P, self.cfg, self.sim
        t0 = time.time()
        specs = [s for s in specs if np.isfinite(chem.neutral_mass(s["prec"], s["adduct"]))][:cfg.max_query_spectra]
        target = self.target_mass(specs)
        empty = (np.zeros(0, np.int64), np.zeros((0, len(FEATURES)), np.float32),
                 dict(target=target, n_cand=0, ppm=np.nan, lib_max=0.0, secs=0.0))
        if not np.isfinite(target):
            return empty
        ppm = cfg.ppm_win
        cand = P.window(target, ppm)
        if len(cand) == 0:
            ppm = cfg.ppm_fallback
            cand = P.window(target, ppm)
        if len(cand) == 0:
            return empty
        nc = len(cand)
        pos = self._pos
        self._cand = cand
        pos[cand] = np.arange(nc)
        X = np.zeros((nc, len(FEATURES)), np.float32)

        # ---------------- library channel: spectra of candidate structures, same polarity ----------------
        a, b = self._window(C.lib_mass, target, ppm)
        rows = C.lib_rows[a:b]
        rpid = P.sid2pid[C.sid[rows]]
        keep = ~self._hidden(rpid, C.lib[rows], hide)
        rows, rpid = rows[keep], rpid[keep]
        rci = self._cand_index(rpid)
        rows, rci = rows[rci >= 0], rci[rci >= 0]
        nq = len(specs)
        M = np.zeros((nq, nc), np.float32)
        MS = np.zeros((nq, nc), np.float32)
        SA = np.zeros(nc, np.float32)
        nrows = np.zeros(nc, np.float32)
        for q, s in enumerate(specs):
            qm, qp = spectra.prep_query(s["mz"], s["it"], sim)
            sel = C.mode[rows] == s["mode"]
            r, ci = rows[sel], rci[sel]
            if len(qm) == 0 or len(r) == 0:
                continue
            shifts = (s["prec"] - C.prec[r]).astype(np.float32)
            d, ds = spectra.sim_rows(qm, qp, r, C.off, C.mz, C.it, shifts, sim.floor, sim.topk, sim.power,
                                     sim.ent_weight, sim.tol_da, sim.tol_ppm)
            np.maximum.at(M[q], ci, d)
            np.maximum.at(MS[q], ci, ds)
            ad = C.adduct_id.get(s["adduct"], -1)
            same = C.adduct[r] == ad
            if same.any():
                np.maximum.at(SA, ci[same], d[same])
            nrows += np.bincount(ci, minlength=nc)
        lib_max = M.max(0)
        srt = np.sort(M, 0)[::-1]
        X[:, COL["lib_max"]] = lib_max
        X[:, COL["lib_mean"]] = M.mean(0)
        X[:, COL["lib_top2"]] = srt[:2].mean(0)
        X[:, COL["lib_shift_max"]] = MS.max(0)
        X[:, COL["lib_same_adduct"]] = SA
        X[:, COL["lib_rows"]] = np.log1p(nrows / max(nq, 1))
        X[:, COL["lib_hit"]] = lib_max > 0
        X[:, COL["lib_rank"]] = rank_norm(lib_max)
        gmax = float(lib_max.max())
        X[:, COL["lib_gap"]] = lib_max - gmax
        X[:, COL["lib_grp_max"]] = gmax
        X[:, COL["lib_grp_gap2"]] = gmax - (np.sort(lib_max)[-2] if nc > 1 else 0.0)
        # structure priors: library spectra / libraries per candidate (all polarities, after hiding)
        X[:, COL["cand_nspec"]] = np.log1p(np.bincount(rci, minlength=nc))
        lib_pairs = np.unique(rci * 64 + C.lib[rows].astype(np.int64))
        X[:, COL["cand_nlib"]] = np.bincount(lib_pairs // 64, minlength=nc)

        # ---------------- analog channel: mass-shifted search over representatives ----------------
        best = {}
        for md in (1, -1):
            grp = [(s["mz"], s["it"]) for s in specs if s["mode"] == md]
            if not grp:
                continue
            mm, ii = spectra.merge_spectra(grp)
            qm, qp = spectra.prep_query(mm, ii, sim)
            if len(qm) == 0:
                continue
            ra, rb = np.searchsorted(C.rep_mass, [target - cfg.analog_win, target + cfg.analog_win])
            idx = np.arange(ra, rb)
            idx = idx[(C.rep_mode[idx] == md) & (self.rep_pid[idx] >= 0)]
            idx = idx[~self._hidden(self.rep_pid[idx], C.rep_lib[idx], hide)]
            if len(idx) == 0:
                continue
            shifts = (target - C.rep_mass[idx]).astype(np.float32)
            sc = spectra.sim_store_shift(qm, qp, idx, C.rep_off, C.rep_mz, C.rep_p, shifts, sim.tol_da, sim.tol_ppm)
            k = min(len(sc), 4 * cfg.n_analog)
            top = np.argpartition(-sc, k - 1)[:k] if k < len(sc) else np.arange(len(sc))
            for j in top:
                ap_ = int(self.rep_pid[idx[j]])
                v = float(sc[j])
                if v > best.get(ap_, (-1.0, 0.0))[0]:
                    best[ap_] = (v, float(shifts[j]))
        items = sorted(best.items(), key=lambda x: -x[1][0])[:cfg.n_analog]
        top_sim = 0.0
        if items:
            apid = np.array([x[0] for x in items])
            asim = np.array([x[1][0] for x in items], np.float32)
            ashift = np.array([x[1][1] for x in items], np.float32)
            cf = _unpack(P.fp[cand])
            af = _unpack(P.fp[apid])
            inter = cf @ af.T
            tan = inter / (cf.sum(1)[:, None] + af.sum(1)[None, :] - inter + 1e-9)
            w = np.clip(asim, 0, None)
            w4 = w ** cfg.sim_power
            ap4 = (tan * w4[None, :]).max(1)
            X[:, COL["ap4"]] = ap4
            X[:, COL["ap1"]] = (tan * w[None, :]).max(1)
            X[:, COL["best_tan"]] = tan.max(1)
            X[:, COL["top_tan"]] = tan[:, 0]
            z0 = np.abs(ashift) < 0.01
            if z0.any():
                X[:, COL["tan_self"]] = (tan[:, z0] * w[None, z0]).max(1)
            X[:, COL["ap_rank"]] = rank_norm(ap4)
            X[:, COL["ap_gap"]] = ap4 - ap4.max()
            top_sim = float(asim[0])
            X[:, COL["n_analog_hi"]] = float((asim > 0.5).sum())
        X[:, COL["top_sim"]] = top_sim

        # ---------------- spectrum -> fingerprint model: f.z of each candidate ----------------
        if self.model is not None:
            z = self.model(specs)
            fz = _unpack(P.fp[cand]) @ z
            X[:, COL["fz"]] = fz / 100.0
            sd = fz.std()
            X[:, COL["fz_z"]] = (fz - fz.mean()) / sd if sd > 1e-9 else 0.0
            X[:, COL["fz_rank"]] = rank_norm(fz)
            X[:, COL["fz_gap"]] = (fz - fz.max()) / 100.0
            e = np.exp((fz - fz.max()) / 4.0)
            X[:, COL["fz_soft"]] = e / e.sum()
            X[:, COL["fz_grp_gap2"]] = (np.sort(fz)[-1] - np.sort(fz)[-2]) / 100.0 if nc > 1 else 0.0

        # ---------------- context ----------------
        X[:, COL["mass_ppm"]] = np.abs(P.mass[cand] - target) / target * 1e6
        X[:, COL["n_cand"]] = np.log(nc)
        X[:, COL["n_query"]] = nq
        X[:, COL["n_pos"]] = sum(1 for s in specs if s["mode"] > 0)
        X[:, COL["n_neg"]] = sum(1 for s in specs if s["mode"] < 0)
        X[:, COL["target_mass"]] = target / 1000.0
        info = dict(target=target, n_cand=nc, ppm=ppm, lib_max=gmax, top_sim=top_sim, secs=time.time() - t0)
        return cand.astype(np.int64), X, info

    def nearest_structures(self, target, n):
        """Pool ids closest in mass to `target` (padding for short candidate lists)."""
        m = self.P.mass_sorted
        i = int(np.searchsorted(m, target))
        idx = np.arange(max(0, i - n), min(len(m), i + n))
        return self.P.order[idx[np.argsort(np.abs(m[idx] - target), kind="mergesort")][:n]]


def rule_score(X, name):
    """Hand-written baselines on the feature matrix (higher = better).
    'gate:<tau>': molecules whose best library match (lib_grp_max) reaches tau keep the class-1 order (lib_analog);
    the others are treated as class-2 and ranked by analog evidence (ap4) with a small library term.
    'fz': the spectrum -> fingerprint model alone. 'gfz:<tau>:<w>': same gate, but class-2 molecules are ranked by
    fz_z + w * (ap4 + 0.2 * lib_max) (w = 0: model only; large w: the gate rule with fz as tie-break)."""
    f = lambda n: X[:, COL[n]]
    tie = 1e-3 * f("ap4") - 1e-6 * f("mass_ppm")
    if name == "lib_max":
        return f("lib_max") + tie
    if name == "lib_max_mean":
        return f("lib_max") + 0.3 * f("lib_mean") + tie
    if name == "lib_analog":
        return f("lib_max") + 0.3 * f("lib_mean") + 0.5 * f("ap4") + tie
    if name == "fz":
        return f("fz") + tie
    if name.startswith("gfz:"):
        tau, w = map(float, name.split(":")[1:])
        class1 = f("lib_grp_max") >= tau
        return np.where(class1, rule_score(X, "lib_analog"), f("fz_z") + w * (f("ap4") + 0.2 * f("lib_max")) + tie)
    if name.startswith("gate:"):
        tau = float(name.split(":")[1])
        class1 = f("lib_grp_max") >= tau
        return np.where(class1, rule_score(X, "lib_analog"), f("ap4") + 0.2 * f("lib_max") + tie)
    raise ValueError(name)


def hybrid_score(X, tau, rk):
    """Per-molecule gate: molecules whose best library match (lib_grp_max) reaches tau keep the class-1 order
    (lib_analog); the others are ranked by the learned score rk (rows aligned with X)."""
    return np.where(X[:, COL["lib_grp_max"]] >= tau, rule_score(X, "lib_analog"), rk)


def zscore_by(v, qid=None):
    """Per-molecule z-score of v (rows grouped by qid; qid=None = all rows are one molecule). Constant -> 0."""
    v = np.asarray(v, np.float64)
    if qid is None:
        qid = np.zeros(len(v), np.int64)
    _, inv = np.unique(qid, return_inverse=True)
    n = np.bincount(inv)
    mu = np.bincount(inv, v) / n
    sd = np.sqrt(np.maximum(np.bincount(inv, v * v) / n - mu ** 2, 0.0))
    return np.where(sd[inv] > 1e-9, (v - mu[inv]) / np.where(sd[inv] > 1e-9, sd[inv], 1.0), 0.0)


def blend_score(X, tau, rk, w, qid=None):
    """hybrid_score whose class-2 branch is z(rk) + w * (ap4 + 0.2 * lib_max): gfz with the learned score rk in place
    of fz_z (z = per-molecule z-score, so w means the same in k08, which scores all molecules at once with qid, and
    k09, which scores one molecule with qid=None)."""
    f = lambda n: X[:, COL[n]]
    tie = 1e-3 * f("ap4") - 1e-6 * f("mass_ppm")
    return hybrid_score(X, tau, zscore_by(rk, qid) + w * (f("ap4") + 0.2 * f("lib_max")) + tie)


RULES = ["lib_max", "lib_max_mean", "lib_analog"]
GATE_RULES = [f"gate:{t}" for t in (0.5, 0.6, 0.7, 0.8, 0.9)]
GFZ_RULES = [f"gfz:{t}:{w}" for t in (0.5, 0.6, 0.7, 0.8, 0.9) for w in (0, 1, 2, 5, 10)]
FZ_RULES = ["fz"] + GFZ_RULES


# ---------------------------------------------------------------- M4: per-molecule router (class 1 or not)
ROUTE_FEATURES = ["lib_grp_max", "lib_grp_gap2", "n_cand", "n_query", "n_pos", "target_mass", "top_sim",
                  "n_analog_hi", "ap4_max", "fz_rank_libtop", "same_top", "libmax_fztop", "fz_soft_libtop",
                  "rk_margin", "rk_libtop", "libmax_rktop", "la_margin"]


def _top2_margin(v, inv, n_groups):
    """Per group: largest - second largest of v (0 for single-row groups). inv = group index of each row."""
    o = np.lexsort((-v, inv))
    vs, gs = v[o], inv[o]
    first = np.r_[True, gs[1:] != gs[:-1]]
    second = np.r_[False, first[:-1]] & ~first
    top = np.zeros(n_groups)
    top[gs[first]] = vs[first]
    sec = top.copy()
    sec[gs[second]] = vs[second]
    return top - sec


def route_features(X, rk, qid=None):
    """Molecule-level features for the router (one row per qid, sorted; qid=None = one molecule): the library
    channel's group context, how well fz and the learned score rk agree with the library's top candidate, and the
    score margins. Uses only the per-candidate features and rk (no provenance). -> (qids, matrix in ROUTE_FEATURES)."""
    f = lambda n: X[:, COL[n]].astype(np.float64)
    n = len(X)
    if qid is None:
        qid = np.zeros(n, np.int64)
    uq, inv = np.unique(qid, return_inverse=True)
    G = len(uq)
    srt = np.lexsort((np.arange(n), inv))  # rows grouped by molecule (stable)
    starts = np.r_[0, np.cumsum(np.bincount(inv, minlength=G))[:-1]]
    lib, fz, rkz = f("lib_max"), f("fz"), zscore_by(rk, qid)
    la = rule_score(X, "lib_analog").astype(np.float64)

    def by_desc(v):
        """Rows sorted by molecule, then by v descending; -> (order, 1-based rank of each row in its molecule)."""
        o = np.lexsort((-v, inv))
        r = np.empty(n)
        r[o] = np.arange(n) - starts[inv[o]] + 1
        return o, r

    o_lib, _ = by_desc(lib)
    o_fz, fz_rank = by_desc(fz)
    o_rk, _ = by_desc(rkz)
    i_lib, i_fz, i_rk = o_lib[starts], o_fz[starts], o_rk[starts]
    R = np.zeros((G, len(ROUTE_FEATURES)))
    col = {c: j for j, c in enumerate(ROUTE_FEATURES)}
    for c in ["lib_grp_max", "lib_grp_gap2", "n_cand", "n_query", "n_pos", "target_mass", "top_sim", "n_analog_hi"]:
        R[:, col[c]] = f(c)[srt[starts]]
    R[:, col["ap4_max"]] = np.maximum.reduceat(f("ap4")[srt], starts)
    R[:, col["fz_rank_libtop"]] = np.minimum(fz_rank[i_lib], 300)
    R[:, col["same_top"]] = (i_lib == i_fz).astype(float)
    R[:, col["libmax_fztop"]] = lib[i_fz]
    R[:, col["fz_soft_libtop"]] = f("fz_soft")[i_lib]
    R[:, col["rk_margin"]] = _top2_margin(rkz, inv, G)
    R[:, col["rk_libtop"]] = rkz[i_lib]
    R[:, col["libmax_rktop"]] = lib[i_rk]
    R[:, col["la_margin"]] = _top2_margin(la, inv, G)
    return uq, R


def route_score(X, rk, p, w, mode, t=0.5, qid=None):
    """Routed score with the router's class-1 probability p (one value per row, i.e. p of the row's molecule).
    The class-2 branch is blend_score's: z(rk) + w * (ap4 + 0.2 * lib_max).
    mode 'hard': lib_analog where p >= t, else the class-2 branch.
    mode 'soft': p * z(lib_analog) + (1 - p) * z(class-2 branch), z per molecule."""
    f = lambda n: X[:, COL[n]]
    tie = 1e-3 * f("ap4") - 1e-6 * f("mass_ppm")
    c2 = zscore_by(rk, qid) + w * (f("ap4") + 0.2 * f("lib_max")) + tie
    la = rule_score(X, "lib_analog")
    if mode == "hard":
        return np.where(p >= t, la, c2)
    if mode == "soft":
        return p * zscore_by(la, qid) + (1 - p) * zscore_by(c2, qid) + tie
    raise ValueError(mode)

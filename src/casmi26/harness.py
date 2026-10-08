"""Local validation that imitates the hidden test (timsTOF queries, 10 adducts, 1-16 spectra per molecule).

Query sets (one query = one structure, truth = its pool id `pid`):
  A    enveda-np-examples structures; the whole np-examples library is hidden.          class-1, primary
  B_A  same queries, and the truth's spectra are hidden in every library (truth stays in the pool).  class-2
  C    enveda-180 structures that also exist in another library; their enveda-180
       spectra are the queries and are hidden from the library.                        class-1, larger N
  B_C  same queries as C with the truth hidden in every library.                        class-2, larger N
  R1   gate / ranker training queries (independent libraries), truth hidden in the query library.   class-1
  R2   same queries with the truth hidden in every library.                                          class-2
  V    validation structures for the spectrum -> fingerprint model (truth hidden everywhere).
A, C, R and V structures form the holdout that the fingerprint model must not be trained on.
"""
import time

import numpy as np
import pandas as pd

from . import chem

NP_LIB = "enveda-np-examples"
E180_LIB = "enveda-180"
QUERY_MAX_ABS_PPM = 20.0
# T のクエリ元は独立に測定されたライブラリに限る。gnps / mona / spectraverse は他ライブラリの再収録が多く、
# 隠したスペクトルの複製がライブラリ側に残って正解が自明になる
T_QUERY_LIBS = ["enveda-180", "pluskal_ms2", "riken", "massbank", "msdial"]


def _query_rows(C):
    """Spectra usable as queries: test adducts, >= 1 peak, consistent precursor."""
    test_ad = np.array([C.adduct_id.get(a, -99) for a in chem.TEST_ADDUCTS])
    ppm = np.abs(C.ppm)
    return np.isin(C.adduct, test_ad) & (C.npk >= 1) & ((ppm <= QUERY_MAX_ABS_PPM) | ~np.isfinite(ppm))


def _k_sampler(rng, k_dist):
    ks, ps = np.array(sorted(k_dist)), np.array([k_dist[k] for k in sorted(k_dist)], float)
    ps /= ps.sum()
    return lambda: int(rng.choice(ks, p=ps))


def build_sets(C, P, k_dist, n_c=1000, n_r=3000, n_v=2000, seed=0):
    """-> (queries DataFrame(qid, set, pid, qlib, rows, hide_lib, hide_pid, hide_pid_lib),
           holdout DataFrame(pid, key, set))."""
    rng = np.random.default_rng(seed)
    draw_k = _k_sampler(rng, k_dist)
    np_id, e180_id = C.lib_id[NP_LIB], C.lib_id[E180_LIB]
    d = pd.DataFrame({"row": np.arange(len(C.sid)), "pid": P.sid2pid[C.sid], "lib": C.lib, "q": _query_rows(C),
                      "keep": C.keep})
    d = d[d.pid >= 0]
    # libraries in which each structure has kept (library-usable) spectra
    kept_libs = d[d.keep].groupby("pid").lib.agg(lambda s: set(s.tolist()))
    qrows = d[d.q].groupby(["pid", "lib"]).row.agg(list)
    out = []

    def add(name, p, lib, rows, hide_lib=-1, hide_pid=-1, hide_pid_lib=-1):
        out.append(dict(set=name, pid=int(p), qlib=int(lib), rows=list(map(int, rows)), hide_lib=hide_lib,
                        hide_pid=hide_pid, hide_pid_lib=hide_pid_lib))

    def sample(p, lib):
        rows = qrows[(p, lib)]
        k = min(len(rows), draw_k())
        return sorted(rng.choice(rows, size=k, replace=False).tolist())

    # A / B_A: np-examples, all their test-adduct spectra (<= 16), whole np library hidden
    a_p = sorted({p for (p, l) in qrows.index if l == np_id})
    for p in a_p:
        rows = qrows[(p, np_id)][:16]
        add("A", p, np_id, rows, hide_lib=np_id)
        add("B_A", p, np_id, rows, hide_lib=np_id, hide_pid=p, hide_pid_lib=-1)
    used = set(a_p)

    # C / B_C: enveda-180 structures with kept spectra in another library
    c_pool = sorted(p for (p, l) in qrows.index
                    if l == e180_id and p not in used and len(kept_libs.get(p, set()) - {e180_id}) > 0)
    c_p = sorted(rng.choice(c_pool, size=min(n_c, len(c_pool)), replace=False).tolist())
    for p in c_p:
        rows = sample(p, e180_id)
        add("C", p, e180_id, rows, hide_pid=p, hide_pid_lib=e180_id)
        add("B_C", p, e180_id, rows, hide_pid=p, hide_pid_lib=-1)
    used |= set(c_p)

    # R1 / R2: independent libraries, enveda-180 (timsTOF) first
    t_libs = {C.lib_id[n] for n in T_QUERY_LIBS if n in C.lib_id}
    by_p = {}
    for (p, l) in qrows.index:
        if p not in used and l in t_libs:
            by_p.setdefault(p, []).append(l)
    r_pool = []
    for p, ls in sorted(by_p.items()):
        other = kept_libs.get(p, set())
        libs = [l for l in ls if len(other - {l}) > 0]
        if libs:
            r_pool.append((p, e180_id if e180_id in libs else libs[rng.integers(len(libs))]))
    e_first = [x for x in r_pool if x[1] == e180_id]
    rest = [x for x in r_pool if x[1] != e180_id]
    rng.shuffle(e_first)
    pick = e_first[:n_r]
    if len(pick) < n_r:
        pick += [rest[i] for i in rng.choice(len(rest), size=min(n_r - len(pick), len(rest)), replace=False)]
    for p, l in pick:
        rows = sample(p, l)
        add("R1", p, l, rows, hide_pid=p, hide_pid_lib=l)
        add("R2", p, l, rows, hide_pid=p, hide_pid_lib=-1)
    used |= {p for p, _ in pick}

    # V: fingerprint-model validation structures (any library but np-examples), truth hidden everywhere
    q_libs = {}
    for (p, l) in qrows.index:
        if l != np_id and p not in used:
            q_libs.setdefault(p, []).append(l)
    v_cand = sorted(q_libs)
    for p in sorted(rng.choice(v_cand, size=min(n_v, len(v_cand)), replace=False).tolist()):
        l = q_libs[p][rng.integers(len(q_libs[p]))]
        add("V", p, l, sample(p, l), hide_pid=p, hide_pid_lib=-1)

    q = pd.DataFrame(out)
    q.insert(0, "qid", np.arange(len(q)))
    hold = q[q.set.isin(["A", "C", "R1", "V"])][["pid", "set"]].drop_duplicates("pid")
    hold["set"] = hold["set"].replace({"R1": "R"})
    hold.insert(1, "key", P.key[hold.pid.values])
    return q, hold.reset_index(drop=True)


TIMS_LIBS = ("enveda-180", "enveda-np-examples")


def query_specs(C, rows):
    """Spectra dicts of cache rows (ce in eV or NaN; tims = acquired on the Enveda timsTOF pipeline)."""
    tims = {C.lib_id[n] for n in TIMS_LIBS if n in C.lib_id}
    return [dict(mz=C.spectrum(r)[0], it=C.spectrum(r)[1], prec=float(C.prec[r]), adduct=C.adducts[C.adduct[r]],
                 mode=int(C.mode[r]), ce=float(C.ce[r]), tims=float(C.lib[r] in tims)) for r in rows]


def test_specs(g):
    """Spectra dicts of one molecule of the competition test table (rows of one molecule_id), with the same
    conditioning as query_specs: ce = mean of collision_energy_ev (NaN when missing), tims = timsTOF instrument."""
    from . import spectra
    out = []
    for r in g.itertuples():
        mz, it = spectra.clean_one(r.ms2_mzs, r.ms2_normalized_intensities, r.precursor_mz)
        ce = getattr(r, "collision_energy_ev", None)
        ce = np.asarray(ce if ce is not None else [], float).ravel()
        ce = float(np.nanmean(ce)) if np.isfinite(ce).any() else np.nan
        out.append(dict(mz=mz, it=it, prec=float(r.precursor_mz), adduct=str(r.adduct),
                        mode=1 if r.ionization_mode == "positive" else -1, ce=ce,
                        tims=float(getattr(r, "instrument_type", "") == "timsTOF")))
    return out


TRAIN_SETS = ("R1", "R2")


def run_queries(searcher, queries, log=print, max_cand_train=300, keep_rule=None):
    """Feature rows for every query. For training sets (R1/R2), candidates are cut to the top `max_cand_train`
    by `keep_rule` (the truth is always kept). keep_rule(X) returns one score array or a list of them; with a list
    the union of each score's top `max_cand_train` is kept. -> (features with qid/pid/label, per-query info)."""
    from .search import FEATURES
    C = searcher.C
    parts, infos = [], []
    t0 = time.time()
    for i, q in enumerate(queries.itertuples()):
        cand, X, info = searcher.features(query_specs(C, q.rows), dict(hide_lib=q.hide_lib, hide_pid=q.hide_pid,
                                                                         hide_pid_lib=q.hide_pid_lib))
        label = (cand == q.pid).astype(np.int8)
        if q.set in TRAIN_SETS and keep_rule is not None and len(cand) > max_cand_train:
            scores = keep_rule(X)
            sel = np.zeros(len(cand), bool)
            for sc in (scores if isinstance(scores, (list, tuple)) else [scores]):
                sel[np.argsort(-sc, kind="mergesort")[:max_cand_train]] = True
            sel |= label > 0
            cand, X, label = cand[sel], X[sel], label[sel]
        df = pd.DataFrame(X, columns=FEATURES)
        df.insert(0, "label", label)
        df.insert(0, "pid", cand.astype(np.int32))
        df.insert(0, "qid", np.int32(q.qid))
        parts.append(df)
        infos.append(dict(qid=q.qid, set=q.set, qlib=C.libs[q.qlib], n_spec=len(q.rows), recall=int(label.any()),
                          **info))
        if (i + 1) % 200 == 0 or i + 1 == len(queries):
            log(f"  {i + 1}/{len(queries)} queries, {time.time() - t0:.0f}s, "
                f"{(time.time() - t0) / (i + 1):.2f}s/query")
    feats = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return feats, pd.DataFrame(infos)


def ranks_of_truth(qid, label, score, pid=None):
    """1-based rank of the truth per qid (inf if absent). Ties broken by pid for determinism."""
    d = pd.DataFrame({"qid": qid, "label": label, "score": score, "pid": pid if pid is not None else 0})
    d = d.sort_values(["qid", "score", "pid"], ascending=[True, False, True], kind="mergesort")
    d["r"] = d.groupby("qid").cumcount() + 1
    r = d[d.label > 0].groupby("qid").r.min()
    allq = pd.Index(np.unique(qid), name="qid")
    return r.reindex(allq).fillna(np.inf)


def metrics(ranks, k=25):
    r = np.asarray(ranks, float)
    rr = np.where(r <= k, 1.0 / r, 0.0)
    n = len(r)
    return dict(n=n, mrr25=rr.mean() if n else np.nan, mrr_se=rr.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan,
                top1=(r <= 1).mean(), top10=(r <= 10).mean(), top25=(r <= 25).mean(), recall=np.isfinite(r).mean())


# ---------------------------------------------------------------- uncertainty (yardstick review, 2026-10-07)
def rr_of(ranks, k=25):
    """Reciprocal rank at k (0 beyond k or when the truth is absent)."""
    r = np.asarray(ranks, float)
    return np.where(r <= k, 1.0 / r, 0.0)


def boot_index(n, n_boot=2000, seed=0):
    """Bootstrap resamples of n queries (n_boot, n). Reuse one index for every model of a set, so that differences
    of replicate means are paired bootstrap differences."""
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def boot_reps(x, idx):
    """Replicate means of x under the resamples idx (from boot_index(len(x)))."""
    return np.asarray(x, float)[idx].mean(1)


def ci(point, reps, level=0.95):
    """dict(est, se, lo, hi) from a point estimate and its bootstrap replicates (percentile interval)."""
    a = (1 - level) / 2
    return dict(est=float(point), se=float(np.std(reps, ddof=1)), lo=float(np.quantile(reps, a)),
                hi=float(np.quantile(reps, 1 - a)))


def paired_ci(x, y, n_boot=2000, seed=0, level=0.95):
    """Mean of x - y over paired queries with its bootstrap interval: dict(est, se, lo, hi, n)."""
    d = np.asarray(x, float) - np.asarray(y, float)
    out = ci(d.mean(), boot_reps(d, boot_index(len(d), n_boot, seed)), level) if len(d) else \
        dict(est=np.nan, se=np.nan, lo=np.nan, hi=np.nan)
    return dict(out, n=len(d))

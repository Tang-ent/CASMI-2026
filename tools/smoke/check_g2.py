"""G2 checks (M6 P0-3, 2026-10-12), in docker on CPU with the smoke data (tools/smoke/run.sh g2):
(a) the atom-set enumeration gives the same unique fragment masses (>= 30 Da, 1e-4 Da) as frag.fragments_of_mol
(b) cosine: all mass on the matched slots -> the explained share of the observed norm; nothing matched -> 0 with
    finite zero gradients; two groups on one observed peak add up; a structure without fragments -> 0, no NaN
(c) one training batch forward / backward is finite; G2Model.score runs on a query
(d) g2.train resumed from its first save point == the straight run (identical weights)
Mounts /src (code), /d (smoke data), /k26 (smoke k26), /w (scratch).
G2_REAL=1 (tools/local/run_check_g2_real.sh): the plan 9.2 checks (e)-(j) on the real inputs instead (see real_checks;
per-spectrum outputs, consensus scores, training-path cosine and default outputs against a reference commit)."""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "/src")
import torch  # noqa: E402

from casmi26 import cache, enc3, fpnet, frag, g2, harness, pool, subform  # noqa: E402

torch.set_num_threads(4)
REAL = os.environ.get("G2_REAL") == "1"


# ---------------------------------------------------------------------------------------------- plan 9.2 (real inputs)
def load_head(name):
    """Module <name>.py of the reference commit (mounted at /head) as casmi26.<name>_head; its relative imports
    resolve to the current package."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"casmi26.{name}_head", f"/head/{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


class SubGraphs:
    """g2.Graphs of a subset of pool ids, indexed by pool id (G2Model uses only .ok[pid] and .get(pid))."""

    def __init__(self, P, pids):
        self.pids = np.unique(np.asarray(pids, np.int64))
        self.G = g2.Graphs.build(list(P.smiles[self.pids]))
        self.loc = {int(p): i for i, p in enumerate(self.pids)}
        self.ok = self

    def __getitem__(self, p):
        i = self.loc.get(int(p))
        return i is not None and bool(self.G.ok[i])

    def get(self, p):
        i = self.loc.get(int(p))
        return None if i is None else self.G.get(i)


def real_checks():
    """Plan 9.2 checks on the real inputs (tools/local/run_check_g2_real.sh; CPU, fp32): k01 cache, k05 pool + k05b
    mass fix, k21 fragment table, casmi26-g2 (g2.pt = the k30 model) under /kaggle/input, the reference commit's
    g2.py / frag.py / spectra.py / search.py under /head. Queries: A / R1 / R2 / V with a window <= MAX_WIN, N_PER of
    each kind (one spectrum, >= 2 finite CEs, >= 2 adducts, >= 2 spectra all without CE).
    (e) default outputs == the reference commit (G2Model.score, match_features, merge_spectra, Searcher.features)
    (f) [task a] mean of score(per_spec=True) per-spectrum cosines == score() (1e-6)
    (g) [task b] consensus: one-spectrum queries (merge is the identity after observed() unless the spectrum has peaks
        closer than 0.01 Da) == score() / frag_expl; a spectrum given twice == the spectrum alone; queries whose
        adducts have one spectrum each == score() / frag_expl (count-weighted mean = plain mean); 1e-6
    (h) [task c] training-path cosine (forward and d/dlogits) == the reference commit on a fixed batch (bit-exact)
    (i) [task d] mean of match_features(per_spec=True) rows == match_features() (1e-6)
    (j) Searcher with spec_stats / consensus: the first columns == the default features; stats sane"""
    from casmi26 import chem, paths, search, spectra
    MAX_WIN, N_PER, TOL = int(os.environ.get("MAX_WIN", 300)), int(os.environ.get("N_PER", 8)), 1e-6
    torch.set_num_threads(int(os.environ.get("THREADS", 8)))
    ok = True
    t0 = time.time()
    gh, fh, sh, srh = (load_head(n) for n in ("g2", "frag", "spectra", "search"))
    C = cache.Cache(cache.Cache.find_dir())
    PD = pool.Pool.find_dir()
    P = pool.Pool(PD, pubchem=False)
    Q = pd.read_csv(f"{PD}/queries.csv")
    Q = Q[Q.set.isin(["A", "R1", "R2", "V"])].sample(frac=1.0, random_state=0)
    FT = frag.FragTable(frag.FragTable.find_dir())
    cfg = search.SearchCfg()
    print(f"[real] cache / pool / frag loaded ({time.time() - t0:.0f}s), {len(Q)} queries in A/R1/R2/V")

    kinds = ["single", "multi_ce", "multi_adduct", "nan_ce"]
    pick = {k: [] for k in kinds}
    qs = {}
    for qid, set_, rows in Q[["qid", "set", "rows"]].itertuples(index=False):
        if all(len(v) >= N_PER for v in pick.values()):
            break
        sp = [s for s in harness.query_specs(C, [int(x) for x in str(rows).split(",")])
              if np.isfinite(chem.neutral_mass(s["prec"], s["adduct"]))][:cfg.max_query_spectra]
        if not sp:
            continue
        target = float(np.median([chem.neutral_mass(s["prec"], s["adduct"]) for s in sp]))
        cand = P.window(target, cfg.ppm_win)
        if len(cand) == 0:
            cand = P.window(target, cfg.ppm_fallback)
        if not 0 < len(cand) <= MAX_WIN:
            continue
        ce = np.array([s["ce"] for s in sp])
        kk = [k for k, c in zip(kinds, [len(sp) == 1, len(set(ce[np.isfinite(ce)])) >= 2,
                                        len({s["adduct"] for s in sp}) >= 2,
                                        len(sp) >= 2 and not np.isfinite(ce).any()]) if c and len(pick[k]) < N_PER]
        if kk:
            qs[qid] = dict(set=set_, specs=sp, cand=np.asarray(cand, np.int64))
            for k in kk:
                pick[k].append(qid)
    # informative: spectra whose polarity disagrees with the adduct name (adduct_groups keys on (adduct, mode))
    n_sp = n_off = 0
    q_off, q_split = set(), set()
    for qid, rows in Q[["qid", "rows"]].itertuples(index=False):
        sp = harness.query_specs(C, [int(x) for x in str(rows).split(",")])
        n_sp += len(sp)
        off = [s for s in sp if (s["adduct"].endswith("+") and s["mode"] < 0)
               or (s["adduct"].endswith("-") and s["mode"] > 0)]
        n_off += len(off)
        if off:
            q_off.add(qid)
            if any(len({t["mode"] for t in sp if t["adduct"] == s["adduct"]}) > 1 for s in off):
                q_split.add(qid)
    print(f"[real] spectra with polarity != adduct sign: {n_off} of {n_sp} in {len(q_off)} queries; queries where "
          f"one adduct label has both polarities (split into two groups): {len(q_split)}")
    print(f"[real] queries per kind {({k: len(v) for k, v in pick.items()})}, {len(qs)} distinct, "
          f"{sum(len(q['cand']) for q in qs.values())} candidates")
    SG = SubGraphs(P, np.concatenate([q["cand"] for q in qs.values()]))
    step = json.load(open(paths.find("result_g2.json")))["step"]
    M = g2.G2Model(paths.find("g2.pt"), G=SG, device="cpu")
    MH = gh.G2Model(paths.find("g2.pt"), G=SG, device="cpu")
    print(f"[real] g2.pt step {M.step} (result_g2.json {step}), graphs {int(SG.G.ok.sum())}/{len(SG.pids)} "
          f"({time.time() - t0:.0f}s)")
    ok &= int(M.step) == int(step)

    # (e) default outputs vs the reference commit; (f) per-spectrum mean; (i) frag per-spectrum mean
    n_eq = dict(score=0, frag=0, merge=0)
    d_f = d_i = 0.0
    n_ps = 0
    for qid, q in qs.items():
        sp, cand = q["specs"], q["cand"]
        s_new, s_head = M.score(sp, cand), MH.score(sp, cand)
        n_eq["score"] += np.array_equal(s_new, s_head, equal_nan=True)
        mean, Mx, ce, used = M.score(sp, cand, per_spec=True)
        ok &= np.array_equal(mean, s_new, equal_nan=True) and Mx.shape == (len(used), len(cand))
        ok &= np.array_equal(ce, [sp[i]["ce"] for i in used], equal_nan=True)
        h = np.isfinite(s_new)
        if len(used) and h.any():
            d_f = max(d_f, float(np.abs(Mx[:, h].astype(np.float64).mean(0) - s_new[h]).max()))
            ok &= bool(np.isnan(Mx[:, ~h]).all()) and bool(np.isfinite(Mx[:, h]).all())
        n_ps += len(used)
        fcsr = FT.csr(cand)
        e0, t0_ = frag.match_features(sp, *fcsr)
        eh, th = fh.match_features(sp, *fcsr)
        n_eq["frag"] += np.array_equal(e0, eh) and np.array_equal(t0_, th)
        e1, t1, E, fu = frag.match_features(sp, *fcsr, per_spec=True)
        ok &= np.array_equal(e1, e0) and np.array_equal(t1, t0_) and E.shape == (len(fu), len(cand))
        if len(fu):
            d_i = max(d_i, float(np.abs(E.astype(np.float64).mean(0) - e0).max()))
        for md in (1, -1):
            grp = [(s["mz"], s["it"]) for s in sp if s["mode"] == md]
            a, b = spectra.merge_spectra(grp), sh.merge_spectra(grp)
            n_eq["merge"] += 0.5 * (np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]))
    print(f"(e) == reference commit on {len(qs)} queries: G2Model.score {n_eq['score']}, match_features "
          f"{n_eq['frag']}, merge_spectra {n_eq['merge']:.0f}")
    ok &= n_eq["score"] == n_eq["frag"] == len(qs) and n_eq["merge"] == len(qs)
    print(f"(f) [a] g2 per-spectrum mean vs score(): max |d| {d_f:.2e} over {n_ps} spectra (tol {TOL:g})")
    ok &= d_f <= TOL
    print(f"(i) [d] frag per-spectrum mean vs match_features(): max |d| {d_i:.2e} (tol {TOL:g})")
    ok &= d_i <= TOL

    # (g) consensus invariants
    def merged_is_identity(s):
        g = spectra.adduct_groups([s], gap=g2.PREC_GAP)[0]
        a, b = g2.observed(s["mz"], s["it"], s["prec"]), g2.observed(g["mz"], g["it"], g["prec"])
        return len(a[0]) == len(b[0]) and np.array_equal(a[0], b[0]) and np.allclose(a[1], b[1], rtol=0, atol=1e-6)

    res = dict(single=[], twice=[], per_adduct=[], f_single=[], f_twice=[], f_per_adduct=[], decomp=[], f_decomp=[])
    n_skip, res_ni = 0, []
    for qid, q in qs.items():
        sp, cand = q["specs"], q["cand"]
        fcsr = FT.csr(cand)
        use = [s for s in sp if len(g2.observed(s["mz"], s["it"], s["prec"])[0])]
        if not use:
            continue
        if not all(merged_is_identity(s) for s in use):
            n_skip += 1
            if len(use) == 1:  # informative: how far the collapse of close peaks moves the consensus
                s0 = M.score(sp, cand)
                h = np.isfinite(s0)
                res_ni.append((np.abs(M.score_consensus(sp, cand)[h] - s0[h]).max(),
                               np.abs(frag.consensus_expl(sp, *fcsr) - frag.match_features(sp, *fcsr)[0]).max()))
            continue
        s0, e0 = M.score(sp, cand), frag.match_features(sp, *fcsr)[0]
        h = np.isfinite(s0)
        if len(use) == 1:
            res["single"].append(np.abs(M.score_consensus(sp, cand)[h] - s0[h]).max())
            res["f_single"].append(np.abs(frag.consensus_expl(sp, *fcsr) - e0).max())
            s1 = M.score(use, cand)
            res["twice"].append(np.abs(M.score_consensus(use * 2, cand)[h] - s1[h]).max())
            res["f_twice"].append(np.abs(frag.consensus_expl(use * 2, *fcsr)
                                         - frag.match_features(use, *fcsr)[0]).max())
        elif len({s["adduct"] for s in use}) == len(use):
            res["per_adduct"].append(np.abs(M.score_consensus(sp, cand)[h] - s0[h]).max())
            res["f_per_adduct"].append(np.abs(frag.consensus_expl(sp, *fcsr) - e0).max())
    # decomposition (any query, merge identity not needed): consensus == count-weighted mean over the adducts of
    # score() / match_features() of the merged spectrum (one member) or the consensus of the members alone (several)
    n_multi = 0
    for qid, q in qs.items():
        sp, cand = q["specs"], q["cand"]
        fcsr = FT.csr(cand)
        gs = [g for g in spectra.adduct_groups(sp, gap=g2.PREC_GAP) if len(g2.observed(g["mz"], g["it"], g["prec"])[0])]
        if not gs:
            continue
        exp_g, exp_f, w = 0.0, 0.0, 0.0
        for g in gs:
            mem = [sp[i] for i in g["idx"]]
            if g["n"] == 1:
                one = [dict(mz=g["mz"], it=g["it"], prec=g["prec"], mode=g["mode"], adduct=g["adduct"], ce=g["ce"][0])]
                cg, cf = M.score(one, cand), frag.match_features(one, *fcsr)[0]
            else:
                cg, cf = M.score_consensus(mem, cand), frag.consensus_expl(mem, *fcsr)
            exp_g, exp_f, w = exp_g + g["n"] * cg.astype(np.float64), exp_f + g["n"] * cf.astype(np.float64), w + g["n"]
        h = np.isfinite(exp_g)
        res["decomp"].append(np.abs(M.score_consensus(sp, cand)[h] - exp_g[h] / w).max())
        res["f_decomp"].append(np.abs(frag.consensus_expl(sp, *fcsr) - exp_f / w).max())
        n_multi += len(gs) >= 2
    for k, v in res.items():
        print(f"(g) [b] consensus {k:13s}: {len(v)} queries, max |d| {max(v, default=0.0):.2e}")
        ok &= max(v, default=0.0) <= TOL
    print(f"    identity checks left out {n_skip} queries (a spectrum with peaks < 0.01 Da apart: the merge collapses "
          f"them, so it is not the identity); decomposition covers {n_multi} queries with >= 2 adducts")
    ok &= len(res["single"]) > 0 and n_multi > 0
    if res_ni:
        ni = np.array(res_ni)
        print(f"    one-spectrum queries whose merge is not the identity ({len(ni)}): max |g2_cons - g2| per query "
              f"{np.round(ni[:, 0], 4).tolist()}, max |frag_expl_cons - frag_expl| {np.round(ni[:, 1], 4).tolist()}")
    # informative: consensus vs mean on the multi-spectrum queries
    dd = []
    for qid in set(pick["multi_ce"]) | set(pick["multi_adduct"]) | set(pick["nan_ce"]):
        q = qs[qid]
        a, b = M.score(q["specs"], q["cand"]), M.score_consensus(q["specs"], q["cand"])
        h = np.isfinite(a)
        if h.sum() > 2:
            dd.append((np.corrcoef(a[h], b[h])[0, 1], float(np.mean(b[h] - a[h]))))
    if dd:
        dd = np.array(dd)
        print(f"    multi-spectrum queries ({len(dd)}): corr(g2, g2_cons) median {np.nanmedian(dd[:, 0]):.3f}, "
              f"mean (g2_cons - g2) {dd[:, 1].mean():+.4f}")

    # (h) training-path cosine vs the reference commit: 2 spectra x 32 structures, logits of the trained net
    torch.set_num_threads(1)
    # the multi-spectrum query with the most candidates scoring > 0.1, its 32 best-scoring candidates
    qid = max(pick["multi_ce"] + pick["multi_adduct"],
              key=lambda i: int((np.nan_to_num(M.score(qs[i]["specs"], qs[i]["cand"])) > 0.1).sum()))
    q = qs[qid]
    use = [s for s in q["specs"] if len(g2.observed(s["mz"], s["it"], s["prec"])[0])][:2]
    sc = np.nan_to_num(M.score(q["specs"], q["cand"]), nan=-1.0)
    ps0 = [g2.piece(SG, p) for p in q["cand"][np.argsort(-sc, kind="mergesort")] if SG.ok[p]][:32]
    ps, jous, obs, ex, cond = [], [], [], [], []
    for k, s in enumerate(use):
        om, ow = g2.observed(s["mz"], s["it"], s["prec"])
        obs.append(np.pad(ow, (0, g2.TOP_PEAKS - len(ow))))
        for p in ps0:
            ps.append(p)
            jous.append(g2.match(p["umz"], g2.ion_of(s["mode"]), om))
            ex.append(k)
            cond.append(g2.cond_row(fpnet.adduct_index(s["adduct"]), s["mode"], s["ce"]))
    b = g2.collate(ps, jous)
    bt = g2.to_torch(b, "cpu")
    with torch.no_grad():
        lg0 = M.net.head(M.net.encode(bt), torch.as_tensor(np.stack(cond)))
    obs_t, ex_t = torch.as_tensor(np.stack(obs)), torch.as_tensor(np.array(ex, np.int64))
    out = []
    for fn in (g2.cosine, gh.cosine):
        lg = lg0.clone().requires_grad_(True)
        c = fn(lg, bt["valid"], bt["u"], bt["jou"], obs_t, ex_t)
        (1 - c).mean().backward()
        out.append((c.detach(), lg.grad))
    same = torch.equal(out[0][0], out[1][0]) and torch.equal(out[0][1], out[1][1])
    print(f"(h) [c] training-path cosine vs reference on S {len(ps)} (2 spectra of qid {qid}), F {b['bits'].shape[1]}: "
          f"cos {'==' if torch.equal(out[0][0], out[1][0]) else '!='}, grad "
          f"{'==' if torch.equal(out[0][1], out[1][1]) else '!='}, cos range {float(out[0][0].min()):.3f}.."
          f"{float(out[0][0].max()):.3f}")
    ok &= same
    torch.set_num_threads(int(os.environ.get("THREADS", 8)))

    # (j) Searcher: default == reference commit, flags only append columns
    S0 = search.Searcher(C, P, frag=FT, g2=M)
    SH = srh.Searcher(C, P, frag=FT, g2=M)
    S1 = search.Searcher(C, P, frag=FT, g2=M, spec_stats=True, consensus=True)
    k = len(S0.feature_names)
    ok &= S0.feature_names == SH.feature_names and S1.feature_names[:k] == S0.feature_names
    ok &= S1.feature_names[k:] == search.FRAG_STATS + search.G2_STATS + search.CONS
    ok &= search.Searcher(C, P, frag=FT, g2=M, spec_stats=True).feature_names[k:] == search.FRAG_STATS + search.G2_STATS
    ok &= search.Searcher(C, P, frag=FT, g2=M, consensus=True).feature_names[k:] == search.CONS
    col = {n: i for i, n in enumerate(S1.feature_names)}
    n_eq = n_q = 0
    bad = []
    for qid in pick["single"][:3] + pick["multi_ce"][:3] + pick["multi_adduct"][:3] + pick["nan_ce"][:3]:
        sp = qs[qid]["specs"]
        c0, X0, _ = S0.features(sp)
        ch, XH, _ = SH.features(sp)
        c1, X1, _ = S1.features(sp)
        n_q += 1
        n_eq += np.array_equal(c0, ch) and np.array_equal(c0, c1) and np.array_equal(X0, XH, equal_nan=True) and \
            np.array_equal(X1[:, :k], X0, equal_nan=True)
        f = lambda n: X1[:, col[n]]
        h = np.isfinite(f("g2"))
        for base, st in (("frag_expl", "frag_expl"), ("g2", "g2")):
            mn, sd, hi = f(st + "_min"), f(st + "_std"), f(st + "_hice")
            mm = f(base)
            hh = h if base == "g2" else np.ones(len(mm), bool)
            if not ((mn[hh] <= mm[hh] + 1e-6).all() and (sd[hh] >= 0).all()):
                bad.append((qid, base, "min/std"))
            if base == "g2" and (np.isfinite(mn[~h]).any() or np.isfinite(hi[~h]).any()):
                bad.append((qid, base, "nan"))
            if len(sp) == 1 and not (np.array_equal(mn[hh], mm[hh]) and (sd[hh] == 0).all()
                                     and np.array_equal(hi[hh], mm[hh])):
                bad.append((qid, base, "single"))
            ce = np.array([s["ce"] for s in sp])
            if not np.isfinite(ce).any() and not np.array_equal(hi[hh], mm[hh]):
                bad.append((qid, base, "nan-ce hice"))
        ok &= bool(np.isfinite(f("frag_expl_cons")).all() and np.isfinite(f("frag_expl_cons_z")).all()
                   and np.isfinite(f("g2_cons_z")).all() and (np.isfinite(f("g2_cons")) == h).all())
    print(f"(j) Searcher: default == reference and flags append only on {n_eq}/{n_q} queries; columns "
          f"{k} -> {len(S1.feature_names)}; stat sanity problems {bad}")
    ok &= n_eq == n_q and not bad

    # (k) ce=None (accepted by cond_row): score() runs and == the reference commit and == ce=NaN; the per-spectrum /
    #     consensus paths record it as NaN
    n_eq = n_q = 0
    for qid in pick["multi_ce"][:3] + pick["multi_adduct"][:3]:
        sp, cand = qs[qid]["specs"], qs[qid]["cand"]
        sn = [dict(s, ce=None) for s in sp]
        sa = [dict(s, ce=np.nan) for s in sp]
        a, b, c = M.score(sn, cand), MH.score(sn, cand), M.score(sa, cand)
        _, _, ce_n, _ = M.score(sn, cand, per_spec=True)
        gce = np.concatenate([g["ce"] for g in spectra.adduct_groups(sn, gap=g2.PREC_GAP)])
        n_q += 1
        n_eq += bool(np.array_equal(a, b, equal_nan=True) and np.array_equal(a, c, equal_nan=True)
                     and np.isnan(ce_n).all() and np.isnan(gce).all()
                     and np.array_equal(M.score_consensus(sn, cand), M.score_consensus(sa, cand), equal_nan=True))
    print(f"(k) ce=None: score() == reference == ce=NaN, per-spectrum / group CE NaN, consensus == ce=NaN on "
          f"{n_eq}/{n_q} queries")
    ok &= n_q > 0 and n_eq == n_q
    print(f"[real] {time.time() - t0:.0f}s")
    print("g2 real check:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if REAL:
    sys.exit(real_checks())

ok = True
P = pool.Pool("/d/pool", pubchem=False)
t0 = time.time()
G = g2.Graphs.build(P.smiles, workers=1)
print(f"graphs: {G.ok.sum()} of {len(G)} structures ({time.time() - t0:.1f}s), atoms {len(G.af)}, bonds {len(G.be)}")

# (a) masses against frag.py (uncapped enumeration; frag.py caps at 2,000 unique masses)
n_cmp = n_bad = n_cap = 0
for p in range(len(P)):
    g = G.get(p)
    if g is None:
        continue
    ref = frag.fragments(P.smiles[p])
    if len(ref) >= frag.MAX_FRAG:
        continue
    fr = g2.fragments(g, maxf=100000)
    mine = np.unique(np.round(fr["mass"], frag.DEC))
    n_cmp += 1
    if len(mine) != len(ref) or not np.array_equal(mine, ref):
        n_bad += 1
        if n_bad <= 3:
            print("  mismatch", P.smiles[p], len(mine), len(ref))
    n_cap += len(g2.fragments(g)["mass"]) >= g2.MAX_FRAG
print(f"(a) fragment masses == frag.py on {n_cmp - n_bad}/{n_cmp} structures; {n_cap} hit the {g2.MAX_FRAG} cap")
ok &= n_bad == 0 and n_cmp > 100


# (b) cosine unit checks: one structure, 2 fragments x 5 shifts, observed peaks [0.6, 0.8]
def cos_of(logit_rows, valid, u, jou):
    lg = torch.tensor(logit_rows, dtype=torch.float32, requires_grad=True)
    c = g2.cosine(lg[None], torch.tensor([valid]), torch.tensor([u]), torch.tensor([jou]),
                  torch.tensor([[0.6, 0.8]]), torch.zeros(1, dtype=torch.long))
    c.sum().backward()
    return float(c.detach()), lg.grad


big = -50.0
u = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
one = [[big] * 5, [big] * 5]
one[0][2] = 50.0  # all mass on (fragment 0, shift 0) = group 2
c, gr = cos_of(one, [True, True], u, [-1, -1, 0, -1, -1, -1, -1, -1, -1, -1])
print(f"(b) all mass on a slot matched to peak 0: cos {c:.4f} (expect 0.6000)")
ok &= abs(c - 0.6) < 1e-4
c, gr = cos_of(one, [True, True], u, [-1] * 10)
print(f"    nothing matched: cos {c:.4f}, grad finite {bool(torch.isfinite(gr).all())}, max |grad| {float(gr.abs().max()):.2e}")
ok &= abs(c) < 1e-6 and bool(torch.isfinite(gr).all())
two = [[big] * 5, [big] * 5]
two[0][2] = two[1][2] = 50.0  # half on group 2, half on group 7, both on peak 0
c, gr = cos_of(two, [True, True], u, [-1, -1, 0, -1, -1, -1, -1, 0, -1, -1])
print(f"    two groups on one peak: cos {c:.4f} (expect 0.6000; 0.8485 if they did not add up)")
ok &= abs(c - 0.6) < 1e-4
c, gr = cos_of(two, [False, False], u, [-1, -1, 0, -1, -1, -1, -1, 0, -1, -1])
print(f"    no valid fragments: cos {c:.4f}, grad finite {bool(torch.isfinite(gr).all())}")
ok &= abs(c) < 1e-6 and bool(torch.isfinite(gr).all())

# (c) one training batch, scoring
C = cache.Cache("/d/work")
FT = subform.FormulaTable("/k26")
hold = set(pd.read_csv("/d/pool/holdout.csv").pid.values)
fpnet.Cfg.max_per_struct, fpnet.Cfg.np_weight, fpnet.Cfg.neg_train_only = 0, 3, True
D = fpnet.TrainData(C, P, hold)
NEG = enc3.SameFormulaNegatives(P, FT, hold)
idx = D.epoch_index(np.random.default_rng(0))[:8]
b, st = g2.example_batch(C, D, G, NEG, idx, 4, np.random.default_rng(1))
net = g2.build_model()
bt = g2.to_torch(b, "cpu")
lg = net.head(net.encode(bt), bt["cond"])
cos = g2.cosine(lg, bt["valid"], bt["u"], bt["jou"], bt["obs"], bt["ex"])
cos.sum().backward()
fin = all(bool(torch.isfinite(p.grad).all()) for p in net.parameters() if p.grad is not None)
print(f"(c) batch of 8 x 5: S {len(b['nat'])}, F {b['bits'].shape[1]}, N {b['nmax']}, ceiling {st['ceiling']:.3f}, "
      f"zero {st['zero_ceiling']:.2f}, cos range {float(cos.min()):.3f}..{float(cos.max()):.3f}, grads finite {fin}")
ok &= fin and bool(torch.isfinite(cos).all())
Q = pd.read_csv("/d/pool/queries.csv")
Q["rows"] = Q.rows.map(lambda s: [int(x) for x in str(s).split(",")])
q = Q.iloc[0]
cand = P.window(P.mass[q.pid], 4000.0)
s = g2.G2Model(G=G, net=net.eval()).score(harness.query_specs(C, q.rows), cand)
print(f"    score: {len(cand)} candidates, finite {np.isfinite(s).sum()}, range {np.nanmin(s):.3f}..{np.nanmax(s):.3f}")
ok &= np.isfinite(s).sum() > 0

# (d) resume: 2 x 6 steps straight vs resumed from the save at step 6 (identical weights; one CPU thread, since the
#     multithreaded index_add / scatter_add sum in a varying order)
torch.set_num_threads(1)
g2.Cfg.batch, g2.Cfg.k_neg, g2.Cfg.total_steps, g2.Cfg.save_every, g2.Cfg.workers = 8, 4, 12, 6, 2
os.makedirs("/w/ga", exist_ok=True)
os.makedirs("/w/gb", exist_ok=True)
_save = g2.ckpt.save_state


def keep(path, *a, **k):
    _save(path, *a, **k)
    import shutil
    shutil.copy(path, path.replace(".pt", f"_{k['step']}.pt"))


g2.ckpt.save_state = keep
g2.train(C, D, G, NEG, "/w/ga/m.pt", log=lambda *a: None)
g2.train(C, D, G, NEG, "/w/gb/m.pt", log=lambda *a: None, resume="/w/ga/m_state_6.pt")
A = torch.load("/w/ga/m.pt", weights_only=False)
B = torch.load("/w/gb/m.pt", weights_only=False)
d = max(float((A["model"][k] - B["model"][k]).abs().max()) for k in A["model"])
print(f"(d) g2.train straight vs resumed at step 6: steps {A['step']} / {B['step']}, max |dw| {d:.3g}")
ok &= d == 0 and A["step"] == B["step"] == 12
print("g2 check:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)

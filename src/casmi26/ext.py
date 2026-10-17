"""M7 P3: external pretrained MS/MS simulators as candidate columns (user decision 2026-10-17: published research
weights only). ms-pred (coleygroup, MIT, commit 708148c): GLACIER trained on MassSpecGym ("gl") and ICEBERG 2.1
msg_all ("ice"). Everything below is fixed before any result (plan 1-3-deep-storm.md):
- K_TOP candidates of the reference order are scored (the rest of the window is NaN).
- Query spectra: the Searcher's spectra (finite neutral mass, first max_query_spectra), then only [M+H]+ / [M+Na]+
  with at least one collision energy (the host's collision_energy_ev list of that row). Instrument: "QTOF" when the
  row's instrument_type contains "tof", else "Orbitrap".
- One prediction per (candidate, adduct, CE value, instrument); a stepped spectrum (several CE values) is compared with
  the sum of its predictions (each normalised to sum 1, identical m/z merged, renormalised).
- Similarity: entropy similarity, greedy one-to-one peak matching within PPM (query peaks in m/z order, each takes the
  first unused predicted peak in its window), predicted spectra cut to the SPARSE_K most intense peaks. Candidate score
  = mean over the query spectra.
- Columns (per kind): value, z over the finite values (NaN -> 0), rank_norm (NaN -> bottom), and within the (molecule,
  neutral formula) group the rank and the gap to the best of the z value with unscored rows at NO_SCORE_Z (bottom).
The ms-pred stack (dgl, torch_scatter, ...) comes from the casmi26-mspred dataset (install_mspred)."""
import glob
import os
import subprocess
import sys

import numpy as np
from numba import njit

KINDS = ("gl", "ice")
FEATURES = {k: [k, f"{k}_z", f"{k}_rank", f"{k}_frank", f"{k}_fgap"] for k in KINDS}
POS_ADDUCTS = ("[M+H]+", "[M+Na]+")
SPARSE_K = 100
PPM = 20.0
K_TOP = 100
MAX_HEAVY = 100
NO_SCORE_Z = -10.0
FEAT_CACHE = 64
MANIFEST = "mspred_manifest.json"
# casmi26-mspred is flat (Kaggle datasets keep no folders): the ms-pred src as a tar.gz, checkpoints by name
_CKPT = {"gl": ["glacier_msg.best.ckpt"], "ice": ["iceberg_msg_all.gen.best.ckpt", "iceberg_msg_all.inten_contr.best.ckpt"]}
SRC_TAR = "ms_pred_src.tar.gz"


def mspred_root():
    from .paths import find
    return os.path.dirname(find(MANIFEST))


def install_mspred(root=None, log=print):
    """pip install the bundled wheels (no index, no deps) and put the ms-pred src on sys.path. -> root.
    Kaggle unpacks ms_pred_src.tar.gz into the folder ms_pred_src/ and drops '+' from file names (dgl-2.5.0cu124...),
    so the wheels are copied to /tmp under their real names (local version '+cu124', '+pt210cu128') first."""
    import re
    import shutil
    import tarfile
    root = root or mspred_root()
    try:
        import dgl  # noqa: F401
        import LinSATNet  # noqa: F401
    except ImportError:
        tmp = "/tmp/mspred_wheels"
        os.makedirs(tmp, exist_ok=True)
        wh = []
        tag = f"-cp{sys.version_info.major}{sys.version_info.minor}-"  # Kaggle runs Python 3.13, the docker image 3.12
        for w in sorted(glob.glob(os.path.join(root, "*.whl"))):
            if tag not in os.path.basename(w) and "-none-any" not in os.path.basename(w):
                continue
            dst = os.path.join(tmp, re.sub(r"-(\d+(?:\.\d+)*)((?:pt\d+)?cu\d+)-", r"-\1+\2-", os.path.basename(w)))
            shutil.copy(w, dst)
            wh.append(dst)
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps"] + wh,
                           capture_output=True, text=True)
        log(f"ms-pred wheels: {[os.path.basename(w) for w in wh]}, pip exit {r.returncode} {r.stderr[-300:]}")
        assert r.returncode == 0, r.stderr[-2000:]
    src = os.path.join(root, "ms_pred_src")
    if not os.path.exists(os.path.join(src, "ms_pred", "__init__.py")):
        src = "/tmp/ms_pred_src"
        if not os.path.exists(os.path.join(src, "ms_pred", "__init__.py")):
            with tarfile.open(os.path.join(root, SRC_TAR)) as t:
                t.extractall(src, filter="data")
    if src not in sys.path:
        sys.path.insert(0, src)
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")  # the ICEBERG ckpt pickles a PosixPath
    os.environ.setdefault("DGLBACKEND", "pytorch")
    return root


def instrument_of(s):
    return "QTOF" if "tof" in str(s).lower() else "Orbitrap"


def ce_values(v):
    """Collision energies (eV) of one row: finite, positive (enveda negative mode stores -20 as 20 already), rounded."""
    a = np.asarray(v if v is not None else [], np.float64).ravel()
    a = np.abs(a[np.isfinite(a)])
    return tuple(sorted({round(float(x), 1) for x in a if x > 0}))


class CETable:
    """Per cache row (== train.parquet row): the collision_energy_ev list and the instrument class."""

    def __init__(self, train_path, n_rows=None):
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(train_path)
        t = pf.read(columns=["collision_energy_ev", "instrument_type"])
        col = t.column("collision_energy_ev").combine_chunks()
        off = np.asarray(col.offsets, np.int64)
        vals = np.asarray(col.values.to_numpy(zero_copy_only=False), np.float64)
        valid = np.asarray(col.is_valid(), bool)
        self.off = off[:-1], off[1:]
        self.vals = vals
        self.valid = valid
        inst = t.column("instrument_type").to_pandas()
        self.qtof = inst.map(lambda s: instrument_of(s) == "QTOF").values.astype(bool)
        assert n_rows is None or len(self.qtof) == n_rows, (len(self.qtof), n_rows)

    def ce(self, row):
        if not self.valid[row]:
            return ()
        a, b = self.off[0][row], self.off[1][row]
        return ce_values(self.vals[a:b])

    def instrument(self, row):
        return "QTOF" if self.qtof[row] else "Orbitrap"


def _keep(spec, ces):
    return spec["mode"] == 1 and str(spec["adduct"]).replace(" ", "") in POS_ADDUCTS and len(ces) > 0


def query_specs_ce(C, rows, cet, max_spectra=16):
    """The Searcher's spectra of a harness query (same filter and cut), restricted to positive [M+H]+ / [M+Na]+ with CE;
    each dict gets ce_list and instrument."""
    from . import chem, harness
    sp = [(r, s) for r, s in zip(rows, harness.query_specs(C, rows))
          if np.isfinite(chem.neutral_mass(s["prec"], s["adduct"]))][:max_spectra]
    out = []
    for r, s in sp:
        ces = cet.ce(r)
        if _keep(s, ces):
            out.append(dict(s, ce_list=ces, instrument=cet.instrument(r)))
    return out


def test_specs_ce(g, max_spectra=16):
    """Same as query_specs_ce for one molecule of the competition test table."""
    from . import chem, harness
    sp = [(r, s) for r, s in zip(g.itertuples(), harness.test_specs(g))
          if np.isfinite(chem.neutral_mass(s["prec"], s["adduct"]))][:max_spectra]
    out = []
    for r, s in sp:
        ces = ce_values(getattr(r, "collision_energy_ev", None))
        if _keep(s, ces):
            out.append(dict(s, ce_list=ces, instrument=instrument_of(getattr(r, "instrument_type", ""))))
    return out


@njit(cache=True)
def _entropy_greedy(qm, qp, pm, pp, ppm):
    """qp, pp: probabilities (sum 1); pm sorted ascending. Entropy similarity with greedy one-to-one matching."""
    n2 = len(pm)
    used = np.zeros(n2, np.bool_)
    hm = 0.0
    for i in range(len(qm)):
        a = qp[i]
        tol = qm[i] * ppm * 1e-6
        lo = qm[i] - tol
        k = np.searchsorted(pm, lo)
        hit = -1
        while k < n2 and pm[k] <= qm[i] + tol:
            if not used[k]:
                hit = k
                break
            k += 1
        if hit >= 0:
            used[hit] = True
            m = 0.5 * (a + pp[hit])
        else:
            m = 0.5 * a
        if m > 0:
            hm -= m * np.log(m)
    for k in range(n2):
        if not used[k]:
            m = 0.5 * pp[k]
            if m > 0:
                hm -= m * np.log(m)
    hp = 0.0
    for i in range(len(qp)):
        if qp[i] > 0:
            hp -= qp[i] * np.log(qp[i])
    hq = 0.0
    for k in range(n2):
        if pp[k] > 0:
            hq -= pp[k] * np.log(pp[k])
    return 1.0 - (2.0 * hm - hp - hq) / np.log(4.0)


def entropy_sim(q_mz, q_it, p_mz, p_it, ppm=PPM):
    """Entropy similarity of a query spectrum and a predicted one (NaN when either is empty)."""
    q_mz, q_it = np.asarray(q_mz, np.float64), np.asarray(q_it, np.float64)
    p_mz, p_it = np.asarray(p_mz, np.float64), np.asarray(p_it, np.float64)
    if len(q_mz) == 0 or len(p_mz) == 0 or q_it.sum() <= 0 or p_it.sum() <= 0:
        return np.nan
    o = np.argsort(p_mz, kind="mergesort")
    return float(_entropy_greedy(q_mz, q_it / q_it.sum(), p_mz[o], p_it[o] / p_it.sum(), ppm))


def merge_predictions(preds):
    """Sum of predicted spectra (each normalised to sum 1; identical m/z, to 1e-4, merged). None entries skipped."""
    preds = [p for p in preds if p is not None and len(p[0]) and p[1].sum() > 0]
    if not preds:
        return None
    if len(preds) == 1:
        return preds[0]
    mz = np.concatenate([p[0] for p in preds])
    it = np.concatenate([p[1] / p[1].sum() for p in preds])
    key = np.round(mz, 4)
    u, inv = np.unique(key, return_inverse=True)
    s = np.bincount(inv, weights=it)
    m = np.bincount(inv, weights=mz) / np.bincount(inv)
    return m, s / s.sum()


def canonical_input(smiles, kind):
    """ms-pred's input filter (valid elements, no '.', <= MAX_HEAVY heavy atoms) -> canonical SMILES or None."""
    from rdkit import Chem
    from ms_pred.common import chem_utils
    if not isinstance(smiles, str) or "." in smiles:
        return None
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return None
        ok = set(chem_utils.VALID_ATOM_NUM)
        if any(a.GetAtomicNum() not in ok for a in m.GetAtoms()):
            return None
        m = Chem.RemoveHs(m)
        if m.GetNumAtoms() > MAX_HEAVY:
            return None
        return Chem.MolToSmiles(m)
    except Exception:
        return None


class ExtModel:
    """predict(entries) with entries = [(canonical SMILES, adduct, CE, instrument, precursor m/z)] -> list of
    (mz, intensity) arrays (the SPARSE_K most intense peaks) or None for a failed entry."""

    def __init__(self, kind, root=None, device=None, batch_size=64, workers=0, log=print):
        import hashlib
        import torch
        assert kind in KINDS, kind
        self.kind, self.batch_size, self.workers, self.log = kind, batch_size, workers, log
        self.root = install_mspred(root, log)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpts = [os.path.join(self.root, p) for p in _CKPT[kind]]
        self.sha = {os.path.basename(p): hashlib.sha256(open(p, "rb").read()).hexdigest()[:16] for p in self.ckpts}
        if kind == "gl":
            from ms_pred.glacier import joint_model, dataset as gd
            self.model = joint_model.JointModel.load_from_checkpoint(self.ckpts[0], map_location="cpu")
            self.model.eval()
            self.model.freeze()
            self.tp = gd.TreeProcessor(pe_embed_k=self.model.pe_embed_k, root_encode="graphormer",
                                       embed_elem_group=self.model.embed_elem_group,
                                       multi_hop_max_dist=self.model.multi_hop_max_dist)
        else:
            from ms_pred.iceberg import gen_model, inten_model, joint_model
            gen = gen_model.FragGNN.load_from_checkpoint(self.ckpts[0], map_location="cpu")
            inten = inten_model.IntenGNN.load_from_checkpoint(self.ckpts[1], map_location="cpu")
            self.model = joint_model.JointModel(gen_model_obj=gen, inten_model_obj=inten)
            self.model.eval()
            self.model.freeze()
        self.model = self.model.to(self.device)
        self.n_fail = 0

    # ---------------------------------------------------------------- GLACIER
    def _gl_items(self, entries):
        from ms_pred.glacier import predict_smis_joint as pj
        tup = [(e[0], str(i), float(e[2]), e[1], e[3], "") for i, e in enumerate(entries)]
        # entries are sorted by (heavy atoms, SMILES), so a small featurisation cache suffices (ms-pred's default 8192
        # Graphormer inputs ran a 20 GB container out of memory, 2026-10-17)
        return pj.GraphormerFeatDataset(tup, tree_processor=self.tp, sort_for_batching=True, feat_cache_size=FEAT_CACHE)

    def _gl_forward(self, items, out):
        import torch
        from ms_pred.glacier import predict_smis_joint as pj
        batch = pj._batch_to_device(pj.collate_predict(items), self.device)
        with torch.inference_mode():
            res = self.model.predict_inten_frag_batch_sparse(batch, SPARSE_K)
        for j, name in enumerate(batch["pred_names"]):
            c = int(res["counts"][j])
            out[int(name)] = (np.asarray(res["masses"][j, :c], np.float64), np.asarray(res["intens"][j, :c], np.float64))

    def _predict_gl(self, entries):
        out = [None] * len(entries)
        ds = self._gl_items(entries)
        n, bs = len(ds), self.batch_size
        if self.workers > 0:
            from torch.utils.data import DataLoader
            from ms_pred.glacier import predict_smis_joint as pj
            try:
                dl = DataLoader(ds, batch_size=bs, num_workers=self.workers, collate_fn=pj.collate_predict,
                                shuffle=False, multiprocessing_context="fork", prefetch_factor=4)
                import torch
                for batch in dl:
                    batch = pj._batch_to_device(batch, self.device)
                    with torch.inference_mode():
                        res = self.model.predict_inten_frag_batch_sparse(batch, SPARSE_K)
                    for j, name in enumerate(batch["pred_names"]):
                        c = int(res["counts"][j])
                        out[int(name)] = (np.asarray(res["masses"][j, :c], np.float64),
                                          np.asarray(res["intens"][j, :c], np.float64))
                return out
            except Exception as e:  # fall back to the isolating loop below
                self.log(f"  gl DataLoader failed ({type(e).__name__}: {str(e)[:200]}); serial fallback")
                out = [None] * len(entries)
        for a in range(0, n, bs):
            idx = range(a, min(a + bs, n))
            try:
                self._gl_forward([ds[i] for i in idx], out)
            except Exception:
                for i in idx:  # isolate the failing entry
                    try:
                        self._gl_forward([ds[i]], out)
                    except Exception:
                        self.n_fail += 1
        return out

    # ---------------------------------------------------------------- ICEBERG
    def _ice_forward(self, ents):
        import torch
        with torch.no_grad():
            o = self.model.predict_mol([e[0] for e in ents], collision_eng=[float(e[2]) for e in ents],
                                       precursor_mz=[float(e[4]) for e in ents], adduct=[e[1] for e in ents],
                                       threshold=0.0, device=self.device, max_nodes=100,
                                       instrument=[e[3] for e in ents], binned_out=False, adduct_shift=False,
                                       canonical_root_smi=True)
        res = []
        for sp in o["spec"]:
            sp = sp.cpu().numpy()
            top = np.argsort(sp[:, 1], kind="mergesort")[::-1][:SPARSE_K]
            res.append((sp[top, 0].astype(np.float64), sp[top, 1].astype(np.float64)))
        return res

    def _predict_ice(self, entries):
        out = [None] * len(entries)
        bs = self.batch_size
        order = sorted(range(len(entries)), key=lambda i: (len(entries[i][0]), entries[i][0]))
        for a in range(0, len(order), bs):
            idx = order[a:a + bs]
            try:
                for i, r in zip(idx, self._ice_forward([entries[i] for i in idx])):
                    out[i] = r
            except Exception:
                for i in idx:
                    try:
                        out[i] = self._ice_forward([entries[i]])[0]
                    except Exception:
                        self.n_fail += 1
        return out

    def predict(self, entries):
        if not entries:
            return []
        return self._predict_gl(entries) if self.kind == "gl" else self._predict_ice(entries)


def score_candidates(model, specs, smiles, pred_cache=None):
    """Scores of candidate SMILES against one molecule's spectra (query_specs_ce / test_specs_ce): mean over the
    spectra of the entropy similarity with the (CE-merged) prediction; NaN when nothing could be predicted.
    pred_cache: dict key -> prediction shared across calls (keys: (canonical SMILES, adduct, CE, instrument, prec))."""
    cache = {} if pred_cache is None else pred_cache
    can = [canonical_input(s, model.kind) for s in smiles]
    need = []
    for c in can:
        if c is None:
            continue
        for s in specs:
            for ce in s["ce_list"]:
                k = entry_key(c, s, ce, model.kind)
                if k not in cache:
                    cache[k] = None
                    need.append(k)
    for k, p in zip(need, model.predict(need)):
        cache[k] = p
    return candidate_scores(can, specs, cache, model.kind)


def entry_key(can, spec, ce, kind):
    """Prediction key; the precursor m/z only matters for ICEBERG (GLACIER ignores it)."""
    prec = round(float(spec["prec"]), 3) if kind == "ice" else 0.0
    return (can, str(spec["adduct"]).replace(" ", ""), float(ce), spec["instrument"], prec)


def candidate_scores(can, specs, cache, kind):
    out = np.full(len(can), np.nan)
    for i, c in enumerate(can):
        if c is None:
            continue
        sims = []
        for s in specs:
            p = merge_predictions([cache.get(entry_key(c, s, ce, kind)) for ce in s["ce_list"]])
            if p is not None:
                v = entropy_sim(s["mz"], s["it"], p[0], p[1])
                if np.isfinite(v):
                    sims.append(v)
        if sims:
            out[i] = float(np.mean(sims))
    return out


def columns(v, fid):
    """The FEATURES block of one molecule's window: v = score per window candidate (NaN = not scored),
    fid = neutral-formula id per candidate -> (n, 5) float32."""
    from .search import _zs_finite, rank_norm
    v = np.asarray(v, np.float64)
    z = _zs_finite(v)
    rk = rank_norm(np.nan_to_num(v, nan=-1.0))
    zg = np.where(np.isfinite(v), z, NO_SCORE_Z).astype(np.float64)
    fid = np.asarray(fid, np.int64)
    _, g = np.unique(fid, return_inverse=True)
    gmax = np.full(g.max() + 1 if len(g) else 0, -np.inf)
    np.maximum.at(gmax, g, zg)
    frank = np.ones(len(v))
    if len(v):  # rank inside the formula group, 1 = best, ties share the best rank (pandas rank method="min")
        import pandas as pd
        frank = pd.Series(zg).groupby(g).rank(method="min", ascending=False).values
    return np.c_[v, z, rk, frank, zg - gmax[g]].astype(np.float32)

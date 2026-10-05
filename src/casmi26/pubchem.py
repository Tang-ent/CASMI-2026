"""PubChem bulk files (FTP Compound/Extras) and PubChemLite: download, streaming readers, the cheap formula filter
and the own natural-product-likeness classifier (COCONUT-only structures vs random PubChem).

Per-compound arrays are dense and indexed by CID (PubChem CIDs run from 1 to roughly 1.8e8).
Status of a CID after the formula filter (`formula_info`, `scan_mass`), checked in this order:
  0 ok, 1 several components, 2 formula does not parse / untabulated element, 3 element outside ALLOWED or no
  carbon, 4 formula mass != PubChem monoisotopic mass (isotope label), 5 mass outside [MASS_LO, MASS_HI], 255 absent.
PubChem writes the formula of a salt as one formula (e.g. C6H8ClN for aniline hydrochloride), so several components
are detected from the formula layer of the standard InChI ('C6H7N.ClH') in `scan_ik14`.
"""
import os
import re
import time

import numpy as np

from . import chem

BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras"
PCL_RECORD = "https://zenodo.org/api/records/22953182"  # PubChemLite for Exposomics, 2026-09-25 (CC BY 4.0)
ALLOWED = {"C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I"}
BIT = {"N": 1, "S": 2, "P": 4, "F": 8, "Cl": 8, "Br": 8, "I": 8}  # + 16 = charged formula
CHARGED = 16
MASS_LO, MASS_HI = 150.0, 1200.0  # official test range 157-1,159 Da with a margin
ISO_TOL = 2e-3  # Da
STATUS = {0: "ok", 1: "multi_component", 2: "unparsable", 3: "elements", 4: "isotope", 5: "mass_range", 255: "absent"}
_TOK = re.compile(r"([A-Z][a-z]?)(\d*)")


# ---------------------------------------------------------------- files
def fetch(url, path, retries=8, log=print):
    """Download url to path, resuming with HTTP Range after a dropped connection. Returns path."""
    import requests
    h = requests.head(url, timeout=60, allow_redirects=True)
    size = int(h.headers.get("Content-Length", -1)) if h.status_code == 200 else -1
    t0 = time.time()
    for attempt in range(retries):
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if size > 0 and have == size:
            break
        try:
            h = {"Range": f"bytes={have}-"} if have else {}
            with requests.get(url, headers=h, stream=True, timeout=120) as r:
                r.raise_for_status()
                mode = "ab" if have and r.status_code == 206 else "wb"
                with open(path, mode) as fh:
                    for chunk in r.iter_content(8 << 20):
                        fh.write(chunk)
            if size < 0:
                break
        except Exception as e:  # noqa: BLE001 (network errors of any kind -> resume)
            log(f"  {os.path.basename(path)}: retry {attempt + 1} after {e!r}")
            time.sleep(15)
    got = os.path.getsize(path)
    if size > 0 and got != size:
        raise RuntimeError(f"{url}: got {got:,} of {size:,} bytes")
    log(f"  downloaded {os.path.basename(path)} {got / 1e9:.2f} GB in {time.time() - t0:.0f}s")
    return path


def source(name, tmp, url=None, log=print):
    """A local copy under /kaggle/input (smoke tests) if there is one, else download to tmp."""
    from .paths import find
    try:
        p = find(name, roots=["/kaggle/input"])
        log(f"  {name}: local {p}")
        return p, False
    except FileNotFoundError:
        return fetch(url or f"{BASE_URL}/{name}", os.path.join(tmp, name), log=log), True


def pubchemlite_url():
    import requests
    files = requests.get(PCL_RECORD, timeout=60).json()["files"]
    f = next(f for f in files if f["key"].lower().endswith(".csv"))
    return f["key"], f["links"]["self"]


def batches(path, names, keep, types=None, block=64 << 20):
    """Record batches of a gzipped tab-separated file without header (rows with a wrong field count are skipped)."""
    import pyarrow.csv as pv
    ro = pv.ReadOptions(column_names=names, block_size=block)
    po = pv.ParseOptions(delimiter="\t", quote_char=False, invalid_row_handler=lambda row: "skip")
    co = pv.ConvertOptions(include_columns=keep, column_types=types or {})
    with pv.open_csv(path, read_options=ro, parse_options=po, convert_options=co) as rd:
        for b in rd:
            yield b


def _cids(b):
    import pyarrow.compute as pc
    return pc.fill_null(b.column("cid"), 0).to_numpy().astype(np.int64)


# ---------------------------------------------------------------- formula filter
def formula_info(f):
    """(mass, bits, status) of a PubChem molecular formula (status 0-3 as in the module docstring)."""
    if not isinstance(f, str):
        return np.nan, 0, 2
    if "." in f:
        return np.nan, 0, 1
    m = chem._FORMULA.match(f.strip())
    if m is None:
        return np.nan, 0, 2
    els = {e for e, _ in _TOK.findall(m.group(1))}
    mass = chem.formula_mass(f)
    if not np.isfinite(mass):
        return np.nan, 0, 2
    bits = CHARGED if m.group(2) else 0
    for e in els:
        bits |= BIT.get(e, 0)
    return mass, bits, (3 if (els - ALLOWED or "C" not in els) else 0)


def scan_mass(path, log=print):
    """CID-Mass.gz -> dict(mass float64, bits uint8, status uint8), each indexed by CID, plus row count."""
    import pyarrow as pa
    import pyarrow.compute as pc
    t0 = time.time()
    parts = []
    n_rows = 0
    names = ["cid", "formula", "mono", "exact"]
    for b in batches(path, names, ["cid", "formula", "mono"], {"cid": pa.int64(), "mono": pa.float64()}):
        cid = _cids(b)
        enc = pc.dictionary_encode(b.column("formula"))
        idx = pc.fill_null(enc.indices, -1).to_numpy().astype(np.int64)
        info = [formula_info(f) for f in enc.dictionary.to_pylist()] + [(np.nan, 0, 2)]
        fm = np.array([x[0] for x in info])[idx]
        bits = np.array([x[1] for x in info], np.uint8)[idx]
        st = np.array([x[2] for x in info], np.uint8)[idx]
        mono = pc.fill_null(b.column("mono"), np.nan).to_numpy()
        st[(st == 0) & ~(np.abs(fm - mono) <= ISO_TOL)] = 4
        st[(st == 0) & ~((mono >= MASS_LO) & (mono <= MASS_HI))] = 5
        parts.append((cid, mono, bits, st))
        n_rows += len(cid)
        if len(parts) % 20 == 0:
            log(f"  CID-Mass: {n_rows:,} rows ({time.time() - t0:.0f}s)")
    cid = np.concatenate([p[0] for p in parts])
    n = int(cid.max()) + 1
    out = dict(mass=np.full(n, np.nan), bits=np.zeros(n, np.uint8), status=np.full(n, 255, np.uint8))
    for k, j in [("mass", 1), ("bits", 2), ("status", 3)]:
        out[k][cid] = np.concatenate([p[j] for p in parts])
    log(f"  CID-Mass: {n_rows:,} rows, max CID {n - 1:,} ({time.time() - t0:.0f}s)")
    return out, n_rows


def scan_ik14(path, n, log=print):
    """CID-InChI-Key.gz -> (standard InChIKey first block per CID (S14, b'' = absent),
    several components per CID (bool, from the InChI formula layer))."""
    import pyarrow as pa
    import pyarrow.compute as pc
    t0 = time.time()
    ik = np.zeros(n, "S14")
    multi = np.zeros(n, bool)
    rows = 0
    for b in batches(path, ["cid", "inchi", "key"], ["cid", "inchi", "key"], {"cid": pa.int64()}):
        cid = _cids(b)
        k = pc.utf8_slice_codeunits(pc.fill_null(b.column("key"), ""), 0, 14)
        k = k.to_numpy(zero_copy_only=False).astype("S14")
        f = pc.struct_field(pc.extract_regex(pc.fill_null(b.column("inchi"), ""), r"^InChI=1S?/(?P<f>[^/]*)"), "f")
        mc = pc.fill_null(pc.match_substring(f, "."), False).to_numpy(zero_copy_only=False)
        ok = (cid > 0) & (cid < n)
        ik[cid[ok]] = k[ok]
        multi[cid[ok]] = mc[ok]
        rows += len(cid)
        if rows // 10_000_000 != (rows - len(cid)) // 10_000_000:
            log(f"  CID-InChI-Key: {rows:,} rows ({time.time() - t0:.0f}s)")
    log(f"  CID-InChI-Key: {rows:,} rows, {(ik != b'').sum():,} CIDs with a key, {multi.sum():,} with several "
        f"components ({time.time() - t0:.0f}s)")
    return ik, multi


def count_by_cid(path, n, ncols, tag, log=print):
    """Number of lines per CID of a 'CID <tab> ...' file (int32, indexed by CID)."""
    import pyarrow as pa
    t0 = time.time()
    counts = np.zeros(n, np.int32)
    rows = 0
    names = ["cid"] + [f"c{i}" for i in range(1, ncols)]
    for b in batches(path, names, ["cid"], {"cid": pa.int64()}):
        cid = _cids(b)
        cid = cid[(cid > 0) & (cid < n)]
        u, k = np.unique(cid, return_counts=True)
        counts[u] += k.astype(np.int32)
        rows += len(cid)
    log(f"  {tag}: {rows:,} lines, {(counts > 0).sum():,} CIDs ({time.time() - t0:.0f}s)")
    return counts


def scan_smiles(path, want, log=print):
    """CID-SMILES.gz -> (cids, smiles) for the CIDs with want[cid] (bool array indexed by CID)."""
    import pyarrow as pa
    import pyarrow.compute as pc
    t0 = time.time()
    n = len(want)
    cs, ss = [], []
    for b in batches(path, ["cid", "smiles"], ["cid", "smiles"], {"cid": pa.int64()}):
        cid = _cids(b)
        ok = (cid > 0) & (cid < n)
        ok[ok] = want[cid[ok]]
        if ok.any():
            cs.append(cid[ok])
            ss.append(pc.filter(b.column("smiles"), pa.array(ok)).to_numpy(zero_copy_only=False))
    cids = np.concatenate(cs) if cs else np.zeros(0, np.int64)
    smiles = np.concatenate(ss) if ss else np.zeros(0, object)
    log(f"  CID-SMILES: {len(cids):,} of {int(want.sum()):,} wanted CIDs ({time.time() - t0:.0f}s)")
    return cids, smiles


def read_pubchemlite(path):
    """PubChemLite CSV -> DataFrame(ik14, pcl_anno, pcl_pubmed, pcl_patent), one row per InChIKey first block."""
    import pandas as pd
    head = pd.read_csv(path, nrows=0).columns
    cols = [c for c in ["FirstBlock", "InChIKey", "AnnoTypeCount", "PubMed_Count", "Patent_Count"] if c in head]
    d = pd.read_csv(path, usecols=cols, low_memory=False)
    ik = d["FirstBlock"] if "FirstBlock" in d else d["InChIKey"].str[:14]
    out = pd.DataFrame({"ik14": ik.astype(str).str[:14],
                        "pcl_anno": d.get("AnnoTypeCount", pd.Series(0, index=d.index)).fillna(0).astype(int),
                        "pcl_pubmed": d.get("PubMed_Count", pd.Series(0, index=d.index)).fillna(0).astype(int),
                        "pcl_patent": d.get("Patent_Count", pd.Series(0, index=d.index)).fillna(0).astype(int)})
    return out.sort_values("pcl_anno", ascending=False).drop_duplicates("ik14")


# ---------------------------------------------------------------- RDKit helpers (multiprocessing workers)
def std_ik14(smiles):
    """Standard InChIKey first block of a SMILES as written (b'' on failure)."""
    r = chem._rdkit()
    try:
        m = r["Chem"].MolFromSmiles(smiles) if isinstance(smiles, str) else None
        k = r["Chem"].MolToInchiKey(m)[:14] if m is not None else ""
    except Exception:  # noqa: BLE001
        k = ""
    return k.encode()


NP_FP_BITS = 1024
NP_DESC = ["heavy", "fsp3", "rings", "arom_rings", "stereo", "n_C", "n_N", "n_O", "n_S", "n_P", "n_hal",
           "o_c", "n_c", "hbd", "hba", "rotb", "charge"]
_NP = {}


def _np_gen():
    if not _NP:
        from rdkit.Chem import rdFingerprintGenerator
        _NP["gen"] = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=NP_FP_BITS)
    return _NP["gen"]


def np_features(smiles):
    """Features of the natural-product classifier: Morgan r=2 count fingerprint (NP_FP_BITS, clipped at 255) +
    NP_DESC. Stereo centres are counted as potential centres, so the value does not depend on whether the SMILES
    carries stereo. Returned as (uint8 fingerprint, float32 descriptors) to keep the transfer between processes
    small; None when RDKit cannot parse."""
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors as D
    try:
        m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        if m is None:
            return None
        fp = np.minimum(_np_gen().GetCountFingerprintAsNumPy(m), 255).astype(np.uint8)
        el = [a.GetSymbol() for a in m.GetAtoms()]
        cnt = {e: el.count(e) for e in ("C", "N", "O", "S", "P")}
        nc = max(cnt["C"], 1)
        stereo = len(Chem.FindMolChiralCenters(m, includeUnassigned=True, useLegacyImplementation=False))
        d = [m.GetNumHeavyAtoms(), D.CalcFractionCSP3(m), D.CalcNumRings(m), D.CalcNumAromaticRings(m), stereo,
             cnt["C"], cnt["N"], cnt["O"], cnt["S"], cnt["P"], sum(e in ("F", "Cl", "Br", "I") for e in el),
             cnt["O"] / nc, cnt["N"] / nc, D.CalcNumHBD(m), D.CalcNumHBA(m), D.CalcNumRotatableBonds(m),
             Chem.GetFormalCharge(m)]
        return fp, np.array(d, np.float32)
    except Exception:  # noqa: BLE001
        return None


def map_many(fn, items, workers, chunksize=500):
    from multiprocessing import Pool
    with Pool(workers) as p:
        return p.map(fn, items, chunksize=chunksize)


def np_matrix(smiles, workers):
    """(X float32 [n_ok, d], ok bool [n]) for a list of SMILES."""
    feats = map_many(np_features, list(smiles), workers)
    ok = np.array([f is not None for f in feats], bool)
    d = NP_FP_BITS + len(NP_DESC)
    X = np.zeros((int(ok.sum()), d), np.float32)
    for i, f in enumerate(f for f in feats if f is not None):
        X[i, :NP_FP_BITS] = f[0]
        X[i, NP_FP_BITS:] = f[1]
    return X, ok


NP_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=50, feature_fraction=0.5,
                 bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=4)
NP_ROUNDS = 400


def np_fit(X, y, seed=0):
    import lightgbm as lgb
    return lgb.train(dict(NP_PARAMS, seed=seed), lgb.Dataset(X, y, free_raw_data=True), num_boost_round=NP_ROUNDS)


def np_score(models, smiles, workers, chunk=200_000):
    """Mean of the models' probabilities per SMILES (NaN when RDKit cannot parse), computed in chunks."""
    out = np.full(len(smiles), np.nan, np.float32)
    smiles = list(smiles)
    for a in range(0, len(smiles), chunk):
        X, ok = np_matrix(smiles[a:a + chunk], workers)
        if ok.any():
            out[a:a + chunk][ok] = np.mean([m.predict(X, num_threads=workers) for m in models], axis=0)
    return out


def auc(y, s):
    """ROC AUC by ranks (ties averaged)."""
    from scipy.stats import rankdata
    y = np.asarray(y, bool)
    r = rankdata(s)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))

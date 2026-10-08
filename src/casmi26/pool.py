"""Candidate pool = train structures U COCONUT, one entry per tautomer-canonical InChIKey14.

Files written to `out_dir`:
  pool.parquet    key, smiles (canonical; COCONUT's form when the structure is in COCONUT), formula, mass,
                  in_train, in_coconut                                       sorted by mass, pid = row index
  pool_fp.npy     packed Morgan r=2 fingerprints of the canonical SMILES (n_pool, FP_BITS // 8) uint8
  sid2pid.npy     train structure id (cache sid) -> pid
  pool_meta.json  source release, counts, check results
in_train / in_coconut are for reporting only; they must never become ranking features (provenance leak).

Mass fix (pool_mass.npy, written by neutral_masses in a separate kernel): COCONUT stores some molecules as protonated
cations ([nH+], [oH+], phosphocholine N+ with P-OH). InChIKey14 ignores protonation, so such an entry shares its key
with the neutral train structure, but its formula mass is off by one proton and the truth falls outside the +-10 ppm
window. The corrected mass is the train formula mass for structures in train, else the RDKit-uncharged formula mass.
pids, fingerprints and the holdout are unchanged; windows are searched through `order` (pids sorted by the
corrected mass).

PubChem part (build_pubchem_part, kernel k05p; M3): spectrum-less PubChem structures appended after the pool, so every
existing pid (queries, holdout, sid2pid, pool_mass, k07 features, FPNet holdout) stays valid. Its own file names do not
collide with the pool's (paths.find picks the shortest match):
  pc_pool.parquet key, smiles, formula, mass (raw formula mass), cid, level, n_pmid, n_syn, n_pat, pcl_anno, np
  pc_fp.npy       packed fingerprints of the canonical SMILES
  pc_mass.npy     corrected (uncharged) masses
  pc_meta.json    counts and the level definitions
level = the smallest PubChem filter level that contains the structure (1 = S': n_pmid >= 1); the base pool has
level 0. Pool.set_level(k) restricts windows to level <= k. cid / annotation counts / np are for reporting only, never
ranking features (harness truths are library compounds, so annotations would leak).
"""
import io
import json
import os
import time
import zipfile

import numpy as np
import pandas as pd

from . import chem

COCONUT_URL = "https://coconut.s3.uni-jena.de/prod/downloads/{ym}/coconut_csv-{m}-{y}.zip"
SMILES_COLUMNS = ["canonical_smiles", "smiles", "SMILES", "Smiles"]


def download_coconut(out_path, releases=("2026-10", "2026-09", "2026-08")):
    """First available monthly CSV release -> (path, release)."""
    import requests
    for ym in releases:
        y, m = ym.split("-")
        url = COCONUT_URL.format(ym=ym, m=m, y=y)
        r = requests.get(url, timeout=600)
        print(url, r.status_code, f"{len(r.content) / 1e6:.1f} MB", flush=True)
        if r.status_code == 200 and len(r.content) > 1e6:
            open(out_path, "wb").write(r.content)
            return out_path, ym
    raise RuntimeError("no COCONUT release could be downloaded")


def read_coconut(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        df = pd.read_csv(io.BytesIO(z.read(name)), low_memory=False)
    print("COCONUT csv", name, df.shape, "columns:", list(df.columns)[:40], flush=True)
    col = next(c for c in SMILES_COLUMNS if c in df.columns)
    return df[col].dropna().astype(str).drop_duplicates().tolist(), col


def _canon_many(smiles, workers, log, tag):
    from multiprocessing import Pool
    t0 = time.time()
    out = []
    with Pool(workers) as p:
        for i, r in enumerate(p.imap(chem.canonical, smiles, chunksize=500)):
            out.append(r)
            if (i + 1) % 100_000 == 0:
                log(f"  {tag}: {i + 1:,}/{len(smiles):,} ({time.time() - t0:.0f}s)")
    log(f"  {tag}: {len(smiles):,} done, failed {sum(r is None for r in out):,} ({time.time() - t0:.0f}s)")
    return out


def _tanimoto_packed(a, b):
    a = np.unpackbits(a, axis=1).astype(bool)
    b = np.unpackbits(b, axis=1).astype(bool)
    inter = (a & b).sum(1)
    union = (a | b).sum(1)
    return np.where(union > 0, inter / np.maximum(union, 1), 1.0)


def build(cache, coconut_zip, out_dir, release="", workers=4, max_coconut=None, log=print):
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    co_smiles, co_col = read_coconut(coconut_zip)
    if max_coconut:
        co_smiles = co_smiles[:max_coconut]
    tr = cache.struct
    tr_c = _canon_many(tr["smiles"].tolist(), workers, log, "train")
    co_c = _canon_many(co_smiles, workers, log, "coconut")

    def frame(cs, src):
        ok = [i for i, c in enumerate(cs) if c is not None]
        return pd.DataFrame({"key": [cs[i]["key"] for i in ok], "smiles": [cs[i]["smiles"] for i in ok],
                             "formula": [cs[i]["formula"] for i in ok], "src": src, "idx": ok,
                             "fp": [cs[i]["fp"] for i in ok], "fp_raw": [cs[i]["fp_raw"] for i in ok]})

    ft = frame(tr_c, "train")
    fc = frame(co_c, "coconut")
    # 正規化に失敗した train 構造は、元の inchikey14 / SMILES / 分子式のまま入れる
    bad = [i for i, c in enumerate(tr_c) if c is None]
    if bad:
        fb = pd.DataFrame({"key": tr["ik14"].values[bad], "smiles": tr["smiles"].values[bad],
                           "formula": tr["formula"].values[bad], "src": "train", "idx": bad,
                           "fp": [chem.morgan_packed(s) for s in tr["smiles"].values[bad]]})
        fb["fp_raw"] = fb["fp"]
        ft = pd.concat([ft, fb], ignore_index=True)

    # check: fingerprints of structures present in both sources, before / after canonicalisation
    shared = ft.drop_duplicates("key").merge(fc.drop_duplicates("key"), on="key", suffixes=("_t", "_c"))
    check = {}
    if len(shared):
        before = _tanimoto_packed(np.stack(shared.fp_raw_t.values), np.stack(shared.fp_raw_c.values))
        after = _tanimoto_packed(np.stack(shared.fp_t.values), np.stack(shared.fp_c.values))
        check = dict(n_shared=int(len(shared)), raw_smiles_differ=float((shared.smiles_t != shared.smiles_c).mean()),
                     tan_before_mean=float(before.mean()), tan_before_lt1=float((before < 0.999).mean()),
                     tan_after_mean=float(after.mean()), tan_after_lt1=float((after < 0.999).mean()))
        log(f"shared train/COCONUT structures: {check}")

    # one entry per key; COCONUT's representation first
    allf = pd.concat([fc, ft], ignore_index=True)
    in_tr = set(ft.key)
    in_co = set(fc.key)
    pool = allf.drop_duplicates("key", keep="first").copy()
    pool["in_train"] = pool.key.isin(in_tr)
    pool["in_coconut"] = pool.key.isin(in_co)
    fm = {f: chem.formula_mass(f) for f in pool.formula.unique()}
    pool["mass"] = pool.formula.map(fm).astype(np.float64)
    pool = pool[np.isfinite(pool.mass)].sort_values(["mass", "key"], kind="mergesort").reset_index(drop=True)
    fp = np.stack(pool.fp.values).astype(np.uint8)
    np.save(os.path.join(out_dir, "pool_fp.npy"), fp)
    pool[["key", "smiles", "formula", "mass", "in_train", "in_coconut"]].to_parquet(
        os.path.join(out_dir, "pool.parquet"), index=False)

    pid_of = pd.Series(np.arange(len(pool)), index=pool.key.values)
    sid_key = np.array(tr["ik14"].values, dtype=object)
    sid_key[ft.idx.values] = ft.key.values
    sid2pid = pid_of.reindex(sid_key).fillna(-1).values.astype(np.int64)
    np.save(os.path.join(out_dir, "sid2pid.npy"), sid2pid)

    meta = dict(release=release, coconut_smiles_column=co_col, n_pool=int(len(pool)),
                n_train_struct=int(len(tr)), n_coconut_input=int(len(co_smiles)),
                n_in_train=int(pool.in_train.sum()), n_in_coconut=int(pool.in_coconut.sum()),
                n_both=int((pool.in_train & pool.in_coconut).sum()), sid_unmapped=int((sid2pid < 0).sum()),
                max_tautomers=chem.MAX_TAUTOMERS, check=check, secs=round(time.time() - t0))
    json.dump(meta, open(os.path.join(out_dir, "pool_meta.json"), "w"), indent=1)
    log(f"pool written: {meta}")
    return meta


def neutral_masses(d, cache, log=print):
    """Corrected (neutral) mass per pid for the pool in directory d. Needs RDKit."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    RDLogger.DisableLog("rdApp.*")
    p = pd.read_parquet(os.path.join(d, "pool.parquet"))
    sid2pid = np.load(os.path.join(d, "sid2pid.npy"))
    mass = p["mass"].values.astype(np.float64).copy()
    stats = dict(n=len(p))
    # 1) structures in train: the train formula mass (the one train precursors agree with)
    ok = sid2pid >= 0
    tm = pd.Series(cache.smass[np.flatnonzero(ok)]).groupby(sid2pid[ok]).first()
    tm = tm[np.isfinite(tm.values)]
    stats["from_train_changed"] = int((np.abs(mass[tm.index.values] - tm.values) > 1e-4).sum())
    mass[tm.index.values] = tm.values
    # 2) other charged entries: neutralise with RDKit's Uncharger
    charged = p["formula"].str.contains(r"[+-]\d*$", regex=True).values
    todo = np.flatnonzero(charged & ~np.isin(np.arange(len(p)), tm.index.values))
    unch = rdMolStandardize.Uncharger()
    changed = 0
    for i in todo:
        m = Chem.MolFromSmiles(p["smiles"].values[i])
        if m is None:
            continue
        try:
            f = CalcMolFormula(unch.uncharge(m))
        except Exception:
            continue
        nm = chem.formula_mass(f)
        if np.isfinite(nm) and abs(nm - mass[i]) > 1e-4:
            mass[i] = nm
            changed += 1
    stats.update(charged_not_in_train=int(len(todo)), uncharged_changed=changed)
    log(f"neutral masses: {stats}")
    return mass, stats


def uncharged_mass(smiles):
    """Formula mass after RDKit's Uncharger (NaN on failure). Needs RDKit."""
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if m is None:
        return np.nan
    try:
        return chem.formula_mass(CalcMolFormula(rdMolStandardize.Uncharger().uncharge(m)))
    except Exception:  # noqa: BLE001
        return np.nan


def build_pubchem_part(cand, base_keys, out_dir, level, level_def, workers=4, log=print, extra_meta=None):
    """Canonicalise PubChem candidates and write the PubChem part (see the module docstring).
    cand: DataFrame with cid, smiles and the reporting columns n_pmid, n_syn, n_pat, pcl_anno, np, in the order of
    preference (the first of several entries with the same key is kept). base_keys: keys of the pool it is appended to."""
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    cs = _canon_many(cand["smiles"].tolist(), workers, log, "pubchem")
    ok = np.array([c is not None for c in cs])
    d = cand[ok].reset_index(drop=True)
    good = [c for c in cs if c is not None]
    d["key"] = [c["key"] for c in good]
    d["smiles"] = [c["smiles"] for c in good]
    d["formula"] = [c["formula"] for c in good]
    fp = np.stack([c["fp"] for c in good]).astype(np.uint8) if good else np.zeros((0, chem.FP_BITS // 8), np.uint8)
    del cs, good
    stats = dict(n_input=int(len(cand)), failed=int((~ok).sum()))
    in_base = d.key.isin(set(base_keys)).values
    dup = d.key.duplicated(keep="first").values
    keep = ~in_base & ~dup
    stats.update(in_base_pool=int(in_base.sum()), duplicate_key=int((dup & ~in_base).sum()))
    d, fp = d[keep].reset_index(drop=True), fp[keep]
    fm = {f: chem.formula_mass(f) for f in d.formula.unique()}
    d["mass"] = d.formula.map(fm).astype(np.float64)
    fin = np.isfinite(d.mass.values)
    stats["no_mass"] = int((~fin).sum())
    d, fp = d[fin].reset_index(drop=True), fp[fin]
    mass = d.mass.values.copy()
    charged = np.flatnonzero(d.formula.str.contains(r"[+-]\d*$", regex=True).values)
    um = np.array(map_many(uncharged_mass, d.smiles.values[charged].tolist(), workers)) if len(charged) else np.zeros(0)
    upd = np.isfinite(um) & (np.abs(um - mass[charged]) > 1e-4)
    mass[charged[upd]] = um[upd]
    stats.update(charged=int(len(charged)), uncharged_changed=int(upd.sum()))
    o = np.argsort(d.mass.values, kind="mergesort")
    d, fp, mass = d.iloc[o].reset_index(drop=True), fp[o], mass[o]
    d["level"] = np.int8(level)
    cols = ["key", "smiles", "formula", "mass", "cid", "level", "n_pmid", "n_syn", "n_pat", "pcl_anno", "np"]
    d[cols].to_parquet(os.path.join(out_dir, "pc_pool.parquet"), index=False)
    np.save(os.path.join(out_dir, "pc_fp.npy"), fp)
    np.save(os.path.join(out_dir, "pc_mass.npy"), mass)
    meta = dict(n=int(len(d)), levels={str(level): level_def}, **stats, max_tautomers=chem.MAX_TAUTOMERS,
                secs=round(time.time() - t0), **(extra_meta or {}))
    json.dump(meta, open(os.path.join(out_dir, "pc_meta.json"), "w"), indent=1)
    log(f"PubChem part written: {meta}")
    return meta


def map_many(fn, items, workers, chunksize=500):
    from multiprocessing import Pool as MPool
    with MPool(workers) as p:
        return p.map(fn, items, chunksize=chunksize)


class Pool:
    """Candidate pool (pid = row index of pool.parquet, which is sorted by the raw formula mass, followed by the
    PubChem part when one is attached). Mass windows use `mass_sorted` / `order`: the active pids (level <= the
    level set with set_level) sorted by the corrected mass when pool_mass.npy / pc_mass.npy are available."""

    def __init__(self, d=None, cache=None, mass_fix=True, pubchem=True):
        if d is None:  # train-only pool straight from the cache (class-1 kernels)
            order = cache.struct_sorted
            s = cache.struct.iloc[order]
            self.key = s["ik14"].values
            self.smiles = s["smiles"].values
            self.mass = s["mass"].values
            self.fp = cache.fp[order]
            self.sid2pid = np.full(len(cache.struct), -1, np.int64)
            self.sid2pid[order] = np.arange(len(order))
            self.in_train = np.ones(len(order), bool)
            self.in_coconut = np.zeros(len(order), bool)
            self.meta = dict(n_pool=len(order), train_only=True)
        else:
            p = pd.read_parquet(os.path.join(d, "pool.parquet"))
            self.key = p["key"].values
            self.smiles = p["smiles"].values
            self.mass = p["mass"].values.astype(np.float64)
            self.in_train = p["in_train"].values
            self.in_coconut = p["in_coconut"].values if "in_coconut" in p else np.zeros(len(p), bool)
            self.fp = np.load(os.path.join(d, "pool_fp.npy"))
            self.sid2pid = np.load(os.path.join(d, "sid2pid.npy"))
            self.meta = json.load(open(os.path.join(d, "pool_meta.json")))
            if mass_fix:
                from .paths import find
                try:
                    fixed = np.load(find("pool_mass.npy"))
                    assert len(fixed) == len(self.mass)
                    print(f"[Pool] mass fix applied to {(np.abs(fixed - self.mass) > 1e-4).sum():,} structures")
                    self.mass = fixed
                except FileNotFoundError:
                    print("[Pool] no pool_mass.npy: raw formula masses")
        self.n_base = len(self.key)
        self.level = np.zeros(self.n_base, np.int8)
        if d is not None and pubchem:
            from .paths import find
            try:
                self._append_pubchem(os.path.dirname(find("pc_meta.json")))
            except FileNotFoundError:
                pass
        self.set_level(int(self.level.max()))
        print(f"[Pool] {len(self.key):,} structures", flush=True)

    def _append_pubchem(self, pd_dir):
        q = pd.read_parquet(os.path.join(pd_dir, "pc_pool.parquet"), columns=["key", "smiles", "level"])
        fp = np.load(os.path.join(pd_dir, "pc_fp.npy"))
        mass = np.load(os.path.join(pd_dir, "pc_mass.npy"))
        assert len(q) == len(fp) == len(mass), "PubChem part files differ in length"
        self.key = np.concatenate([self.key, q["key"].values])
        self.smiles = np.concatenate([self.smiles, q["smiles"].values])
        self.mass = np.concatenate([self.mass, mass])
        self.in_train = np.concatenate([self.in_train, np.zeros(len(q), bool)])
        self.in_coconut = np.concatenate([self.in_coconut, np.zeros(len(q), bool)])
        self.fp = np.concatenate([self.fp, fp])
        self.level = np.concatenate([self.level, q["level"].values.astype(np.int8)])
        self.pc_meta = json.load(open(os.path.join(pd_dir, "pc_meta.json")))
        print(f"[Pool] PubChem part: {len(q):,} structures from {pd_dir}, levels {self.pc_meta.get('levels')}")

    def set_level(self, k):
        """Windows (and nearest_structures) use only pids with level <= k (0 = the pool without PubChem)."""
        act = np.flatnonzero(self.level <= k)
        self.active_level = k
        self.order = act[np.argsort(self.mass[act], kind="mergesort")]
        self.mass_sorted = self.mass[self.order]
        print(f"[Pool] level <= {k}: {len(self.order):,} active of {len(self.key):,}", flush=True)

    def window(self, target, ppm):
        """pids whose (corrected) mass is within +-ppm of target."""
        tol = target * ppm * 1e-6
        lo, hi = np.searchsorted(self.mass_sorted, [target - tol, target + tol])
        return self.order[lo:hi]

    def __len__(self):
        return len(self.key)

    @staticmethod
    def find_dir():
        from .paths import find
        return os.path.dirname(find("pool_meta.json"))

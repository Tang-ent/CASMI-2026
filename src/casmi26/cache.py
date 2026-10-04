"""train.parquet -> cleaned CSR peaks + spectrum metadata + structure table + search indices.

Files written to `out_dir` (all loaded fully into RAM by Cache.load):
  spec_peaks.npz   off (int64), mz, it (float32)            cleaned peaks of every train spectrum
  spec_meta.npz    sid, lib, adduct, mode, prec, ppm, ce, keep, npk
  struct.parquet   ik14, smiles, formula, mass               one row per inchikey14 (sid = row index)
  index.npz        lib_rows/lib_mass (kept spectra by structure mass), struct_sorted/struct_mass (by mass)
  reps.npz         one representative per (structure, polarity, library), prepped for analog search
  struct_fp.npy    packed Morgan r=2 fingerprints (n_struct, FP_BITS // 8) uint8
  cache_meta.json  libraries, adducts, parameters, counts
"""
import json
import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import chem, spectra

VERSION = 1
MIN_PEAKS = 3
MAX_ABS_PPM = 50.0
COLUMNS = ["ingest_lib", "inchikey14", "normalized_smiles", "molecular_formula", "adduct", "ionization_mode",
           "precursor_mz", "precursor_error_ppm", "ms2_mzs", "ms2_normalized_intensities", "collision_energy_ev"]


def _peaks(col):
    """(offsets from 0, flat float64 values) of a list<double> column."""
    col = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    off = col.offsets.to_numpy().astype(np.int64)
    vals = col.values.to_numpy(zero_copy_only=False).astype(np.float64)
    return off - off[0], vals[off[0]:off[-1]]


def _list_mean(col):
    """Mean of each list in a list<double> column (NaN for null / empty)."""
    off, vals = _peaks(col)
    n = np.diff(off)
    starts = np.minimum(off[:-1], len(vals))
    sums = np.add.reduceat(np.r_[vals, 0.0], starts)
    return np.where(n > 0, sums / np.maximum(n, 1), np.nan)


def _fp_worker(smiles):
    return chem.morgan_packed(smiles)


def build(train_path, out_dir, with_fp=True, max_row_groups=None, workers=4, log=print):
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    pf = pq.ParquetFile(train_path)
    n_rg = pf.num_row_groups if max_row_groups is None else min(max_row_groups, pf.num_row_groups)
    offs, mzs, its, metas = [], [], [], []
    p0 = 0
    for g in range(n_rg):
        t = pf.read_row_group(g, columns=COLUMNS)
        off, mz = _peaks(t.column("ms2_mzs"))
        off2, it = _peaks(t.column("ms2_normalized_intensities"))
        assert np.array_equal(off, off2), "m/z and intensity lists differ in length"
        prec = t.column("precursor_mz").to_numpy(zero_copy_only=False).astype(np.float64)
        noff, nmz, nit = spectra.clean_csr(off, mz, it, prec, spectra.MAX_OVER_PREC,
                                           spectra.CACHE_FLOOR, spectra.CACHE_TOPK)
        offs.append(noff[1:] + p0)
        p0 += int(noff[-1])
        mzs.append(nmz)
        its.append(nit)
        m = t.select([c for c in COLUMNS if not c.startswith("ms2_") and c != "collision_energy_ev"]).to_pandas()
        m["ce"] = _list_mean(t.column("collision_energy_ev"))
        metas.append(m)
        log(f"  row group {g + 1}/{n_rg}: {t.num_rows:,} spectra, {p0:,} peaks kept ({time.time() - t0:.0f}s)")
        del t
    off = np.concatenate([[0], *offs]).astype(np.int64)
    mz = np.concatenate(mzs)
    it = np.concatenate(its)
    meta = pd.concat(metas, ignore_index=True)
    del offs, mzs, its, metas
    np.savez(os.path.join(out_dir, "spec_peaks.npz"), off=off, mz=mz, it=it)
    npk = np.diff(off).astype(np.int32)

    # structures (sid = factorized inchikey14)
    sid, ik_uniq = pd.factorize(meta["inchikey14"])
    first = np.unique(sid, return_index=True)[1]
    struct = pd.DataFrame({"ik14": np.asarray(ik_uniq, dtype=object),
                           "smiles": meta["normalized_smiles"].values[first],
                           "formula": meta["molecular_formula"].values[first]})
    fmass = {f: chem.formula_mass(f) for f in struct["formula"].unique()}
    struct["mass"] = struct["formula"].map(fmass).astype(np.float64)
    struct.to_parquet(os.path.join(out_dir, "struct.parquet"), index=False)
    smass = struct["mass"].values
    log(f"structures {len(struct):,}; formula mass ok {np.isfinite(smass).mean():.2%}")

    lib, libs = pd.factorize(meta["ingest_lib"])
    adduct, adducts = pd.factorize(meta["adduct"].fillna("?"))
    mode = np.where(meta["ionization_mode"].values == "positive", 1, -1).astype(np.int8)
    ppm = meta["precursor_error_ppm"].values.astype(np.float32)
    keep = (np.isfinite(smass[sid]) & (npk >= MIN_PEAKS)
            & ((np.abs(ppm) <= MAX_ABS_PPM) | ~np.isfinite(ppm)))
    np.savez(os.path.join(out_dir, "spec_meta.npz"), sid=sid.astype(np.int32), lib=lib.astype(np.int8),
             adduct=adduct.astype(np.int16), mode=mode, prec=meta["precursor_mz"].values.astype(np.float64),
             ppm=ppm, ce=meta["ce"].values.astype(np.float32), keep=keep, npk=npk)
    log(f"spectra {len(meta):,}; kept for the library {keep.mean():.2%}")

    # library index: kept spectra sorted by their structure's formula mass
    rows = np.flatnonzero(keep)
    rows = rows[np.argsort(smass[sid[rows]], kind="mergesort")]
    ok = np.isfinite(smass)
    s_sorted = np.flatnonzero(ok)
    s_sorted = s_sorted[np.argsort(smass[s_sorted], kind="mergesort")]
    np.savez(os.path.join(out_dir, "index.npz"), lib_rows=rows.astype(np.int64), lib_mass=smass[sid[rows]],
             struct_sorted=s_sorted.astype(np.int64), struct_mass=smass[s_sorted])

    # analog representatives: per (structure, polarity, library) the kept spectrum with the most peaks
    d = pd.DataFrame({"row": rows, "sid": sid[rows], "mode": mode[rows], "lib": lib[rows], "npk": npk[rows]})
    d = d.sort_values(["sid", "mode", "lib", "npk", "row"], ascending=[True, True, True, False, True])
    rep = d.drop_duplicates(["sid", "mode", "lib"])["row"].values
    rep = rep[np.argsort(smass[sid[rep]], kind="mergesort")]
    c = spectra.SimCfg
    roff, rmz, rp = spectra.prep_store(rep.astype(np.int64), off, mz, it, c.floor, c.topk, c.power, c.ent_weight)
    np.savez(os.path.join(out_dir, "reps.npz"), rep_rows=rep.astype(np.int64), rep_mass=smass[sid[rep]],
             rep_sid=sid[rep].astype(np.int32), rep_mode=mode[rep], rep_lib=lib[rep].astype(np.int8),
             rep_off=roff, rep_mz=rmz, rep_p=rp)
    log(f"representatives {len(rep):,} ({time.time() - t0:.0f}s)")

    if with_fp:
        from multiprocessing import Pool
        with Pool(workers) as p:
            fps = p.map(_fp_worker, list(struct["smiles"].values), chunksize=2000)
        fp = np.stack(fps).astype(np.uint8)
        log(f"fingerprints {fp.shape}, empty {(fp.sum(1) == 0).mean():.2%} ({time.time() - t0:.0f}s)")
    else:
        fp = np.zeros((len(struct), chem.FP_BITS // 8), np.uint8)
    np.save(os.path.join(out_dir, "struct_fp.npy"), fp)

    info = dict(version=VERSION, libs=[str(x) for x in libs], adducts=[str(x) for x in adducts],
                n_spectra=int(len(meta)), n_struct=int(len(struct)), n_kept=int(keep.sum()), n_reps=int(len(rep)),
                n_peaks=int(len(mz)), with_fp=bool(with_fp), min_peaks=MIN_PEAKS, max_abs_ppm=MAX_ABS_PPM,
                cache_floor=spectra.CACHE_FLOOR, cache_topk=spectra.CACHE_TOPK, secs=round(time.time() - t0))
    json.dump(info, open(os.path.join(out_dir, "cache_meta.json"), "w"), indent=1)
    log(f"cache written to {out_dir} ({time.time() - t0:.0f}s)")
    return info


class Cache:
    """All cache arrays in RAM (numba needs plain ndarrays)."""

    def __init__(self, d):
        t0 = time.time()
        self.dir = d
        self.info = json.load(open(os.path.join(d, "cache_meta.json")))
        z = np.load(os.path.join(d, "spec_peaks.npz"))
        self.off, self.mz, self.it = z["off"], z["mz"], z["it"]
        z = np.load(os.path.join(d, "spec_meta.npz"))
        for k in ["sid", "lib", "adduct", "mode", "prec", "ppm", "ce", "keep", "npk"]:
            setattr(self, k, z[k])
        z = np.load(os.path.join(d, "index.npz"))
        for k in ["lib_rows", "lib_mass", "struct_sorted", "struct_mass"]:
            setattr(self, k, z[k])
        z = np.load(os.path.join(d, "reps.npz"))
        for k in ["rep_rows", "rep_mass", "rep_sid", "rep_mode", "rep_lib", "rep_off", "rep_mz", "rep_p"]:
            setattr(self, k, z[k])
        self.fp = np.load(os.path.join(d, "struct_fp.npy"))
        self.struct = pd.read_parquet(os.path.join(d, "struct.parquet"))
        self.smass = self.struct["mass"].values
        self.libs = self.info["libs"]
        self.lib_id = {n: i for i, n in enumerate(self.libs)}
        self.adducts = self.info["adducts"]
        self.adduct_id = {n: i for i, n in enumerate(self.adducts)}
        print(f"[Cache] {self.info['n_spectra']:,} spectra, {self.info['n_struct']:,} structures, "
              f"{self.info['n_reps']:,} reps ({time.time() - t0:.0f}s)", flush=True)

    def spectrum(self, row):
        a, b = self.off[row], self.off[row + 1]
        return self.mz[a:b], self.it[a:b]

    @staticmethod
    def find_dir():
        from .paths import find
        return os.path.dirname(find("cache_meta.json"))

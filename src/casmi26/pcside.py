"""M7 P2: the PubChem S' side list (casmi26-pubchem-part: literature-linked PubChem structures that are not in the
pool, 2,142,653 rows). S' never enters the pool (M3: the pool merge cost 0.034 on the LB); for molecules whose best
library match is weak, the S' structures in the pool's mass window are ranked by spectral evidence only and a few of
them take fixed slots of the final list (casmi26.slots).

Score (fixed before any result, plan 2026-10-14): z(fz) + z(fzr) + z(fwd) + z(g2) + z(ap4) over the window's S'
structures (z within the list; NaN g2 -> 0). fz / fwd / ap4 use pc_fp (Morgan r2 of the canonical tautomer, as the
pool), fzr the Morgan + chem.rich_fp_tail layout of pool_fp2, g2 G2Model.score_smiles. Windows larger than max_cand
are cut to the top max_cand by fz first (time). No provenance (n_pmid, n_pat, np...) enters the score."""
import os

import numpy as np

from . import chem
from .search import _zs, _zs_finite


def _unpack(fp, nbits=chem.FP_BITS):
    return np.unpackbits(np.atleast_2d(np.asarray(fp)), axis=1)[:, :nbits].astype(np.float32)


class PubChemSide:
    def __init__(self, d=None):
        import pandas as pd
        d = d or self.find_dir()
        t = pd.read_parquet(os.path.join(d, "pc_pool.parquet"), columns=["key", "smiles"])
        self.key, self.smiles = t.key.values, t.smiles.values
        self.mass = np.load(os.path.join(d, "pc_mass.npy"))
        self.fp = np.load(os.path.join(d, "pc_fp.npy"), mmap_mode="r")
        assert len(self.key) == len(self.mass) == len(self.fp), "PubChem part files differ in length"
        self.order = np.argsort(self.mass, kind="mergesort")
        self.mass_sorted = self.mass[self.order]
        print(f"[PubChemSide] {len(self.key):,} S' structures from {d}")

    @staticmethod
    def find_dir():
        from .paths import find
        return os.path.dirname(find("pc_pool.parquet"))

    def __len__(self):
        return len(self.key)

    def window(self, target, ppm):
        """S' row ids within +-ppm of target (sorted)."""
        if not (np.isfinite(target) and np.isfinite(ppm)):
            return np.zeros(0, np.int64)
        tol = target * ppm * 1e-6
        lo, hi = np.searchsorted(self.mass_sorted, [target - tol, target + tol])
        return np.sort(self.order[lo:hi])

    def ranked(self, S, specs, info, fwd=None, fprich=None, g2=None, exclude_keys=(), max_cand=300, sim_power=4.0):
        """S' structures of one molecule, best first, right after S.features(specs) (uses S.last and info's target and
        ppm). fprich = (FPModel, None) or the query's z_rich; g2 = G2Model. -> (smiles list, key list, score array)."""
        idx = self.window(info["target"], info.get("ppm", np.nan))
        z = S.last.get("z")
        if len(idx) == 0 or z is None:
            return [], [], np.zeros(0)
        F = _unpack(self.fp[idx])
        fz = F @ z
        if len(idx) > max_cand:
            keep = np.sort(np.argsort(-fz, kind="mergesort")[:max_cand])
            idx, F, fz = idx[keep], F[keep], fz[keep]
        comb = _zs(fz)
        apid, asim = S.last.get("apid", np.zeros(0, np.int64)), S.last.get("asim", np.zeros(0))
        if len(apid):
            A = _unpack(S.P.fp[apid])
            inter = F @ A.T
            tan = inter / (F.sum(1)[:, None] + A.sum(1)[None, :] - inter + 1e-9)
            comb += _zs((tan * (np.clip(asim, 0, None) ** sim_power)[None, :]).max(1))
        smiles = [str(s) for s in self.smiles[idx]]
        if fwd is not None:
            comb += _zs(fwd.score(specs, np.asarray(self.fp[idx])))
        if fprich is not None:
            zr = S.last.get("z_rich")
            if zr is None:
                zr = fprich[0](specs)
            Fr = np.stack([np.r_[F[i], chem.rich_fp_tail(s)] for i, s in enumerate(smiles)]).astype(np.float32)
            comb += _zs(Fr[:, :len(zr)] @ zr)
        if g2 is not None:
            comb += _zs_finite(g2.score_smiles(specs, smiles))
        o = np.argsort(-comb, kind="mergesort")
        ex = set(exclude_keys)
        o = [i for i in o if self.key[idx[i]] not in ex]
        return [smiles[i] for i in o], [str(self.key[idx[i]]) for i in o], comb[o]

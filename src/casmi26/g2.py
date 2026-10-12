"""G2: fragment-level forward model (M6 P0-3, k29_g2; ICEBERG-like, own training only).

Fragments of a structure = the frag.py definition (cut one or two acyclic single bonds, or two bonds of the same
non-aromatic ring; components >= 30 Da; fragment mass = its atoms incl. the H they carry in the intact molecule).
Here every fragment keeps its atom set (unique atom sets; one-cut fragments first, then two-cut ones, up to MAX_FRAG
per structure) so that a model can see which atoms it holds. Structures with more than MAX_ATOMS heavy atoms, or that
RDKit cannot read, get no score.

Model (torch only): atom / bond embeddings, an MPNN of `layers` layers (condition-free), and a fragment head:
mean of the projected atom states over the fragment + the cut bonds + the molecule mean + size features (mass share,
atom share, depth), plus the spectrum's condition (adduct, polarity, CE: fwd.cond_row) -> one logit per hydrogen shift
-2..+2. Softmax over all (fragment, shift) of a structure = predicted intensities.

Cosine with a spectrum: observed = the top TOP_PEAKS peaks below precursor - 1.5 Da (sqrt intensity, L2-normalised).
Predicted ion m/z = fragment mass + shift * H +- proton (sign by polarity; like frag.py, Na / K fragment ions are not
generated). Predictions with the same m/z (1e-4 Da) form one group; each group goes to the nearest observed peak within
max(PPM, ABS_DA) and predictions on the same observed peak add up; unmatched groups stay in the norm.
Score of a candidate for a query = mean cosine over the query's usable spectra.

Training (fwd.train_rank form): (1 - cosine of the truth) + an in-window softmax over the truth and k_neg train
structures with the truth's formula (enc3.SameFormulaNegatives), temperature temp. The training state is saved
every save_every steps (ckpt); train(..., resume=) continues it.
"""
import math
import time

import numpy as np
from numba import njit

from . import ckpt
from .chem import MONO, PROTON
from .fpnet import adduct_index, unpack_bits_torch
from .fwd import N_COND, cond_row

H = MONO["H"]
SHIFTS = np.arange(-2, 3)
NS = len(SHIFTS)
MIN_MASS = 30.0
MAX_ATOMS = 160
NBY = MAX_ATOMS // 8
MAX_FRAG = 512
TOP_PEAKS = 64
PREC_GAP = 1.5
PPM, ABS_DA = 10.0, 0.002
DEC = 4

ELEMS = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "Si", "B", "Se", "H"]
EIX = {e: i for i, e in enumerate(ELEMS)}
ATOM_CARD = [len(ELEMS) + 1, 7, 5, 5, 2, 2, 6]  # element, degree, H count, charge + 2, aromatic, ring, hybridisation
BOND_CARD = [5, 2, 2]  # type (single, double, triple, aromatic, other), ring, conjugated
ZOB = np.random.default_rng(20261012).integers(1, 2 ** 62, MAX_ATOMS, dtype=np.int64)  # atom-set hashes


class Cfg:
    d = 256
    hid = 128
    layers = 4
    drop = 0.1
    batch = 64
    k_neg = 16
    temp = 0.05
    lr = 5e-4
    wd = 0.01
    warmup = 1000
    total_steps = 60000
    workers = 3
    save_every = 2000


# ---------------------------------------------------------------------------------------------- graphs
def mol_graph(smiles):
    """(atom features (n, 7) int8, atom masses incl. own H (n,), bond ends (m, 2) int16, bond features (m, 3) int8,
    acyclic single bonds (s,) int16, non-aromatic rings [bond index arrays]) or None."""
    from rdkit import Chem
    from rdkit.Chem import rdchem
    m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if m is None or m.GetNumAtoms() == 0 or m.GetNumAtoms() > MAX_ATOMS:
        return None
    hyb = {rdchem.HybridizationType.SP: 0, rdchem.HybridizationType.SP2: 1, rdchem.HybridizationType.SP3: 2,
           rdchem.HybridizationType.SP3D: 3, rdchem.HybridizationType.SP3D2: 4}
    bt = {Chem.BondType.SINGLE: 0, Chem.BondType.DOUBLE: 1, Chem.BondType.TRIPLE: 2, Chem.BondType.AROMATIC: 3}
    n = m.GetNumAtoms()
    af = np.zeros((n, 7), np.int8)
    am = np.zeros(n)
    for a in m.GetAtoms():
        el = a.GetSymbol()
        if el not in MONO:
            return None
        i, h = a.GetIdx(), a.GetTotalNumHs()
        am[i] = MONO[el] + H * h
        af[i] = (EIX.get(el, len(ELEMS)), min(a.GetDegree(), 6), min(h, 4), min(max(a.GetFormalCharge(), -2), 2) + 2,
                 a.GetIsAromatic(), a.IsInRing(), hyb.get(a.GetHybridization(), 5))
    bonds = list(m.GetBonds())
    be = np.array([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in bonds], np.int16).reshape(-1, 2)
    bf = np.array([(bt.get(b.GetBondType(), 4), b.IsInRing(), b.GetIsConjugated()) for b in bonds], np.int8).reshape(-1, 3)
    sa = np.array([b.GetIdx() for b in bonds if not b.IsInRing() and b.GetBondType() == Chem.BondType.SINGLE], np.int16)
    rings = [np.array(r, np.int16) for r in m.GetRingInfo().BondRings()
             if not any(m.GetBondWithIdx(i).GetIsAromatic() for i in r)]
    return af, am, be, bf, sa, rings


class Graphs:
    """mol_graph of many structures in CSR arrays (index = position in the SMILES list given to build)."""

    def __init__(self, graphs):
        n = len(graphs)
        self.ok = np.array([g is not None for g in graphs])
        empty = (np.zeros((0, 7), np.int8), np.zeros(0), np.zeros((0, 2), np.int16), np.zeros((0, 3), np.int8),
                 np.zeros(0, np.int16), [])
        gs = [g if g is not None else empty for g in graphs]

        def csr(xs):
            off = np.zeros(n + 1, np.int64)
            off[1:] = np.cumsum([len(x) for x in xs])
            return off

        self.a_off = csr([g[1] for g in gs])
        self.af = np.concatenate([g[0] for g in gs]) if n else np.zeros((0, 7), np.int8)
        self.am = np.concatenate([g[1] for g in gs]) if n else np.zeros(0)
        self.b_off = csr([g[2] for g in gs])
        self.be = np.concatenate([g[2] for g in gs]) if n else np.zeros((0, 2), np.int16)
        self.bf = np.concatenate([g[3] for g in gs]) if n else np.zeros((0, 3), np.int8)
        self.s_off = csr([g[4] for g in gs])
        self.sa = np.concatenate([g[4] for g in gs]) if n else np.zeros(0, np.int16)
        rings = [r for g in gs for r in g[5]]
        self.r_moff = csr([g[5] for g in gs])  # rings of structure i: r_moff[i]:r_moff[i + 1]
        self.r_off = np.zeros(len(rings) + 1, np.int64)
        self.r_off[1:] = np.cumsum([len(r) for r in rings])
        self.rb = np.concatenate(rings) if rings else np.zeros(0, np.int16)

    @classmethod
    def build(cls, smiles, workers=1, chunksize=2000):
        smiles = list(smiles)
        if workers > 1:
            from .pool import map_many
            return cls(map_many(mol_graph, smiles, workers, chunksize))
        return cls([mol_graph(s) for s in smiles])

    def __len__(self):
        return len(self.ok)

    def get(self, i):
        """Local arrays of structure i (af, am, be, bf, sa, ring offsets from 0, ring bonds) or None."""
        if not self.ok[i]:
            return None
        a, b = self.a_off[i], self.a_off[i + 1]
        c, d = self.b_off[i], self.b_off[i + 1]
        e, f = self.s_off[i], self.s_off[i + 1]
        r0, r1 = self.r_moff[i], self.r_moff[i + 1]
        ro = self.r_off[r0:r1 + 1]
        return (self.af[a:b], self.am[a:b], self.be[c:d], self.bf[c:d], self.sa[e:f], ro - ro[0],
                self.rb[ro[0]:ro[-1]])


# ---------------------------------------------------------------------------------------------- fragments
@njit(nogil=True, cache=True)
def _find(par, i):
    while par[i] != i:
        par[i] = par[par[i]]
        i = par[i]
    return i


@njit(nogil=True, cache=True)
def _cut(c1, c2, depth, am, be, min_mass, par, rid, cid, tab_key, tab_used, k, maxf, bits, mass, dep, cuts, size):
    """Components of the cut set (c1, c2) appended as fragments (new atom sets >= min_mass); returns the new count."""
    n, m = len(am), len(be)
    for i in range(n):
        par[i] = i
        rid[i] = -1
    for b in range(m):
        if b == c1 or b == c2:
            continue
        ra, rb = _find(par, be[b, 0]), _find(par, be[b, 1])
        if ra != rb:
            par[ra] = rb
    nc = 0
    for i in range(n):
        r = _find(par, i)
        if rid[r] < 0:
            rid[r] = nc
            nc += 1
        cid[i] = rid[r]
    if nc < 2:
        return k
    cm = np.zeros(nc)
    ch = np.zeros(nc, np.int64)
    cs = np.zeros(nc, np.int64)
    for i in range(n):
        cm[cid[i]] += am[i]
        ch[cid[i]] ^= ZOB[i]
        cs[cid[i]] += 1
    tmask = len(tab_key) - 1
    for c in range(nc):
        if math.floor(cm[c] * 1e4 + 0.5) / 1e4 < min_mass:
            continue
        h = ch[c]
        slot = h & tmask
        dup = False
        while tab_used[slot]:
            if tab_key[slot] == h:
                dup = True
                break
            slot = (slot + 1) & tmask
        if dup:
            continue
        tab_used[slot] = True
        tab_key[slot] = h
        for j in range(bits.shape[1]):
            bits[k, j] = 0
        for i in range(n):
            if cid[i] == c:
                bits[k, i >> 3] |= np.uint8(1 << (7 - (i & 7)))
        mass[k] = cm[c]
        dep[k] = depth
        size[k] = cs[c]
        for t in range(2):
            cb = c1 if t == 0 else c2
            cuts[k, t] = -1
            if cb >= 0 and (cid[be[cb, 0]] == c) != (cid[be[cb, 1]] == c):
                cuts[k, t] = cb
        k += 1
        if k >= maxf:
            return k
    return k


@njit(nogil=True, cache=True)
def _enumerate(am, be, sa, r_off, rb, min_mass, maxf, tsize, bits, mass, dep, cuts, size):
    n = len(am)
    if len(be) == 0:
        return 0
    par = np.empty(n, np.int64)
    rid = np.empty(n, np.int64)
    cid = np.empty(n, np.int64)
    tab_key = np.zeros(tsize, np.int64)
    tab_used = np.zeros(tsize, np.bool_)
    k = 0
    s = len(sa)
    for x in range(s):
        k = _cut(sa[x], -1, 1, am, be, min_mass, par, rid, cid, tab_key, tab_used, k, maxf, bits, mass, dep, cuts, size)
        if k >= maxf:
            return k
    for x in range(s):
        for y in range(x + 1, s):
            k = _cut(sa[x], sa[y], 2, am, be, min_mass, par, rid, cid, tab_key, tab_used, k, maxf, bits, mass, dep,
                     cuts, size)
            if k >= maxf:
                return k
    for r in range(len(r_off) - 1):
        for x in range(r_off[r], r_off[r + 1]):
            for y in range(x + 1, r_off[r + 1]):
                k = _cut(rb[x], rb[y], 2, am, be, min_mass, par, rid, cid, tab_key, tab_used, k, maxf, bits, mass,
                         dep, cuts, size)
                if k >= maxf:
                    return k
    return k


def fragments(g, maxf=MAX_FRAG):
    """Fragments of one structure (Graphs.get output) -> dict(bits (k, NBY) uint8, mass (k,), depth (k,) int8,
    cuts (k, 2) int64 local cut bonds on the fragment's boundary (-1 = none), size (k,) atoms, capped (bool))."""
    af, am, be, bf, sa, r_off, rb = g
    tsize = 1 << max(11, int(math.ceil(math.log2(4 * maxf))))
    bits = np.zeros((maxf, NBY), np.uint8)
    mass, dep = np.zeros(maxf), np.zeros(maxf, np.int8)
    cuts, size = np.zeros((maxf, 2), np.int64), np.zeros(maxf, np.int64)
    k = _enumerate(am, be.astype(np.int64), sa.astype(np.int64), r_off.astype(np.int64), rb.astype(np.int64),
                   MIN_MASS, maxf, tsize, bits, mass, dep, cuts, size)
    return dict(bits=bits[:k], mass=mass[:k], depth=dep[:k], cuts=cuts[:k], size=size[:k], capped=k >= maxf)


def groups(mass):
    """(group of every (fragment, shift) (k, NS) int32, neutral m/z of each group (U,)): fragment mass + shift * H,
    one group per value rounded to 1e-4 Da."""
    v = np.round(mass[:, None] + SHIFTS[None, :] * H, DEC)
    umz, inv = np.unique(v, return_inverse=True)
    return inv.reshape(v.shape).astype(np.int32), umz


def piece(G, pid):
    """Graph + fragments + groups of one structure (None without a graph)."""
    g = G.get(pid)
    if g is None:
        return None
    fr = fragments(g)
    u, umz = groups(fr["mass"])
    return dict(g=g, nat=len(g[0]), molmass=float(g[1].sum()), u=u, umz=umz, **fr)


def observed(mz, it, prec):
    """Top TOP_PEAKS peaks below prec - PREC_GAP -> (m/z ascending, sqrt-intensity weights L2-normalised)."""
    mz, it = np.asarray(mz, np.float64), np.asarray(it, np.float64)
    k = (mz < float(prec) - PREC_GAP) & (it > 0)
    mz, it = mz[k], it[k]
    if len(mz) == 0:
        return np.zeros(0), np.zeros(0, np.float32)
    o = np.argsort(-it, kind="mergesort")[:TOP_PEAKS]
    mz, w = mz[o], np.sqrt(it[o])
    s = np.argsort(mz, kind="mergesort")
    return mz[s], (w[s] / np.linalg.norm(w)).astype(np.float32)


def match(umz, ion, obs_mz):
    """Nearest observed peak within max(PPM, ABS_DA) of every group's ion m/z (umz + ion) -> (U,) int32, -1 = none."""
    x = np.asarray(umz) + ion
    if len(obs_mz) == 0 or len(x) == 0:
        return np.full(len(x), -1, np.int32)
    i = np.searchsorted(obs_mz, x)
    lo, hi = np.clip(i - 1, 0, len(obs_mz) - 1), np.clip(i, 0, len(obs_mz) - 1)
    dl, dh = np.abs(x - obs_mz[lo]), np.abs(x - obs_mz[hi])
    j = np.where(dh < dl, hi, lo)
    tol = np.maximum(obs_mz[j] * PPM * 1e-6, ABS_DA)
    return np.where(np.minimum(dl, dh) <= tol, j, -1).astype(np.int32)


def ion_of(mode):
    return PROTON if float(mode) > 0 else -PROTON


def collate(pieces, jous=None):
    """Structures (pieces, None = no graph) -> numpy batch. jous: per structure the observed peak of each group
    (match output) or None (graph-only batch for scoring; the groups / matches are then added per spectrum)."""
    S = len(pieces)
    nat = np.array([p["nat"] if p else 0 for p in pieces], np.int64)
    nfr = np.array([len(p["mass"]) if p else 0 for p in pieces], np.int64)
    N, F = max(1, int(nat.max(initial=0))), max(1, int(nfr.max(initial=0)))
    nby = (N + 7) // 8
    a_off = np.r_[0, np.cumsum(nat)]
    nb = np.array([len(p["g"][2]) if p else 0 for p in pieces], np.int64)
    b_off = np.r_[0, np.cumsum(nb)]
    live = [p for p in pieces if p]
    af = np.concatenate([p["g"][0] for p in live]).astype(np.int64) if live else np.zeros((0, 7), np.int64)
    bf = np.concatenate([p["g"][3] for p in live]).astype(np.int64) if live else np.zeros((0, 3), np.int64)
    ends = [p["g"][2].astype(np.int64) + a_off[s] for s, p in enumerate(pieces) if p]
    be = np.concatenate(ends) if ends else np.zeros((0, 2), np.int64)
    a_s = np.repeat(np.arange(S), nat)
    a_l = np.concatenate([np.arange(n) for n in nat]) if S else np.zeros(0, np.int64)
    bits = np.zeros((S, F, nby), np.uint8)
    valid = np.zeros((S, F), bool)
    feat = np.zeros((S, F, 3), np.float32)
    cut = np.full((S, F, 2), -1, np.int64)
    u = np.full((S, F, NS), -1, np.int32)
    for s, p in enumerate(pieces):
        k = nfr[s]
        if not k:
            continue
        bits[s, :k] = p["bits"][:, :nby]
        valid[s, :k] = True
        feat[s, :k, 0] = p["mass"] / max(p["molmass"], 1.0)
        feat[s, :k, 1] = p["size"] / p["nat"]
        feat[s, :k, 2] = p["depth"] - 1
        c = p["cuts"]
        cut[s, :k] = np.where(c >= 0, c + b_off[s], -1)
        u[s, :k] = p["u"]
    out = dict(af=af, bf=bf, be=be, a_s=a_s, a_l=a_l, nat=nat, bits=bits, valid=valid, feat=feat, cut=cut, u=u, nmax=N)
    if jous is not None:
        out["jou"] = jou_array(jous)
    return out


def jou_array(jous):
    U = max(1, max((len(j) for j in jous if j is not None), default=1))
    out = np.full((len(jous), U), -1, np.int32)
    for s, j in enumerate(jous):
        if j is not None:
            out[s, :len(j)] = j
    return out


# ---------------------------------------------------------------------------------------------- model
def build_model(cfg=Cfg):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class G2Net(nn.Module):
        def __init__(self):
            super().__init__()
            d, h = cfg.d, cfg.hid
            self.aemb = nn.ModuleList([nn.Embedding(c, d) for c in ATOM_CARD])
            self.bemb = nn.ModuleList([nn.Embedding(c, d) for c in BOND_CARD])
            self.msg = nn.ModuleList([nn.Linear(d, d) for _ in range(cfg.layers)])
            self.ff = nn.ModuleList([nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(cfg.drop),
                                                   nn.Linear(2 * d, d)) for _ in range(cfg.layers)])
            self.wa = nn.Linear(d, h)
            self.wb = nn.Linear(d, h, bias=False)
            self.wg = nn.Linear(d, h, bias=False)
            self.ws = nn.Linear(3, h, bias=False)
            self.wc = nn.Linear(N_COND, h, bias=False)
            self.out = nn.Linear(h, NS)

        def encode(self, b):
            """Condition-free fragment inputs (S, F, hid) of a batch of torch tensors (collate keys)."""
            x = sum(e(b["af"][:, i]) for i, e in enumerate(self.aemb))
            eb = sum(e(b["bf"][:, i]) for i, e in enumerate(self.bemb))
            e0, e1 = b["be"][:, 0], b["be"][:, 1]
            src, dst, ebb = torch.cat([e0, e1]), torch.cat([e1, e0]), torch.cat([eb, eb])
            # index_select (backward = index_add_) instead of x[idx]: the indexing backward sorts the indices and was
            # 90% of the step time with many repeated indices
            for msg, ff in zip(self.msg, self.ff):
                m = torch.zeros_like(x).index_add_(0, dst, msg(x.index_select(0, src) + ebb).float())
                x = x + ff(x + m).float()
            S, N = b["bits"].shape[0], b["nmax"]
            ha = self.wa(x)
            hp = ha.new_zeros(S, N, ha.shape[1])
            hp[b["a_s"], b["a_l"]] = ha
            M = unpack_bits_torch(b["bits"], N).to(hp.dtype)
            F_ = M.shape[1]
            pooled = torch.bmm(M.reshape(S, -1, N), hp) / M.sum(-1, keepdim=True).clamp(min=1)
            br = self.wb(x.index_select(0, e0) + x.index_select(0, e1) + eb)
            cut = b["cut"].reshape(-1)
            pos = torch.nonzero(cut >= 0).squeeze(1)  # only the real cut bonds (no padding rows)
            cutrep = br.new_zeros(S * F_, br.shape[1]).index_add_(0, pos // 2, br.index_select(0, cut[pos]))
            g = torch.zeros(S, x.shape[1], device=x.device).index_add_(0, b["a_s"], x) / \
                b["nat"].clamp(min=1)[:, None].float()
            return pooled + cutrep.view(S, F_, -1) + self.wg(g)[:, None] + self.ws(b["feat"])

        def head(self, pre, cond):
            """pre (S, F, hid), cond (S, N_COND) -> logits (S, F, NS)."""
            return self.out(F.gelu(pre + self.wc(cond)[:, None]))

    return G2Net()


def cosine(logits, valid, u, jou, obs, ex):
    """logits (S, F, NS), valid (S, F), u (S, F, NS) groups, jou (S, U) observed peak of each group (-1 none),
    obs (B, J) L2-normalised observed weights, ex (S,) row of obs for each structure -> (S,) cosine (0 without
    fragments, finite gradients)."""
    import torch
    S, J = logits.shape[0], obs.shape[1]
    vm = valid[..., None].expand(-1, -1, NS).reshape(S, -1)
    lg = logits.float().reshape(S, -1).masked_fill(~vm, -1e4)
    p = torch.softmax(lg, -1) * vm
    pu = p.new_zeros(S, jou.shape[1]).scatter_add_(1, u.reshape(S, -1).clamp(min=0).long(), p)
    jj = torch.where(jou >= 0, jou, J).long()
    pj = p.new_zeros(S, J + 1).scatter_add_(1, jj, pu)[:, :J]
    sq = (pj ** 2).sum(1) + (pu ** 2 * (jou < 0)).sum(1)
    return (pj * obs[ex]).sum(1) / torch.sqrt(sq + 1e-12)


def to_torch(b, dev):
    import torch
    return {k: (torch.as_tensor(v).to(dev, non_blocking=True) if isinstance(v, np.ndarray) else v) for k, v in b.items()}


# ---------------------------------------------------------------------------------------------- inference
class G2Model:
    """score(specs, pids) -> (n,) mean cosine over the usable query spectra (NaN for structures without a graph).
    Inference is fp32 unless amp=True, so that GPU and CPU give the same features (as fwd; the k29 gate was scored
    with amp). score_smiles scores structures given as SMILES (generated products)."""

    def __init__(self, path=None, G=None, net=None, cfg=None, device=None, amp=False):
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.amp = bool(amp)
        self.G = G
        if net is None:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            self.cfg = type("G2Cfg", (), dict(ck["cfg"]))
            self.step = ck.get("step", -1)
            net = build_model(self.cfg).to(self.device)
            net.load_state_dict(ck["model"])
        else:
            self.cfg, self.step = cfg or Cfg, -1
        self.net = net.eval()

    def score(self, specs, pids, chunk=256):
        import torch
        n = len(pids)
        qs = []
        for s in specs:
            om, ow = observed(s["mz"], s["it"], s["prec"])
            if len(om):
                qs.append((om, ow, ion_of(s["mode"]), cond_row(adduct_index(s["adduct"]), s["mode"], s.get("ce", np.nan))))
        out = np.full(n, np.nan, np.float32)
        if n == 0:
            return out
        has = np.array([bool(self.G.ok[p]) for p in pids])
        out[has] = 0.0
        if not qs:
            return out
        dev = self.device
        amp = self.amp and dev == "cuda"
        with torch.no_grad():
            for a in range(0, n, chunk):
                ps = [piece(self.G, p) for p in pids[a:a + chunk]]
                b = collate(ps)
                bt = to_torch(b, dev)
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    pre = self.net.encode(bt)
                acc = np.zeros(len(ps))
                for om, ow, ion, cond in qs:
                    jou = jou_array([match(p["umz"], ion, om) if p else None for p in ps])
                    c = torch.as_tensor(cond, device=dev).expand(len(ps), -1)
                    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                        lg = self.net.head(pre, c)
                    obs = torch.as_tensor(np.pad(ow, (0, TOP_PEAKS - len(ow)))[None], device=dev)
                    cos = cosine(lg, bt["valid"], bt["u"], torch.as_tensor(jou, device=dev), obs,
                                 torch.zeros(len(ps), dtype=torch.long, device=dev))
                    acc += cos.float().cpu().numpy()
                sl = out[a:a + chunk]
                sl[has[a:a + chunk]] = (acc / len(qs))[has[a:a + chunk]]
        return out

    def score_smiles(self, specs, smiles, chunk=256):
        """score() for structures given as SMILES (their graphs are built here, in this process)."""
        G0 = self.G
        try:
            self.G = Graphs.build(list(smiles))
            return self.score(specs, np.arange(len(smiles)), chunk)
        finally:
            self.G = G0


# ---------------------------------------------------------------------------------------------- training
def example_batch(C, D, G, NEG, idx, k_neg, prng):
    """Training batch of TrainData rows idx: each spectrum with its truth and k_neg same-formula train structures."""
    pids = D.pid[idx]
    cand = np.concatenate([pids[:, None], NEG(pids, k_neg, prng)], 1)
    B = len(idx)
    obs = np.zeros((B, TOP_PEAKS), np.float32)
    cond = np.zeros((B, N_COND), np.float32)
    has_obs = np.zeros(B, bool)
    pieces, jous = [], []
    ceil, capped = np.zeros(B), np.zeros(B, bool)
    for b, j in enumerate(idx):
        r = D.rows[j]
        mz, it = C.spectrum(r)
        om, ow = observed(mz, it, C.prec[r])
        obs[b, :len(ow)] = ow
        has_obs[b] = len(ow) > 0
        cond[b] = cond_row(D.adduct[j], C.mode[r], C.ce[r])
        ion = ion_of(C.mode[r])
        for t, p in enumerate(cand[b]):
            pc = piece(G, p) if p >= 0 else None
            jo = match(pc["umz"], ion, om) if pc is not None else None
            pieces.append(pc)
            jous.append(jo)
            if t == 0 and pc is not None:
                hit = np.unique(jo[jo >= 0])
                ceil[b] = float(np.sqrt((ow[hit] ** 2).sum())) if len(hit) else 0.0
                capped[b] = pc["capped"]
    bt = collate(pieces, jous)
    s_ok = bt["valid"].any(1).reshape(B, -1) & (cand >= 0)
    bt.update(obs=obs, cond=np.repeat(cond, cand.shape[1], 0), ex=np.repeat(np.arange(B), cand.shape[1]),
              s_ok=s_ok, ex_ok=s_ok[:, 0] & has_obs)
    stats = dict(ceiling=float(ceil[bt["ex_ok"]].mean()) if bt["ex_ok"].any() else 0.0,
                 zero_ceiling=float((ceil[bt["ex_ok"]] == 0).mean()) if bt["ex_ok"].any() else 0.0,
                 truth_ok=float(bt["ex_ok"].mean()), capped=float(capped.mean()))
    return bt, stats


def train(C, D, G, NEG, out_path, budget_secs=6.5 * 3600, log=print, seed=0, track=None, resume=None):
    """Train G2 on the TrainData draw D (fpnet.Cfg v2b settings) with same-formula negatives NEG. Batch i is a fixed
    slice of consecutive epoch draws and its negatives come from a generator seeded with (seed, i); producer w builds
    the steps i with i % workers == w and the loop reads them in step order, so a resumed run sees the same batches as
    an uninterrupted one. Saves the weights (out_path) and the training state (ckpt) every Cfg.save_every steps and at
    the end. Returns the history list."""
    import queue
    import threading

    import torch
    import torch.nn.functional as F

    from .enc3 import FIT_AT, FIT_FROM, fit_horizon

    t0 = time.time()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = build_model().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=Cfg.lr, weight_decay=Cfg.wd)
    total = Cfg.total_steps
    horizon = [total]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / Cfg.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / horizon[0]))))
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    order = []
    while sum(len(o) for o in order) < total * Cfg.batch:
        order.append(D.epoch_index(rng))
    n_draw = len(order[0])
    order = np.concatenate(order)[:total * Cfg.batch]
    cfg_dump = {k: getattr(Cfg, k) for k in dir(Cfg) if not k.startswith("_")}
    hist, step = [], 0
    if resume:
        x = ckpt.load_state(resume, net, opt, sched, scaler, rng, cfg=cfg_dump)
        hist, step, horizon[0] = x["hist"], x["step"], x["horizon"]
        log(f"[g2] resumed from {resume}: step {step:,}, horizon {horizon[0]:,}")
    step0 = step
    log(f"[g2] {len(D):,} training spectra, {total:,} steps of {Cfg.batch} spectra x (1 + {Cfg.k_neg}) structures "
        f"({total * Cfg.batch / max(n_draw, 1):.2f} epoch draws), device {dev}, "
        f"params {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M")

    todo = np.arange(step0, horizon[0])
    W = max(1, Cfg.workers)
    qs = [queue.Queue(maxsize=4) for _ in range(W)]

    def producer(w):
        try:
            for i in todo[w::W]:
                prng = np.random.default_rng([seed, int(i)])
                qs[w].put(example_batch(C, D, G, NEG, order[i * Cfg.batch:(i + 1) * Cfg.batch], Cfg.k_neg, prng))
        except Exception as e:  # the training loop re-raises it
            qs[w].put(e)

    threads = [threading.Thread(target=producer, args=(w,), daemon=True) for w in range(W)]
    for th in threads:
        th.start()

    def save():
        net.eval()
        torch.save(dict(model=net.state_dict(), cfg=cfg_dump, step=step, horizon=horizon[0]), out_path)
        ckpt.save_state(ckpt.state_path(out_path), net, opt, sched, scaler, rng, cfg=cfg_dump, step=step, hist=hist,
                        horizon=horizon[0])
        net.train()

    s_reg = s_rank = 0.0
    s_st = dict(ceiling=0.0, zero_ceiling=0.0, truth_ok=0.0, capped=0.0)
    n_log, te = 0, time.time()
    net.train()
    for i in todo:
        if step >= horizon[0] or time.time() - t0 > budget_secs:
            break
        item = qs[(i - step0) % W].get()
        if isinstance(item, Exception):
            raise item
        b, st = item
        bt = to_torch(b, dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            lg = net.head(net.encode(bt), bt["cond"])
        cos = cosine(lg, bt["valid"], bt["u"], bt["jou"], bt["obs"], bt["ex"]).view(len(b["ex_ok"]), -1)
        ok = bt["ex_ok"]
        if bool(ok.any()):
            reg = (1 - cos[ok, 0]).mean()
            rank = F.cross_entropy((cos[ok] / Cfg.temp).masked_fill(~bt["s_ok"][ok], -1e4),
                                   torch.zeros(int(ok.sum()), dtype=torch.long, device=dev))
            opt.zero_grad(set_to_none=True)
            scaler.scale(reg + rank).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            s_reg += float(reg.detach())
            s_rank += float(rank.detach())
        sched.step()
        step += 1
        for k in s_st:
            s_st[k] += st[k]
        n_log += 1
        if step == FIT_FROM:
            t_fit = time.time()
        if step == FIT_AT and step0 < FIT_FROM:
            horizon[0] = fit_horizon(horizon[0], step, (time.time() - t_fit) / (FIT_AT - FIT_FROM),
                                     budget_secs - (time.time() - t0), log)
        if step % 500 == 0:
            rec = dict(step=step, reg=s_reg / n_log, rank=s_rank / n_log, **{k: v / n_log for k, v in s_st.items()},
                       s_per_step=(time.time() - te) / n_log, secs=time.time() - t0, horizon=horizon[0])
            hist.append(rec)
            log(f"  [g2] step {step}/{horizon[0]} reg {rec['reg']:.4f} rank {rec['rank']:.4f} ceiling "
                f"{rec['ceiling']:.3f} (zero {rec['zero_ceiling']:.3f}, capped {rec['capped']:.3f}) "
                f"{rec['s_per_step']:.3f}s/step ({rec['secs']:.0f}s)")
            if track is not None:
                track.log({f"g2/{k}": v for k, v in rec.items()}, step=step)
            s_reg = s_rank = 0.0
            s_st = {k: 0.0 for k in s_st}
            n_log, te = 0, time.time()
        if step % Cfg.save_every == 0:
            save()
    if time.time() - t0 > budget_secs:
        log("[g2] time budget reached")
    save()
    log(f"[g2] {step:,} steps ({time.time() - t0:.0f}s)")
    return hist

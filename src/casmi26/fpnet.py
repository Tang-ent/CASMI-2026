"""Spectrum -> fingerprint model (peak transformer) trained on train spectra, scored by f.z on pool candidates.

Each spectrum is a sequence of peak tokens [sin(m/z), sin(precursor - m/z), sqrt(intensity)] plus one global token
(precursor mass, adduct, polarity, collision energy, timsTOF flag). Output: one logit z_b per fingerprint bit of the
pool fingerprint (Morgan r=2 of the tautomer-canonical SMILES). A candidate with bits f scores f.z, which equals the
Bernoulli log-likelihood of f up to a candidate-independent constant.

Loss = BCE over bits + in-window softmax over (truth + K pool structures within +-10 ppm of the truth's mass).
"""
import math
import time

import numpy as np

from . import chem, ckpt
from .harness import TIMS_LIBS

ADDUCT_LIST = ["[M+H]+", "[M+NH4]+", "[M+Na]+", "[M+K]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M]+", "[M+2H]2+",
               "[2M+H]+", "[2M+Na]+", "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+C2H4O2-H]-", "[M+Cl]-", "[M]-",
               "[2M-H]-", "<unk>"]
ADDUCT_IX = {a: i for i, a in enumerate(ADDUCT_LIST)}
MAX_PEAKS = 128


class Cfg:
    d = 256
    layers = 6
    heads = 8
    drop = 0.1
    batch = 512
    lr = 5e-4
    wd = 0.01
    epochs = 20
    warmup = 2000
    k_neg = 16
    ppm = 10.0
    rank_w = 1.0
    rank_temp = 4.0
    workers = 3
    # 前処理と timsTOF の印（v2〜。チェックポイントの cfg に入り、推論では FPModel が自分の値を使う）
    peak_floor = 0.0  # 基準ピークに対する強度の床（キャッシュの 0.1% の上に足す。0 = なし）
    max_peaks = MAX_PEAKS
    tims_fixed = None  # None = スペクトルの印のまま、0.0 / 1.0 = 学習でも推論でも常にその値
    # 学習の抽出（v2〜。既定値は k06 と同じ動き）
    max_per_struct = 0  # エポックごとに各構造から引くスペクトルの上限（0 = 全部）
    np_weight = 1  # COCONUT にもある train の構造を 1 エポックに何回分引くか
    neg_train_only = False  # 順位の損失の負例を、窓の中の train の構造（ホールドアウト以外）からだけ引く
    total_steps = 0  # > 0 なら、エポック数をこの総ステップ数から決める


def adduct_index(a):
    return ADDUCT_IX.get(str(a).replace(" ", ""), ADDUCT_IX["<unk>"])


def peak_tokens(mz, it, max_peaks=MAX_PEAKS, floor=0.0):
    """Peaks with it >= floor * max, then the top-`max_peaks` by intensity, m/z sorted, sqrt(it / max)."""
    mz = np.asarray(mz, np.float32)
    it = np.asarray(it, np.float32)
    if floor > 0 and len(it):
        k = it >= floor * it.max()
        mz, it = mz[k], it[k]
    if len(mz) > max_peaks:
        k = np.argpartition(-it, max_peaks - 1)[:max_peaks]
        k.sort()
        mz, it = mz[k], it[k]
    if len(it) and it.max() > 0:
        it = np.sqrt(it / it.max())
    return mz, it


def collate(items, max_peaks=MAX_PEAKS):
    """items: dicts(mz, it, prec, adduct_ix, mode, ce, tims) -> dict of numpy arrays (padded)."""
    n = len(items)
    L = max(1, max(len(x["mz"]) for x in items))
    mz = np.zeros((n, L), np.float32)
    it = np.zeros((n, L), np.float32)
    pad = np.ones((n, L), bool)
    for i, x in enumerate(items):
        k = len(x["mz"])
        mz[i, :k] = x["mz"]
        it[i, :k] = x["it"]
        pad[i, :k] = False
    g = np.array([[x["prec"], x["mode"], x["ce"], x["ce_known"], x["tims"]] for x in items], np.float32)
    ad = np.array([x["adduct_ix"] for x in items], np.int64)
    return dict(mz=mz, it=it, pad=pad, g=g, ad=ad)


def spec_item(mz, it, prec, adduct, mode, ce=np.nan, tims=1.0, floor=0.0, max_peaks=MAX_PEAKS):
    m, i = peak_tokens(mz, it, max_peaks, floor)
    ce_ok = ce is not None and np.isfinite(ce)
    return dict(mz=m, it=i, prec=float(prec), adduct_ix=adduct_index(adduct), mode=float(mode),
                ce=float(ce) / 100.0 if ce_ok else 0.0, ce_known=1.0 if ce_ok else 0.0, tims=float(tims))


# ---------------------------------------------------------------------------------------------- model
def build_model(nbits, cfg=Cfg):
    import torch
    import torch.nn as nn

    class SinEmb(nn.Module):
        def __init__(self, dim, lo=-2.0, hi=3.2):
            super().__init__()
            wav = torch.pow(10.0, torch.linspace(lo, hi, dim // 2))
            self.register_buffer("inv", 2 * math.pi / wav)

        def forward(self, x):
            a = x.unsqueeze(-1) * self.inv
            return torch.cat([torch.sin(a), torch.cos(a)], -1)

    class FPNet(nn.Module):
        def __init__(self):
            super().__init__()
            d = cfg.d
            self.mz = SinEmb(d)
            self.nl = SinEmb(d)
            self.pk = nn.Linear(2 * d + 1, d)
            self.prec = SinEmb(d)
            self.ad = nn.Embedding(len(ADDUCT_LIST), d)
            self.gl = nn.Linear(d + 4, d)
            layer = nn.TransformerEncoderLayer(d, cfg.heads, 4 * d, cfg.drop, activation="gelu", batch_first=True,
                                               norm_first=True)
            self.enc = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d)
            self.head = nn.Sequential(nn.Linear(2 * d, 1024), nn.GELU(), nn.Dropout(cfg.drop), nn.Linear(1024, nbits))

        def forward(self, mz, it, pad, g, ad):
            prec = g[:, 0]
            nl = (prec[:, None] - mz).clamp(min=0)
            p = self.pk(torch.cat([self.mz(mz), self.nl(nl), it.unsqueeze(-1)], -1))
            glob = self.gl(torch.cat([self.prec(prec), g[:, 1:5]], -1)) + self.ad(ad)
            x = torch.cat([glob.unsqueeze(1), p], 1)
            mask = torch.cat([torch.zeros_like(pad[:, :1]), pad], 1)
            x = self.norm(self.enc(x, src_key_padding_mask=mask))
            keep = (~pad).float().unsqueeze(-1)
            mean = (x[:, 1:] * keep).sum(1) / keep.sum(1).clamp(min=1)
            return self.head(torch.cat([x[:, 0], mean], -1))

    return FPNet()


def unpack_bits_torch(packed, nbits):
    """(..., nbytes) uint8 tensor -> (..., nbits) float tensor in {0, 1} (on the same device)."""
    import torch
    shifts = torch.arange(7, -1, -1, device=packed.device, dtype=torch.uint8)
    bits = (packed.unsqueeze(-1) >> shifts) & 1
    return bits.reshape(*packed.shape[:-1], -1)[..., :nbits].float()


# ---------------------------------------------------------------------------------------------- training data
class TrainData:
    """Training spectra = kept train spectra whose structure maps to a pool entry outside the holdout.

    v2 options (Cfg; the defaults reproduce k06): each epoch draws at most `max_per_struct` spectra per structure,
    structures that are also in COCONUT are drawn `np_weight` times (independent draws), and with `neg_train_only`
    the in-window negatives come only from train structures outside the holdout (COCONUT-only structures then never
    appear in training). in_coconut / in_train shape only the training sample here, never a ranking feature."""

    def __init__(self, C, P, holdout_pids, frac=1.0, seed=0):
        pid = P.sid2pid[C.sid]
        hold = np.asarray(sorted(holdout_pids), np.int64)
        ok = C.keep & (pid >= 0) & ~np.isin(pid, hold)
        rows = np.flatnonzero(ok)
        if frac < 1.0:
            rows = np.random.default_rng(seed).choice(rows, int(len(rows) * frac), replace=False)
        self.rows = rows
        self.pid = pid[rows]
        self.C = C
        tims_ids = [C.lib_id[n] for n in TIMS_LIBS if n in C.lib_id]
        self.tims = np.isin(C.lib[rows], tims_ids).astype(np.float32)
        self.adduct = np.array([adduct_index(a) for a in C.adducts], np.int64)[C.adduct[rows]]
        self.is_np = P.in_coconut[self.pid]
        self.by_pid = np.argsort(self.pid, kind="mergesort")  # spectra grouped by structure (for per-epoch draws)
        # +-ppm window of every pool entry (for in-window negatives), in the corrected-mass order
        tol = P.mass * Cfg.ppm * 1e-6
        if Cfg.neg_train_only:
            keep = P.in_train[P.order] & ~np.isin(P.order, hold)
            self.order = P.order[keep]
        else:
            self.order = P.order
        ms = P.mass[self.order]
        self.win_lo = np.searchsorted(ms, P.mass - tol, "left")
        self.win_hi = np.searchsorted(ms, P.mass + tol, "right")

    def __len__(self):
        return len(self.rows)

    def epoch_index(self, rng):
        """Shuffled spectrum indices for one epoch: all spectra (k06), or the capped / NP-weighted draw (v2)."""
        cap, w = int(Cfg.max_per_struct), int(Cfg.np_weight)
        if cap <= 0 and w <= 1:
            return rng.permutation(len(self.rows))
        parts = []
        for k in range(max(w, 1)):
            o = self.by_pid if k == 0 else self.by_pid[self.is_np[self.by_pid]]
            if cap > 0:
                o = o[np.lexsort((rng.random(len(o)), self.pid[o]))]  # random order inside each structure
                p = self.pid[o]
                first = np.r_[True, p[1:] != p[:-1]]
                start = np.maximum.accumulate(np.where(first, np.arange(len(o)), 0))
                o = o[np.arange(len(o)) - start < cap]
            parts.append(o)
        idx = np.concatenate(parts)
        return idx[rng.permutation(len(idx))]

    def items(self, idx):
        C = self.C
        tf = Cfg.tims_fixed
        out = []
        for j in idx:
            r = self.rows[j]
            m, i = peak_tokens(*C.spectrum(r), Cfg.max_peaks, Cfg.peak_floor)
            ce = float(C.ce[r])
            ok = np.isfinite(ce)
            out.append(dict(mz=m, it=i, prec=float(C.prec[r]), adduct_ix=int(self.adduct[j]), mode=float(C.mode[r]),
                            ce=ce / 100.0 if ok else 0.0, ce_known=1.0 if ok else 0.0,
                            tims=float(self.tims[j]) if tf is None else float(tf)))
        return out

    def negatives(self, pids, k, rng):
        """(n, k) pool ids drawn from each truth's mass window (truth excluded; -1 = none)."""
        lo, hi = self.win_lo[pids], self.win_hi[pids]
        size = hi - lo
        pos = lo[:, None] + (rng.random((len(pids), k)) * size[:, None]).astype(np.int64)
        neg = self.order[np.minimum(pos, len(self.order) - 1)]
        neg[(neg == pids[:, None]) | (size[:, None] <= 1)] = -1
        return neg


def _batches(o, batch):
    """Full batches of the (already shuffled) index array o."""
    return [o[i:i + batch] for i in range(0, len(o) - batch + 1, batch)]


# ---------------------------------------------------------------------------------------------- evaluation
def window_mrr(model_fn, C, P, queries, ppm=10.0, k=25, fp=None):
    """MRR@k of the truth among pool candidates within +-ppm of the truth's mass, scored by f.z
    (molecule logits = mean of per-spectrum logits). queries: DataFrame with pid, rows.
    Ties count in the truth's favour (rank = 1 + number of strictly better candidates)."""
    from .harness import query_specs
    fp = P.fp if fp is None else fp
    rr = []
    for q in queries.itertuples():
        z = model_fn(query_specs(C, q.rows))
        cand = P.window(P.mass[q.pid], ppm)
        f = np.unpackbits(fp[cand], axis=1)[:, :len(z)].astype(np.float32)
        s = f @ z
        rank = 1 + int((s > s[np.flatnonzero(cand == q.pid)[0]]).sum())
        rr.append(1.0 / rank if rank <= k else 0.0)
    return float(np.mean(rr)) if rr else float("nan")


def truth_ranks(s, cand, pid):
    """Rank of the truth `pid` among candidates `cand` with scores `s` under three tie rules:
    opt (ties in the truth's favour, as window_mrr), pid (ties by pid, as harness.ranks_of_truth), pess (against)."""
    t = s[np.flatnonzero(cand == pid)[0]]
    better, tie = int((s > t).sum()), (s == t) & (cand != pid)
    return 1 + better, 1 + better + int((tie & (cand < pid)).sum()), 1 + better + int(tie.sum())


def window_ranks(Z, P, pids, ppm=10.0, fp=None, keep=None):
    """Ranks of each truth pid among the pool candidates within +-ppm of its mass, scored by f.Z[i].
    fp: packed fingerprints per pid (default P.fp). keep(cand, pid) -> bool mask: optional candidate filter (M6: the
    same-formula candidates); the truth must pass it. -> DataFrame(n_cand, r_opt, r_pid, r_pess), one row per query."""
    import pandas as pd
    fp = P.fp if fp is None else fp
    out = []
    for z, p in zip(Z, pids):
        cand = P.window(P.mass[p], ppm)
        if keep is not None:
            cand = cand[keep(cand, p)]
        s = np.unpackbits(fp[cand], axis=1)[:, :len(z)].astype(np.float32) @ z
        out.append((len(cand), *truth_ranks(s, cand, p)))
    return pd.DataFrame(out, columns=["n_cand", "r_opt", "r_pid", "r_pess"])


TAN_BINS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.01)


def provenance_coef(Z, P, pids, holdout, ppm=10.0, n_boot=2000, seed=2, score_fn=None):
    """Does the model score structures that only appear as negatives (COCONUT-only) lower than comparable
    never-trained structures (holdout train structures)? For every query window (candidates within +-ppm of the
    truth's mass, truth excluded) the scores f.z are standardised; the result is the weighted mean over bins of the
    Tanimoto similarity to the truth of mean(z | COCONUT-only) - mean(z | holdout), weights = holdout counts per bin,
    with a bootstrap over queries. -> dict(est, se, lo, hi, n_co, n_ho) (k14 section 3; used by k15 for v2b).
    score_fn (plan 10.3, MSAlign gate S): score_fn(i, cand) -> scores of the window candidates of query i, used in
    place of f.Z[i] (Z is then ignored and may be None). Bins, standardisation and bootstrap are unchanged."""
    from .harness import boot_index, ci
    pop = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1)
    hold = np.asarray(sorted(holdout), np.int64)
    nb = len(TAN_BINS) - 1
    S, N = np.zeros((len(pids), 2, nb)), np.zeros((len(pids), 2, nb))
    for i, (z, p) in enumerate(zip(Z, pids) if score_fn is None else ((None, p) for p in pids)):
        c = P.window(P.mass[p], ppm)
        F = P.fp[c]
        if score_fn is None:
            s = np.unpackbits(F, axis=1)[:, :len(z)].astype(np.float32) @ z
        else:
            s = np.asarray(score_fn(i, c))
        if len(s) < 3 or s.std() == 0:
            continue
        zz = (s - s.mean()) / s.std()
        ft = P.fp[p]
        b = np.clip(np.digitize(pop[F & ft].sum(1) / np.maximum(pop[F | ft].sum(1), 1), TAN_BINS) - 1, 0, nb - 1)
        for k, m in enumerate([P.in_coconut[c] & ~P.in_train[c], np.isin(c, hold) & (c != p)]):
            np.add.at(S[i, k], b[m], zz[m])
            np.add.at(N[i, k], b[m], 1)

    def coef(Ss, Ns):
        ok = (Ns[..., 0, :] > 0) & (Ns[..., 1, :] > 0)
        d = np.where(ok, Ss[..., 0, :] / np.maximum(Ns[..., 0, :], 1) - Ss[..., 1, :] / np.maximum(Ns[..., 1, :], 1), 0)
        w = np.where(ok, Ns[..., 1, :], 0)
        return (d * w).sum(-1) / np.maximum(w.sum(-1), 1)

    M = np.stack([np.bincount(r, minlength=len(pids)) for r in boot_index(len(pids), n_boot, seed)]).astype(np.float32)
    out = ci(coef(S.sum(0), N.sum(0)), coef(np.tensordot(M, S, axes=1), np.tensordot(M, N, axes=1)))
    return dict(out, n_co=int(N[:, 0].sum()), n_ho=int(N[:, 1].sum()))


def cfg_of(d):
    """An instance copy of Cfg with the checkpoint's values (does not touch the module-level Cfg)."""
    c = type("CfgView", (), {})()
    for k in dir(Cfg):
        if not k.startswith("_"):
            setattr(c, k, getattr(Cfg, k))
    for k, v in d.items():
        setattr(c, k, v)
    return c


class FPModel:
    """Inference wrapper: molecule logits z (nbits,) from a list of spectra dicts. The preprocessing and the
    timsTOF setting come from the checkpoint's cfg (defaults = k06), stored per instance in `prep`."""

    def __init__(self, path, device=None):
        import torch
        ck = torch.load(path, map_location="cpu", weights_only=False)
        # M6: the checkpoint's cfg lives on the instance (several models in one process, e.g. v2b and fprich);
        # the module-level Cfg is left alone
        self.cfg = cfg_of(ck["cfg"])
        self.nbits, self.epoch = ck["nbits"], ck.get("epoch", -1)
        cfg = ck["cfg"]
        self.prep = dict(floor=cfg.get("peak_floor", 0.0), max_peaks=cfg.get("max_peaks", MAX_PEAKS),
                         tims_fixed=cfg.get("tims_fixed"))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = build_model(self.nbits, self.cfg).to(self.device).eval()
        self.net.load_state_dict(ck["model"])

    def _items(self, specs):
        p = self.prep
        tf = p["tims_fixed"]
        items = [spec_item(s["mz"], s["it"], s["prec"], s["adduct"], s["mode"], s.get("ce", np.nan),
                           s.get("tims", 1.0) if tf is None else tf, p["floor"], p["max_peaks"]) for s in specs]
        return [x for x in items if len(x["mz"])]

    def _forward(self, items):
        import torch
        b = {k: torch.as_tensor(v, device=self.device) for k, v in collate(items).items()}
        with torch.no_grad():
            return self.net(b["mz"], b["it"], b["pad"], b["g"], b["ad"]).float().cpu().numpy()

    def __call__(self, specs):
        items = self._items(specs)
        if not items:
            return np.zeros(self.nbits, np.float32)
        return self._forward(items).mean(0)

    def zs(self, spec_lists, batch=256):
        """Molecule logits for many molecules at once (spectra batched across molecules, sorted by length).
        -> (n_molecules, nbits); zeros for a molecule without usable spectra."""
        items, owner = [], []
        for i, specs in enumerate(spec_lists):
            for x in self._items(specs):
                items.append(x)
                owner.append(i)
        owner = np.asarray(owner, np.int64)
        Z = np.zeros((len(spec_lists), self.nbits), np.float32)
        if not items:
            return Z
        order = np.argsort([len(x["mz"]) for x in items], kind="mergesort")
        out = np.zeros((len(items), self.nbits), np.float32)
        for a in range(0, len(order), batch):
            ix = order[a:a + batch]
            out[ix] = self._forward([items[j] for j in ix])
        np.add.at(Z, owner, out)
        Z /= np.maximum(np.bincount(owner, minlength=len(spec_lists)), 1)[:, None]
        return Z


def load_checked(name="fpnet.pt", min_epoch=10, threads=2):
    """FPModel on CPU from the k06 output; refuses checkpoints before min_epoch (e.g. the 1-epoch pilot).
    torch gets few threads: inference is milliseconds per query and the library search uses numba's cores."""
    import torch
    from . import paths
    torch.set_num_threads(threads)
    path = paths.find(name)
    m = FPModel(path, device="cpu")
    print(f"[fpnet] {path}: epoch {m.epoch}, d {m.cfg.d}, layers {m.cfg.layers}, {m.nbits} bits")
    if m.epoch < min_epoch:
        raise RuntimeError(f"checkpoint epoch {m.epoch} < {min_epoch}: not the full k06 run")
    return m


# ---------------------------------------------------------------------------------------------- training loop
def train(C, P, holdout_pids, val_queries, out_path, frac=1.0, epochs=None, budget_secs=10.5 * 3600,
          extra_val=None, log=print, seed=0, track=None, select="best", eval_every=1, fp=None, nbits=None,
          resume=None, init=None):
    """Train on the non-holdout spectra (all of them, or the v2 per-epoch draw); validate (window MRR on
    val_queries and extra_val) every `eval_every` epochs and on the last one. Returns the history list.
    select="best": out_path = the best checkpoint by val MRR (k06); select="last": out_path = the last epoch (v2,
    so that no evaluation set picks the epoch). The latest epoch is always also written to <out>_last.pt.
    Epochs: Cfg.total_steps > 0 sets them from the epoch size, else `epochs` or Cfg.epochs.
    track: optional casmi26.track.Run; losses every 500 steps and the epoch records are logged to it.
    fp / nbits (M6 k28): packed target fingerprints per pid and their bit count (default P.fp, chem.FP_BITS).
    Every epoch also writes the full training state to <out>_state.pt (ckpt); resume=<that path> continues after its
    epoch with the same cfg (the time budget counts from the resumed start).
    init (M7 k39): a checkpoint whose weights start the network (same nbits); the optimizer, the warmup and the cosine
    start fresh (a warm restart). Ignored when resuming (the state has the weights)."""
    import threading
    import queue

    import torch
    import torch.nn.functional as F

    t0 = time.time()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    D = TrainData(C, P, holdout_pids, frac=frac, seed=seed)
    nbits = nbits or chem.FP_BITS
    fp = P.fp if fp is None else fp
    net = build_model(nbits).to(dev)
    if init and not resume:
        ck0 = torch.load(init, map_location="cpu", weights_only=False)
        assert ck0["nbits"] == nbits, f"init {init} has {ck0['nbits']} bits, training {nbits}"
        net.load_state_dict(ck0["model"])
        log(f"[fpnet] warm start from {init} (epoch {ck0.get('epoch')})")
    opt = torch.optim.AdamW(net.parameters(), lr=Cfg.lr, weight_decay=Cfg.wd)
    probe = D.epoch_index(np.random.default_rng(seed + 12345))  # epoch size (the draw sizes are fixed)
    steps_per_epoch = max(1, len(probe) // Cfg.batch)
    epochs = math.ceil(Cfg.total_steps / steps_per_epoch) if Cfg.total_steps > 0 else (epochs or Cfg.epochs)
    log(f"[fpnet] epoch draw: {len(probe):,} of {len(D):,} spectra, NP share {D.is_np[probe].mean():.3f} "
        f"(all spectra {D.is_np.mean():.3f}); cap {Cfg.max_per_struct}, NP weight {Cfg.np_weight}, "
        f"negatives from {'train only' if Cfg.neg_train_only else 'the pool'}, tims {Cfg.tims_fixed}, "
        f"floor {Cfg.peak_floor}, max peaks {Cfg.max_peaks}")
    del probe
    total = max(1, steps_per_epoch * epochs)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / Cfg.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    fp_gpu = torch.as_tensor(fp, device=dev)
    log(f"[fpnet] {len(D):,} training spectra, {steps_per_epoch:,} steps/epoch, {epochs} epochs, device {dev}, "
        f"params {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M")

    def producer(batches, qout, seed_):
        prng = np.random.default_rng(seed_)  # numpy の Generator はスレッド間で共有しない
        for idx in batches:
            pids = D.pid[idx]
            neg = D.negatives(pids, Cfg.k_neg, prng)
            qout.put((collate(D.items(idx)), pids, neg))
        qout.put(None)

    hist, best, gstep, ep0 = [], -1.0, 0, 0  # gstep: 全エポック通しのステップ数（wandb の step）
    cfg_dump = {k: getattr(Cfg, k) for k in dir(Cfg) if not k.startswith("_")}
    if resume:
        x = ckpt.load_state(resume, net, opt, sched, scaler, rng, cfg=dict(cfg_dump, nbits=nbits))
        hist, best, gstep, ep0 = x["hist"], x["best"], x["gstep"], x["epoch"] + 1
        log(f"[fpnet] resumed from {resume}: epoch {x['epoch']} done, step {gstep:,}")
    for ep in range(ep0, epochs):
        net.train()
        batches = _batches(D.epoch_index(rng), Cfg.batch)
        qb = queue.Queue(maxsize=8)
        # 前処理（Python のピーク切り出し）を複数スレッドで並行させる
        chunks = [batches[i::Cfg.workers] for i in range(Cfg.workers)]
        threads = [threading.Thread(target=producer, args=(c, qb, seed * 1000 + ep * 10 + i), daemon=True)
                   for i, c in enumerate(chunks)]
        for th in threads:
            th.start()
        done, n, te = 0, 0, time.time()
        s_bce, s_rank = torch.zeros((), device=dev), torch.zeros((), device=dev)  # GPU 同期を避けてテンソルで足し込む
        while done < len(threads):
            item = qb.get()
            if item is None:
                done += 1
                continue
            b, pids, neg = item
            bt = {k: torch.as_tensor(v).to(dev, non_blocking=True) for k, v in b.items()}
            pids_t = torch.as_tensor(pids, device=dev)
            neg_t = torch.as_tensor(neg, device=dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                z = net(bt["mz"], bt["it"], bt["pad"], bt["g"], bt["ad"])
            z = z.float()
            f_true = unpack_bits_torch(fp_gpu[pids_t], nbits)
            bce = F.binary_cross_entropy_with_logits(z, f_true)
            f_neg = unpack_bits_torch(fp_gpu[neg_t.clamp(min=0)], nbits)            # (n, k, bits)
            s = torch.cat([(f_true * z).sum(-1, keepdim=True), torch.einsum("nkb,nb->nk", f_neg, z)], 1)
            s = s.masked_fill(torch.cat([torch.zeros_like(neg_t[:, :1], dtype=torch.bool), neg_t < 0], 1), -1e4)
            rank = F.cross_entropy(s / Cfg.rank_temp, torch.zeros(len(s), dtype=torch.long, device=dev))
            loss = bce + Cfg.rank_w * rank
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            n += 1
            gstep += 1
            s_bce += bce.detach()
            s_rank += rank.detach()
            if n % 500 == 0:
                mb, mr, sps = float(s_bce) / n, float(s_rank) / n, (time.time() - te) / n
                log(f"  ep {ep} step {n}/{steps_per_epoch} bce {mb:.4f} rank {mr:.4f} "
                    f"{sps:.3f}s/step ({time.time() - t0:.0f}s)")
                if track is not None:
                    track.log({"train/bce": mb, "train/rank": mr, "train/lr": sched.get_last_lr()[0],
                               "train/s_per_step": sps, "epoch": ep + n / max(steps_per_epoch, 1)}, step=gstep)
            if time.time() - t0 > budget_secs:
                break
        net.eval()
        ck = dict(model=net.state_dict(), nbits=nbits, cfg=cfg_dump, epoch=ep)
        torch.save(ck, out_path.replace(".pt", "_last.pt"))
        out_of_time = time.time() - t0 > budget_secs
        if select == "last":
            torch.save(ck, out_path)
        rec = dict(epoch=ep, steps=n, bce=float(s_bce) / max(n, 1), rank=float(s_rank) / max(n, 1),
                   epoch_secs=time.time() - te)
        if (ep + 1) % eval_every == 0 or ep == epochs - 1 or out_of_time:
            wrap = _Wrap(net, dev, nbits)
            rec["val_mrr"] = v = window_mrr(wrap, C, P, val_queries, fp=fp)
            if extra_val is not None:
                for name, qx in extra_val.items():
                    rec[f"{name}_mrr"] = window_mrr(wrap, C, P, qx, fp=fp)
            if select == "best" and v > best:
                best = v
                torch.save(ck, out_path)
        hist.append(rec)
        ckpt.save_state(ckpt.state_path(out_path), net, opt, sched, scaler, rng, cfg=dict(cfg_dump, nbits=nbits),
                        epoch=ep, gstep=gstep, hist=hist, best=best)
        if track is not None:
            track.log({f"epoch/{k}": x for k, x in rec.items()}, step=gstep)
            ev = [r for r in hist if "val_mrr" in r]
            if ev:
                track.summary({"best_val_mrr": max(r["val_mrr"] for r in ev),
                               "best_epoch": max(ev, key=lambda r: r["val_mrr"])["epoch"]})
        log(f"[fpnet] epoch {ep}: {rec} ({time.time() - t0:.0f}s)")
        if out_of_time:
            log("[fpnet] time budget reached")
            break
    return hist


class _Wrap:
    """FPModel-like callable around an in-memory network (validation during training)."""

    def __init__(self, net, device, nbits):
        self.net, self.device, self.nbits = net, device, nbits
        self.prep = dict(floor=Cfg.peak_floor, max_peaks=Cfg.max_peaks, tims_fixed=Cfg.tims_fixed)

    _items = FPModel._items
    _forward = FPModel._forward
    __call__ = FPModel.__call__
    zs = FPModel.zs

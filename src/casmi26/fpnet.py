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

from . import chem
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


def adduct_index(a):
    return ADDUCT_IX.get(str(a).replace(" ", ""), ADDUCT_IX["<unk>"])


def peak_tokens(mz, it, max_peaks=MAX_PEAKS):
    """Top-`max_peaks` peaks by intensity, m/z sorted, sqrt(it / max)."""
    mz = np.asarray(mz, np.float32)
    it = np.asarray(it, np.float32)
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


def spec_item(mz, it, prec, adduct, mode, ce=np.nan, tims=1.0):
    m, i = peak_tokens(mz, it)
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
    """Training spectra = kept train spectra whose structure maps to a pool entry outside the holdout."""

    def __init__(self, C, P, holdout_pids, frac=1.0, seed=0):
        pid = P.sid2pid[C.sid]
        ok = C.keep & (pid >= 0) & ~np.isin(pid, np.asarray(list(holdout_pids), np.int64))
        rows = np.flatnonzero(ok)
        if frac < 1.0:
            rows = np.random.default_rng(seed).choice(rows, int(len(rows) * frac), replace=False)
        self.rows = rows
        self.pid = pid[rows]
        self.C = C
        tims_ids = [C.lib_id[n] for n in TIMS_LIBS if n in C.lib_id]
        self.tims = np.isin(C.lib[rows], tims_ids).astype(np.float32)
        self.adduct = np.array([adduct_index(a) for a in C.adducts], np.int64)[C.adduct[rows]]
        # +-ppm window of every pool entry (for in-window negatives), in the corrected-mass order
        tol = P.mass * Cfg.ppm * 1e-6
        self.order = P.order
        self.win_lo = np.searchsorted(P.mass_sorted, P.mass - tol, "left")
        self.win_hi = np.searchsorted(P.mass_sorted, P.mass + tol, "right")

    def __len__(self):
        return len(self.rows)

    def items(self, idx):
        C = self.C
        out = []
        for j in idx:
            r = self.rows[j]
            m, i = peak_tokens(*C.spectrum(r))
            ce = float(C.ce[r])
            ok = np.isfinite(ce)
            out.append(dict(mz=m, it=i, prec=float(C.prec[r]), adduct_ix=int(self.adduct[j]), mode=float(C.mode[r]),
                            ce=ce / 100.0 if ok else 0.0, ce_known=1.0 if ok else 0.0, tims=float(self.tims[j])))
        return out

    def negatives(self, pids, k, rng):
        """(n, k) pool ids drawn from each truth's mass window (truth excluded; -1 = none)."""
        lo, hi = self.win_lo[pids], self.win_hi[pids]
        size = hi - lo
        pos = lo[:, None] + (rng.random((len(pids), k)) * size[:, None]).astype(np.int64)
        neg = self.order[np.minimum(pos, len(self.order) - 1)]
        neg[(neg == pids[:, None]) | (size[:, None] <= 1)] = -1
        return neg


def _batches(n, batch, rng):
    o = rng.permutation(n)
    return [o[i:i + batch] for i in range(0, n - batch + 1, batch)]


# ---------------------------------------------------------------------------------------------- evaluation
def window_mrr(model_fn, C, P, queries, ppm=10.0, k=25):
    """MRR@k of the truth among pool candidates within +-ppm of the truth's mass, scored by f.z
    (molecule logits = mean of per-spectrum logits). queries: DataFrame with pid, rows."""
    from .harness import query_specs
    rr = []
    for q in queries.itertuples():
        z = model_fn(query_specs(C, q.rows))
        cand = P.window(P.mass[q.pid], ppm)
        f = np.unpackbits(P.fp[cand], axis=1)[:, :len(z)].astype(np.float32)
        s = f @ z
        rank = 1 + int((s > s[np.flatnonzero(cand == q.pid)[0]]).sum())
        rr.append(1.0 / rank if rank <= k else 0.0)
    return float(np.mean(rr)) if rr else float("nan")


class FPModel:
    """Inference wrapper: molecule logits z (nbits,) from a list of spectra dicts."""

    def __init__(self, path, device=None):
        import torch
        ck = torch.load(path, map_location="cpu", weights_only=False)
        for k, v in ck["cfg"].items():
            setattr(Cfg, k, v)
        self.nbits, self.epoch = ck["nbits"], ck.get("epoch", -1)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = build_model(self.nbits).to(self.device).eval()
        self.net.load_state_dict(ck["model"])

    def __call__(self, specs):
        import torch
        items = [spec_item(s["mz"], s["it"], s["prec"], s["adduct"], s["mode"], s.get("ce", np.nan),
                           s.get("tims", 1.0)) for s in specs]
        items = [x for x in items if len(x["mz"])]
        if not items:
            return np.zeros(self.nbits, np.float32)
        b = {k: torch.as_tensor(v, device=self.device) for k, v in collate(items).items()}
        with torch.no_grad():
            z = self.net(b["mz"], b["it"], b["pad"], b["g"], b["ad"]).float()
        return z.mean(0).cpu().numpy()


def load_checked(name="fpnet.pt", min_epoch=10, threads=2):
    """FPModel on CPU from the k06 output; refuses checkpoints before min_epoch (e.g. the 1-epoch pilot).
    torch gets few threads: inference is milliseconds per query and the library search uses numba's cores."""
    import torch
    from . import paths
    torch.set_num_threads(threads)
    path = paths.find(name)
    m = FPModel(path, device="cpu")
    print(f"[fpnet] {path}: epoch {m.epoch}, d {Cfg.d}, layers {Cfg.layers}, {m.nbits} bits")
    if m.epoch < min_epoch:
        raise RuntimeError(f"checkpoint epoch {m.epoch} < {min_epoch}: not the full k06 run")
    return m


# ---------------------------------------------------------------------------------------------- training loop
def train(C, P, holdout_pids, val_queries, out_path, frac=1.0, epochs=None, budget_secs=10.5 * 3600,
          extra_val=None, log=print, seed=0, track=None):
    """Train on all non-holdout spectra; validate (window MRR on val_queries) and checkpoint every epoch.
    Returns the history list. The best checkpoint (by val MRR) is written to out_path.
    track: optional casmi26.track.Run; losses every 500 steps and the epoch records are logged to it."""
    import threading
    import queue

    import torch
    import torch.nn.functional as F

    t0 = time.time()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    D = TrainData(C, P, holdout_pids, frac=frac, seed=seed)
    nbits = chem.FP_BITS
    net = build_model(nbits).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=Cfg.lr, weight_decay=Cfg.wd)
    epochs = epochs or Cfg.epochs
    steps_per_epoch = len(D) // Cfg.batch
    total = max(1, steps_per_epoch * epochs)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / Cfg.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    fp_gpu = torch.as_tensor(P.fp, device=dev)
    log(f"[fpnet] {len(D):,} training spectra, {steps_per_epoch:,} steps/epoch, {epochs} epochs, device {dev}, "
        f"params {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M")

    def producer(batches, qout, seed_):
        prng = np.random.default_rng(seed_)  # numpy の Generator はスレッド間で共有しない
        for idx in batches:
            pids = D.pid[idx]
            neg = D.negatives(pids, Cfg.k_neg, prng)
            qout.put((collate(D.items(idx)), pids, neg))
        qout.put(None)

    hist, best, gstep = [], -1.0, 0  # gstep: 全エポック通しのステップ数（wandb の step）
    cfg_dump = {k: getattr(Cfg, k) for k in dir(Cfg) if not k.startswith("_")}
    for ep in range(epochs):
        net.train()
        batches = _batches(len(D), Cfg.batch, rng)
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
        wrap = _Wrap(net, dev, nbits)
        v = window_mrr(wrap, C, P, val_queries)
        rec = dict(epoch=ep, steps=n, bce=float(s_bce) / max(n, 1), rank=float(s_rank) / max(n, 1), val_mrr=v,
                   epoch_secs=time.time() - te)
        if extra_val is not None:
            for name, qx in extra_val.items():
                rec[f"{name}_mrr"] = window_mrr(wrap, C, P, qx)
        hist.append(rec)
        if v > best:
            best = v
            torch.save(ck, out_path)
        if track is not None:
            track.log({f"epoch/{k}": x for k, x in rec.items()}, step=gstep)
            track.summary({"best_val_mrr": best, "best_epoch": max(hist, key=lambda r: r["val_mrr"])["epoch"]})
        log(f"[fpnet] epoch {ep}: {rec} ({time.time() - t0:.0f}s)")
        if time.time() - t0 > budget_secs:
            log("[fpnet] time budget reached")
            break
    return hist


class _Wrap:
    """FPModel-like callable around an in-memory network (validation during training)."""

    def __init__(self, net, device, nbits):
        self.net, self.device, self.nbits = net, device, nbits

    __call__ = FPModel.__call__

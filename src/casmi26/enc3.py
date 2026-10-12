"""Formula-annotated spectrum -> fingerprint encoder (M6 P0-1, k27_enc3; MIST-like).

Same task and target as FPNet v2b (Morgan r=2 2048 logits z, candidate score f.z), but every peak token also carries
the subformula of the precursor ion that explains it (casmi26.subform): element counts of the fragment and of the
neutral loss, the ppm error and an "assigned" flag. Attention gets a per-head bias from the element-count difference
of every pair of assigned peaks (MIST's pairwise neutral losses). The global token also sees the precursor ion formula.

The annotation depends on the candidate's formula, so a query has one z per formula in its window (z_F); a candidate is
scored with its own formula's z. Training annotates each spectrum with the truth's formula, and the in-window
negatives of the rank loss are train structures (outside the holdout) with the same formula (the encoder is conditioned
on the formula, so other formulas would need their own forward pass; k18: window negatives were 74% isomers anyway).

Everything else is the v2b recipe (fpnet.TrainData: holdout, NP x 3, no per-structure cap, timsTOF flag 0; BCE + rank
loss; fixed total steps; the last checkpoint).
"""
import math
import time

import numpy as np

from . import chem, ckpt, subform
from .fpnet import ADDUCT_LIST, MAX_PEAKS, _batches, adduct_index, peak_tokens, truth_ranks, unpack_bits_torch

NE = subform.NE


class Cfg:
    d = 256
    layers = 6
    heads = 8
    drop = 0.1
    batch = 512
    lr = 5e-4
    wd = 0.01
    warmup = 2000
    total_steps = 60000
    k_neg = 16
    rank_w = 1.0
    rank_temp = 4.0
    workers = 3
    max_peaks = MAX_PEAKS
    peak_floor = 0.0
    tims_fixed = 0.0
    pair_hidden = 32
    ppm = subform.PPM
    abs_da = subform.ABS_DA
    max_steps_per_epoch = 0  # > 0: stop each epoch early (smoke test only)


def cfg_of(d):
    c = type("Enc3CfgView", (), {})()
    for k in dir(Cfg):
        if not k.startswith("_"):
            setattr(c, k, getattr(Cfg, k))
    for k, v in d.items():
        setattr(c, k, v)
    return c


# ---------------------------------------------------------------------------------------------- items
def spec_item(mz, it, prec, adduct, mode, ce, tims, counts, supported, cfg=Cfg):
    """One spectrum annotated with the neutral formula `counts` (supported = all elements in subform.ELEM)."""
    m, i = peak_tokens(mz, it, cfg.max_peaks, cfg.peak_floor)
    ce_ok = ce is not None and np.isfinite(ce)
    ion, sign = subform.ion_counts(counts, adduct) if supported else (None, 0)
    fc, err, ok = subform.annotate(m, ion, sign, cfg.ppm, cfg.abs_da)
    if ion is not None and len(m):
        lc = (ion[None, :] - fc).astype(np.int16)
        lc[~ok] = 0
    else:
        lc = np.zeros_like(fc)
    return dict(mz=m, it=i, prec=float(prec), adduct_ix=adduct_index(adduct), mode=float(mode),
                ce=float(ce) / 100.0 if ce_ok else 0.0, ce_known=1.0 if ce_ok else 0.0, tims=float(tims),
                fc=fc, lc=lc, err=err, asg=ok,
                ion=(ion if ion is not None else np.zeros(NE, np.int64)).astype(np.int16), ann=ion is not None)


def collate(items):
    n = len(items)
    L = max(1, max(len(x["mz"]) for x in items))
    mz = np.zeros((n, L), np.float32)
    it = np.zeros((n, L), np.float32)
    pad = np.ones((n, L), bool)
    fc = np.zeros((n, L, NE), np.int16)
    lc = np.zeros((n, L, NE), np.int16)
    err = np.zeros((n, L), np.float32)
    asg = np.zeros((n, L), bool)
    for j, x in enumerate(items):
        k = len(x["mz"])
        mz[j, :k], it[j, :k], pad[j, :k] = x["mz"], x["it"], False
        fc[j, :k], lc[j, :k], err[j, :k], asg[j, :k] = x["fc"], x["lc"], x["err"], x["asg"]
    g = np.array([[x["prec"], x["mode"], x["ce"], x["ce_known"], x["tims"]] for x in items], np.float32)
    ad = np.array([x["adduct_ix"] for x in items], np.int64)
    ion = np.stack([x["ion"] for x in items]).astype(np.int16)
    return dict(mz=mz, it=it, pad=pad, g=g, ad=ad, fc=fc, lc=lc, err=err, asg=asg, ion=ion)


# ---------------------------------------------------------------------------------------------- model
def build_model(nbits, cfg=Cfg):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class SinEmb(nn.Module):
        def __init__(self, dim, lo=-2.0, hi=3.2):
            super().__init__()
            wav = torch.pow(10.0, torch.linspace(lo, hi, dim // 2))
            self.register_buffer("inv", 2 * math.pi / wav)

        def forward(self, x):
            a = x.unsqueeze(-1) * self.inv
            return torch.cat([torch.sin(a), torch.cos(a)], -1)

    class Block(nn.Module):
        """Pre-norm transformer layer whose attention takes an additive (B, heads, L, L) bias."""

        def __init__(self, d, h, drop):
            super().__init__()
            self.h, self.p = h, drop
            self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
            self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, d))
            self.drop = nn.Dropout(drop)

        def forward(self, x, bias):
            B, L, d = x.shape
            q, k, v = self.qkv(self.ln1(x)).view(B, L, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype),
                                               dropout_p=self.p if self.training else 0.0)
            x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, L, d)))
            return x + self.drop(self.ff(self.ln2(x)))

    class Enc3(nn.Module):
        def __init__(self):
            super().__init__()
            d = cfg.d
            self.mz, self.nl, self.prec = SinEmb(d), SinEmb(d), SinEmb(d)
            self.pk = nn.Linear(2 * d + 1, d)
            self.fm = nn.Linear(2 * NE + 2, d)
            self.ad = nn.Embedding(len(ADDUCT_LIST), d)
            self.gl = nn.Linear(d + 4, d)
            self.ionf = nn.Linear(NE + 1, d)
            self.pair = nn.Sequential(nn.Linear(NE, cfg.pair_hidden), nn.GELU(), nn.Linear(cfg.pair_hidden, cfg.heads))
            self.blocks = nn.ModuleList([Block(d, cfg.heads, cfg.drop) for _ in range(cfg.layers)])
            self.norm = nn.LayerNorm(d)
            self.head = nn.Sequential(nn.Linear(2 * d, 1024), nn.GELU(), nn.Dropout(cfg.drop), nn.Linear(1024, nbits))

        def forward(self, mz, it, pad, g, ad, fc, lc, err, asg, ion):
            prec = g[:, 0]
            nl = (prec[:, None] - mz).clamp(min=0)
            a = asg.float()
            fcf = fc.float()
            ff = torch.cat([torch.log1p(fcf), torch.log1p(lc.float()), (err / 10.0).unsqueeze(-1), a.unsqueeze(-1)], -1)
            p = self.pk(torch.cat([self.mz(mz), self.nl(nl), it.unsqueeze(-1)], -1)) + self.fm(ff)
            ionf = ion.float()
            ann = (ionf.sum(-1, keepdim=True) > 0).float()
            glob = self.gl(torch.cat([self.prec(prec), g[:, 1:5]], -1)) + self.ad(ad) + \
                self.ionf(torch.cat([torch.log1p(ionf), ann], -1))
            x = torch.cat([glob.unsqueeze(1), p], 1)
            B, L = x.shape[:2]
            diff = fcf[:, :, None, :] - fcf[:, None, :, :]
            pb = self.pair(torch.sign(diff) * torch.log1p(diff.abs()))  # (B, L-1, L-1, heads)
            pb = (pb * (a[:, :, None] * a[:, None, :]).unsqueeze(-1)).permute(0, 3, 1, 2)
            bias = torch.zeros(B, pb.shape[1], L, L, device=x.device, dtype=pb.dtype)
            bias[:, :, 1:, 1:] = pb
            keypad = torch.cat([torch.zeros_like(pad[:, :1]), pad], 1)
            bias = bias.masked_fill(keypad[:, None, None, :], -1e4)
            for blk in self.blocks:
                x = blk(x, bias)
            x = self.norm(x)
            keep = (~pad).float().unsqueeze(-1)
            mean = (x[:, 1:] * keep).sum(1) / keep.sum(1).clamp(min=1)
            return self.head(torch.cat([x[:, 0], mean], -1))

    return Enc3()


_KEYS = ["mz", "it", "pad", "g", "ad", "fc", "lc", "err", "asg", "ion"]


def forward(net, batch, device):
    import torch
    b = {k: torch.as_tensor(batch[k]).to(device, non_blocking=True) for k in _KEYS}
    return net(*[b[k] for k in _KEYS])


# ---------------------------------------------------------------------------------------------- training data
class SameFormulaNegatives:
    """In-window negatives with the truth's formula: train structures outside the holdout, grouped by formula id."""

    def __init__(self, P, FT, holdout_pids):
        hold = np.asarray(sorted(holdout_pids), np.int64)
        ok = P.in_train[:len(FT.fid)] & ~np.isin(np.arange(len(FT.fid)), hold) & (FT.fid >= 0)
        cand = np.flatnonzero(ok)
        self.order = cand[np.argsort(FT.fid[cand], kind="mergesort")]
        fs = FT.fid[self.order]
        self.lo = np.searchsorted(fs, FT.fid, "left")
        self.hi = np.searchsorted(fs, FT.fid, "right")
        self.hi[FT.fid < 0] = self.lo[FT.fid < 0]

    def __call__(self, pids, k, rng):
        lo, hi = self.lo[pids], self.hi[pids]
        size = hi - lo
        pos = lo[:, None] + (rng.random((len(pids), k)) * size[:, None]).astype(np.int64)
        neg = self.order[np.minimum(pos, len(self.order) - 1)] if len(self.order) else np.full((len(pids), k), -1)
        neg[(neg == pids[:, None]) | (size[:, None] <= 1)] = -1
        return neg


def train_items(D, FT, idx, cfg=Cfg):
    """Annotated items of TrainData rows idx (formula of the row's structure, adduct of the row)."""
    C = D.C
    out = []
    for j in idx:
        r = D.rows[j]
        fid = FT.fid[D.pid[j]]
        counts, sup = (FT.counts[fid], bool(FT.supported[fid])) if fid >= 0 else (np.zeros(NE, np.int64), False)
        mz, it = C.spectrum(r)
        out.append(spec_item(mz, it, float(C.prec[r]), C.adducts[C.adduct[r]], float(C.mode[r]), float(C.ce[r]),
                             cfg.tims_fixed if cfg.tims_fixed is not None else float(D.tims[j]), counts, sup, cfg))
    return out


# ---------------------------------------------------------------------------------------------- inference
class Enc3Model:
    """z per (spectra, formula id): z_F = mean over the spectra of the logits with the spectra annotated by F."""

    def __init__(self, path=None, FT=None, net=None, cfg=None, nbits=None, device=None):
        import torch
        self.FT = FT
        if net is not None:
            self.net, self.cfg, self.nbits, self.epoch = net, cfg or Cfg, nbits, -1
            self.device = device
        else:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            self.cfg = cfg_of(ck["cfg"])
            self.nbits, self.epoch = ck["nbits"], ck.get("epoch", -1)
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.net = build_model(self.nbits, self.cfg).to(self.device).eval()
            self.net.load_state_dict(ck["model"])

    def _items(self, specs, fid):
        FT, c = self.FT, self.cfg
        counts, sup = (FT.counts[fid], bool(FT.supported[fid])) if fid >= 0 else (np.zeros(NE, np.int64), False)
        tf = c.tims_fixed
        items = [spec_item(s["mz"], s["it"], s["prec"], s["adduct"], s["mode"], s.get("ce", np.nan),
                           s.get("tims", 1.0) if tf is None else tf, counts, sup, c) for s in specs]
        return [x for x in items if len(x["mz"])]

    def z_many(self, jobs, batch=256):
        """jobs: list of (specs, fid) -> (len(jobs), nbits) float32 (zeros for a job without usable spectra)."""
        import torch
        items, owner = [], []
        for i, (specs, fid) in enumerate(jobs):
            for x in self._items(specs, int(fid)):
                items.append(x)
                owner.append(i)
        owner = np.asarray(owner, np.int64)
        Z = np.zeros((len(jobs), self.nbits), np.float32)
        if not items:
            return Z
        order = np.argsort([len(x["mz"]) for x in items], kind="mergesort")
        out = np.zeros((len(items), self.nbits), np.float32)
        was = self.net.training
        self.net.eval()
        with torch.no_grad():
            for a in range(0, len(order), batch):
                ix = order[a:a + batch]
                out[ix] = forward(self.net, collate([items[j] for j in ix]), self.device).float().cpu().numpy()
        self.net.train(was)
        np.add.at(Z, owner, out)
        Z /= np.maximum(np.bincount(owner, minlength=len(jobs)), 1)[:, None]
        return Z

    def zs_by_formula(self, specs, fids):
        """{fid: z} for the distinct formula ids."""
        u = [int(f) for f in dict.fromkeys(int(x) for x in fids)]
        Z = self.z_many([(specs, f) for f in u])
        return dict(zip(u, Z))


# ---------------------------------------------------------------------------------------------- evaluation
def window_ranks(model, P, FT, spec_lists, pids, ppm=10.0, fp=None, batch_queries=256):
    """Truth ranks per query (window = pool structures within +-ppm of the truth's mass, as fpnet.window_ranks):
    same-formula candidates scored with the truth's formula's z (n_same, r_*), and all candidates with each
    candidate's own formula's z (n_cand, a_*). -> DataFrame, one row per query."""
    import pandas as pd
    fp = P.fp if fp is None else fp
    rows = []
    for a in range(0, len(pids), batch_queries):
        jobs, slots = [], []
        part = list(range(a, min(a + batch_queries, len(pids))))
        for i in part:
            cand = P.window(P.mass[pids[i]], ppm)
            fids = FT.fid[cand]
            u = list(dict.fromkeys(int(f) for f in fids))
            slots.append((cand, fids, {f: len(jobs) + k for k, f in enumerate(u)}))
            jobs.extend((spec_lists[i], f) for f in u)
        Z = model.z_many(jobs)
        for i, (cand, fids, pos) in zip(part, slots):
            p = int(pids[i])
            ft = int(FT.fid[p])
            F = np.unpackbits(fp[cand], axis=1)[:, :Z.shape[1]].astype(np.float32)
            s_all = np.einsum("nb,nb->n", F, Z[[pos[int(f)] for f in fids]])
            same = fids == ft
            r_same = truth_ranks(s_all[same], cand[same], p)
            r_all = truth_ranks(s_all, cand, p)
            rows.append((len(cand), int(same.sum()), *r_same, *r_all))
    return pd.DataFrame(rows, columns=["n_cand", "n_same", "r_opt", "r_pid", "r_pess", "a_opt", "a_pid", "a_pess"])


def same_formula_ranks(Z, P, FT, pids, ppm=10.0, fp=None):
    """fpnet.window_ranks restricted to the truth's formula (for v2b on the same candidate sets)."""
    from .fpnet import window_ranks as wr
    return wr(Z, P, pids, ppm, fp=fp, keep=lambda cand, p: FT.fid[cand] == FT.fid[p])


# ---------------------------------------------------------------------------------------------- training loop
def train(C, P, FT, holdout_pids, out_path, val=None, budget_secs=11 * 3600, log=print, seed=0, track=None,
          eval_every=2, resume=None):
    """v2b recipe with annotated items and same-formula negatives. val: {name: (spec_lists, pids)} for the
    monitoring MRR (same-formula, r_pid) every `eval_every` epochs and at the end; never used to pick the
    checkpoint (out_path = the last epoch). Returns the history list.
    Every epoch also writes the full training state to <out>_state.pt (ckpt); resume=<that path> continues after its
    epoch with the same cfg and horizon (fit_horizon is not run again)."""
    import queue
    import threading

    import torch
    import torch.nn.functional as F

    from .fpnet import Cfg as FCfg, TrainData

    t0 = time.time()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # the training draw of v2b: no per-structure cap, NP x 3, timsTOF flag fixed (fpnet.Cfg drives TrainData)
    FCfg.max_per_struct, FCfg.np_weight, FCfg.neg_train_only = 0, 3, True
    D = TrainData(C, P, holdout_pids, seed=seed)
    NEG = SameFormulaNegatives(P, FT, holdout_pids)
    nbits = chem.FP_BITS
    net = build_model(nbits).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=Cfg.lr, weight_decay=Cfg.wd)
    probe = D.epoch_index(np.random.default_rng(seed + 12345))
    steps_per_epoch = max(1, len(probe) // Cfg.batch)
    epochs = max(1, math.ceil(Cfg.total_steps / steps_per_epoch))
    has_neg = (NEG.hi - NEG.lo)[D.pid[probe]] > 1
    log(f"[enc3] epoch draw {len(probe):,} of {len(D):,} spectra, NP share {D.is_np[probe].mean():.3f}; "
        f"same-formula negatives available for {has_neg.mean():.3f} of the draw")
    del probe
    total = max(1, steps_per_epoch * epochs)
    horizon = [total]  # shortened once (fit_horizon) when the projected time exceeds the budget
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / Cfg.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / horizon[0]))))
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    fp_gpu = torch.as_tensor(P.fp, device=dev)
    log(f"[enc3] {steps_per_epoch:,} steps/epoch, {epochs} epochs ({total:,} steps), device {dev}, "
        f"params {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M")

    def producer(batches, qout, seed_):
        prng = np.random.default_rng(seed_)
        for idx in batches:
            pids = D.pid[idx]
            items = train_items(D, FT, idx)
            qout.put((collate(items), pids, NEG(pids, Cfg.k_neg, prng), float(np.mean([x["asg"].mean() if len(x["asg"])
                                                                                       else 0.0 for x in items]))))
        qout.put(None)

    cfg_dump = {k: getattr(Cfg, k) for k in dir(Cfg) if not k.startswith("_")}
    hist, gstep, ep0 = [], 0, 0
    if resume:
        x = ckpt.load_state(resume, net, opt, sched, scaler, rng, cfg=cfg_dump)
        hist, gstep, ep0, horizon[0] = x["hist"], x["gstep"], x["epoch"] + 1, x["horizon"]
        log(f"[enc3] resumed from {resume}: epoch {x['epoch']} done, step {gstep:,}, horizon {horizon[0]:,}")
        if x["last"]:
            ep0 = epochs
    for ep in range(ep0, epochs):
        net.train()
        batches = _batches(D.epoch_index(rng), Cfg.batch)  # whole epochs, as fpnet.train (v2b: 10 x 6,469 steps)
        if Cfg.max_steps_per_epoch:
            batches = batches[:Cfg.max_steps_per_epoch]
        qb = queue.Queue(maxsize=8)
        chunks = [batches[i::Cfg.workers] for i in range(Cfg.workers)]
        threads = [threading.Thread(target=producer, args=(c, qb, seed * 1000 + ep * 10 + i), daemon=True)
                   for i, c in enumerate(chunks)]
        for th in threads:
            th.start()
        done, n, te = 0, 0, time.time()
        s_bce, s_rank, s_asg = torch.zeros((), device=dev), torch.zeros((), device=dev), 0.0
        out_of_time = False
        while done < len(threads):
            item = qb.get()
            if item is None:
                done += 1
                continue
            if out_of_time or gstep >= horizon[0]:
                continue  # drain the producers
            b, pids, neg, asg = item
            pids_t = torch.as_tensor(pids, device=dev)
            neg_t = torch.as_tensor(neg, device=dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                z = forward(net, b, dev)
            z = z.float()
            f_true = unpack_bits_torch(fp_gpu[pids_t], nbits)
            bce = F.binary_cross_entropy_with_logits(z, f_true)
            f_neg = unpack_bits_torch(fp_gpu[neg_t.clamp(min=0)], nbits)
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
            s_asg += asg
            if n % 500 == 0:
                mb, mr, sps = float(s_bce) / n, float(s_rank) / n, (time.time() - te) / n
                log(f"  ep {ep} step {n}/{steps_per_epoch} bce {mb:.4f} rank {mr:.4f} assigned {s_asg / n:.3f} "
                    f"{sps:.3f}s/step ({time.time() - t0:.0f}s)")
                if track is not None:
                    track.log({"train/bce": mb, "train/rank": mr, "train/assigned": s_asg / n,
                               "train/lr": sched.get_last_lr()[0], "train/s_per_step": sps,
                               "epoch": ep + n / max(steps_per_epoch, 1)}, step=gstep)
            if gstep == FIT_FROM:
                t_fit = time.time()
            if gstep == FIT_AT:
                horizon[0] = fit_horizon(horizon[0], gstep, (time.time() - t_fit) / (FIT_AT - FIT_FROM),
                                         budget_secs - (time.time() - t0), log)
            if time.time() - t0 > budget_secs:
                out_of_time = True
        net.eval()
        ck = dict(model=net.state_dict(), nbits=nbits, cfg=cfg_dump, epoch=ep, steps=gstep)
        torch.save(ck, out_path)
        rec = dict(epoch=ep, steps=n, gstep=gstep, bce=float(s_bce) / max(n, 1), rank=float(s_rank) / max(n, 1),
                   assigned=s_asg / max(n, 1), epoch_secs=time.time() - te, horizon=horizon[0])
        last = ep == epochs - 1 or out_of_time or gstep >= horizon[0]
        if val and ((ep + 1) % eval_every == 0 or last):
            m = Enc3Model(FT=FT, net=net, cfg=Cfg, nbits=nbits, device=dev)
            for name, (sl, pids) in val.items():
                r = window_ranks(m, P, FT, sl, pids)
                rec[f"{name}_same_mrr"] = float(np.where(r.r_pid <= 25, 1.0 / r.r_pid, 0.0).mean())
        hist.append(rec)
        ckpt.save_state(ckpt.state_path(out_path), net, opt, sched, scaler, rng, cfg=cfg_dump, epoch=ep, gstep=gstep,
                        hist=hist, horizon=horizon[0], last=bool(gstep >= horizon[0] or ep == epochs - 1))
        if track is not None:
            track.log({f"epoch/{k}": x for k, x in rec.items()}, step=gstep)
        log(f"[enc3] epoch {ep}: {rec} ({time.time() - t0:.0f}s)")
        if last:
            if out_of_time:
                log("[enc3] time budget reached")
            break
    return hist


FIT_FROM, FIT_AT = 100, 600  # steps between which the time per step is measured
RESERVE_SECS = 2700  # kept for the epoch-end monitoring, the last save and the kernel's own evaluation


def fit_horizon(total, step, sps, remaining, log=print):
    """M6 rule fixed before any result (2026-10-12): keep the planned total steps if they fit in the remaining
    budget; otherwise shorten the schedule once so that it ends inside the budget (the cosine decay then completes).
    The decision is logged."""
    need = (total - step) * sps
    if need <= remaining - RESERVE_SECS:
        log(f"[fit] {sps:.3f}s/step: {total:,} steps fit ({need / 3600:.1f} h of {remaining / 3600:.1f} h left)")
        return total
    new = step + max(1, int((remaining - RESERVE_SECS) / sps))
    log(f"[fit] {sps:.3f}s/step: {total:,} steps need {need / 3600:.1f} h but {remaining / 3600:.1f} h are left -> "
        f"horizon {new:,} steps")
    return new

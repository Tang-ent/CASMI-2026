"""Structure -> MS/MS forward model on the pool fingerprint (M5 G1, k20_fwd).

NEIMS-like: an MLP maps (Morgan r=2 2048 bits of the pool fingerprint, adduct, polarity, collision energy) to a
spectrum on 0.1 Da bins up to 1,200 Da. Two heads: a forward head (absolute m/z) and a reverse head (distance from
the precursor m/z, i.e. neutral losses), mixed per bin by a learned gate. Bins above the precursor + 2 Da are zero
(the cache keeps peaks <= precursor + 2 Da). Target = sqrt intensity, max per bin, L2-normalised; loss 1 - cosine.
No timsTOF flag (M4: the flag meant "synthetic" in training). The precursor m/z only places the reverse head.

A candidate's score for a molecule = mean over the query spectra of the cosine between the query spectrum (binned the
same way) and the prediction for that spectrum's adduct / polarity / collision energy / precursor m/z.
"""
import math
import time

import numpy as np

from . import chem
from .fpnet import ADDUCT_LIST, adduct_index, unpack_bits_torch

BIN = 0.1
MAX_MZ = 1200.0
NB = int(round(MAX_MZ / BIN))
MARGIN_BINS = int(round(2.0 / BIN))
N_COND = len(ADDUCT_LIST) + 3  # adduct one-hot, polarity, ce / 100, ce known


class Cfg:
    hidden = 1536
    layers = 3
    drop = 0.1
    batch = 512
    lr = 5e-4
    wd = 0.01
    warmup = 2000
    epochs = 10
    workers = 3
    max_steps_per_epoch = 0  # > 0: stop each epoch early (smoke test only)


def bin_peaks(mz, it):
    """(bins int64, vals float32): sqrt intensity, max per 0.1 Da bin, bins < NB; vals L2-normalised."""
    mz = np.asarray(mz, np.float64)
    v = np.sqrt(np.clip(np.asarray(it, np.float64), 0, None))
    b = np.floor(mz / BIN).astype(np.int64)
    k = (b >= 0) & (b < NB) & (v > 0)
    b, v = b[k], v[k]
    if len(b) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.float32)
    o = np.lexsort((-v, b))
    b, v = b[o], v[o]
    first = np.r_[True, b[1:] != b[:-1]]
    b, v = b[first], v[first]
    return b, (v / np.linalg.norm(v)).astype(np.float32)


def cond_row(adduct_ix, mode, ce):
    c = np.zeros(N_COND, np.float32)
    c[int(adduct_ix)] = 1.0
    c[-3] = float(mode)
    ok = ce is not None and np.isfinite(ce)
    c[-2] = float(ce) / 100.0 if ok else 0.0
    c[-1] = 1.0 if ok else 0.0
    return c


def prec_bin(prec):
    return int(min(max(math.floor(float(prec) / BIN), 0), NB - 1))


def build_model(cfg=Cfg):
    import torch
    import torch.nn as nn

    class FwdNet(nn.Module):
        def __init__(self):
            super().__init__()
            h = cfg.hidden
            self.inp = nn.Linear(chem.FP_BITS + N_COND, h)
            self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.GELU(), nn.Dropout(cfg.drop))
                                         for _ in range(cfg.layers)])
            self.norm = nn.LayerNorm(h)
            self.fwd = nn.Linear(h, NB)
            self.rev = nn.Linear(h, NB)
            self.gate = nn.Linear(h, NB)
            self.register_buffer("j", torch.arange(NB), persistent=False)

        def forward(self, fp, cond, pbin):
            """fp (n, bits) float, cond (n, N_COND), pbin (n,) int64 -> (n, NB) non-negative spectrum."""
            x = self.inp(torch.cat([fp, cond], -1))
            for blk in self.blocks:
                x = x + blk(x)
            x = torch.nn.functional.gelu(self.norm(x))
            f, r, g = self.fwd(x), self.rev(x), torch.sigmoid(self.gate(x))
            idx = pbin[:, None] - self.j[None, :]  # reverse head index k = precursor bin - bin
            r_abs = torch.gather(r, 1, idx.clamp(min=0)) * (idx >= 0)
            out = torch.relu(g * f + (1 - g) * r_abs)
            return out * (self.j[None, :] <= pbin[:, None] + MARGIN_BINS)

    return FwdNet()


def cosine_loss(pred, bins, vals):
    """1 - cosine between predictions (n, NB) and sparse targets (bins / vals (n, L), padded with bin 0 / val 0)."""
    import torch
    t = torch.zeros_like(pred).scatter_(1, bins, vals)
    cos = (pred * t).sum(-1) / (pred.norm(dim=-1) * t.norm(dim=-1) + 1e-8)
    return 1 - cos.mean()


class FwdModel:
    """Inference wrapper. score(specs, fp_packed) -> (n_cand,) mean cosine over the usable query spectra."""

    def __init__(self, path=None, net=None, device=None):
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if net is None:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            for k, v in ck["cfg"].items():
                setattr(Cfg, k, v)
            assert ck["nb"] == NB and abs(ck["bin"] - BIN) < 1e-9, "bin layout changed"
            self.epoch = ck.get("epoch", -1)
            net = build_model().to(self.device)
            net.load_state_dict(ck["model"])
        else:
            self.epoch = -1
        self.net = net.eval()

    def score(self, specs, fp_packed, chunk=4096):
        import torch
        n = len(fp_packed)
        out = np.zeros(n, np.float32)
        qs = [(bin_peaks(s["mz"], s["it"]), s) for s in specs]
        qs = [(b, v, s) for (b, v), s in qs if len(b)]
        if not qs or n == 0:
            return out
        fp = torch.as_tensor(np.ascontiguousarray(fp_packed), device=self.device)
        with torch.no_grad():
            for b, v, s in qs:
                q = torch.zeros(NB, device=self.device)
                q[torch.as_tensor(b, device=self.device)] = torch.as_tensor(v, device=self.device)
                c = torch.as_tensor(cond_row(adduct_index(s["adduct"]), s["mode"], s.get("ce", np.nan)),
                                    device=self.device)
                pb = prec_bin(s["prec"])
                for a in range(0, n, chunk):
                    f = unpack_bits_torch(fp[a:a + chunk], chem.FP_BITS)
                    m = len(f)
                    with torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda"):
                        p = self.net(f, c.expand(m, -1), torch.full((m,), pb, device=self.device, dtype=torch.long))
                    p = p.float()
                    out[a:a + m] += ((p @ q) / (p.norm(dim=-1) + 1e-8)).cpu().numpy()
        return out / len(qs)


def window_ranks(model, C, P, queries, ppm=10.0, Z=None, log=None):
    """Truth ranks among the pool candidates within +-ppm of the truth's mass (pid ties, fpnet.truth_ranks r_pid)
    by fwd alone and, when Z (molecule logits of the FPNet, rows aligned with queries) is given, by fz and by
    z(fz) + z(fwd). queries: DataFrame with pid, rows. -> dict name -> (n,) ranks."""
    from .fpnet import truth_ranks
    from .harness import query_specs

    def zs(x):
        sd = x.std()
        return (x - x.mean()) / sd if sd > 1e-9 else np.zeros_like(x)

    out = {"fwd": [], "fz": [], "comb": []}
    t0 = time.time()
    for i, q in enumerate(queries.itertuples()):
        cand = P.window(P.mass[q.pid], ppm)
        s_fwd = model.score(query_specs(C, q.rows), P.fp[cand])
        out["fwd"].append(truth_ranks(s_fwd, cand, q.pid)[1])
        if Z is not None:
            s_fz = np.unpackbits(P.fp[cand], axis=1)[:, :Z.shape[1]].astype(np.float32) @ Z[i]
            out["fz"].append(truth_ranks(s_fz, cand, q.pid)[1])
            out["comb"].append(truth_ranks(zs(s_fz) + zs(s_fwd), cand, q.pid)[1])
        if log is not None and (i + 1) % 1000 == 0:
            log(f"  {i + 1}/{len(queries)} queries ({time.time() - t0:.0f}s)")
    return {k: np.asarray(v, np.float64) for k, v in out.items() if v}


def train(C, D, P, out_path, val=None, budget_secs=6 * 3600, log=print, seed=0, track=None, eval_every=2):
    """Train on the spectra of D (fpnet.TrainData: rows, pid, adduct, epoch_index = the v2 draw). The last epoch is
    written to out_path (no evaluation set picks the epoch). val: dict name -> queries; window MRR of fwd alone is
    logged every `eval_every` epochs and on the last one (display only). Returns the history list."""
    import queue
    import threading

    import torch
    from .harness import rr_of

    t0 = time.time()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = build_model().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=Cfg.lr, weight_decay=Cfg.wd)
    n_epoch = len(D.epoch_index(np.random.default_rng(seed + 12345)))
    steps_per_epoch = max(1, n_epoch // Cfg.batch)
    if Cfg.max_steps_per_epoch > 0:
        steps_per_epoch = min(steps_per_epoch, Cfg.max_steps_per_epoch)
    total = steps_per_epoch * Cfg.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / Cfg.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    fp_gpu = torch.as_tensor(P.fp, device=dev)
    adducts = np.asarray(D.adduct)
    log(f"[fwd] {n_epoch:,} spectra per epoch draw, {steps_per_epoch:,} steps/epoch, {Cfg.epochs} epochs, device {dev}, "
        f"params {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M")

    def make_batch(idx):
        rows = D.rows[idx]
        bl, vl = [], []
        for r in rows:
            b, v = bin_peaks(*C.spectrum(r))
            bl.append(b)
            vl.append(v)
        L = max(1, max(len(b) for b in bl))
        bins = np.zeros((len(idx), L), np.int64)
        vals = np.zeros((len(idx), L), np.float32)
        for i, (b, v) in enumerate(zip(bl, vl)):
            bins[i, :len(b)] = b
            vals[i, :len(v)] = v
        cond = np.stack([cond_row(adducts[j], C.mode[r], C.ce[r]) for j, r in zip(idx, rows)])
        pb = np.array([prec_bin(C.prec[r]) for r in rows], np.int64)
        return D.pid[idx], cond, pb, bins, vals

    def producer(batches, qout):
        for idx in batches:
            qout.put(make_batch(idx))
        qout.put(None)

    hist, gstep = [], 0
    cfg_dump = {k: getattr(Cfg, k) for k in dir(Cfg) if not k.startswith("_")}
    for ep in range(Cfg.epochs):
        net.train()
        o = D.epoch_index(rng)
        batches = [o[i:i + Cfg.batch] for i in range(0, steps_per_epoch * Cfg.batch, Cfg.batch)]
        batches = [b for b in batches if len(b) == Cfg.batch]
        qb = queue.Queue(maxsize=16)
        threads = [threading.Thread(target=producer, args=(batches[i::Cfg.workers], qb), daemon=True)
                   for i in range(Cfg.workers)]
        for th in threads:
            th.start()
        done, n, te = 0, 0, time.time()
        s_loss = torch.zeros((), device=dev)
        while done < len(threads):
            item = qb.get()
            if item is None:
                done += 1
                continue
            pids, cond, pb, bins, vals = (torch.as_tensor(x).to(dev, non_blocking=True) for x in item)
            fp = unpack_bits_torch(fp_gpu[pids], chem.FP_BITS)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                pred = net(fp, cond, pb)
            loss = cosine_loss(pred.float(), bins, vals)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            n += 1
            gstep += 1
            s_loss += loss.detach()
            if n % 500 == 0:
                ml, sps = float(s_loss) / n, (time.time() - te) / n
                log(f"  ep {ep} step {n}/{steps_per_epoch} loss {ml:.4f} {sps:.3f}s/step ({time.time() - t0:.0f}s)")
                if track is not None:
                    track.log({"train/loss": ml, "train/lr": sched.get_last_lr()[0], "train/s_per_step": sps,
                               "epoch": ep + n / steps_per_epoch}, step=gstep)
            if time.time() - t0 > budget_secs:
                break
        net.eval()
        torch.save(dict(model=net.state_dict(), cfg=cfg_dump, epoch=ep, nb=NB, bin=BIN), out_path)
        out_of_time = time.time() - t0 > budget_secs
        rec = dict(epoch=ep, steps=n, loss=float(s_loss) / max(n, 1), epoch_secs=time.time() - te)
        if val and ((ep + 1) % eval_every == 0 or ep == Cfg.epochs - 1 or out_of_time):
            m = FwdModel(net=net, device=dev)
            for name, qx in val.items():
                rec[f"{name}_mrr"] = float(rr_of(window_ranks(m, C, P, qx)["fwd"]).mean())
        hist.append(rec)
        if track is not None:
            track.log({f"epoch/{k}": x for k, x in rec.items()}, step=gstep)
        log(f"[fwd] epoch {ep}: {rec} ({time.time() - t0:.0f}s)")
        if out_of_time:
            log("[fwd] time budget reached")
            break
    return hist

"""Class-3 structure decoder: fingerprint logits + neutral formula -> SMILES (M6 P0-4, k33_decoder / k34_dec_eval).

GPT-style decoder (pre-norm, causal) over SMILES tokens (regex tokenizer, stereo removed, randomised SMILES as
augmentation). The condition (FPNet logits z of Morgan r=2 2048 bits, scaled and clipped, and log1p of the formula's
element counts) goes through an MLP into N_PREFIX prefix vectors in front of the BOS token.

Training uses structures only (pool structures outside the FPNet holdout); the condition is the structure's own
fingerprint turned into noisy logits whose on/off-bit distributions match v2b's logits on R1 (holdout, not in B_N /
G_N): logit = N(m1, s1) for on bits and N(m0, s0) for off bits, times a random scale in [0.7, 1.3] per sample.
At test time the condition is v2b's z of the query and one of the formulas of its pool window.

Formula-constrained sampling (M6, user decision 2026-10-12 after the k33 report: unconstrained samples matched the
formula only 4.6% of the time): during sampling the heavy atoms of the generated tokens are counted per element; tokens
that would exceed the target formula are masked, and EOS is allowed only once every heavy-atom count is reached
(MSNovelist-style). Ring-closure digits and branches are tracked too: once every heavy atom is placed, only the
closures of open rings, ')' while a branch is open, and finally EOS are allowed ('.' is never allowed). Two grammar
rules: after a bond symbol only an atom or a ring digit, after '(' only an atom or a bond symbol. Hydrogens, valences
and aromaticity are not constrained; the product is still parsed and its exact formula checked.
"""
import math
import re
import time

import numpy as np

from . import ckpt, subform

NE = subform.NE
PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]
_TOK = re.compile(r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])")


class Cfg:
    d = 512
    layers = 6
    heads = 8
    drop = 0.1
    n_prefix = 8
    max_len = 160  # SMILES tokens (+ BOS / EOS); longer structures are not used for training
    batch = 256
    lr = 3e-4
    wd = 0.01
    warmup = 2000
    total_steps = 60000
    n_random = 5  # randomised SMILES per structure
    z_scale = 5.0
    z_clip = 3.0
    workers = 3


def cfg_of(d):
    c = type("DecCfgView", (), {})()
    for k in dir(Cfg):
        if not k.startswith("_"):
            setattr(c, k, getattr(Cfg, k))
    for k, v in d.items():
        setattr(c, k, v)
    return c


def tokenize(smiles):
    toks = _TOK.findall(smiles)
    return toks if "".join(toks) == smiles else None


def build_vocab(token_lists, min_count=20):
    from collections import Counter
    cnt = Counter(t for tl in token_lists if tl for t in tl)
    return SPECIAL + sorted(t for t, n in cnt.items() if n >= min_count)


def encode(toks, stoi, max_len):
    """Token list -> int16 ids (without BOS / EOS), or None if too long or a token is not in the vocabulary."""
    if toks is None or len(toks) > max_len:
        return None
    ids = [stoi.get(t, UNK) for t in toks]
    return None if UNK in ids else np.array(ids, np.int16)


def decode(ids, itos):
    out = []
    for i in ids:
        i = int(i)
        if i == EOS:
            break
        if i > UNK:
            out.append(itos[i])
    return "".join(out)


_BRACKET = re.compile(r"^\[\d*([A-Z][a-z]?|[a-z][a-z]?)")
_ORGANIC = {"B": "B", "C": "C", "N": "N", "O": "O", "S": "S", "P": "P", "F": "F", "I": "I", "Cl": "Cl", "Br": "Br",
            "b": "B", "c": "C", "n": "N", "o": "O", "s": "S", "p": "P"}
UNKNOWN_ATOM = 999


def token_atoms(vocab):
    """(len(vocab), NE) int heavy-atom counts per token (H column 0); UNKNOWN_ATOM in the C column for atoms outside
    subform.ELEM (never allowed under the constraint)."""
    out = np.zeros((len(vocab), NE), np.int64)
    for i, t in enumerate(vocab):
        el = None
        if t in _ORGANIC:
            el = _ORGANIC[t]
        elif t.startswith("["):
            m = _BRACKET.match(t)
            if m:
                e = m.group(1)
                el = e if e in subform.EIX else e.capitalize()
                el = el if el in subform.EIX else "?"
        if el is None:
            continue
        if el == "?" or el == "H":
            out[i, subform.EIX["C"]] = UNKNOWN_ATOM if el == "?" else 0
            continue
        out[i, subform.EIX[el]] = 1
    return out


BOND_TOKENS = ("-", "=", "#", "/", "\\", ":", "$", "~")


def token_kinds(vocab):
    """(ring slot per token (-1 = not a ring digit), id of '(', id of ')', id of '.' or -1, bond-token mask)."""
    ring = [t for t in vocab if re.fullmatch(r"[0-9]|%[0-9]{2}", t)]
    slot = np.array([ring.index(t) if t in ring else -1 for t in vocab], np.int64)
    ix = {t: i for i, t in enumerate(vocab)}
    bond = np.array([t in BOND_TOKENS for t in vocab], bool)
    return slot, ix.get("(", -1), ix.get(")", -1), ix.get(".", -1), bond


def random_smiles(smiles, n, seed=0):
    """Canonical (stereo removed) + n - 1 randomised SMILES of one structure (RDKit); [] when it does not parse."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if m is None:
        return []
    out = [Chem.MolToSmiles(m, isomericSmiles=False)]
    for s in Chem.MolToRandomSmilesVect(m, 4 * n, randomSeed=seed, isomericSmiles=False):
        if len(out) >= n:
            break
        if s not in out:
            out.append(s)
    return out


def cond_formula(counts, supported):
    """(NE + 1,) float32: log1p of the element counts and the supported flag."""
    return np.r_[np.log1p(np.maximum(np.asarray(counts, np.float32), 0)), float(supported)].astype(np.float32)


def cond_z(z, cfg=Cfg):
    return np.clip(np.asarray(z, np.float32) / cfg.z_scale, -cfg.z_clip, cfg.z_clip)


# ---------------------------------------------------------------------------------------------- model
def build_model(vocab, nbits=2048, cfg=Cfg):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Block(nn.Module):
        def __init__(self, d, h, drop):
            super().__init__()
            self.h, self.p = h, drop
            self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
            self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, d))
            self.drop = nn.Dropout(drop)

        def forward(self, x, cache=None):
            """cache None: causal over x. cache (k, v): x are new positions after the cached ones (no mask needed
            for a single new position). Returns (x, (k, v))."""
            B, L, d = x.shape
            q, k, v = self.qkv(self.ln1(x)).view(B, L, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
            if cache is not None:
                k, v = torch.cat([cache[0], k], 2), torch.cat([cache[1], v], 2)
                a = F.scaled_dot_product_attention(q, k, v, is_causal=False) if L == 1 else \
                    F.scaled_dot_product_attention(q, k, v, attn_mask=_tail_mask(L, k.shape[2], x.device))
            else:
                a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.p if self.training else 0.0)
            x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, L, d)))
            return x + self.drop(self.ff(self.ln2(x))), (k, v)

    def _tail_mask(L, T, device):
        i = torch.arange(L, device=device)[:, None] + (T - L)
        return torch.arange(T, device=device)[None, :] <= i

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            d = cfg.d
            self.cond = nn.Sequential(nn.Linear(nbits + NE + 1, 1024), nn.GELU(), nn.Linear(1024, cfg.n_prefix * d))
            self.tok = nn.Embedding(len(vocab), d)
            self.pos = nn.Embedding(cfg.n_prefix + cfg.max_len + 2, d)
            self.blocks = nn.ModuleList([Block(d, cfg.heads, cfg.drop) for _ in range(cfg.layers)])
            self.norm = nn.LayerNorm(d)
            self.out = nn.Linear(d, len(vocab))

        def prefix(self, zc, fc):
            B = zc.shape[0]
            return self.cond(torch.cat([zc, fc], -1)).view(B, cfg.n_prefix, -1)

        def forward(self, zc, fc, ids):
            """ids (B, T) starting with BOS -> logits (B, T, vocab) for the next token at each position."""
            x = torch.cat([self.prefix(zc, fc), self.tok(ids)], 1)
            x = x + self.pos(torch.arange(x.shape[1], device=x.device))[None]
            for blk in self.blocks:
                x, _ = blk(x)
            return self.out(self.norm(x[:, cfg.n_prefix:]))

        @torch.no_grad()
        def sample(self, zc, fc, max_len, temp=1.0, gen=None, target=None, contrib=None, kinds=None):
            """Ancestral sampling with a KV cache. -> (B, max_len) int64 token ids (EOS-terminated).
            target (B, NE) heavy-atom counts, contrib (vocab, NE) token_atoms and kinds token_kinds:
            formula-constrained sampling (see the module docstring)."""
            B, dev = zc.shape[0], zc.device
            if target is not None:
                V = contrib.shape[0]
                used = torch.zeros_like(target)
                special = torch.zeros(V, dtype=torch.bool, device=dev)
                special[[PAD, BOS, UNK]] = True
                slot, lp, rp, dot, bond = kinds
                slot = torch.as_tensor(slot, device=dev)
                bond = torch.as_tensor(bond, device=dev)
                atom = contrib.sum(-1) > 0
                prev_bond = torch.zeros(B, dtype=torch.bool, device=dev)
                prev_lp = torch.zeros(B, dtype=torch.bool, device=dev)
                if dot >= 0:
                    special[dot] = True
                is_ring = slot >= 0
                n_ring = int(is_ring.sum())
                open_r = torch.zeros(B, max(n_ring, 1), dtype=torch.bool, device=dev)
                depth = torch.zeros(B, dtype=torch.long, device=dev)
                ar = torch.arange(B, device=dev)
            x = self.prefix(zc, fc)
            bos = self.tok(torch.full((B, 1), BOS, dtype=torch.long, device=dev))
            x = torch.cat([x, bos], 1)
            x = x + self.pos(torch.arange(x.shape[1], device=dev))[None]
            caches = []
            for blk in self.blocks:
                x, kv = blk(x)
                caches.append(kv)
            h = x[:, -1:]
            out = torch.full((B, max_len), EOS, dtype=torch.long, device=dev)
            done = torch.zeros(B, dtype=torch.bool, device=dev)
            pos0 = cfg.n_prefix + 1
            for t in range(max_len):
                logits = self.out(self.norm(h[:, -1])).float() / temp
                if target is not None:
                    over = ((used[:, None, :] + contrib[None, :, :]) > target[:, None, :]).any(-1)  # (B, vocab)
                    over |= special[None, :]
                    complete = (used == target).all(-1)
                    no_open = ~open_r.any(-1) & (depth == 0)
                    if rp >= 0:
                        over[:, rp] |= depth == 0
                    # all heavy atoms placed: only closures of open rings, ')' and finally EOS
                    allow_c = torch.zeros(B, V, dtype=torch.bool, device=dev)
                    if n_ring:
                        allow_c[:, is_ring] = open_r[:, :n_ring]
                    if rp >= 0:
                        allow_c[:, rp] = depth > 0
                    allow_c[:, EOS] = no_open
                    over = torch.where(complete[:, None], ~allow_c, over)
                    over[:, EOS] = ~(complete & no_open)
                    # grammar: after a bond only an atom or a ring digit; after '(' only an atom or a bond
                    over |= prev_bond[:, None] & ~(atom | is_ring)[None, :]
                    over |= prev_lp[:, None] & ~(atom | bond)[None, :]
                    logits = logits.masked_fill(over, float("-inf"))
                    dead = torch.isinf(logits).all(-1)  # nothing allowed (should not happen): end the sequence
                    logits[dead, EOS] = 0.0
                nxt = torch.multinomial(torch.softmax(logits, -1), 1, generator=gen).squeeze(1)
                if target is not None:
                    used = used + contrib[nxt]
                    s_ = slot[nxt]
                    hit = s_ >= 0
                    if n_ring and bool(hit.any()):
                        open_r[ar[hit], s_[hit]] ^= True
                    if lp >= 0:
                        depth = depth + (nxt == lp).long()
                    if rp >= 0:
                        depth = depth - (nxt == rp).long()
                    prev_bond = bond[nxt]
                    prev_lp = nxt == lp
                nxt = torch.where(done, torch.full_like(nxt, EOS), nxt)
                out[:, t] = nxt
                done |= nxt == EOS
                if bool(done.all()) or t == max_len - 1:
                    break
                h = self.tok(nxt[:, None]) + self.pos.weight[pos0 + t][None, None]
                for j, blk in enumerate(self.blocks):
                    h, caches[j] = blk(h, caches[j])
            return out

    return Decoder()


# ---------------------------------------------------------------------------------------------- training data
class SeqData:
    """Encoded SMILES variants of the training structures: ids (concatenated int16), off, owner pid.
    Batches are drawn from length buckets (less padding): a bucket with probability proportional to its size, then
    uniformly inside it."""

    def __init__(self, ids, off, pid, n_buckets=16):
        self.ids, self.off, self.pid = ids, off, pid
        lens = np.diff(off)
        o = np.argsort(lens, kind="mergesort")
        self.buckets = [b for b in np.array_split(o, n_buckets) if len(b)]
        self.p = np.array([len(b) for b in self.buckets], float) / max(len(o), 1)

    def __len__(self):
        return len(self.pid)

    def draw(self, rng, n):
        b = self.buckets[rng.choice(len(self.buckets), p=self.p)]
        return b[rng.integers(0, len(b), n)]

    def batch(self, idx, max_len):
        n = len(idx)
        T = max(int(self.off[i + 1] - self.off[i]) for i in idx) + 2
        x = np.zeros((n, min(T, max_len + 2)), np.int64)
        for r, i in enumerate(idx):
            s = self.ids[self.off[i]:self.off[i + 1]]
            x[r, 0] = BOS
            x[r, 1:1 + len(s)] = s
            x[r, 1 + len(s)] = EOS
        return x


def random_smiles_many(smiles, n_random, workers=4):
    """random_smiles of many structures in parallel (list of lists)."""
    from .pool import map_many
    return map_many(_rand_worker, [(x, n_random) for x in smiles], workers, chunksize=500)


def build_seqdata(variants, pids, stoi, max_len, log=print):
    """Encoded SMILES variants -> SeqData (variants that do not encode are dropped)."""
    ids, lens, owner = [], [], []
    for p, var in zip(pids, variants):
        for x in var:
            e = encode(tokenize(x), stoi, max_len)
            if e is not None:
                ids.append(e)
                lens.append(len(e))
                owner.append(p)
    off = np.r_[0, np.cumsum(lens)].astype(np.int64)
    log(f"[decoder] {len(owner):,} sequences of {len(set(owner)):,} structures, median length "
        f"{np.median(lens) if lens else 0:.0f}")
    return SeqData(np.concatenate(ids) if ids else np.zeros(0, np.int16), off, np.asarray(owner, np.int64))


def _rand_worker(a):
    return random_smiles(a[0], a[1])


def noise_stats(Z, F):
    """On/off-bit means and standard deviations of logits Z (n, bits) against true bits F (n, bits) in {0, 1}."""
    on = F > 0.5
    return dict(m1=float(Z[on].mean()), s1=float(Z[on].std()), m0=float(Z[~on].mean()), s0=float(Z[~on].std()))


# ---------------------------------------------------------------------------------------------- training loop
def train(SD, fp, FT, vocab, noise, out_path, val=None, budget_secs=8 * 3600, log=print, seed=0, track=None,
          eval_every=5000, resume=None):
    """SD: SeqData; fp: packed Morgan fingerprints per pid; FT: subform.FormulaTable; noise: noise_stats dict.
    val: optional (zc, fc, ids) arrays of held-out structures with real v2b logits (teacher-forced loss, monitoring
    only). Saves the last checkpoint. Returns the history list.
    Every save point (eval_every and the end) also writes the full training state to <out>_state.pt (ckpt);
    resume=<that path> continues from its step with the same cfg and horizon. The batches after a resume come from
    producers seeded with the resumed step (the same distribution, not the same draws)."""
    import queue
    import threading

    import torch
    import torch.nn.functional as F

    from .enc3 import FIT_AT, FIT_FROM, fit_horizon
    from .fpnet import unpack_bits_torch

    t0 = time.time()
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = build_model(vocab).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=Cfg.lr, weight_decay=Cfg.wd)
    total = Cfg.total_steps
    horizon = [total]  # shortened once (enc3.fit_horizon) when the projected time exceeds the budget
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / Cfg.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / horizon[0]))))
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    fp_gpu = torch.as_tensor(fp, device=dev)
    fcond = np.stack([cond_formula(FT.counts[f], FT.supported[f]) for f in range(len(FT.keys))])
    fcond_gpu = torch.as_tensor(fcond, device=dev)
    fid_gpu = torch.as_tensor(FT.fid.astype(np.int64), device=dev)
    log(f"[decoder] {len(SD):,} sequences, vocab {len(vocab)}, {total:,} steps of {Cfg.batch}, device {dev}, "
        f"params {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M, noise {noise}")

    rng = np.random.default_rng(seed)
    steps_per_epoch = max(1, len(SD) // Cfg.batch)

    def producer(qout, seed_, n_batches):
        prng = np.random.default_rng(seed_)
        for _ in range(n_batches):
            idx = SD.draw(prng, Cfg.batch)
            qout.put((SD.batch(idx, Cfg.max_len), SD.pid[idx]))
        qout.put(None)

    cfg_dump = {k: getattr(Cfg, k) for k in dir(Cfg) if not k.startswith("_")}
    hist, step, s_loss, n_loss, te = [], 0, 0.0, 0, time.time()
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    if resume:
        x = ckpt.load_state(resume, net, opt, sched, scaler, rng, gen, cfg=cfg_dump)
        hist, step, horizon[0] = x["hist"], x["step"], x["horizon"]
        log(f"[decoder] resumed from {resume}: step {step:,}, horizon {horizon[0]:,}")
    step0 = step
    qb = queue.Queue(maxsize=16)
    per = math.ceil(max(horizon[0] - step0, 0) / Cfg.workers)
    seeds = [seed * 100 + i if step0 == 0 else [seed, i, step0] for i in range(Cfg.workers)]
    threads = [threading.Thread(target=producer, args=(qb, seeds[i], per), daemon=True) for i in range(Cfg.workers)]
    for th in threads:
        th.start()
    done, out_of_time = 0, False
    net.train()
    while done < len(threads):
        item = qb.get()
        if item is None:
            done += 1
            continue
        if out_of_time or step >= horizon[0]:
            continue
        ids, pids = item
        ids_t = torch.as_tensor(ids, device=dev)
        pid_t = torch.as_tensor(pids, device=dev)
        f = unpack_bits_torch(fp_gpu[pid_t], 2048)
        scale = 0.7 + 0.6 * torch.rand(len(pids), 1, device=dev, generator=gen)
        eps = torch.randn(f.shape, device=dev, generator=gen)
        z = torch.where(f > 0.5, noise["m1"] + noise["s1"] * eps, noise["m0"] + noise["s0"] * eps) * scale
        zc = (z / Cfg.z_scale).clamp(-Cfg.z_clip, Cfg.z_clip)
        fc = fcond_gpu[fid_gpu[pid_t].clamp(min=0)]
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            logits = net(zc, fc, ids_t[:, :-1])
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), ids_t[:, 1:].reshape(-1),
                               ignore_index=PAD)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1
        if step == FIT_FROM:
            t_fit = time.time()
        if step == FIT_AT:
            horizon[0] = fit_horizon(horizon[0], step, (time.time() - t_fit) / (FIT_AT - FIT_FROM),
                                     budget_secs - (time.time() - t0), log)
        s_loss += float(loss.detach()) if step % 50 == 0 else 0.0
        n_loss += step % 50 == 0
        if step % 1000 == 0:
            sps = (time.time() - te) / (step - step0)
            log(f"  step {step}/{horizon[0]} loss {s_loss / max(n_loss, 1):.4f} {sps:.3f}s/step ({time.time() - t0:.0f}s)")
            if track is not None:
                track.log({"train/loss": s_loss / max(n_loss, 1), "train/lr": sched.get_last_lr()[0],
                           "train/s_per_step": sps, "epoch": step / steps_per_epoch}, step=step)
            s_loss, n_loss = 0.0, 0
        last = step >= horizon[0] or time.time() - t0 > budget_secs
        if step % eval_every == 0 or last:
            rec = dict(step=step, secs=time.time() - t0, horizon=horizon[0])
            if val is not None:
                rec["val_loss"] = val_loss(net, val, dev)
                net.train()
            torch.save(dict(model=net.state_dict(), cfg=cfg_dump, vocab=vocab, noise=noise, step=step), out_path)
            hist.append(rec)
            ckpt.save_state(ckpt.state_path(out_path), net, opt, sched, scaler, rng, gen, cfg=cfg_dump, step=step,
                            hist=hist, horizon=horizon[0])
            log(f"[decoder] {rec}")
            if track is not None:
                track.log({f"eval/{k}": v for k, v in rec.items()}, step=step)
        if time.time() - t0 > budget_secs and not out_of_time:
            out_of_time = True
            log("[decoder] time budget reached")
    return hist


def val_loss(net, val, dev, batch=256):
    import torch
    import torch.nn.functional as F
    zc, fc, ids = val
    net.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for a in range(0, len(ids), batch):
            z_ = torch.as_tensor(zc[a:a + batch], device=dev)
            f_ = torch.as_tensor(fc[a:a + batch], device=dev)
            i_ = torch.as_tensor(ids[a:a + batch], device=dev)
            lg = net(z_, f_, i_[:, :-1]).float()
            loss = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), i_[:, 1:].reshape(-1), ignore_index=PAD,
                                   reduction="sum")
            tot += float(loss)
            n += int((i_[:, 1:] != PAD).sum())
    return tot / max(n, 1)


# ---------------------------------------------------------------------------------------------- inference
class DecoderModel:
    def __init__(self, path, device=None):
        import torch
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.cfg = cfg_of(ck["cfg"])
        self.vocab, self.noise, self.step = ck["vocab"], ck["noise"], ck.get("step", -1)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = build_model(self.vocab, cfg=self.cfg).to(self.device).eval()
        self.net.load_state_dict(ck["model"])

    def sample(self, z, formulas, n_per, temp=1.0, seed=0, chunk=512, constrain=False):
        """z: v2b logits (2048,); formulas: list of (counts, supported); n_per: samples per formula.
        constrain: formula-constrained sampling (heavy atoms). -> list of (formula index, SMILES string) (raw
        decoder strings, unchecked)."""
        import torch
        jobs = [(j, n) for j, n in enumerate(n_per) for _ in range(n)]
        if not jobs:
            return []
        zc = torch.as_tensor(np.repeat(cond_z(z, self.cfg)[None], len(jobs), 0), device=self.device)
        fc = torch.as_tensor(np.stack([cond_formula(*formulas[j]) for j, _ in jobs]), device=self.device)
        tg = ct = None
        if constrain:
            if not hasattr(self, "_contrib"):
                self._contrib = torch.as_tensor(token_atoms(self.vocab), device=self.device)
                self._kinds = token_kinds(self.vocab)
            heavy = np.stack([np.asarray(formulas[j][0], np.int64) for j, _ in jobs])
            heavy[:, subform.H_IX] = 0
            tg, ct = torch.as_tensor(heavy, device=self.device), self._contrib
        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed)
        out = []
        for a in range(0, len(jobs), chunk):
            ids = self.net.sample(zc[a:a + chunk], fc[a:a + chunk], self.cfg.max_len + 1, temp, gen,
                                  target=None if tg is None else tg[a:a + chunk], contrib=ct,
                                  kinds=self._kinds if constrain else None).cpu().numpy()
            out.extend((jobs[a + r][0], decode(ids[r], self.vocab)) for r in range(len(ids)))
        return out


def split_samples(n_by_formula, total=512, max_formulas=8, min_per=32):
    """Samples per formula: the max_formulas most frequent formulas of the window, `total` split in proportion to
    their candidate counts with at least min_per each. n_by_formula: list of counts (descending order)."""
    n = np.asarray(n_by_formula[:max_formulas], float)
    if not len(n):
        return []
    k = np.maximum(min_per, np.floor(total * n / n.sum())).astype(int)
    while k.sum() > total and (k > min_per).any():
        k[np.argmax(k)] -= 1
    return k.tolist()

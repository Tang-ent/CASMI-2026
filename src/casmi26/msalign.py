"""MSAlign-type spectrum <-> molecule alignment on precomputed embeddings (plan section 10, 2026-10-24; k50_msalign).

Inputs are frozen tables, built by the embedding kernel (k49) and resident on the GPU during training:
  spectrum table  E_spec fp16 (n_emb, 1024): DreaMS ssl_model.ckpt embeddings of cache rows; emb_rows (n_emb,) = the
                  k01 cache row of every table row (row_map turns it into cache row -> table row, -1 = none)
  molecule table  MolTable: 'cb' = ChemBERTa-100M-MLM token means fp16 (n_pid, 768), or 'fp' = the packed pool_fp
                  bits (Morgan r=2 2048, no stereo), indexed by pid

Training spectra (plan 10.2): train_rows = the fpnet.TrainData rule (C.keep & pid >= 0 & pid not in the holdout),
then at most `max_per_struct` (4) rows per structure, drawn once with a fixed seed (the embedding kernel calls the same
function to know which rows to embed). Per row: adduct id (vocabulary = sorted adducts of the selected rows + '<unk>';
during training 10% of the rows of a step are set to '<unk>'), polarity (from the cache mode: -, +, unknown) and
collision energy (MSAlign's encoder: e / 100 / 10000^(2i/32), 16 interleaved sin / cos pairs, a learned vector when NaN).
No timsTOF flag.

Negatives (NegTable, by pid): train-only structures outside the holdout. 'mass' = the +-10 ppm window of the truth's
corrected mass (fpnet.TrainData windows built with Cfg.neg_train_only=True), 'formula' = the truth's formula
(enc3.SameFormulaNegatives). Every step draws up to k_cand - 1 = 127 of them per row afresh, without replacement; the
truth is always candidate 0 and missing candidates are masked (-inf logits).

Model (plan 10.1, the public MSAlign code d365d1f): blocks Linear -> LayerNorm -> GELU -> Dropout(0.3), each tower ends
with one Linear. Spectrum: ms_pre = Linear 1024 -> 512, concat metadata (CE 32 + adduct 16 + polarity 16 = 64),
ms_post = 576 -> 2048 -> 2048 -> 512. Molecule: d_mol -> 2048 x 3 -> 512. Both L2-normalised; logit = cosine / tau,
log tau learned (init log 0.07). blind (the shortcut check of 10.2): the spectrum tower input (the 1024-d embedding and
the 64-d metadata) is a learned constant, so the spectrum side never sees the spectrum or its metadata.
Loss = candidate cross-entropy with target 0 (one direction, spectrum -> its candidates).

Training: AdamW lr 1e-4, wd 1e-6 (all parameters, log tau included), the MSAlign schedule (linear warmup from 0 over
2,000 steps, then cosine to 0 at max_steps = 30,000), clip 5.0, B = 128, AMP fp16 + GradScaler, fixed seed. Step i
uses a fixed slice of concatenated seeded epoch permutations, and its negatives and adduct dropout come from
default_rng([seed, i]); torch dropout uses the global RNG, which ckpt saves. So train(resume=) after an interruption
gives the same weights as an uninterrupted run (checked bit-identical on CPU with one thread and on an RTX 4070 Ti
with AMP; multi-threaded CPU kernels can differ in the last bits even between two straight runs).
The weights (<out>.pt) and the state (<out>_state.pt, casmi26.ckpt) are written every save_every steps and at the end;
the last step's model is the one used (no evaluation set picks the step).

Scoring (plan 10.2): per query, the mean over its spectra of the raw cosine between the spectrum projection and each
candidate's projection -> msa, msa_z (z over the full window, search._zs), msa_rank (search.rank_norm, 0 = best).
Gates (plan 10.3): window_rr / window_mrr (MRR@25 inside full windows), random_mrr, provenance_coef (fpnet's
coefficient with this model's score function) and paired_ci (harness).
"""
import math
import time
from dataclasses import asdict, dataclass

import numpy as np

from . import ckpt

MSA_FEATURES = ["msa", "msa_z", "msa_rank"]
UNK = "<unk>"
D_SPEC = 1024  # DreaMS embedding size


@dataclass
class Cfg:
    """One variant's recipe (plan 10.1 / 10.2). Plain scalars only (ckpt compares the dump on resume)."""
    name: str = "cb_mass"
    blind: bool = False
    d_shared: int = 512
    d_hidden: int = 2048
    d_ce: int = 32
    d_adduct: int = 16
    d_pol: int = 16
    n_pre: int = 0
    n_post: int = 2
    n_mol: int = 3
    dropout: float = 0.3
    temp_init: float = 0.07
    batch: int = 128
    k_cand: int = 128  # truth + up to k_cand - 1 negatives
    lr: float = 1e-4
    wd: float = 1e-6
    warmup: int = 2000
    max_steps: int = 30000
    clip: float = 5.0
    adduct_drop: float = 0.1
    save_every: int = 1000
    log_every: int = 500


# ---------------------------------------------------------------------------------------------- data
def train_rows(C, P, holdout_pids, max_per_struct=4, seed=0):
    """Training spectra of plan 10.2: cache rows with C.keep & pid >= 0 & pid not in the holdout (the fpnet.TrainData
    rule), then at most max_per_struct rows per structure, uniformly at random with default_rng(seed) (0 = all).
    -> (rows sorted, pid of each row). Independent of fpnet.Cfg."""
    pid = P.sid2pid[C.sid]
    hold = np.asarray(sorted(holdout_pids), np.int64)
    ok = C.keep & (pid >= 0) & ~np.isin(pid, hold)
    rows = np.flatnonzero(ok)
    if max_per_struct > 0:
        key = np.random.default_rng(seed).random(len(rows))
        o = np.lexsort((key, pid[rows]))
        p = pid[rows][o]
        first = np.r_[True, p[1:] != p[:-1]]
        start = np.maximum.accumulate(np.where(first, np.arange(len(o)), 0))
        rows = np.sort(rows[o[np.arange(len(o)) - start < max_per_struct]])
    return rows, pid[rows].astype(np.int64)


def row_map(emb_rows, n_rows):
    """Cache row -> spectrum table row (-1 = not embedded), from emb_rows (n_emb,) = the cache row of each table row."""
    m = np.full(int(n_rows), -1, np.int64)
    emb_rows = np.asarray(emb_rows, np.int64)
    assert len(np.unique(emb_rows)) == len(emb_rows), "a cache row appears twice in the spectrum table"
    m[emb_rows] = np.arange(len(emb_rows))
    return m


def norm_adduct(a):
    return str(a).replace(" ", "")


def make_vocab(C, rows):
    """Sorted adduct names of the rows, then '<unk>' (its index = len - 1)."""
    return sorted({norm_adduct(a) for a in np.asarray(C.adducts, object)[np.unique(C.adduct[rows])]} - {UNK}) + [UNK]


def meta_of(C, rows, vocab):
    """-> (adduct index (n,), polarity index (n,) 0 = -, 1 = +, 2 = unknown, CE in eV with NaN (n,) float32)."""
    ix = {a: i for i, a in enumerate(vocab)}
    unk = ix[UNK]
    by_cache = np.array([ix.get(norm_adduct(a), unk) for a in C.adducts], np.int64)
    rows = np.asarray(rows, np.int64)
    mode = np.asarray(C.mode[rows])
    pol = np.where(mode < 0, 0, np.where(mode > 0, 1, 2)).astype(np.int64)
    return by_cache[C.adduct[rows]], pol, np.asarray(C.ce[rows], np.float32)


def emb_index(rows, rmap):
    e = rmap[np.asarray(rows, np.int64)]
    if (e < 0).any():
        bad = np.asarray(rows)[e < 0][:5]
        raise KeyError(f"{int((e < 0).sum())} cache rows have no spectrum embedding (first: {bad.tolist()})")
    return e


@dataclass
class TrainSet:
    rows: np.ndarray      # cache rows
    pid: np.ndarray       # their structures
    emb_ix: np.ndarray    # spectrum table rows
    adduct_ix: np.ndarray
    pol_ix: np.ndarray
    ce: np.ndarray
    vocab: list

    def __len__(self):
        return len(self.rows)

    def subset(self, idx):
        return TrainSet(self.rows[idx], self.pid[idx], self.emb_ix[idx], self.adduct_ix[idx], self.pol_ix[idx],
                        self.ce[idx], self.vocab)


def train_set(C, P, holdout_pids, rmap, max_per_struct=4, seed=0, vocab=None):
    """TrainSet of train_rows(...): every selected row must be in the spectrum table (rmap = row_map(...))."""
    rows, pid = train_rows(C, P, holdout_pids, max_per_struct, seed)
    vocab = make_vocab(C, rows) if vocab is None else list(vocab)
    ad, pol, ce = meta_of(C, rows, vocab)
    return TrainSet(rows, pid, emb_index(rows, rmap), ad, pol, ce, vocab)


def query_rows(C, rows, max_spec=16):
    """The cache rows of a query that k49 embeds and k50 scores: the Searcher's spectra (search.py:178), i.e. the rows
    (queries.csv order; a list or the comma-separated string) whose neutral mass is finite, the first max_spec
    (SearchCfg.max_query_spectra) of them. -> int64 array."""
    from .chem import neutral_mass
    if isinstance(rows, str):
        rows = rows.split(",")
    ok = [int(r) for r in rows if np.isfinite(neutral_mass(float(C.prec[int(r)]), C.adducts[C.adduct[int(r)]]))]
    return np.asarray(ok[:max_spec], np.int64)


def query_inputs(C, rows, rmap, vocab):
    """Scoring input of one query (its spectra = cache rows, as chosen by the caller) -> dict(emb_ix, adduct_ix,
    pol_ix, ce)."""
    rows = np.asarray(rows, np.int64)
    ad, pol, ce = meta_of(C, rows, vocab)
    return dict(emb_ix=emb_index(rows, rmap), adduct_ix=ad, pol_ix=pol, ce=ce)


class NegTable:
    """Negative pools by pid: the candidates of pid p are order[lo[p]:hi[p]] (p itself is skipped when drawing)."""

    def __init__(self, order, lo, hi, kind):
        self.order, self.lo, self.hi, self.kind = np.asarray(order, np.int64), lo, hi, kind

    @classmethod
    def mass(cls, C, P, holdout_pids):
        """+-10 ppm (fpnet.Cfg.ppm) of the corrected mass, train structures outside the holdout (fpnet.TrainData with
        Cfg.neg_train_only=True; the global is restored afterwards)."""
        from . import fpnet
        old = fpnet.Cfg.neg_train_only
        fpnet.Cfg.neg_train_only = True
        try:
            D = fpnet.TrainData(C, P, holdout_pids)
        finally:
            fpnet.Cfg.neg_train_only = old
        return cls(D.order, D.win_lo, D.win_hi, "mass")

    @classmethod
    def formula(cls, P, FT, holdout_pids):
        """Same formula id (enc3.SameFormulaNegatives: train structures outside the holdout)."""
        from .enc3 import SameFormulaNegatives
        N = SameFormulaNegatives(P, FT, holdout_pids)
        return cls(N.order, N.lo, N.hi, "formula")

    def sizes(self, pids):
        """Number of negatives available to each pid (its pool minus itself)."""
        pids = np.asarray(pids, np.int64)
        lo, hi = self.lo[pids].astype(np.int64), self.hi[pids].astype(np.int64)
        pos = np.full(len(self.lo), -1, np.int64)
        pos[self.order] = np.arange(len(self.order))
        inside = (pos[pids] >= lo) & (pos[pids] < hi)
        return np.maximum(hi - lo, 0) - inside

    def draw(self, pids, k, rng):
        """(n, 1 + k) candidates: column 0 = the truth, then up to k distinct negatives drawn without replacement
        (in random order), -1 = padding."""
        pids = np.asarray(pids, np.int64)
        out = np.full((len(pids), 1 + k), -1, np.int64)
        out[:, 0] = pids
        for b, p in enumerate(pids):
            w = self.order[self.lo[p]:self.hi[p]]
            w = w[w != p]
            if len(w) > k:
                w = rng.choice(w, k, replace=False)
            else:
                w = w[rng.permutation(len(w))]
            out[b, 1:1 + len(w)] = w
        return out


class MolTable:
    """Molecule inputs by pid: 'cb' dense fp16 (n_pid, 768) or 'fp' packed bits (n_pid, nbits / 8) uint8."""

    def __init__(self, arr, kind, nbits=None):
        assert kind in ("cb", "fp")
        self.arr, self.kind = arr, kind
        self.nbits = int(nbits or arr.shape[1] * 8) if kind == "fp" else None
        self.d = self.nbits if kind == "fp" else int(arr.shape[1])

    def to(self, dev):
        import torch
        return torch.as_tensor(np.ascontiguousarray(self.arr), device=dev)

    def gather(self, t, pids):
        """Rows of the device table t (from .to) as float32 inputs."""
        if self.kind == "fp":
            from .fpnet import unpack_bits_torch
            return unpack_bits_torch(t[pids], self.nbits)
        return t[pids].float()


# ---------------------------------------------------------------------------------------------- model
def build_model(d_mol, n_adducts, cfg=None):
    import torch
    import torch.nn as nn
    cfg = cfg or Cfg()

    def mlp(d_in, n_hidden):
        layers, w = [], d_in
        for _ in range(n_hidden):
            layers += [nn.Linear(w, cfg.d_hidden), nn.LayerNorm(cfg.d_hidden), nn.GELU(), nn.Dropout(cfg.dropout)]
            w = cfg.d_hidden
        layers.append(nn.Linear(w, cfg.d_shared))
        return nn.Sequential(*layers)

    class CEEnc(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.register_buffer("freq", 10_000.0 ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
            self.missing = nn.Parameter(torch.empty(d))
            nn.init.normal_(self.missing, std=0.02)

        def forward(self, e):
            miss = torch.isnan(e)
            a = e.nan_to_num().unsqueeze(-1) / 100.0 / self.freq
            enc = torch.stack((a.sin(), a.cos()), -1).flatten(-2)
            return torch.where(miss.unsqueeze(-1), self.missing, enc)

    class MSAlignNet(nn.Module):
        def __init__(self):
            super().__init__()
            d_meta = cfg.d_ce + cfg.d_adduct + cfg.d_pol
            self.blind = bool(cfg.blind)
            self.ms_pre = mlp(D_SPEC, cfg.n_pre)
            self.ce = CEEnc(cfg.d_ce)
            self.adduct = nn.Embedding(n_adducts, cfg.d_adduct)
            self.pol = nn.Embedding(3, cfg.d_pol)
            nn.init.normal_(self.adduct.weight, std=0.02)
            nn.init.normal_(self.pol.weight, std=0.02)
            self.ms_post = mlp(cfg.d_shared + d_meta, cfg.n_post)
            self.mol = mlp(d_mol, cfg.n_mol)
            self.log_temp = nn.Parameter(torch.tensor(float(cfg.temp_init)).log())
            if self.blind:  # the learned constant that replaces the spectrum embedding and its metadata
                self.const_spec = nn.Parameter(torch.empty(D_SPEC))
                self.const_meta = nn.Parameter(torch.empty(d_meta))
                nn.init.normal_(self.const_spec, std=0.02)
                nn.init.normal_(self.const_meta, std=0.02)

        def encode_spectrum(self, x, ad, pol, ce):
            """x (n, 1024) (ignored when blind; n is then taken from ad) -> unnormalised (n, d_shared)."""
            if self.blind:
                n = ad.shape[0]
                h = self.ms_pre(self.const_spec.expand(n, -1))
                m = self.const_meta.expand(n, -1)
            else:
                h = self.ms_pre(x.float())
                m = torch.cat([self.ce(ce.float()), self.adduct(ad), self.pol(pol)], -1)
            return self.ms_post(torch.cat([h, m.to(h.dtype)], -1))

        def encode_molecule(self, m):
            return self.mol(m.float())

    return MSAlignNet()


def candidate_logits(net, spec_x, ad, pol, ce, mol_in, cand):
    """(B, K) logits cosine / tau with -inf where cand < 0; mol_in = molecule inputs of cand[cand >= 0] (row-major).
    Towers may run under autocast; normalisation, cosine and the temperature are float32."""
    import torch
    import torch.nn.functional as F
    mask = cand >= 0
    assert bool(mask[:, 0].all()), "candidate 0 (the truth) must be valid"
    u = F.normalize(net.encode_spectrum(spec_x, ad, pol, ce).float(), dim=-1)
    m = F.normalize(net.encode_molecule(mol_in).float(), dim=-1)
    r = mask.nonzero(as_tuple=False)[:, 0]
    s = torch.full(mask.shape, float("-inf"), device=u.device, dtype=torch.float32)
    s[mask] = (u[r] * m).sum(-1) / net.log_temp.float().exp()
    return s


def candidate_loss(s):
    import torch
    import torch.nn.functional as F
    return F.cross_entropy(s, torch.zeros(len(s), dtype=torch.long, device=s.device))


def lr_lambda(warmup, max_steps):
    """MSAlign's cosine_with_warmup (model.py configure_optimizers), stepped every optimizer step."""
    def f(step):
        if step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, max_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))
    return f


def step_order(n, steps, batch, seed):
    """Row indices of every step: concatenated permutations of range(n) from default_rng(seed), cut to steps * batch."""
    rng = np.random.default_rng(seed)
    parts, tot = [], 0
    while tot < steps * batch:
        parts.append(rng.permutation(n))
        tot += n
    return np.concatenate(parts)[:steps * batch].reshape(steps, batch)


def train(spec_table, ts, mol, neg, out_path, cfg=None, budget_secs=8.0 * 3600, log=print, seed=0, track=None,
          resume=None, device=None, stop_at=None, tag=None):
    """Train one variant. spec_table: E_spec (n_emb, 1024) fp16 (ignored by a blind cfg, may be None); ts: TrainSet;
    mol: MolTable; neg: NegTable. Writes out_path (dict(model, cfg, vocab, d_mol, step, hist)) and <out>_state.pt
    every cfg.save_every steps and at the end; resume=<state path> continues with the same cfg and data.
    stop_at: stop (and save) after this step, as if interrupted (tests / split runs); budget_secs: the same on time.
    tag: optional JSON-able identity of the input tables (k50 passes k49's rows sha / fp16 blocks / ChemBERTa sha);
    when given it is stored in the cfg dump of the weights and state files as cfg["tag"], so a resume on other tables
    is refused (ckpt.load_state compares the cfg); None (default) leaves the cfg dump as before.
    Returns the history (one record per log_every steps)."""
    import torch

    cfg = cfg or Cfg()
    t0 = time.time()
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    net = build_model(mol.d, len(ts.vocab), cfg).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(cfg.warmup, cfg.max_steps))
    amp = dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    cfg_dump = dict(asdict(cfg), d_mol=int(mol.d), mol=mol.kind, neg=neg.kind, n_rows=int(len(ts)),
                    n_adducts=int(len(ts.vocab)), seed=int(seed))
    if tag is not None:
        cfg_dump["tag"] = tag
    hist, step = [], 0
    if resume:
        x = ckpt.load_state(resume, net, opt, sched, scaler, cfg=cfg_dump)
        hist, step = x["hist"], x["step"]
        log(f"[msalign {cfg.name}] resumed from {resume}: step {step:,}")
    order = step_order(len(ts), cfg.max_steps, cfg.batch, seed)
    S = None if cfg.blind else torch.as_tensor(np.ascontiguousarray(spec_table), device=dev)
    M = mol.to(dev)
    T = {k: torch.as_tensor(getattr(ts, k), device=dev) for k in ("emb_ix", "adduct_ix", "pol_ix", "ce")}
    unk = len(ts.vocab) - 1
    n_neg = neg.sizes(np.unique(ts.pid))
    log(f"[msalign {cfg.name}] {len(ts):,} spectra of {len(np.unique(ts.pid)):,} structures, mol {mol.kind} "
        f"(d {mol.d}), negatives {neg.kind} (median {np.median(n_neg):.0f}, none for {(n_neg == 0).mean():.3f} of "
        f"structures), {cfg.max_steps:,} steps x {cfg.batch} x {cfg.k_cand}, blind {cfg.blind}, device {dev}, "
        f"params {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M", flush=True)

    def save():
        net.eval()
        torch.save(dict(model=net.state_dict(), cfg=cfg_dump, vocab=list(ts.vocab), d_mol=int(mol.d), step=step,
                        hist=hist), out_path)
        ckpt.save_state(ckpt.state_path(out_path), net, opt, sched, scaler, cfg=cfg_dump, step=step, hist=hist)
        net.train()

    acc = dict(loss=0.0, hit=0.0, n_ok=0, n=0, k=0.0)
    te = time.time()
    net.train()
    while step < cfg.max_steps:
        if (stop_at is not None and step >= stop_at) or time.time() - t0 > budget_secs:
            break
        idx = order[step]
        prng = np.random.default_rng([seed, step])
        cand = neg.draw(ts.pid[idx], cfg.k_cand - 1, prng)
        drop = prng.random(len(idx)) < cfg.adduct_drop
        it = torch.as_tensor(idx, device=dev)
        ad = T["adduct_ix"][it].masked_fill(torch.as_tensor(drop, device=dev), unk)
        cand_t = torch.as_tensor(cand, device=dev)
        mask = cand_t >= 0
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            x = None if S is None else S[T["emb_ix"][it]]
            s = candidate_logits(net, x, ad, T["pol_ix"][it], T["ce"][it], mol.gather(M, cand_t[mask]), cand_t)
        loss = candidate_loss(s)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.clip)
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1
        with torch.no_grad():
            ok = mask[:, 1:].any(1)
            hit = ((s[:, 1:] >= s[:, :1]) & mask[:, 1:]).sum(1) == 0  # strict: a tie with a negative is a miss
            acc["loss"] += float(loss)
            acc["hit"] += float((hit & ok).sum())
            acc["n_ok"] += int(ok.sum())
            acc["n"] += 1
            acc["k"] += float(mask.sum(1).float().mean())
        if step % cfg.log_every == 0 or step == cfg.max_steps:
            rec = dict(step=step, loss=acc["loss"] / acc["n"], acc=acc["hit"] / max(acc["n_ok"], 1),
                       temp=float(net.log_temp.detach().exp()), lr=sched.get_last_lr()[0],
                       cand=acc["k"] / acc["n"], with_neg=acc["n_ok"] / (acc["n"] * cfg.batch),
                       s_per_step=(time.time() - te) / acc["n"])
            hist.append(rec)
            log(f"  [msalign {cfg.name}] step {step:,} loss {rec['loss']:.4f} acc {rec['acc']:.4f} "
                f"temp {rec['temp']:.4f} lr {rec['lr']:.2e} cand {rec['cand']:.1f} {rec['s_per_step']:.3f}s/step "
                f"({time.time() - t0:.0f}s)", flush=True)
            if track is not None:
                track.log({f"{cfg.name}/{k}": v for k, v in rec.items() if k != "step"}, step=step)
            acc = dict(loss=0.0, hit=0.0, n_ok=0, n=0, k=0.0)
            te = time.time()
        if step % cfg.save_every == 0:
            save()
    save()
    log(f"[msalign {cfg.name}] stopped at step {step:,} of {cfg.max_steps:,} ({time.time() - t0:.0f}s)")
    return hist


# ---------------------------------------------------------------------------------------------- scoring
class MSAModel:
    """Inference around a trained net: unit projections of spectra (table rows + metadata) and of pids (eval mode,
    float32, no autocast). path: a weights file of train(); or net + cfg + vocab directly."""

    def __init__(self, path=None, spec_table=None, mol=None, device=None, net=None, cfg=None, vocab=None):
        import torch
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if path is not None:
            ck = torch.load(path, map_location="cpu", weights_only=True)
            c = Cfg(**{k: v for k, v in ck["cfg"].items() if k in Cfg.__dataclass_fields__})
            assert ck["d_mol"] == mol.d, f"{path} expects d_mol {ck['d_mol']}, table has {mol.d}"
            net = build_model(ck["d_mol"], len(ck["vocab"]), c)
            net.load_state_dict(ck["model"])
            cfg, vocab, self.step = c, ck["vocab"], ck["step"]
        self.net = net.to(self.dev).eval()
        self.cfg, self.vocab, self.mol = cfg, list(vocab), mol
        self.S = None if spec_table is None or cfg.blind else torch.as_tensor(np.ascontiguousarray(spec_table),
                                                                             device=self.dev)
        self.M = mol.to(self.dev)

    def spec_proj(self, q):
        """Unit spectrum projections (n_spec, d_shared) float32 of a query_inputs dict (on the device)."""
        import torch
        import torch.nn.functional as F
        with torch.no_grad():
            ad = torch.as_tensor(q["adduct_ix"], device=self.dev)
            x = None if self.S is None else self.S[torch.as_tensor(q["emb_ix"], device=self.dev)]
            u = self.net.encode_spectrum(x, ad, torch.as_tensor(q["pol_ix"], device=self.dev),
                                         torch.as_tensor(q["ce"], device=self.dev))
            return F.normalize(u.float(), dim=-1)

    def mol_proj(self, pids, chunk=8192):
        """Unit molecule projections (n, d_shared) float32 (on the device)."""
        import torch
        import torch.nn.functional as F
        pids = torch.as_tensor(np.asarray(pids, np.int64), device=self.dev)
        out = []
        with torch.no_grad():
            for i in range(0, len(pids), chunk):
                out.append(F.normalize(self.net.encode_molecule(self.mol.gather(self.M, pids[i:i + chunk])).float(),
                                       dim=-1))
        return torch.cat(out) if out else torch.zeros((0, self.cfg.d_shared), device=self.dev)


def score_windows(model, queries, windows, group=512):
    """queries: query_inputs dicts; windows: pid arrays (the full windows). -> list of (n_cand, 3) float32 arrays
    [msa, msa_z, msa_rank] (MSA_FEATURES): msa = mean over the query's spectra of the raw cosine, msa_z = search._zs
    over the window, msa_rank = search.rank_norm (0 = best)."""
    import torch

    from .search import _zs, rank_norm
    out = []
    for g in range(0, len(queries), group):
        ws = [np.asarray(w, np.int64) for w in windows[g:g + group]]
        allp = np.unique(np.concatenate(ws)) if ws else np.zeros(0, np.int64)
        Mp = model.mol_proj(allp)
        for q, w in zip(queries[g:g + group], ws):
            U = model.spec_proj(q)
            loc = torch.as_tensor(np.searchsorted(allp, w), device=Mp.device)
            s = (Mp[loc] @ U.T).mean(1).double().cpu().numpy() if len(w) else np.zeros(0)
            out.append(np.c_[s, _zs(s), rank_norm(s)].astype(np.float32))
    return out


def query_vectors(model, queries):
    """(n_q, d_shared) mean of each query's unit spectrum projections (not renormalised): v . unit_mol = msa."""
    return [model.spec_proj(q).mean(0) for q in queries]


def msa_score_fn(model, queries):
    """score_fn(i, cand) -> msa of window candidates cand for query i (for provenance_coef)."""
    V = query_vectors(model, queries)

    def f(i, cand):
        return (model.mol_proj(cand) @ V[i]).double().cpu().numpy()
    return f


# ---------------------------------------------------------------------------------------------- gates (plan 10.3)
def window_ranks(scores, windows, truths):
    """Truth ranks inside each full window under fpnet.truth_ranks' three tie rules -> DataFrame(n_cand, in_window,
    r_opt, r_pid, r_pess); ranks are 0 when the truth is not in the window."""
    import pandas as pd
    from .fpnet import truth_ranks
    out = []
    for s, w, t in zip(scores, windows, truths):
        s, w = np.asarray(s), np.asarray(w)
        s = s[:, 0] if s.ndim == 2 else s
        if (w == t).any():
            out.append((len(w), True, *truth_ranks(s, w, t)))
        else:
            out.append((len(w), False, 0, 0, 0))
    return pd.DataFrame(out, columns=["n_cand", "in_window", "r_opt", "r_pid", "r_pess"])


def window_rr(scores, windows, truths, k=25, ties="opt"):
    """Per-query reciprocal rank @k (0 beyond k or when the truth is not in the window); ties as fpnet.window_mrr
    ('opt', in the truth's favour) unless 'pid' / 'pess'. scores: per-query score arrays (or score_windows output:
    column 0 = msa)."""
    R = window_ranks(scores, windows, truths)
    r = R[f"r_{ties}"].values
    return np.where(R.in_window.values & (r <= k) & (r > 0), 1.0 / np.maximum(r, 1), 0.0)


def window_mrr(scores, windows, truths, k=25, ties="opt"):
    rr = window_rr(scores, windows, truths, k, ties)
    return float(rr.mean()) if len(rr) else float("nan")


def random_mrr(n, k=25):
    """Expected reciprocal rank @k of a random order, per query: (1/n) * sum_{r <= min(n, k)} 1/r (0 for n = 0;
    pass n = 0 for a window without the truth)."""
    n = np.asarray(n, np.int64)
    H = np.r_[0.0, np.cumsum(1.0 / np.arange(1, k + 1))]
    return np.where(n > 0, H[np.minimum(n, k)] / np.maximum(n, 1), 0.0)


def provenance_coef(score_fn, P, pids, holdout, ppm=10.0, n_boot=2000, seed=2):
    """fpnet.provenance_coef with a score function score_fn(i, cand) (e.g. msa_score_fn) in place of f.z: same windows
    (+-ppm of the truth's mass), TAN_BINS, standardisation and bootstrap."""
    from .fpnet import provenance_coef as pc
    return pc(None, P, pids, holdout, ppm=ppm, n_boot=n_boot, seed=seed, score_fn=score_fn)


def paired_ci(x, y, n_boot=2000, seed=0, level=0.95):
    """harness.paired_ci (mean of x - y over paired queries, bootstrap interval)."""
    from .harness import paired_ci as pc
    return pc(x, y, n_boot=n_boot, seed=seed, level=level)

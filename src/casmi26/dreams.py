"""DreaMS self-supervised spectrum encoder (plan 10.2, M9 MSAlign-type training; 2026-10-24). torch + numpy only.

A re-implementation of the forward pass of the official DreaMS code (github.com/pluskal-lab/DreaMS, commit dbec3a0,
MIT): dreams/models/dreams/dreams.py (DreaMS.forward), dreams/models/dreams/layers.py (MultiheadAttention,
FeedForward, TransformerEncoder), dreams/models/layers/{feed_forward,fourier_features}.py and the preprocessing of
dreams/utils/data.py (SpectrumPreprocessor) / dreams/utils/spectra.py. Weights: ssl_model.ckpt (HF roman-bushuiev/DreaMS,
sha256 SSL_SHA256), the only purely self-supervised checkpoint (GeMS-A10, masked-peak m/z + retention order; no
structure labels). The supervised embedding_model*.ckpt files are NOT to be used (trained on MoNA / MassSpecGym
structures, which overlap our libraries).

SSL hyper-parameters (read from the checkpoint): 7 layers, 8 heads, d_peak 44 + d_fourier 980 = d_model 1024,
ff_fourier 5 x 512, Fourier 'lin_float_int' (frequencies 1/(1e-4 i), i = 2..9998 step 2, and 1/i, i = 2..999; not
trained), graphormer m/z-difference bias without parameters, pre-norm, no transformer bias, ReLU everywhere,
n_highest_peaks 60, prec_intens 1.1, max_mz 1000, SSL head ff_out = Linear(1024, 20000) over 0.05-Da m/z bins.

Faithfulness notes (all reproduced on purpose; checked against the official module by tools/smoke/check_dreams.py):
- padding = tokens with m/z == 0. The official attention masks the QUERY axis (mask.unsqueeze(1).unsqueeze(-1)):
  padded rows attend uniformly to everything and padded tokens stay visible as keys / values of the real tokens.
- the graphormer bias sum_d(F_i,d - F_j,d) is computed as s_i - s_j with s = F.sum(-1) (no 4-D tensor; exact up to
  summation order). naive_bias=True keeps the official 4-D form for the equivalence test.
- the Fourier phase (2 pi x) * b is an element-wise product (the official `x @ b` has K = 1, the same single product
  in fp32, but a matmul could be routed through TF32). Fourier features, ff_fourier and ff_peak always run in fp32.
- fp16_blocks=True runs the transformer blocks under autocast fp16, but the attention scores, the bias, the mask and
  the softmax stay fp32 (s_i - s_j reaches O(1e2-1e3); -1e9 does not fit in fp16).
- the embedding is the precursor token of the last layer (after the final LayerNorm): forward(x)[:, 0].

Loading never runs code from the pickle: load_ssl reads the zip archive itself with an unpickler that only resolves
collections.OrderedDict, the torch tensor / parameter rebuild functions, torch storage types and numpy
reconstructors; every other global (msml.*, argparse.Namespace, pathlib, builtins, ...) becomes an inert stub. Only
the tensors of the state_dict are kept.
"""
import collections
import hashlib
import math
import pickle
import sys
import zipfile

import numpy as np

SSL_SHA256 = "4b73da583a4b4e4abef4bb3ab496dc12f716ed484ea0e4066ad45d6952856fef"
SSL_URL = "https://huggingface.co/roman-bushuiev/DreaMS/resolve/main/ssl_model.ckpt"
N_PEAKS = 60            # n_highest_peaks of the SSL pre-training (fine-tuned models use 100)
PREC_INTENS = 1.1
MAX_MZ = 1000.0         # DataFormatA.max_mz (m/z normalisation of ff_peak, top of the integer Fourier range)
X_MIN = 1e-4            # DataFormatA.max_tbxic_stdev (bottom of the float Fourier range)
D_PEAK, D_FOURIER, D_MODEL = 44, 980, 1024
N_LAYERS, N_HEADS = 7, 8
FF_FOURIER_DEPTH, FF_FOURIER_D = 5, 512
HOT_BIN = 0.05          # hot_mz_bin_size; ff_out has MAX_MZ / HOT_BIN = 20000 classes
N_HOT = 20000
# masked-peak objective of the checkpoint (hparams: train_objective mask_mz_hot, frac_masks 0.3, mask_val -1,
# mask_intens_strategy intens_p, mask_prec False, bert801010 False; MaskedSpectraDataset default min_n_masks 2)
MASK_VAL = -1.0
FRAC_MASKS = 0.3
MIN_N_MASKS = 2
BACKBONE_PREFIXES = ("fourier_enc.b", "ff_fourier.", "ff_peak.", "transformer_encoder.")
HEAD_PREFIXES = ("ff_out.",)


def fourier_freqs(x_min=X_MIN, x_max=MAX_MZ):
    """FourierFeatures(strategy='lin_float_int').b as float32 (1, 5997): python floats -> torch.tensor (float32)."""
    f = ([1 / (x_min * i) for i in range(2, math.ceil(1 / x_min), 2)]
         + [1 / (1 * i) for i in range(2, math.ceil(x_max), 1)])
    return np.asarray(f, np.float32)[None]


# ---------------------------------------------------------------------------------------------- checkpoint loading
def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class _Stub:
    """Inert stand-in for a global outside the allow-list: accepts any construction / state and does nothing."""

    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        pass

    def __setitem__(self, k, v):
        pass

    def append(self, v):
        pass

    def extend(self, v):
        pass


class _StorageType:
    def __init__(self, name):
        self.name = name


class _LazyStorage:
    def __init__(self, key, stype, numel):
        self.key, self.stype, self.numel = key, stype, numel


class _LazyTensor:
    def __init__(self, storage, offset, size, stride):
        self.storage, self.offset, self.size, self.stride = storage, int(offset), tuple(size), tuple(stride)


def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad=False, backward_hooks=None,
                       metadata=None):
    return _LazyTensor(storage, storage_offset, size, stride)


def _rebuild_parameter(data, requires_grad=False, backward_hooks=None, state=None):
    return data


_NUMPY_OK = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
             ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
             ("numpy", "ndarray"), ("numpy", "dtype")}


class _RestrictedUnpickler(pickle.Unpickler):
    """Resolves only OrderedDict, tensor / parameter rebuilds, torch storage types and numpy reconstructors."""

    def __init__(self, f):
        super().__init__(f)
        self.stubbed = collections.Counter()

    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2
        if module == "torch._utils" and name in ("_rebuild_parameter", "_rebuild_parameter_with_state"):
            return _rebuild_parameter
        if module == "torch" and name.endswith("Storage"):
            return _StorageType(name)
        if (module, name) in _NUMPY_OK:
            import importlib
            return getattr(importlib.import_module(module), name)
        self.stubbed[(module, name)] += 1
        return type(name, (_Stub,), {"__module__": "casmi26.dreams.stub." + module})

    def persistent_load(self, pid):
        if not (isinstance(pid, tuple) and len(pid) >= 5 and pid[0] == "storage"):
            raise pickle.UnpicklingError(f"unexpected persistent id {pid!r}")
        stype = pid[1]
        name = stype.name if isinstance(stype, _StorageType) else str(stype)
        return _LazyStorage(str(pid[2]), name, int(pid[4]))


_STORAGE_DTYPES = {"FloatStorage": "float32", "DoubleStorage": "float64", "HalfStorage": "float16",
                   "BFloat16Storage": "bfloat16", "LongStorage": "int64", "IntStorage": "int32",
                   "ShortStorage": "int16", "CharStorage": "int8", "ByteStorage": "uint8", "BoolStorage": "bool"}


def load_ssl(path, sha256=SSL_SHA256, heads=True, full=False, log=print):
    """ssl_model.ckpt -> {name: float32 tensor} without executing anything from the pickle.

    sha256: expected hash (None skips the check). heads=True keeps the SSL m/z head ff_out.* (for the self-check);
    full=True returns the whole Lightning state_dict (68 tensors, incl. ro_out and the duplicate
    'fourier_enc.Fourier frequencies'; only for loading the official module in the equivalence test)."""
    import torch
    if sha256 is not None:
        got = sha256_file(path)
        if got != sha256:
            raise ValueError(f"sha256 mismatch for {path}: {got} != {sha256}")
    if sys.byteorder != "little":
        raise RuntimeError("little-endian host required")
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        pkl = [n for n in names if n.endswith("/data.pkl") or n == "data.pkl"]
        if len(pkl) != 1:
            raise ValueError(f"not a torch zip checkpoint: {pkl}")
        prefix = pkl[0][: -len("data.pkl")]
        if prefix + "byteorder" in names and z.read(prefix + "byteorder").decode().strip() != "little":
            raise ValueError("big-endian checkpoint")
        with z.open(pkl[0]) as f:
            up = _RestrictedUnpickler(f)
            obj = up.load()
        sd = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
        if not isinstance(sd, dict):
            raise ValueError("no state_dict in the checkpoint")
        cache = {}
        out = collections.OrderedDict()
        for k, v in sd.items():
            if not isinstance(v, _LazyTensor):
                continue
            keep = full or k.startswith(BACKBONE_PREFIXES) or (heads and k.startswith(HEAD_PREFIXES))
            if k == "fourier_enc.Fourier frequencies" and not full:
                keep = False
            if not keep:
                continue
            s = v.storage
            if s.key not in cache:
                dt = getattr(torch, _STORAGE_DTYPES[s.stype])
                raw = bytearray(z.read(f"{prefix}data/{s.key}"))
                flat = torch.frombuffer(raw, dtype=dt) if len(raw) else torch.empty(0, dtype=dt)
                if flat.numel() != s.numel:
                    raise ValueError(f"storage {s.key}: {flat.numel()} elements, expected {s.numel}")
                cache[s.key] = flat
            out[k] = torch.as_strided(cache[s.key], v.size, v.stride, v.offset).clone()
        # the duplicate frequency parameter must equal fourier_enc.b
        dup = sd.get("fourier_enc.Fourier frequencies")
        if isinstance(dup, _LazyTensor) and isinstance(sd.get("fourier_enc.b"), _LazyTensor):
            b = sd["fourier_enc.b"]
            if (dup.storage.key, dup.offset, dup.size) != (b.storage.key, b.offset, b.size):
                raise ValueError("fourier_enc.b and 'Fourier frequencies' are different storages")
    if log:
        st = ", ".join(f"{m}.{n} x{c}" for (m, n), c in sorted(up.stubbed.items()))
        log(f"[dreams] {len(out)} tensors kept from {path} (stubbed globals: {st or 'none'})")
    return out


def save_plain(sd, path):
    """Plain {name: tensor} file (for a private Kaggle dataset); read back with load_plain (weights_only=True)."""
    import torch
    torch.save({k: v.contiguous() for k, v in sd.items()}, path)


def load_plain(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)


# ---------------------------------------------------------------------------------------------- preprocessing
def preprocess(mz, it, prec, n=N_PEAKS, prec_intens=PREC_INTENS):
    """SpectrumPreprocessor(DataFormatA(), prec_intens=1.1, n_highest_peaks=n)(np.stack([mz, it]), prec_mz=prec,
    high_form=False) -> float32 (n + 1, 2): the n most intense peaks (np.argsort default kind, the kept peaks in their
    input order, i.e. ascending m/z for sorted input), zero padding, intensity / max, then [prec, prec_intens] in
    front. Raw m/z values (not normalised). Arithmetic in the input dtype as in the official code. A spectrum without
    peaks (or with max intensity <= 0) gives all-padding rows (the official code would fail)."""
    spec = np.array([np.asarray(mz), np.asarray(it)])
    out = np.zeros((n + 1, 2), np.float32)
    out[0, 0], out[0, 1] = np.float64(prec), prec_intens
    if spec.shape[1] == 0:
        return out
    idx = np.argsort(spec[1])[-n:]
    spec = spec[:, np.sort(idx)]
    mx = spec[1].max()
    if not mx > 0:
        return out
    k = spec.shape[1]
    out[1:k + 1, 0] = spec[0]
    out[1:k + 1, 1] = spec[1] / mx
    return out


def preprocess_batch(specs, n=N_PEAKS):
    """[(mz, it, prec), ...] -> float32 (B, n + 1, 2)."""
    return np.stack([preprocess(m, i, p, n) for m, i, p in specs]) if len(specs) else np.zeros((0, n + 1, 2),
                                                                                                  np.float32)


# ---------------------------------------------------------------------------------------------- SSL self-check
def ssl_mask(x, rs, frac=FRAC_MASKS, min_n=MIN_N_MASKS, mask_val=MASK_VAL):
    """MaskedSpectraDataset.get_spec masking (mask_mz_hot, intens_p, mask_prec False, no 80/10/10, no m/z shift) of
    one preprocessed spectrum x (n + 1, 2). rs: np.random.RandomState (same draws as the official np.random.choice
    after np.random.seed). Eligible peaks: 0 < intensity < 1 (no padding, no base peak, no precursor token);
    n_masks = max(min_n, round(n_eligible * frac)) drawn without replacement with p ~ intensity; with
    n_eligible <= n_masks every eligible peak is masked. -> (x with m/z of masked peaks = mask_val, bool mask)."""
    mask = x[:, 1] > 0
    mask &= x[:, 1] < 1
    n_peaks = mask.sum()
    n_masks = max(min_n, round(n_peaks * frac))
    if n_peaks > n_masks:
        idx = np.where(mask)[0]
        p = x[idx, 1] / x[idx, 1].sum()
        mask_i = rs.choice(idx, size=n_masks, p=p, replace=False)
        mask[:] = False
        mask[mask_i] = True
    xm = x.copy()
    xm[mask, 0] = mask_val
    return xm, mask


def mz_classes(mz):
    """su.to_classes(mz, max_val=1000, bin_size=0.05): round(mz / 0.05) clamped to 19999 (torch, float32)."""
    import torch
    c = torch.round(torch.as_tensor(mz, dtype=torch.float32) / HOT_BIN).long()
    return c.clamp(max=N_HOT - 1)


def ssl_accuracy(enc, X, frac=FRAC_MASKS, seed=0, batch=128, device="cuda", topk=(1, 5), mask_context=False,
                 per_spec=False):
    """Masked-peak m/z accuracy of the SSL head on preprocessed spectra X (B, n + 1, 2): spectrum i is masked with
    RandomState(seed + i). -> dict(n_spec, n_masked, top1, top5, within1 (|class diff| <= 1), uniform chance
    1 / 20000, majority (share of the most frequent target class among the masked peaks)).
    mask_context=True (control): the m/z of every eligible peak (0 < intensity < 1) is masked as well, scored on the
    same positions (only the precursor token, the base peak and the intensities remain as context).
    per_spec=True adds hit1 / n_mask arrays per spectrum (int32)."""
    import torch
    assert enc.ff_out is not None, "build the encoder with with_head=True"
    xs, ms = [], []
    for i in range(len(X)):
        xm, m = ssl_mask(X[i], np.random.RandomState(seed + i), frac=frac)
        if mask_context:
            xm[(X[i, :, 1] > 0) & (X[i, :, 1] < 1), 0] = MASK_VAL
        xs.append(xm)
        ms.append(m)
    XM, M = np.stack(xs), np.stack(ms)
    hit = {k: 0 for k in topk}
    near, tot, tgt_all = 0, 0, []
    hit1_spec = np.zeros(len(X), np.int32)
    enc.eval()
    with torch.inference_mode():
        for a in range(0, len(XM), batch):
            xb = torch.from_numpy(XM[a:a + batch]).to(device)
            mb = torch.from_numpy(M[a:a + batch]).to(device)
            if not bool(mb.any()):
                continue
            tok = enc.forward_tokens(xb)
            logits = enc.ff_out(tok[mb].float())
            tgt = mz_classes(torch.from_numpy(X[a:a + batch, :, 0]).to(device)[mb])
            top = logits.topk(max(topk), -1).indices
            for k in topk:
                hit[k] += int((top[:, :k] == tgt[:, None]).any(-1).sum())
            sidx = torch.nonzero(mb)[:, 0].cpu().numpy() + a
            np.add.at(hit1_spec, sidx, (top[:, 0] == tgt).cpu().numpy().astype(np.int32))
            near += int(((top[:, 0] - tgt).abs() <= 1).sum())
            tot += int(tgt.numel())
            tgt_all.append(tgt.cpu().numpy())
    t = np.concatenate(tgt_all) if tgt_all else np.zeros(0, np.int64)
    maj = float(np.bincount(t).max() / len(t)) if len(t) else float("nan")
    res = dict(n_spec=int(len(X)), n_masked=int(tot), within1=near / max(tot, 1), chance_uniform=1.0 / N_HOT,
               majority=maj, frac=frac, mask_context=bool(mask_context))
    for k in topk:
        res[f"top{k}"] = hit[k] / max(tot, 1)
    if per_spec:
        res["hit1"], res["n_mask"] = hit1_spec, M.sum(1).astype(np.int32)
    return res


# ---------------------------------------------------------------------------------------------- model
_CLASSES = {}


def _classes():
    if _CLASSES:
        return _CLASSES
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class FeedForward(nn.Module):
        """dreams/models/layers/feed_forward.py: depth x (Linear [, Dropout, ReLU]); ReLU after the last Linear when
        act_last (module indices kept so the official keys load: ff.0 / ff.3 / ... )."""

        def __init__(self, in_dim, out_dim, hidden_dim, depth, act_last=True, bias=True):
            super().__init__()
            layers = []
            for i in range(depth):
                d1 = hidden_dim if i else in_dim
                d2 = hidden_dim if i != depth - 1 else out_dim
                layers.append(nn.Linear(d1, d2, bias=bias))
                if i != depth - 1:
                    layers.append(nn.Dropout(0.0))
                if i != depth - 1 or act_last:
                    layers.append(nn.ReLU())
            self.ff = nn.Sequential(*layers)

        def forward(self, x):
            return self.ff(x)

    class Fourier(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("b", torch.from_numpy(fourier_freqs()))

        def forward(self, x):
            """x (..., 1) raw m/z -> (..., 2 F) [cos, sin] of (2 pi x) * b, fp32."""
            ph = (2 * torch.pi * x.float()) * self.b
            return torch.cat((torch.cos(ph), torch.sin(ph)), dim=-1)

    class Attention(nn.Module):
        """dreams/models/dreams/layers.py MultiheadAttention (dot-product, no bias, non-parametrised graphormer)."""

        def __init__(self, d=D_MODEL, h=N_HEADS):
            super().__init__()
            self.d, self.h, self.hd = d, h, d // h
            self.scale = self.hd ** -0.5
            self.weights = nn.Parameter(torch.empty(4 * d, d))
            nn.init.normal_(self.weights, mean=0.0, std=(2 / (5 * d)) ** 0.5)   # official init (unused once loaded)

        def forward(self, x, pad, bias):
            bs, n, d = x.shape
            q, k, v = F.linear(x, self.weights[: 3 * d]).chunk(3, dim=-1)

            def split(t):
                return t.reshape(bs, n, self.h, self.hd).transpose(1, 2)

            q, k, v = split(q), split(k), split(v)
            with torch.autocast(device_type=x.device.type, enabled=False):
                att = torch.einsum("bhnd,bhdm->bhnm", q.float(), k.float().transpose(-2, -1))
                att = att * self.scale
                att = att + bias
                att.masked_fill_(pad.unsqueeze(1).unsqueeze(-1), -1e9)   # query rows, as in the official code
                att = F.softmax(att, dim=-1)
            att = att.to(v.dtype).reshape(-1, n, n)
            out = torch.bmm(att, v.reshape(bs * self.h, -1, self.hd))
            out = out.reshape(bs, self.h, n, self.hd).transpose(1, 2).reshape(bs, n, -1)
            return F.linear(out, self.weights[3 * d:])

    class TFeedForward(nn.Module):
        def __init__(self, d=D_MODEL):
            super().__init__()
            self.in_proj = nn.Linear(d, 4 * d, bias=False)
            self.out_proj = nn.Linear(4 * d, d, bias=False)

        def forward(self, x):
            return self.out_proj(F.relu(self.in_proj(x)))

    class TransformerEncoder(nn.Module):
        def __init__(self, d=D_MODEL, n_layers=N_LAYERS):
            super().__init__()
            self.n_layers = n_layers
            self.atts = nn.ModuleList([Attention(d) for _ in range(n_layers)])
            self.ffs = nn.ModuleList([TFeedForward(d) for _ in range(n_layers)])
            self.scales = nn.ModuleList([nn.LayerNorm(d) for _ in range(2 * n_layers + 1)])

        def forward(self, x, pad, bias):
            for i in range(self.n_layers):
                r = x
                x = r + self.atts[i](self.scales[2 * i](x), pad, bias)
                r = x
                x = r + self.ffs[i](self.scales[2 * i + 1](x))
            return self.scales[-1](x)

    class DreamsEncoder(nn.Module):
        """forward(x [B, n+1, 2] preprocessed, raw m/z) -> [B, 1024] (last layer, precursor token);
        forward_tokens -> [B, n+1, 1024]. with_head adds ff_out (SSL m/z head, 20000 classes)."""

        def __init__(self, with_head=False, fp16_blocks=False, naive_bias=False, n_peaks=N_PEAKS):
            super().__init__()
            self.fp16_blocks, self.naive_bias, self.n_peaks = fp16_blocks, naive_bias, n_peaks
            self.fourier_enc = Fourier()
            self.ff_fourier = FeedForward(2 * self.fourier_enc.b.shape[1], D_FOURIER, FF_FOURIER_D, FF_FOURIER_DEPTH)
            self.ff_peak = FeedForward(2, D_PEAK, D_PEAK, 1)
            self.transformer_encoder = TransformerEncoder()
            self.ff_out = FeedForward(D_MODEL, N_HOT, D_MODEL, 1, act_last=False) if with_head else None

        def load_ssl_state(self, sd):
            """Load a load_ssl / load_plain state dict (backbone keys, plus ff_out.* when with_head)."""
            want = set(self.state_dict().keys())
            sub = {k: v for k, v in sd.items() if k in want}
            missing = sorted(want - set(sub))
            if missing:
                raise KeyError(f"missing DreaMS parameters: {missing[:5]} ... ({len(missing)})")
            self.load_state_dict(sub, strict=True)
            return self

        def forward_tokens(self, x):
            x = x.float()
            pad = x[:, :, 0] == 0
            with torch.autocast(device_type=x.device.type, enabled=False):
                peak = self.ff_peak(x / torch.tensor([MAX_MZ, 1.0], device=x.device, dtype=x.dtype))
                four = self.ff_fourier(self.fourier_enc(x[..., [0]]))
                h = torch.cat([peak, four], dim=-1)
                if self.naive_bias:
                    bias = (four.unsqueeze(2) - four.unsqueeze(1)).sum(dim=-1).unsqueeze(1)
                else:
                    s = four.sum(dim=-1)
                    bias = (s.unsqueeze(2) - s.unsqueeze(1)).unsqueeze(1)
            if self.fp16_blocks and x.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = self.transformer_encoder(h, pad, bias)
                return out.float()
            return self.transformer_encoder(h, pad, bias)

        def forward(self, x):
            return self.forward_tokens(x)[:, 0]

    _CLASSES.update(DreamsEncoder=DreamsEncoder, FeedForward=FeedForward)
    return _CLASSES


def __getattr__(name):
    if name == "DreamsEncoder":
        return _classes()["DreamsEncoder"]
    raise AttributeError(name)


def build_encoder(sd=None, with_head=False, fp16_blocks=False, naive_bias=False, device=None):
    """DreamsEncoder in eval mode, loaded from a load_ssl / load_plain state dict when given."""
    enc = _classes()["DreamsEncoder"](with_head=with_head, fp16_blocks=fp16_blocks, naive_bias=naive_bias)
    if sd is not None:
        enc.load_ssl_state(sd)
    if device is not None:
        enc = enc.to(device)
    return enc.eval()


def embed(enc, specs_iter, batch=128, device="cuda", log=None, log_every=100000):
    """Embeddings of an iterable of spectra -> float16 numpy (N, 1024). Items are (mz, it, prec) tuples or
    preprocessed (n + 1, 2) arrays."""
    import time

    import torch
    out, buf, n_done, t0 = [], [], 0, time.time()

    def flush():
        nonlocal n_done
        x = torch.from_numpy(np.stack(buf)).to(device, non_blocking=True)
        out.append(enc(x).to(torch.float16).cpu().numpy())
        n_prev, n_done = n_done, n_done + len(buf)
        buf.clear()
        if log and n_done // log_every > n_prev // log_every:
            log(f"[dreams] {n_done:,} spectra embedded ({n_done / (time.time() - t0):.0f}/s)")

    enc.eval()
    with torch.inference_mode():
        for s in specs_iter:
            buf.append(s if isinstance(s, np.ndarray) else preprocess(s[0], s[1], s[2], enc.n_peaks))
            if len(buf) == batch:
                flush()
        if buf:
            flush()
    return np.concatenate(out) if out else np.zeros((0, D_MODEL), np.float16)

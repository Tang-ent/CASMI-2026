"""msalign checks (plan 10.2 / 10.3, 2026-10-24), in docker with the real k01 cache, k05 pool (+ k05b mass fix) and
k26 formula table under /kaggle/input, synthetic spectrum / ChemBERTa-like tables (tools/local/run_check_msalign.sh):
(a) train_rows: the fpnet.TrainData rule, <= 4 rows per structure, fixed by the seed
(b) negatives: fpnet.Cfg.neg_train_only restored; mass = train non-holdout within +-10 ppm, formula = same fid; distinct,
    without the truth, min(127, available) per row
(c) shapes: towers of plan 10.1, CE Fourier features (numpy formula, NaN -> learned vector), blind ignores the spectrum
(d) masking: padded logits are -inf, the loss equals the cross-entropy over the valid candidates, gradients finite
(e) toy problem (spectrum embedding = a linear map of the truth's molecule vector + noise): the loss falls and the
    in-batch accuracy rises for cb and fp; the blind model stays near chance
(f) resume: N steps straight == N/2 (stop_at) + resume (CPU fp32: identical weights; GPU AMP: reported); cfg change
    refused
(g) scoring: score_windows == a brute-force numpy forward (float64) on real query windows; msa_score_fn agrees;
    blind scores are the same for every spectrum
(h) provenance: msalign.provenance_coef with the f.z score == fpnet.provenance_coef; the default fpnet.provenance_coef
    equals the reference commit's (mounted at /head/fpnet.py)
(i) random_mrr closed form and a simulation; window_mrr ties
(j) speed of the full recipe (B 128 x K 128) on the GPU
"""
import copy
import importlib.util
import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "/src")
import torch  # noqa: E402

from casmi26 import cache, fpnet, msalign, pool, search, subform  # noqa: E402

T0 = time.time()
OK = []


def check(name, cond, detail=""):
    OK.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}", flush=True)


def tm():
    return f"({time.time() - T0:.0f}s)"


GPU = torch.cuda.is_available()
print("torch", torch.__version__, "| cuda", GPU, torch.cuda.get_device_name(0) if GPU else "", flush=True)
C = cache.Cache(cache.Cache.find_dir())
PD = pool.Pool.find_dir()
P = pool.Pool(PD, pubchem=False)
FT = subform.FormulaTable(subform.FormulaTable.find_dir())
Q = pd.read_csv(f"{PD}/queries.csv")
hold_df = pd.read_csv(f"{PD}/holdout.csv")
HOLD = set(hold_df.pid.values)
hold_arr = np.asarray(sorted(HOLD), np.int64)
print(f"inputs loaded {tm()}", flush=True)

# ---------------------------------------------------------------------------------------------- (a) train rows
rows, pids = msalign.train_rows(C, P, HOLD, 4, seed=0)
rows2, _ = msalign.train_rows(C, P, HOLD, 4, seed=0)
rows3, _ = msalign.train_rows(C, P, HOLD, 4, seed=1)
D = fpnet.TrainData(C, P, HOLD)
cnt_all = pd.Series(D.pid).value_counts()
cnt_sel = pd.Series(pids).value_counts()
check("a1 rows are a subset of the TrainData rows", np.isin(rows, D.rows).all(), f"{len(rows):,} of {len(D):,}")
check("a2 <= 4 per structure and min(4, all) for every structure",
      (cnt_sel.reindex(cnt_all.index).fillna(0).values == np.minimum(cnt_all.values, 4)).all(),
      f"{len(cnt_sel):,} structures")
check("a3 no holdout structure, pid = sid2pid", (~np.isin(pids, hold_arr)).all() and
      (P.sid2pid[C.sid[rows]] == pids).all() and C.keep[rows].all())
check("a4 fixed by the seed", np.array_equal(rows, rows2) and not np.array_equal(rows, rows3))
del D

# ---------------------------------------------------------------------------------------------- (b) negatives
before = fpnet.Cfg.neg_train_only
NM = msalign.NegTable.mass(C, P, HOLD)
check("b1 fpnet.Cfg.neg_train_only restored", fpnet.Cfg.neg_train_only == before, f"{before}")
NF = msalign.NegTable.formula(P, FT, HOLD)
rng = np.random.default_rng(5)
tp = rng.choice(np.unique(pids), 2000, replace=False)
for N, nm in [(NM, "mass"), (NF, "formula")]:
    cand = N.draw(tp, 127, np.random.default_rng(7))
    avail = N.sizes(tp)
    good = True
    for b, p in enumerate(tp):
        c = cand[b, 1:]
        c = c[c >= 0]
        good &= cand[b, 0] == p and len(c) == min(127, avail[b]) and len(np.unique(c)) == len(c) and (c != p).all()
        good &= bool((P.in_train[c] & ~np.isin(c, hold_arr)).all())
        if nm == "mass":
            good &= bool((np.abs(P.mass[c] - P.mass[p]) <= P.mass[p] * 10e-6 * (1 + 1e-9)).all())
        else:
            good &= bool((FT.fid[c] == FT.fid[p]).all())
        good &= bool((cand[b, 1 + len(c):] == -1).all())
    full = (cand[:, 1:] >= 0).sum(1)
    check(f"b2 {nm} negatives", good, f"median avail {np.median(avail):.0f}, none {np.mean(avail == 0):.3f}, "
          f"full 127 {np.mean(full == 127):.3f}")
    c2 = N.draw(tp, 127, np.random.default_rng(8))
    check(f"b3 {nm} drawn afresh (another rng differs)", not np.array_equal(cand, c2))

# ---------------------------------------------------------------------------------------------- shared synthetic data
N_TOY = 20000
qsel = Q[Q.set.isin(["A", "V", "R2"])].groupby("set").head(15)
q_rows = [np.asarray([int(x) for x in str(r).split(",")][:16], np.int64) for r in qsel.rows]
toy_ix = np.sort(rng.choice(len(rows), N_TOY, replace=False))
toy_rows = rows[toy_ix]
emb_rows = np.unique(np.concatenate([toy_rows] + q_rows))
rmap = msalign.row_map(emb_rows, len(C.sid))
n_pid = len(P.key)
g = torch.Generator().manual_seed(0)
CB = (torch.randn(n_pid, 768, generator=g)).half().numpy()  # ChemBERTa-like random table
W_cb = (torch.randn(768, 1024, generator=g) / 768 ** 0.5).numpy()
W_fp = (torch.randn(2048, 1024, generator=g) / 45.0).numpy()
mol_cb = msalign.MolTable(CB, "cb")
mol_fp = msalign.MolTable(P.fp, "fp", 2048)
pid_of = np.full(len(C.sid), -1, np.int64)
pid_of[emb_rows] = P.sid2pid[C.sid[emb_rows]]


def spec_table(kind, noise=0.5):
    """Synthetic E_spec for emb_rows: a linear map of the truth's molecule input + noise (fp16)."""
    pp = pid_of[emb_rows]
    pp = np.where(pp >= 0, pp, 0)
    if kind == "cb":
        X = CB[pp].astype(np.float32) @ W_cb
    else:
        X = np.unpackbits(P.fp[pp], axis=1)[:, :2048].astype(np.float32) @ W_fp
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    return (X + noise * np.random.default_rng(1).standard_normal(X.shape)).astype(np.float16)


vocab = msalign.make_vocab(C, rows)
ad, pol, ce = msalign.meta_of(C, toy_rows, vocab)
TS = msalign.TrainSet(toy_rows, P.sid2pid[C.sid[toy_rows]], msalign.emb_index(toy_rows, rmap), ad, pol, ce, vocab)
print(f"vocab {len(vocab)} adducts, toy {len(TS):,} rows, CE NaN {np.isnan(ce).mean():.3f}, pol {np.bincount(pol)} "
      f"{tm()}", flush=True)

# ---------------------------------------------------------------------------------------------- (c) shapes
cfg = msalign.Cfg()
net = msalign.build_model(768, len(vocab), cfg).eval()
lin = lambda s: [tuple(m.weight.shape[::-1]) for m in s if isinstance(m, torch.nn.Linear)]  # noqa: E731
check("c1 towers", lin(net.ms_pre) == [(1024, 512)] and lin(net.ms_post) == [(576, 2048), (2048, 2048), (2048, 512)]
      and lin(net.mol) == [(768, 2048), (2048, 2048), (2048, 2048), (2048, 512)] and
      [type(m).__name__ for m in net.ms_post][:4] == ["Linear", "LayerNorm", "GELU", "Dropout"] and
      net.ms_post[3].p == 0.3 and abs(float(net.log_temp.exp()) - 0.07) < 1e-7,
      f"params {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M")
e = torch.tensor([0.0, 35.0, float("nan"), 120.5])
enc = net.ce(e).detach().numpy()
fr = 10000.0 ** (np.arange(0, 32, 2) / 32)
ref = np.zeros((4, 32))
for i, v in enumerate([0.0, 35.0, np.nan, 120.5]):
    a = (0 if np.isnan(v) else v) / 100 / fr
    ref[i, 0::2], ref[i, 1::2] = np.sin(a), np.cos(a)
ref[2] = net.ce.missing.detach().numpy()
check("c2 CE Fourier features", np.abs(enc - ref).max() < 1e-5, f"max diff {np.abs(enc - ref).max():.2e}")
x = torch.randn(5, 1024)
a5, p5, c5 = torch.zeros(5, dtype=torch.long), torch.ones(5, dtype=torch.long), torch.full((5,), 30.0)
u = net.encode_spectrum(x, a5, p5, c5)
m = net.encode_molecule(torch.randn(7, 768))
check("c3 projection shapes", tuple(u.shape) == (5, 512) and tuple(m.shape) == (7, 512))
bnet = msalign.build_model(768, len(vocab), msalign.Cfg(blind=True)).eval()
ub1 = bnet.encode_spectrum(None, a5, p5, c5)
ub2 = bnet.encode_spectrum(torch.randn(5, 1024), torch.arange(5) % len(vocab), torch.tensor([0, 1, 2, 0, 1]),
                           torch.tensor([float("nan"), 1, 2, 3, 4.0]))
check("c4 blind ignores spectrum and metadata", torch.equal(ub1, ub2) and torch.equal(ub1[0], ub1[4]))

# ---------------------------------------------------------------------------------------------- (d) masking
net.train()
cand = torch.tensor([[0, 1, 2, -1, -1], [3, 4, -1, -1, -1], [5, 6, 7, 8, 9]])
mol_in = torch.randn(int((cand >= 0).sum()), 768)
xs = torch.randn(3, 1024)
torch.manual_seed(3)
s = msalign.candidate_logits(net.eval(), xs, a5[:3], p5[:3], c5[:3], mol_in, cand)
loss = msalign.candidate_loss(s)
loss.backward()
uu = torch.nn.functional.normalize(net.encode_spectrum(xs, a5[:3], p5[:3], c5[:3]), dim=-1)
mm = torch.nn.functional.normalize(net.encode_molecule(mol_in), dim=-1)
man, j = [], 0
for b in range(3):
    k = int((cand[b] >= 0).sum())
    lg = (mm[j:j + k] @ uu[b]) / net.log_temp.exp()
    man.append(-torch.log_softmax(lg, 0)[0])
    j += k
man = torch.stack(man).mean()
grads = [p.grad for p in net.parameters() if p.grad is not None]
check("d1 padding -inf, loss = CE over valid candidates, finite gradients",
      bool(torch.isinf(s[cand < 0]).all() and torch.isfinite(s[cand >= 0]).all()) and abs(float(loss - man)) < 1e-5
      and all(bool(torch.isfinite(gr).all()) for gr in grads), f"loss {float(loss):.5f} manual {float(man):.5f}")
try:
    msalign.candidate_logits(net, xs, a5[:3], p5[:3], c5[:3], mol_in, torch.tensor([[-1, 1, 2, 3, 4]] * 3))
    check("d2 invalid truth refused", False)
except AssertionError:
    check("d2 invalid truth refused", True)
net.zero_grad()

# ---------------------------------------------------------------------------------------------- (e) toy training
W = tempfile.mkdtemp(dir="/w")
toy = {}
dev = "cuda" if GPU else "cpu"
E_cb, E_fp = spec_table("cb"), spec_table("fp")
for name, E, mol, blind in [("cb_mass", E_cb, mol_cb, False), ("fp_mass", E_fp, mol_fp, False),
                            ("cb_blind", E_cb, mol_cb, True)]:
    c = msalign.Cfg(name=name, blind=blind, max_steps=600, warmup=50, log_every=100, save_every=300)
    h = msalign.train(E, TS, mol, NM, f"{W}/{name}.pt", cfg=c, device=dev, seed=0)
    toy[name] = h
    print(pd.DataFrame(h)[["step", "loss", "acc", "temp", "cand", "with_neg"]].to_string(index=False), flush=True)
for name in ["cb_mass", "fp_mass"]:
    h = toy[name]
    check(f"e1 {name} loss falls, accuracy rises", h[-1]["loss"] < 0.5 * h[0]["loss"] and h[-1]["acc"] > 0.5,
          f"loss {h[0]['loss']:.3f} -> {h[-1]['loss']:.3f}, acc {h[0]['acc']:.3f} -> {h[-1]['acc']:.3f}")
hb = toy["cb_blind"]
check("e2 blind stays near chance on the toy (no molecule prior in random cb)",
      hb[-1]["acc"] < 0.1 and hb[-1]["loss"] > 0.8 * hb[0]["loss"],
      f"loss {hb[0]['loss']:.3f} -> {hb[-1]['loss']:.3f}, acc {hb[-1]['acc']:.3f}")
check("e3 weights + state written", os.path.exists(f"{W}/cb_mass.pt") and os.path.exists(f"{W}/cb_mass_state.pt"))

# ---------------------------------------------------------------------------------------------- (f) resume
def weights(path):
    return torch.load(path, map_location="cpu", weights_only=True)["model"]


def max_diff(a, b):
    A, B = weights(a), weights(b)
    assert A.keys() == B.keys()
    return max(float((A[k].double() - B[k].double()).abs().max()) for k in A)


small = dict(batch=16, k_cand=16, d_hidden=256, max_steps=20, warmup=5, log_every=5, save_every=7)
cs = msalign.Cfg(name="res", **small)
sub = TS.subset(np.arange(2000))
n_thr = torch.get_num_threads()
torch.set_num_threads(1)  # CPU kernels with one thread: run-to-run bit-identical
msalign.train(E_cb, sub, mol_cb, NF, f"{W}/r_full.pt", cfg=cs, device="cpu", seed=3)
msalign.train(E_cb, sub, mol_cb, NF, f"{W}/r_full2.pt", cfg=cs, device="cpu", seed=3)
d_ctl = max_diff(f"{W}/r_full.pt", f"{W}/r_full2.pt")
check("f0 CPU control: two straight runs are identical", d_ctl == 0.0, f"max |dw| {d_ctl:.3g}")
msalign.train(E_cb, sub, mol_cb, NF, f"{W}/r_half.pt", cfg=cs, device="cpu", seed=3, stop_at=10)
st_half = torch.load(f"{W}/r_half_state.pt", map_location="cpu", weights_only=False)["extra"]["step"]
h_res = msalign.train(E_cb, sub, mol_cb, NF, f"{W}/r_half.pt", cfg=cs, device="cpu", seed=3,
                      resume=f"{W}/r_half_state.pt")
d_cpu = max_diff(f"{W}/r_full.pt", f"{W}/r_half.pt")
h_full = torch.load(f"{W}/r_full.pt", weights_only=True)["hist"]
same_hist = [{k: v for k, v in r.items() if k != "s_per_step"} for r in h_full] == \
    [{k: v for k, v in r.items() if k != "s_per_step"} for r in h_res]
check("f1 CPU: 20 straight == 10 + resume 10 (weights, history)", st_half == 10 and d_cpu == 0.0 and same_hist,
      f"max |dw| {d_cpu:.3g}, history equal {same_hist}")
torch.set_num_threads(n_thr)
try:
    msalign.train(E_cb, sub, mol_cb, NF, f"{W}/r_half.pt", cfg=msalign.Cfg(name="res", **dict(small, lr=2e-4)),
                  device="cpu", seed=3, resume=f"{W}/r_half_state.pt")
    check("f2 changed cfg refused", False)
except ValueError:
    check("f2 changed cfg refused", True)
# (f4, 2026-10-24) tag=: the identity of the input tables goes into cfg["tag"] of the weights and state files; the
# default (tag=None) cfg dump has no such key; a resume with another tag (or none) is refused; the tag does not change
# the training; k50 reads it from both files with weights_only=True
TAG = dict(rows_sha="abc", spec_sha="def", fp16_blocks=True, chemberta_sha256=None)
torch.set_num_threads(1)
msalign.train(E_cb, sub, mol_cb, NF, f"{W}/t_half.pt", cfg=cs, device="cpu", seed=3, stop_at=10, tag=TAG)
ck_w = torch.load(f"{W}/t_half.pt", map_location="cpu", weights_only=True)
ck_s = torch.load(f"{W}/t_half_state.pt", map_location="cpu", weights_only=True)
no_tag = "tag" not in torch.load(f"{W}/r_full.pt", map_location="cpu", weights_only=True)["cfg"]
check("f4a tag stored in weights and state (weights_only readable); default cfg has no tag",
      ck_w["cfg"].get("tag") == TAG and ck_s["extra"]["cfg"].get("tag") == TAG and no_tag,
      f"weights {ck_w['cfg'].get('tag')}, state {ck_s['extra']['cfg'].get('tag')}, default without tag {no_tag}")
refused = []
for bad in [dict(TAG, spec_sha="xyz"), None]:
    try:
        msalign.train(E_cb, sub, mol_cb, NF, f"{W}/t_bad.pt", cfg=cs, device="cpu", seed=3,
                      resume=f"{W}/t_half_state.pt", tag=bad)
        refused.append(False)
    except ValueError:
        refused.append(True)
check("f4b resume with another tag / no tag refused", all(refused), str(refused))
msalign.train(E_cb, sub, mol_cb, NF, f"{W}/t_half.pt", cfg=cs, device="cpu", seed=3, resume=f"{W}/t_half_state.pt",
              tag=TAG)
d_tag = max_diff(f"{W}/r_full.pt", f"{W}/t_half.pt")
check("f4c same tag resumes; weights == the untagged straight run", d_tag == 0.0, f"max |dw| {d_tag:.3g}")
torch.set_num_threads(n_thr)
if GPU:
    cg = msalign.Cfg(name="resg", max_steps=40, warmup=10, log_every=10, save_every=100)
    msalign.train(E_cb, TS, mol_cb, NM, f"{W}/g_full.pt", cfg=cg, device="cuda", seed=4)
    msalign.train(E_cb, TS, mol_cb, NM, f"{W}/g_half.pt", cfg=cg, device="cuda", seed=4, stop_at=20)
    msalign.train(E_cb, TS, mol_cb, NM, f"{W}/g_half.pt", cfg=cg, device="cuda", seed=4,
                  resume=f"{W}/g_half_state.pt")
    d_gpu = max_diff(f"{W}/g_full.pt", f"{W}/g_half.pt")
    check("f3 GPU AMP full recipe: 40 straight vs 20 + resume 20 (report; <= 1e-3)", d_gpu <= 1e-3,
          f"max |dw| {d_gpu:.3g}")

# ---------------------------------------------------------------------------------------------- (g) scoring
from scipy.special import erf  # noqa: E402


def np_mlp(seq, x):
    x = np.asarray(x, np.float64)
    for mod in seq:
        if isinstance(mod, torch.nn.Linear):
            x = x @ mod.weight.detach().double().numpy().T + mod.bias.detach().double().numpy()
        elif isinstance(mod, torch.nn.LayerNorm):
            mu, var = x.mean(-1, keepdims=True), x.var(-1, keepdims=True)
            x = (x - mu) / np.sqrt(var + mod.eps) * mod.weight.detach().double().numpy() + \
                mod.bias.detach().double().numpy()
        elif isinstance(mod, torch.nn.GELU):
            x = 0.5 * x * (1 + erf(x / np.sqrt(2)))
    return x


def np_spec(netc, E, q):
    if netc.blind:
        n = len(q["adduct_ix"])
        h = np_mlp(netc.ms_pre, np.repeat(netc.const_spec.detach().double().numpy()[None], n, 0))
        mt = np.repeat(netc.const_meta.detach().double().numpy()[None], n, 0)
    else:
        h = np_mlp(netc.ms_pre, E[q["emb_ix"]].astype(np.float64))
        fr = 10000.0 ** (np.arange(0, netc.ce.freq.numel() * 2, 2) / (netc.ce.freq.numel() * 2))
        cev = np.zeros((len(q["ce"]), len(fr) * 2))
        for i, v in enumerate(q["ce"]):
            if np.isnan(v):
                cev[i] = netc.ce.missing.detach().double().numpy()
            else:
                a = v / 100 / fr
                cev[i, 0::2], cev[i, 1::2] = np.sin(a), np.cos(a)
        mt = np.c_[cev, netc.adduct.weight.detach().double().numpy()[q["adduct_ix"]],
                   netc.pol.weight.detach().double().numpy()[q["pol_ix"]]]
    u = np_mlp(netc.ms_post, np.c_[h, mt])
    return u / np.linalg.norm(u, axis=1, keepdims=True)


def np_mol(netc, mol, p):
    xin = CB[p].astype(np.float64) if mol.kind == "cb" else np.unpackbits(P.fp[p], axis=1)[:, :2048].astype(float)
    v = np_mlp(netc.mol, xin)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


queries = [msalign.query_inputs(C, r, rmap, vocab) for r in q_rows]
windows = [P.window(P.mass[p], 10.0) for p in qsel.pid]
truths = qsel.pid.values
for name, E, mol in [("cb_mass", E_cb, mol_cb), ("fp_mass", E_fp, mol_fp), ("cb_blind", E_cb, mol_cb)]:
    M = msalign.MSAModel(f"{W}/{name}.pt", spec_table=E, mol=mol, device=dev)
    out = msalign.score_windows(M, queries, windows, group=7)
    netc = copy.deepcopy(M.net).cpu().eval()
    md, mz, mr, sh = 0.0, 0.0, 0.0, True
    for q, w, o in zip(queries, windows, out):
        U = np_spec(netc, E, q)
        V = np_mol(netc, mol, w)
        s = (V @ U.T).mean(1)
        md = max(md, float(np.abs(o[:, 0] - s).max()))
        mz = max(mz, float(np.abs(o[:, 1] - search._zs(s)).max()))
        mr = max(mr, float(np.abs(o[:, 2] - search.rank_norm(o[:, 0])).max()))
        sh &= o.shape == (len(w), 3)
    f = msalign.msa_score_fn(M, queries)
    mf = max(float(np.abs(f(i, w) - o[:, 0]).max()) for i, (w, o) in enumerate(zip(windows, out)))
    check(f"g1 {name} score_windows == numpy brute force", sh and md < 2e-5 and mz < 2e-3 and mr == 0 and mf < 2e-5,
          f"msa {md:.2e}, z {mz:.2e}, rank {mr}, score_fn {mf:.2e}, windows {np.median([len(w) for w in windows]):.0f}")
    if name == "cb_blind":
        Us = [M.spec_proj(q).cpu().numpy() for q in queries]
        dmax = max(float(np.abs(u_ - Us[0][0]).max()) for u_ in Us)  # batch size changes the GEMM path: ~1e-8
        check("g2 blind: one projection for every spectrum (|diff| < 1e-6)", dmax < 1e-6, f"max |diff| {dmax:.2e}")
    rr = msalign.window_rr(out, windows, truths)
    print(f"   {name}: window MRR@25 {rr.mean():.4f} vs random {msalign.random_mrr([len(w) for w in windows]).mean():.4f}"
          f" (toy model, real windows; report only)", flush=True)

# ---------------------------------------------------------------------------------------------- (h) provenance
spec = importlib.util.spec_from_file_location("casmi26.fpnet_head", "/head/fpnet.py")
fh = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fh
spec.loader.exec_module(fh)
pq = hold_df[hold_df.set.isin(["R", "V"])].pid.values[:60]
Z = np.random.default_rng(11).standard_normal((len(pq), 2048)).astype(np.float32)
a_ = fpnet.provenance_coef(Z, P, pq, HOLD, n_boot=300)
b_ = fh.provenance_coef(Z, P, pq, HOLD, n_boot=300)
fz = lambda i, c: np.unpackbits(P.fp[c], axis=1)[:, :2048].astype(np.float32) @ Z[i]  # noqa: E731
c_ = msalign.provenance_coef(fz, P, pq, HOLD, n_boot=300)
check("h1 default provenance_coef == reference commit", a_ == b_, f"{a_}")
check("h2 provenance_coef(score_fn = f.z) == fpnet.provenance_coef", a_ == c_, f"{c_}")
M = msalign.MSAModel(f"{W}/cb_mass.pt", spec_table=E_cb, mol=mol_cb, device=dev)
pq2 = qsel.pid.values
d_ = msalign.provenance_coef(msalign.msa_score_fn(M, queries), P, pq2, HOLD, n_boot=300)
check("h3 provenance of a toy MSAlign model is finite", np.isfinite(d_["est"]), f"{d_}")

# ---------------------------------------------------------------------------------------------- (i) random / ties
rm = msalign.random_mrr([0, 1, 2, 30, 25])
H25 = sum(1 / r for r in range(1, 26))
check("i1 random_mrr closed form", np.allclose(rm, [0, 1, 0.75, H25 / 30, H25 / 25]), f"{rm}")
sim_rng = np.random.default_rng(0)
ns = sim_rng.integers(1, 200, 4000)
ws = [np.arange(n) for n in ns]
tr = [int(sim_rng.integers(0, n)) for n in ns]
sc = [sim_rng.random(n) for n in ns]
sim = msalign.window_rr(sc, ws, tr).mean()
exp_ = msalign.random_mrr(ns).mean()
check("i2 random scores reach random_mrr", abs(sim - exp_) < 0.01, f"sim {sim:.4f} vs {exp_:.4f}")
tie = msalign.window_ranks([np.array([1.0, 1.0, 0.5])], [np.array([5, 9, 7])], [9])
check("i3 ties: opt 1, pid 2, pess 2; truth absent -> rr 0",
      tuple(tie.iloc[0][["r_opt", "r_pid", "r_pess"]]) == (1, 2, 2) and
      msalign.window_rr([np.array([1.0])], [np.array([3])], [4])[0] == 0.0)
pc = msalign.paired_ci(np.ones(50), np.zeros(50))
check("i4 paired_ci", pc["est"] == 1.0 and pc["n"] == 50)

# ---------------------------------------------------------------------------------------------- (j) speed
if GPU:
    cj = msalign.Cfg(name="speed", max_steps=30000, log_every=100, save_every=10 ** 9)
    for name, E, mol in [("cb", E_cb, mol_cb), ("fp", E_fp, mol_fp)]:
        hj = msalign.train(E, TS, mol, NM, f"{W}/speed_{name}.pt", cfg=cj, device="cuda", seed=0, stop_at=300)
        sps = np.median([r["s_per_step"] for r in hj[1:]])
        print(f"   speed {name}: {sps:.4f} s/step -> 30,000 steps {sps * 30000 / 60:.0f} min on "
              f"{torch.cuda.get_device_name(0)}; peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB",
              flush=True)

n_fail = sum(not ok for _, ok in OK)
print(f"\n{len(OK) - n_fail}/{len(OK)} checks passed {tm()}")
sys.exit(1 if n_fail else 0)

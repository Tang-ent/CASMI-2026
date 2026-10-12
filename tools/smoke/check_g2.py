"""G2 checks (M6 P0-3, 2026-10-12), in docker on CPU with the smoke data (tools/smoke/run.sh g2):
(a) the atom-set enumeration gives the same unique fragment masses (>= 30 Da, 1e-4 Da) as frag.fragments_of_mol
(b) cosine: all mass on the matched slots -> the explained share of the observed norm; nothing matched -> 0 with
    finite zero gradients; two groups on one observed peak add up; a structure without fragments -> 0, no NaN
(c) one training batch forward / backward is finite; G2Model.score runs on a query
(d) g2.train resumed from its first save point == the straight run (identical weights)
Mounts /src (code), /d (smoke data), /k26 (smoke k26), /w (scratch)."""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "/src")
import torch  # noqa: E402

from casmi26 import cache, enc3, fpnet, frag, g2, harness, pool, subform  # noqa: E402

torch.set_num_threads(4)
ok = True
P = pool.Pool("/d/pool", pubchem=False)
t0 = time.time()
G = g2.Graphs.build(P.smiles, workers=1)
print(f"graphs: {G.ok.sum()} of {len(G)} structures ({time.time() - t0:.1f}s), atoms {len(G.af)}, bonds {len(G.be)}")

# (a) masses against frag.py (uncapped enumeration; frag.py caps at 2,000 unique masses)
n_cmp = n_bad = n_cap = 0
for p in range(len(P)):
    g = G.get(p)
    if g is None:
        continue
    ref = frag.fragments(P.smiles[p])
    if len(ref) >= frag.MAX_FRAG:
        continue
    fr = g2.fragments(g, maxf=100000)
    mine = np.unique(np.round(fr["mass"], frag.DEC))
    n_cmp += 1
    if len(mine) != len(ref) or not np.array_equal(mine, ref):
        n_bad += 1
        if n_bad <= 3:
            print("  mismatch", P.smiles[p], len(mine), len(ref))
    n_cap += len(g2.fragments(g)["mass"]) >= g2.MAX_FRAG
print(f"(a) fragment masses == frag.py on {n_cmp - n_bad}/{n_cmp} structures; {n_cap} hit the {g2.MAX_FRAG} cap")
ok &= n_bad == 0 and n_cmp > 100


# (b) cosine unit checks: one structure, 2 fragments x 5 shifts, observed peaks [0.6, 0.8]
def cos_of(logit_rows, valid, u, jou):
    lg = torch.tensor(logit_rows, dtype=torch.float32, requires_grad=True)
    c = g2.cosine(lg[None], torch.tensor([valid]), torch.tensor([u]), torch.tensor([jou]),
                  torch.tensor([[0.6, 0.8]]), torch.zeros(1, dtype=torch.long))
    c.sum().backward()
    return float(c.detach()), lg.grad


big = -50.0
u = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
one = [[big] * 5, [big] * 5]
one[0][2] = 50.0  # all mass on (fragment 0, shift 0) = group 2
c, gr = cos_of(one, [True, True], u, [-1, -1, 0, -1, -1, -1, -1, -1, -1, -1])
print(f"(b) all mass on a slot matched to peak 0: cos {c:.4f} (expect 0.6000)")
ok &= abs(c - 0.6) < 1e-4
c, gr = cos_of(one, [True, True], u, [-1] * 10)
print(f"    nothing matched: cos {c:.4f}, grad finite {bool(torch.isfinite(gr).all())}, max |grad| {float(gr.abs().max()):.2e}")
ok &= abs(c) < 1e-6 and bool(torch.isfinite(gr).all())
two = [[big] * 5, [big] * 5]
two[0][2] = two[1][2] = 50.0  # half on group 2, half on group 7, both on peak 0
c, gr = cos_of(two, [True, True], u, [-1, -1, 0, -1, -1, -1, -1, 0, -1, -1])
print(f"    two groups on one peak: cos {c:.4f} (expect 0.6000; 0.8485 if they did not add up)")
ok &= abs(c - 0.6) < 1e-4
c, gr = cos_of(two, [False, False], u, [-1, -1, 0, -1, -1, -1, -1, 0, -1, -1])
print(f"    no valid fragments: cos {c:.4f}, grad finite {bool(torch.isfinite(gr).all())}")
ok &= abs(c) < 1e-6 and bool(torch.isfinite(gr).all())

# (c) one training batch, scoring
C = cache.Cache("/d/work")
FT = subform.FormulaTable("/k26")
hold = set(pd.read_csv("/d/pool/holdout.csv").pid.values)
fpnet.Cfg.max_per_struct, fpnet.Cfg.np_weight, fpnet.Cfg.neg_train_only = 0, 3, True
D = fpnet.TrainData(C, P, hold)
NEG = enc3.SameFormulaNegatives(P, FT, hold)
idx = D.epoch_index(np.random.default_rng(0))[:8]
b, st = g2.example_batch(C, D, G, NEG, idx, 4, np.random.default_rng(1))
net = g2.build_model()
bt = g2.to_torch(b, "cpu")
lg = net.head(net.encode(bt), bt["cond"])
cos = g2.cosine(lg, bt["valid"], bt["u"], bt["jou"], bt["obs"], bt["ex"])
cos.sum().backward()
fin = all(bool(torch.isfinite(p.grad).all()) for p in net.parameters() if p.grad is not None)
print(f"(c) batch of 8 x 5: S {len(b['nat'])}, F {b['bits'].shape[1]}, N {b['nmax']}, ceiling {st['ceiling']:.3f}, "
      f"zero {st['zero_ceiling']:.2f}, cos range {float(cos.min()):.3f}..{float(cos.max()):.3f}, grads finite {fin}")
ok &= fin and bool(torch.isfinite(cos).all())
Q = pd.read_csv("/d/pool/queries.csv")
Q["rows"] = Q.rows.map(lambda s: [int(x) for x in str(s).split(",")])
q = Q.iloc[0]
cand = P.window(P.mass[q.pid], 4000.0)
s = g2.G2Model(G=G, net=net.eval()).score(harness.query_specs(C, q.rows), cand)
print(f"    score: {len(cand)} candidates, finite {np.isfinite(s).sum()}, range {np.nanmin(s):.3f}..{np.nanmax(s):.3f}")
ok &= np.isfinite(s).sum() > 0

# (d) resume: 2 x 6 steps straight vs resumed from the save at step 6 (identical weights; one CPU thread, since the
#     multithreaded index_add / scatter_add sum in a varying order)
torch.set_num_threads(1)
g2.Cfg.batch, g2.Cfg.k_neg, g2.Cfg.total_steps, g2.Cfg.save_every, g2.Cfg.workers = 8, 4, 12, 6, 2
os.makedirs("/w/ga", exist_ok=True)
os.makedirs("/w/gb", exist_ok=True)
_save = g2.ckpt.save_state


def keep(path, *a, **k):
    _save(path, *a, **k)
    import shutil
    shutil.copy(path, path.replace(".pt", f"_{k['step']}.pt"))


g2.ckpt.save_state = keep
g2.train(C, D, G, NEG, "/w/ga/m.pt", log=lambda *a: None)
g2.train(C, D, G, NEG, "/w/gb/m.pt", log=lambda *a: None, resume="/w/ga/m_state_6.pt")
A = torch.load("/w/ga/m.pt", weights_only=False)
B = torch.load("/w/gb/m.pt", weights_only=False)
d = max(float((A["model"][k] - B["model"][k]).abs().max()) for k in A["model"])
print(f"(d) g2.train straight vs resumed at step 6: steps {A['step']} / {B['step']}, max |dw| {d:.3g}")
ok &= d == 0 and A["step"] == B["step"] == 12
print("g2 check:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)

"""Resume check (2026-10-12): a training run that is interrupted and resumed from <out>_state.pt equals the straight run.
- fpnet / enc3: 2 epochs straight vs epoch 0 + resume (CPU, one producer thread, so the batches are deterministic)
  -> identical weights and history.
- decoder: 2 x 10 steps; resumed at step 10 it continues the same LR schedule and step count (its batches come from
  producers seeded with the resumed step, so the weights differ by design).
- resuming a finished run trains nothing; a changed cfg is refused.
Run in docker on CPU (tools/smoke/run.sh resume): mounts /src (code), /d (smoke data), /k26 (smoke k26), /w (scratch).
"""
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/src")
import torch  # noqa: E402

from casmi26 import cache, ckpt, decoder, enc3, fpnet, pool, subform  # noqa: E402

torch.set_num_threads(4)
C = cache.Cache("/d/work")
P = pool.Pool("/d/pool", pubchem=False)
FT = subform.FormulaTable("/k26")
Q = pd.read_csv("/d/pool/queries.csv")
Q["rows"] = Q.rows.map(lambda s: [int(x) for x in str(s).split(",")])
hold = set(pd.read_csv("/d/pool/holdout.csv").pid.values)
W = "/w"

_save = ckpt.save_state


def keep_copies(path, *a, **k):
    """Also keep the state of every save point (<state>_<epoch or step>.pt)."""
    _save(path, *a, **k)
    shutil.copy(path, path.replace(".pt", f"_{k.get('epoch', k.get('step'))}.pt"))


ckpt.save_state = keep_copies


def weights(path):
    return torch.load(path, map_location="cpu", weights_only=False)["model"]


def max_diff(a, b):
    A, B = weights(a), weights(b)
    assert A.keys() == B.keys()
    return max(float((A[k].double() - B[k].double()).abs().max()) for k in A)


def out(name):
    os.makedirs(f"{W}/{name}", exist_ok=True)
    return f"{W}/{name}/m.pt"


ok = True

# ---- fpnet: 2 epochs of a 5% draw, batch 32
fpnet.Cfg.batch, fpnet.Cfg.workers, fpnet.Cfg.total_steps = 32, 1, 0
val_q = Q[Q.set == "V"].head(5)
kw = dict(frac=0.05, epochs=2, select="last", eval_every=1, log=lambda *a: None)
ha = fpnet.train(C, P, hold, val_q, out("fa"), **kw)
hb = fpnet.train(C, P, hold, val_q, out("fb"), resume=f"{W}/fa/m_state_0.pt", **kw)
d = max_diff(f"{W}/fa/m.pt", f"{W}/fb/m.pt")
same_hist = [r["bce"] for r in ha] == [r["bce"] for r in hb]
print(f"fpnet: straight vs resumed after epoch 0: max |dw| {d:.3g}, history equal {same_hist}")
ok &= d == 0 and same_hist
hc = fpnet.train(C, P, hold, val_q, out("fc"), resume=f"{W}/fa/m_state_1.pt", **kw)
print(f"fpnet: resume of the finished run: epochs {len(hc)} (expect 2), no new weights {not os.path.exists(f'{W}/fc/m.pt')}")
ok &= len(hc) == 2 and not os.path.exists(f"{W}/fc/m.pt")
fpnet.Cfg.lr *= 2
try:
    fpnet.train(C, P, hold, val_q, out("fd"), resume=f"{W}/fa/m_state_0.pt", **kw)
    print("fpnet: changed cfg was NOT refused")
    ok = False
except ValueError as e:
    print("fpnet: changed cfg refused:", str(e)[:100])
fpnet.Cfg.lr /= 2

# ---- enc3: 2 epochs of 12 steps, batch 16
fpnet.Cfg.max_per_struct, fpnet.Cfg.np_weight, fpnet.Cfg.neg_train_only = 0, 3, True
D = fpnet.TrainData(C, P, hold, seed=0)
enc3.Cfg.batch, enc3.Cfg.workers, enc3.Cfg.max_steps_per_epoch = 16, 1, 12
enc3.Cfg.total_steps = 2 * (len(D.epoch_index(np.random.default_rng(12345))) // enc3.Cfg.batch)
del D
kw = dict(log=lambda *a: None)
ha = enc3.train(C, P, FT, hold, out("ea"), **kw)
hb = enc3.train(C, P, FT, hold, out("eb"), resume=f"{W}/ea/m_state_0.pt", **kw)
d = max_diff(f"{W}/ea/m.pt", f"{W}/eb/m.pt")
same_hist = [r["bce"] for r in ha] == [r["bce"] for r in hb]
print(f"enc3: straight vs resumed after epoch 0: max |dw| {d:.3g}, history equal {same_hist}, epochs {len(hb)}")
ok &= d == 0 and same_hist and len(hb) == 2

# ---- decoder: 20 steps of 8, small net, save points every 10 steps
pids = np.array([p for p in range(len(P)) if p not in hold and FT.fid[p] >= 0 and FT.supported[FT.fid[p]]])
variants = decoder.random_smiles_many(P.smiles[pids].tolist(), 2, 4)
vocab = decoder.build_vocab([decoder.tokenize(v[0]) for v in variants if v])
SD = decoder.build_seqdata(variants, pids, {t: i for i, t in enumerate(vocab)}, decoder.Cfg.max_len)
noise = dict(m1=-1.0, s1=5.0, m0=-6.7, s0=3.0)
decoder.Cfg.d, decoder.Cfg.layers, decoder.Cfg.batch, decoder.Cfg.workers, decoder.Cfg.total_steps = 128, 2, 8, 1, 20
kw = dict(eval_every=10, log=lambda *a: None)
ha = decoder.train(SD, P.fp, FT, vocab, noise, out("da"), **kw)
hb = decoder.train(SD, P.fp, FT, vocab, noise, out("db"), resume=f"{W}/da/m_state_10.pt", **kw)
sa = torch.load(f"{W}/da/m_state.pt", map_location="cpu", weights_only=False)
sb = torch.load(f"{W}/db/m_state.pt", map_location="cpu", weights_only=False)
lr_same = sa["sched"]["last_epoch"] == sb["sched"]["last_epoch"] == 20 and sa["sched"]["_last_lr"] == sb["sched"]["_last_lr"]
steps = [r["step"] for r in hb]
print(f"decoder: resumed at step 10 -> save points {steps} (expect [10, 20]), schedule equal {lr_same}, "
      f"max |dw| vs straight {max_diff(f'{W}/da/m.pt', f'{W}/db/m.pt'):.3g} (differs by design)")
ok &= steps == [10, 20] and lr_same
print("resume check:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)

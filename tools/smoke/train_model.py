"""Tiny FPNet for the smoke run (inside the Kaggle image; src at /src, smoke data at /d, output /kaggle/working).
The checkpoint epoch is set to 14 so that fpnet.load_checked accepts it like the real k06 model."""
import sys

sys.path.insert(0, "/src")
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from casmi26 import cache, fpnet, pool  # noqa: E402

fpnet.Cfg.batch, fpnet.Cfg.d, fpnet.Cfg.layers, fpnet.Cfg.warmup, fpnet.Cfg.workers = 8, 32, 1, 50, 1
C = cache.Cache("/d/work")
P = pool.Pool("/d/pool", cache=C)
Q = pd.read_csv("/d/pool/queries.csv")
Q["rows"] = Q.rows.map(lambda s: [int(x) for x in str(s).split(",")])
hold = pd.read_csv("/d/pool/holdout.csv")
fpnet.train(C, P, set(hold.pid.values), Q[Q.set == "V"], "/kaggle/working/fpnet.pt", frac=1.0, epochs=2)
ck = torch.load("/kaggle/working/fpnet.pt", weights_only=False)
ck["epoch"] = 14
torch.save(ck, "/kaggle/working/fpnet.pt")
print("model saved")

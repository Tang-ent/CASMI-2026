"""Rebuild an FPNet training run (k06) from its Kaggle log and upload it to wandb, for runs made without tracking.

    python tools/wandb_fpnet_log.py yuanyuan98/casmi26-k06-fpnet --group M2 [--epochs 18] [--dry-run]
Logs the same keys as fpnet.train(track=...): train/bce, train/rank (running means within the epoch, every 500 steps),
train/s_per_step, train/lr (recomputed from the schedule in fpnet.train), epoch, and epoch/* at each epoch end,
on a global step counter. Only the latest kernel version's log is available, so run this before re-pushing the kernel.
"""
import argparse
import ast
import json
import math
import os
import re
import sys

from kaggle_out import list_output

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from casmi26 import fpnet  # noqa: E402
from casmi26.track import ENTITY, PROJECT  # noqa: E402

STEP = re.compile(r"^\s+ep (\d+) step (\d+)/(\d+) bce ([\d.]+) rank ([\d.]+) ([\d.]+)s/step \((\d+)s\)")
EPOCH = re.compile(r"^\[fpnet\] epoch (\d+): (\{.*\}) \((\d+)s\)")
HEAD = re.compile(r"^\[fpnet\] ([\d,]+) training spectra, ([\d,]+) steps/epoch, (\d+) epochs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kernel")
    ap.add_argument("--group", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    text = "".join(e.get("data", "") for e in json.loads(list_output(a.kernel).log))
    head = next(HEAD.match(l) for l in text.splitlines() if HEAD.match(l))
    n_spec, spe, epochs = int(head[1].replace(",", "")), int(head[2].replace(",", "")), int(head[3])
    total = spe * epochs
    lr_at = lambda s: fpnet.Cfg.lr * min(1.0, (s + 1) / fpnet.Cfg.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))
    recs, hist, last_t = [], [], 0
    for line in text.splitlines():
        if m := STEP.match(line):
            ep, n = int(m[1]), int(m[2])
            g = ep * spe + n
            recs.append((g, {"train/bce": float(m[4]), "train/rank": float(m[5]), "train/lr": lr_at(g),
                             "train/s_per_step": float(m[6]), "epoch": ep + n / spe}))
            last_t = int(m[7])
        elif m := EPOCH.match(line):
            rec = ast.literal_eval(m[2])
            hist.append(rec)
            recs.append(((int(m[1]) + 1) * spe, {f"epoch/{k}": v for k, v in rec.items()}))
            last_t = int(m[3])
    best = max(hist, key=lambda r: r["val_mrr"])
    print(f"{n_spec:,} spectra, {spe} steps/epoch, {epochs} epochs planned; parsed {len(recs)} records, "
          f"{len(hist)} epochs, best epoch {best['epoch']} val {best['val_mrr']:.4f}, log ends at {last_t}s")
    if a.dry_run:
        return
    import pandas as pd
    import wandb
    cfg = {k: getattr(fpnet.Cfg, k) for k in dir(fpnet.Cfg) if not k.startswith("_")}
    cfg |= dict(epochs=epochs, n_train_spectra=n_spec, steps_per_epoch=spe,
                backfilled_from=f"Kaggle log of {a.kernel} (latest version), rebuilt by tools/wandb_fpnet_log.py",
                note="no tracking in this run; lr recomputed from the schedule")
    run = wandb.init(entity=ENTITY, project=PROJECT, job_type="train", group=a.group,
                     name=a.kernel.split("/")[1].removeprefix("casmi26-"), config=cfg,
                     settings=wandb.Settings(console="off"))
    for g, d in sorted(recs, key=lambda x: x[0]):
        run.log(d, step=g)
    run.log({"history": wandb.Table(dataframe=pd.DataFrame(hist))}, step=recs[-1][0])
    run.summary.update(dict(best_val_mrr=best["val_mrr"], best_epoch=best["epoch"], total_secs=last_t,
                            log_end_step=max(g for g, _ in recs)))
    print("uploaded", run.url)
    run.finish()


if __name__ == "__main__":
    main()

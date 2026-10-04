"""Experiment tracking with wandb in offline mode.

Most kernels run without internet, so runs are written to /kaggle/working/wandb/offline-run-* and uploaded
later from the local machine (tools/wandb_sync.py). Tracking must never break a kernel: if wandb is missing
or any call fails, the error is printed and the run degrades to a no-op.
"""
import os

PROJECT = "casmi26"


class Run:
    def __init__(self, run=None):
        self.run = run

    def _call(self, what, fn):
        if self.run is None:
            return
        try:
            fn()
        except Exception as e:  # ログのために本処理を止めない
            print(f"[track] {what} failed: {e!r}")

    def log(self, d, step=None):
        self._call("log", lambda: self.run.log(d, step=step))

    def summary(self, d):
        self._call("summary", lambda: self.run.summary.update(d))

    def table(self, name, df):
        import wandb
        self._call("table", lambda: self.run.log({name: wandb.Table(dataframe=df)}))

    def finish(self):
        self._call("finish", lambda: self.run.finish())


def init(job_type, group=None, config=None, out_dir="/kaggle/working"):
    """Start an offline run named after the kernel slug (set by tools/build_kernel.py). Returns a Run."""
    try:
        import wandb

        os.environ.setdefault("WANDB_MODE", "offline")
        os.environ.setdefault("WANDB_SILENT", "true")
        kernel = os.environ.get("CASMI26_KERNEL", "local")
        cfg = dict(config or {}, src_sha=os.environ.get("CASMI26_SHA"), kernel=kernel)
        run = wandb.init(project=PROJECT, job_type=job_type, group=group, name=kernel.removeprefix("casmi26-"),
                         config=cfg, dir=out_dir if os.path.isdir(out_dir) else None, save_code=False,
                         settings=wandb.Settings(console="off"))
        print(f"[track] wandb {wandb.__version__} {os.environ['WANDB_MODE']} run {run.id} ({run.dir})")
        return Run(run)
    except Exception as e:
        print(f"[track] disabled: {e!r}")
        return Run()

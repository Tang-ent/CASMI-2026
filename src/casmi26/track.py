"""Experiment tracking with wandb: online when possible, offline otherwise.

Online (live curves) needs internet on the kernel and the API key, from the WANDB_API_KEY environment variable or
the Kaggle secret of the same name attached to the notebook. Otherwise the run is written offline to
/kaggle/working/wandb/offline-run-* and uploaded later from the local machine (tools/wandb_sync.py).
Tracking must never break a kernel: if wandb is missing or any call fails, the error is printed and the run
degrades to offline or to a no-op. The key is never printed or written to the outputs.
"""
import os

PROJECT = "casmi26"
ENTITY = "CASMI-2026"  # wandb のチーム
SECRET = "WANDB_API_KEY"


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


def _api_key():
    """API key from the environment or the Kaggle secret (None when neither is available)."""
    if os.environ.get(SECRET):
        return os.environ[SECRET]
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(SECRET) or None
    except Exception:
        return None


def _reachable(host="api.wandb.ai", timeout=5.0):
    import socket
    try:
        socket.create_connection((host, 443), timeout=timeout).close()
        return True
    except OSError:
        return False


def init(job_type, group=None, config=None, out_dir="/kaggle/working", online=True):
    """Start a run named after the kernel slug (set by tools/build_kernel.py). Returns a Run.
    online=True tries live logging first (key + network); online=False always writes offline."""
    try:
        import wandb
    except Exception as e:
        print(f"[track] disabled: {e!r}")
        return Run()
    os.environ.setdefault("WANDB_SILENT", "true")
    kernel = os.environ.get("CASMI26_KERNEL", "local")
    cfg = dict(config or {}, src_sha=os.environ.get("CASMI26_SHA"), kernel=kernel)
    key = _api_key() if online else None
    modes = ["online", "offline"] if key and _reachable() else ["offline"]
    why = "" if len(modes) == 2 else (" (no API key)" if online and not key else " (no network)" if online else "")
    for mode in modes:
        try:
            if mode == "online":
                os.environ[SECRET] = key  # wandb がプロセス内で読む。表示も保存もしない
            run = wandb.init(entity=ENTITY, project=PROJECT, job_type=job_type, group=group, mode=mode,
                             name=kernel.removeprefix("casmi26-"), config=cfg, save_code=False,
                             dir=out_dir if os.path.isdir(out_dir) else None,
                             settings=wandb.Settings(console="off", init_timeout=120))
            where = run.url if mode == "online" else run.dir
            why = " (online init failed)" if mode == "offline" and len(modes) == 2 else why
            print(f"[track] wandb {wandb.__version__} {mode}{why} run {run.id} ({where})")
            return Run(run)
        except Exception as e:
            print(f"[track] {mode} init failed: {e!r}")
    return Run()

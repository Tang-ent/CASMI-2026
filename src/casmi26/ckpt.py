"""Resumable training state (2026-10-12, after a local k28 run was lost to a PC shutdown).

The weights-only checkpoints (<out>.pt, <out>_last.pt) stay as they are for inference. Next to them the training loops
write <out>_state.pt: model, AdamW, LR schedule, GradScaler, the numpy / torch RNG states and the loop's own counters
(epoch or step, history, horizon, cfg). The write goes to a temp file and is then renamed, so an interruption during the
write leaves the previous state intact. train(..., resume=<state path>) continues from it; the cfg must match.
"""
import os


def state_path(out_path):
    return out_path.replace(".pt", "_state.pt")


def save_state(path, net, opt, sched, scaler, rng=None, gen=None, **extra):
    import torch
    st = dict(model=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), scaler=scaler.state_dict(),
              torch_rng=torch.get_rng_state(),
              cuda_rng=torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
              np_rng=rng.bit_generator.state if rng is not None else None,
              gen=gen.get_state() if gen is not None else None, extra=extra)
    tmp = path + ".tmp"
    torch.save(st, tmp)
    os.replace(tmp, path)


def load_state(path, net, opt, sched, scaler, rng=None, gen=None, cfg=None):
    """Restore in place; returns the loop's counters (the `extra` of save_state). cfg: the current cfg dump, which must
    equal the saved one (a resumed run keeps its recipe)."""
    import torch
    st = torch.load(path, map_location="cpu", weights_only=False)
    x = st["extra"]
    if cfg is not None and x.get("cfg") != cfg:
        diff = {k: (x.get("cfg", {}).get(k), v) for k, v in cfg.items() if x.get("cfg", {}).get(k) != v}
        raise ValueError(f"resume: cfg differs from the saved state (saved, now): {diff}")
    net.load_state_dict(st["model"])
    opt.load_state_dict(st["opt"])
    sched.load_state_dict(st["sched"])
    scaler.load_state_dict(st["scaler"])
    torch.set_rng_state(st["torch_rng"])
    if st["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(st["cuda_rng"])
    if rng is not None:
        rng.bit_generator.state = st["np_rng"]
    if gen is not None:
        gen.set_state(st["gen"])
    return x

"""casmi26.molenc checks on the real pool (M9, plan 10.2), in the Kaggle docker image (tools/local/run_check_molenc.sh):
(a) canon_text (default, taut=True) over all pool SMILES with canon_pool (CPU, WORKERS processes): failures, wall
    time, and uniformity by source (train_only / coconut_only / both from pool.parquet in_train / in_coconut; reporting
    only): share with '@', '/' or '\\', '.', a bracket charge (CHARGE_RX; any '+' or '-' as sign_any); character
    length (p10 / median / p90) and the share of 'c' characters, before and after canonicalisation
(b) the tautomer pass: the whole pool once more with taut=False; the share of strings the pass changes, by source and
    by raw charge / '.'; ms per molecule (one process, N_SINGLE seeded random pids) with / without it and of
    chem.canonical. The embedding sample (d) is N_SAMPLE seeded random pids
(c) ChemBERTa tokens of all canonical strings: the share above MAX_TOKENS (510) and above 512, by source;
    max_length=512 on the longest string, on the CPU (expected to fail: position table of 512, padding_idx 1)
(d) embed_smiles on the sample on the GPU (fp16): throughput, peak memory; input order restored (shuffled input ->
    the same rows), padding (64 molecules one by one vs in one batch), fp16 vs fp32 (256 molecules)
(e) fp_bits: shape, {0, 1}, popcount = packed popcount, Morgan of pool.smiles vs pool_fp on 1,000 pids
Mounts: /src (code), /kaggle/input/casmi26-k05-pool, casmi26-k05b-massfix, casmi26-chemberta, /w (output).
Writes /w/molenc_check.json and /w/canon_text.parquet (pid, text; text null when canon_text failed)."""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "/src")
from casmi26 import chem, molenc, pool  # noqa: E402

WORKERS = int(os.environ.get("WORKERS", "14"))
N_SAMPLE = int(os.environ.get("N_SAMPLE", "20000"))
N_SINGLE = int(os.environ.get("N_SINGLE", "1000"))
BATCH = int(os.environ.get("BATCH", "256"))
LIMIT = int(os.environ.get("LIMIT", "0"))  # > 0: only the first LIMIT pool rows (a quick run)
CB_DIR = os.environ.get("CB_DIR", "/kaggle/input/casmi26-chemberta")
SEED = 0
# a formal charge inside a bracket atom ([N+], [O-], [nH+] ...); a bare "-" is also the explicit single bond RDKit
# writes between aromatic atoms, so r"[+-]" (kept as sign_any) overcounts (review 2026-10-24)
CHARGE_RX = r"\[[^\]]*[+-]"
R = {}


def log(*a):
    print(*a, flush=True)


def q(x, ps=(10, 50, 90)):
    return {f"p{p}": float(np.percentile(x, p)) for p in ps} if len(x) else {}


def text_stats(texts):
    s = pd.Series([t if isinstance(t, str) else "" for t in texts])
    ok = s.str.len() > 0
    s = s[ok]
    ln = s.str.len().values
    return dict(n=int(ok.sum()), at=float(s.str.contains("@", regex=False).mean()),
                slash=float(s.str.contains(r"[/\\]", regex=True).mean()),
                dot=float(s.str.contains(".", regex=False).mean()),
                charge=float(s.str.contains(CHARGE_RX, regex=True).mean()),
                sign_any=float(s.str.contains(r"[+-]", regex=True).mean()),
                len=q(ln), c_share=float(s.str.count("c").sum() / max(ln.sum(), 1)))


def by_source(texts, src):
    texts = np.asarray(texts, dtype=object)
    return {k: text_stats(texts[src == k]) for k in ("train_only", "coconut_only", "both")}


def main():
    t00 = time.time()
    pdir = "/kaggle/input/casmi26-k05-pool/pool"
    p = pd.read_parquet(os.path.join(pdir, "pool.parquet"), columns=["smiles", "in_train", "in_coconut"])
    if LIMIT:
        p = p.iloc[:LIMIT]
    raw = p.smiles.tolist()
    src = np.where(p.in_train & p.in_coconut, "both", np.where(p.in_train, "train_only", "coconut_only"))
    R["n_pool"] = len(raw)
    R["n_by_source"] = {k: int((src == k).sum()) for k in ("train_only", "coconut_only", "both")}
    R["workers"] = WORKERS
    log(f"pool {len(raw):,}  {R['n_by_source']}  workers {WORKERS}")

    # (a) canon_text on the whole pool: the default (taut=True) and without the tautomer pass
    R["raw_stats"] = by_source(raw, src)
    runs = {}
    for taut in (False, True):
        t0 = time.time()
        c = molenc.canon_pool(raw, workers=WORKERS, taut=taut, chunksize=2000 if not taut else 100, log=log,
                              tag=f"pool taut={taut}")
        runs[taut] = (c, time.time() - t0)
    canon, wall = runs[True]
    fail = np.array([c is None for c in canon])
    R["canon"] = dict(taut=True, wall_s=round(wall, 1), ms_per_mol_x_workers=1000 * wall * WORKERS / len(raw),
                      est_full_pool_s_4cpu=round(wall * WORKERS / 4),
                      fail=int(fail.sum()), fail_share=float(fail.mean()),
                      fail_by_source={k: float(fail[src == k].mean()) for k in R["n_by_source"]},
                      changed_vs_pool_smiles=float(np.mean([c != s for c, s in zip(canon, raw)])))
    R["canon_stats"] = by_source(canon, src)
    log("canon", json.dumps(R["canon"]))
    for k in ("train_only", "coconut_only", "both"):
        log(f"  {k:13s} raw   {json.dumps(R['raw_stats'][k])}")
        log(f"  {k:13s} canon {json.dumps(R['canon_stats'][k])}")
    pd.DataFrame({"pid": np.arange(len(canon), dtype=np.int64), "text": canon}).to_parquet("/w/canon_text.parquet",
                                                                                          index=False)
    bad = [raw[i] for i in np.flatnonzero(fail)[:10]]
    R["fail_examples"] = bad
    log("failed examples:", bad)

    # (b) what the tautomer pass changes (whole pool) and what it costs (one process, seeded random pids)
    nt, w_nt = runs[False]
    R["canon_no_taut_stats"] = by_source(nt, src)
    both_ok = np.array([x is not None and y is not None for x, y in zip(nt, canon)])
    diff = np.array([x != y for x, y in zip(nt, canon)]) & both_ok
    ch = pd.Series(raw).str.contains(CHARGE_RX, regex=True, na=False).values.astype(bool)
    dot = np.array(["." in x for x in raw])
    rng = np.random.default_rng(SEED)
    one = [raw[i] for i in rng.choice(len(raw), size=min(N_SINGLE, len(raw)), replace=False)]
    ms = {}
    for name, fn in (("no_taut", lambda x: molenc.canon_text(x, taut=False)), ("taut", molenc.canon_text),
                     ("chem_canonical", chem.canonical)):
        t0 = time.time()
        for x in one:
            fn(x)
        ms[name] = 1000 * (time.time() - t0) / len(one)
    R["taut"] = dict(
        n=len(raw), wall_pool_no_taut=round(w_nt, 1), wall_pool_taut=round(wall, 1), ms_single_random=ms,
        fail_no_taut=int(sum(x is None for x in nt)), fail_taut=int(fail.sum()),
        changed_share=float(diff[both_ok].mean()), n_changed=int(diff.sum()),
        changed_by_source={k: float(diff[both_ok & (src == k)].mean()) for k in R["n_by_source"]},
        changed_raw_charged=float(diff[both_ok & ch].mean()), changed_raw_uncharged=float(diff[both_ok & ~ch].mean()),
        changed_raw_dot=float(diff[both_ok & dot].mean()), n_raw_charged=int(ch.sum()), n_raw_dot=int(dot.sum()),
        examples=[(nt[i], canon[i]) for i in np.flatnonzero(diff)[:8]])
    del runs, nt
    samp = np.sort(rng.choice(len(raw), size=min(N_SAMPLE, len(raw)), replace=False))
    s_nt = [canon[i] for i in samp]
    log("taut", json.dumps(R["taut"]))

    # (c) tokens
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CB_DIR)
    R["tokenizer"] = dict(cls=type(tok).__name__, model_max_length=float(tok.model_max_length),
                          padding_side=tok.padding_side, pad_id=tok.pad_token_id)
    ok = np.flatnonzero(~fail)
    texts = [canon[i] for i in ok]
    t0 = time.time()
    tl = molenc.token_lengths(tok, texts)
    R["tokens"] = dict(secs=round(time.time() - t0, 1), n=len(tl), over_510=float((tl > 510).mean()),
                       over_512=float((tl > 512).mean()), n_over_510=int((tl > 510).sum()),
                       max=int(tl.max()), len=q(tl, (10, 50, 90, 99)),
                       over_510_by_source={k: float((tl[src[ok] == k] > 510).mean()) for k in R["n_by_source"]},
                       len_by_source={k: q(tl[src[ok] == k]) for k in R["n_by_source"]})
    log("tokens", json.dumps(R["tokens"]))
    longest = texts[int(np.argmax(tl))]

    # (d) the GPU tower
    import transformers
    R["versions"] = dict(torch=torch.__version__, transformers=transformers.__version__,
                         rdkit=__import__("rdkit").__version__, gpu=torch.cuda.get_device_name(0))
    log("versions", R["versions"])
    tok, model = molenc.load_chemberta(CB_DIR)
    R["model"] = dict(cls=type(model).__name__, params=int(sum(x.numel() for x in model.parameters())),
                      dtype=str(next(model.parameters()).dtype))
    s_txt = [t for t in s_nt if t is not None]
    molenc.embed_smiles(s_txt[:512], tok, model, batch=BATCH)  # warm-up
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    E = molenc.embed_smiles(s_txt, tok, model, batch=BATCH, log=log)
    torch.cuda.synchronize()
    sec = time.time() - t0
    R["embed"] = dict(n=len(s_txt), batch=BATCH, secs=round(sec, 1), mol_per_s=len(s_txt) / sec,
                      peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9, dtype=str(E.dtype), shape=list(E.shape),
                      finite=bool(np.isfinite(E).all()), norm=q(np.linalg.norm(E.astype(np.float32), axis=1)),
                      est_full_pool_s=round(len(raw) / (len(s_txt) / sec)))
    log("embed", json.dumps(R["embed"]))

    # the checks below use seeded random rows of the sample (s_txt is in pid = mass order, so a prefix would be small)
    sel = np.random.default_rng(4).choice(len(s_txt), min(2000, len(s_txt)), replace=False)
    sub = [s_txt[i] for i in sel]
    perm = np.random.default_rng(1).permutation(len(sub))
    E1 = E[sel].astype(np.float32)
    E2 = molenc.embed_smiles([sub[i] for i in perm], tok, model, batch=BATCH).astype(np.float32)
    back = np.empty_like(E2)
    back[perm] = E2

    def cos(a, b):
        return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))

    # E was computed on the whole sample; the shuffled 2,000 are padded in other batches -> fp16 noise only
    R["order"] = dict(max_abs=float(np.abs(back - E1).max()), min_cos=float(cos(back, E1).min()))
    rng2 = np.random.default_rng(2)
    pick = [sub[i] for i in rng2.choice(len(sub), 64, replace=False)]
    Eb = molenc.embed_smiles(pick, tok, model, batch=64).astype(np.float32)
    Es = np.concatenate([molenc.embed_smiles([t], tok, model, batch=1) for t in pick]).astype(np.float32)
    lens = [len(t) for t in pick]
    R["padding"] = dict(max_abs=float(np.abs(Eb - Es).max()), min_cos=float(cos(Eb, Es).min()),
                        char_len_min_max=[min(lens), max(lens)])
    del model
    torch.cuda.empty_cache()
    tok32, m32 = molenc.load_chemberta(CB_DIR, fp16=False)
    p256 = sub[:256]
    E32 = molenc.embed_smiles(p256, tok32, m32, batch=64).astype(np.float32)
    E16 = E1[:256]
    R["fp16_vs_fp32"] = dict(max_abs=float(np.abs(E32 - E16).max()), min_cos=float(cos(E32, E16).min()),
                             mean_cos=float(cos(E32, E16).mean()))
    # a neighbour sanity number: cosine of random pairs (mean pooling of RoBERTa is known to be anisotropic)
    a, b = rng2.choice(len(E), 2000), rng2.choice(len(E), 2000)
    Ef = E.astype(np.float32)
    R["random_pair_cos"] = q(cos(Ef[a], Ef[b]))
    log("order", R["order"], "padding", R["padding"], "fp16_vs_fp32", R["fp16_vs_fp32"],
        "random_pair_cos", R["random_pair_cos"])
    del m32
    torch.cuda.empty_cache()
    # max_length 512 vs 510 on the longest string, on the CPU (an index error on the GPU would poison the context)
    tokc, mc = molenc.load_chemberta(CB_DIR, device="cpu", fp16=False)
    res = {}
    for ml in (512, molenc.MAX_TOKENS):
        try:
            e = molenc.embed_smiles([longest], tokc, mc, batch=1, device="cpu", max_length=ml)
            res[str(ml)] = f"ok, finite={bool(np.isfinite(e).all())}"
        except Exception as ex:  # noqa: BLE001
            res[str(ml)] = f"{type(ex).__name__}: {str(ex)[:200]}"
    R["longest_max_length"] = res
    log("longest string (tokens", int(tl.max()), "):", res)
    del mc

    # (e) fp_bits
    P = pool.Pool(pdir, pubchem=False)
    pids = np.sort(np.random.default_rng(3).choice(len(raw), 1000, replace=False))
    F = molenc.fp_bits(P, pids)
    pop = np.unpackbits(P.fp[pids], axis=1).sum(1)
    mp = np.stack([chem.morgan_packed(P.smiles[i]) for i in pids])
    R["fp_bits"] = dict(shape=list(F.shape), dtype=str(F.dtype), values=sorted(np.unique(F).tolist()),
                        popcount_ok=bool((F.sum(1) == pop).all()), bits_mean=float(F.sum(1).mean()),
                        morgan_of_smiles_equal=float((mp == P.fp[pids]).all(1).mean()))
    log("fp_bits", R["fp_bits"])
    R["secs"] = round(time.time() - t00)
    json.dump(R, open("/w/molenc_check.json", "w"), indent=1, default=str)
    log("done", R["secs"], "s")


if __name__ == "__main__":
    main()

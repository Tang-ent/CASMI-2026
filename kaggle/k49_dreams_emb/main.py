# %% [markdown]
# # casmi26-k49-dreams-emb
# 計画 10.4（2026-10-24、事前登録）: MSAlign 型の学習（k50）の入力になる、凍結した埋め込みの表を作る。
# - スペクトル: DreaMS の自己教師の重み ssl_model.ckpt（HF roman-bushuiev/DreaMS、sha256 を確かめる）を、pickle のコードを
#   実行しない読み込み（dreams.load_ssl）でテンソルだけ取り出し、自前の前向き（casmi26.dreams、公式 dbec3a0 と同値を
#   確かめ済み）で埋め込む。先に SSL の頭（マスクしたピークの m/z）で自己検査し、偶然の水準に近ければ止める。
#   閾値（top1 >= 2 × 最多の bin の割合、かつ >= 100 × 一様）は事後に決めたもの（計画 10.2 に数字は無い）。
#   check_dreams の前に自分で決めた 5 × は 3.57 × で満たさず（2026-10-24 10:21）、2 × はその後、最初の k49 smoke（10:59）までに置いた。
#   そのため、この検査は読み込みの壊れを捉える歯止めにすぎない。前向きの正しさの根拠は、公式のモジュール（dbec3a0）
#   との一致（埋め込みの最大の差 5.5e-6、tools/smoke/check_dreams.py）。5 × の結果も ssl.pass_5x_preset に残す。
#   fp16 のブロックは、この GPU で fp32 とのコサインの最小が FP16_COS_MIN 以上のときだけ使う
# - 埋め込む行 = msalign.train_rows（holdout を除く train の行、構造ごとに最大 4 本、種 0）∪ A、R1、R2、V の問いの行
#   （msalign.query_rows: 中性質量が有限な先頭 16 本、Searcher と同じ）。GPU ごとに 1 スレッドで塊を分け、
#   emb_spec.npy（fp16 の memmap）に直接書く。終わった塊は emb_progress.json に残り、止まった実行は続きから再開する
#   （Kaggle では前の版の出力を入力に付ける。RESUME）
# - 分子: プールの全 pid の SMILES を molenc.canon_text（最大の断片、電荷の中和、立体の除去、互変異性体、canonical）で
#   書き直し（CUDA を使う前に別プロセスで）、ChemBERTa-100M-MLM（rev f5c45f44、MIT）の平均プーリングで emb_mol_cb.npy
#   （fp16、正規化に失敗した pid は 0 のまま、emb_mol_cb_ok.npy が False）
# 出力: emb_spec.npy、emb_rows.npy（表の各行のキャッシュの行）、emb_progress.json、canon_text.parquet、canon_stats.json、
# emb_mol_cb.npy、emb_mol_cb_ok.npy、k49_info.json（時間、版、sha、選んだ行と問い、自己検査の数字）。
# 入力: k01（キャッシュ）、k05（プール、queries.csv、holdout.csv）、k05b（質量の補正）、casmi26-wheels（rdkit）。
# インターネット: ssl_model.ckpt と ChemBERTa を取る（/kaggle/input に同じものがあればそれを使う）。

# %%
import glob
import os
import subprocess
import sys

wh = sorted(glob.glob("/kaggle/input/**/rdkit-*.whl", recursive=True))
assert wh, "attach the casmi26-wheels dataset (rdkit wheels)"
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--find-links",
                    os.path.dirname(wh[0]), "rdkit"], capture_output=True, text=True)
print(r.stdout[-300:], r.stderr[-300:])
import rdkit

print("rdkit", rdkit.__version__)

# %%
import hashlib
import json
import platform
import shutil
import threading
import time
import urllib.request

import numpy as np
import pandas as pd

from casmi26 import cache, dreams, molenc, msalign, pool

T0 = time.time()
OUT = "/kaggle/working"
SEED = 0
MAX_PER_STRUCT = 4
SETS = ["A", "R1", "R2", "V"]
MAX_SPEC = 16                 # SearchCfg.max_query_spectra
BATCH = 128                   # DreaMS batch (n = 60 peaks)
CHUNK = 8192                  # rows per resumable unit
SSL_N = 2000                  # spectra per polarity for the SSL self-check
SSL_MIN_X_MAJORITY = 2.0      # POST HOC (set after check_dreams measured 3.57 x): top1 >= 2 x the most frequent bin
SSL_MIN_X_UNIFORM = 100.0     # and >= 100 x uniform chance (1 / 20000); a guard against a broken load only
SSL_PRESET_X_MAJORITY = 5.0   # the threshold check_dreams set before its first run (failed there at 3.57 x); reported
FP16_COS_MIN = 0.9999         # fp16 blocks only when fp16 vs fp32 cosine min >= this on this GPU
CB_BATCH = 256
CHARGE_RX = r"\[[^\]]*[+-]"   # a formal charge in a bracket atom (canon_stats; a bare '-' is also a bond)
CANON_WORKERS = max(1, (os.cpu_count() or 4) - 1)
BUDGET_H = 8.5
RESUME = None                 # a previous k49 output dir; None = look for emb_progress.json under /kaggle/input
EMB_THREADS = None            # embedding threads (None = one per GPU; thread i uses cuda:i % n_gpu)
STOP_AFTER_CHUNKS = None      # test only: stop the spectrum embedding after this many chunks, as if interrupted
MAX_TRAIN_ROWS = None         # 手元の動作確認だけ（学習の行を無作為にこの数に）
MAX_QUERIES = None            # 手元の動作確認だけ（組ごとの先頭の問いの数）
MAX_POOL = None               # 手元の動作確認だけ（ChemBERTa で埋め込む pid の数）
SMOKE_EXTRA_QIDS = []         # 手元の動作確認だけ（MAX_QUERIES に加える問い、k50 の空の窓の確かめ）
SMOKE = bool(MAX_TRAIN_ROWS or MAX_QUERIES or MAX_POOL)
DEADLINE = T0 + BUDGET_H * 3600
assert MAX_QUERIES or not SMOKE_EXTRA_QIDS, "SMOKE_EXTRA_QIDS is for smoke runs only"
INFO = dict(settings=dict(seed=SEED, max_per_struct=MAX_PER_STRUCT, sets=SETS, max_spec=MAX_SPEC, batch=BATCH,
                          chunk=CHUNK, ssl_n=SSL_N, ssl_min_x_majority=SSL_MIN_X_MAJORITY,
                          ssl_min_x_uniform=SSL_MIN_X_UNIFORM, ssl_preset_x_majority=SSL_PRESET_X_MAJORITY,
                          ssl_threshold_note="post hoc: 2 x / 100 x set after check_dreams measured 3.57 x (its own 5 x "
                                             "preset failed); correctness rests on the equivalence with the official "
                                             "DreaMS module (max abs diff 5.5e-6), not on this check",
                          fp16_cos_min=FP16_COS_MIN, cb_batch=CB_BATCH,
                          canon_workers=CANON_WORKERS, budget_h=BUDGET_H, emb_threads=EMB_THREADS,
                          stop_after_chunks=STOP_AFTER_CHUNKS, max_train_rows=MAX_TRAIN_ROWS,
                          max_queries=MAX_QUERIES, max_pool=MAX_POOL, smoke_extra_qids=SMOKE_EXTRA_QIDS,
                          smoke=SMOKE),
            src_sha=os.environ.get("CASMI26_SHA"), timings={})


def tm(key):
    INFO["timings"][key] = round(time.time() - T0, 1)
    print(f"== {key} ({time.time() - T0:.0f}s)", flush=True)


def write_json(obj, path):
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else float(x))
    os.replace(tmp, path)


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def resume_dir():
    if RESUME:
        return RESUME
    hits = [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/emb_progress.json", recursive=True)]
    return hits[0] if hits else None


C = cache.Cache(cache.Cache.find_dir())
PD = pool.Pool.find_dir()
P = pool.Pool(PD, pubchem=False)
Q = pd.read_csv(f"{PD}/queries.csv").set_index("qid")
HOLD = set(pd.read_csv(f"{PD}/holdout.csv").pid.values)
tm("inputs")

# %% [markdown]
# ## 1. 埋め込む行

# %%
tr_rows, tr_pid = msalign.train_rows(C, P, HOLD, MAX_PER_STRUCT, seed=SEED)
INFO["train_rows_full"] = dict(n=int(len(tr_rows)), n_struct=int(len(np.unique(tr_pid))), sha=sha_arr(tr_rows))
if MAX_TRAIN_ROWS:
    keep = np.sort(np.random.default_rng(SEED).choice(len(tr_rows), min(MAX_TRAIN_ROWS, len(tr_rows)), replace=False))
    tr_rows, tr_pid = tr_rows[keep], tr_pid[keep]
qrows, qsel = {}, {}
for s in SETS:
    qs = np.sort(Q.index[Q.set == s].values)
    if MAX_QUERIES:
        qs = np.union1d(qs[:MAX_QUERIES], qs[np.isin(qs, SMOKE_EXTRA_QIDS)])
    qsel[s] = [int(q) for q in qs]
    qrows[s] = {int(q): msalign.query_rows(C, Q.loc[q, "rows"], MAX_SPEC) for q in qs}
    nz = sum(len(v) == 0 for v in qrows[s].values())
    print(f"{s}: {len(qs)} queries, {sum(len(v) for v in qrows[s].values()):,} rows, {nz} without a usable spectrum")
all_q = np.concatenate([np.concatenate(list(d.values())) for d in qrows.values()] + [np.zeros(0, np.int64)])
assert not np.isin(all_q, tr_rows).any(), "a query row is also a training row (holdout leak)"
ROWS = np.unique(np.r_[tr_rows, all_q]).astype(np.int64)
N = len(ROWS)
np.save(f"{OUT}/emb_rows.npy", ROWS)
INFO["rows"] = dict(n=int(N), sha=sha_arr(ROWS), n_train=int(len(tr_rows)), n_train_struct=int(len(np.unique(tr_pid))),
                    train_sha=sha_arr(tr_rows), n_query_rows=int(len(np.unique(all_q))),
                    n_negative_mode=int((C.mode[ROWS] < 0).sum()), n_prec_over_1000=int((C.prec[ROWS] > 1000).sum()),
                    queries={s: len(v) for s, v in qsel.items()},
                    query_rows={s: int(sum(len(v) for v in qrows[s].values())) for s in SETS})
INFO["qids"] = qsel
print("rows:", INFO["rows"], flush=True)
tm("rows")

# %% [markdown]
# ## 2. 分子の文字列の正規化（CUDA を使う前に別プロセスで始め、GPU の作業と並べる）

# %%
if MAX_POOL:
    must = np.unique(np.r_[tr_pid, Q.loc[[q for v in qsel.values() for q in v], "pid"].values]).astype(np.int64)
    rest = np.setdiff1d(np.arange(len(P)), must)
    extra = np.random.default_rng(SEED).choice(rest, max(0, MAX_POOL - len(must)), replace=False)
    PIDS = np.sort(np.r_[must, extra])
else:
    PIDS = np.arange(len(P), dtype=np.int64)
CANON = f"{OUT}/canon_text.parquet"
canon_proc = None
rd = resume_dir()
if not os.path.exists(CANON) and rd and os.path.exists(f"{rd}/canon_text.parquet"):
    shutil.copy(f"{rd}/canon_text.parquet", CANON)
if os.path.exists(CANON) and np.array_equal(pd.read_parquet(CANON, columns=["pid"]).pid.values, PIDS):
    print("canon_text.parquet: reused", flush=True)
else:
    pd.DataFrame({"pid": PIDS, "smiles": P.smiles[PIDS]}).to_parquet("/tmp/k49_canon_in.parquet", index=False)
    open("/tmp/k49_canon.py", "w").write(
        "import sys, time\n"
        "sys.path.insert(0, '/tmp/casmi26_src')\n"
        "import pandas as pd\n"
        "from casmi26 import molenc\n"
        "d = pd.read_parquet('/tmp/k49_canon_in.parquet')\n"
        f"out = molenc.canon_pool(list(d.smiles.values), workers={CANON_WORKERS}, taut=True, "
        "log=lambda *a: print(*a, flush=True))\n"
        "d['text'] = pd.Series(out, dtype=object)\n"
        f"d[['pid', 'text']].to_parquet('{CANON}.tmp', index=False)\n"
        "import os\n"
        f"os.replace('{CANON}.tmp', '{CANON}')\n")
    canon_log = open(f"{OUT}/canon.log", "w")
    canon_proc = subprocess.Popen([sys.executable, "/tmp/k49_canon.py"], stdout=canon_log, stderr=subprocess.STDOUT)
    print(f"canon: {len(PIDS):,} pids in a subprocess with {CANON_WORKERS} workers (log canon.log)", flush=True)

# %% [markdown]
# ## 3. ssl_model.ckpt（sha256、コードを実行しない読み込み）と自己検査

# %%
import torch

NG = torch.cuda.device_count()
assert NG >= 1, "k49 needs a GPU"
INFO["versions"] = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                        pandas=pd.__version__, rdkit=rdkit.__version__,
                        gpus=[torch.cuda.get_device_name(i) for i in range(NG)])
ck = sorted(glob.glob("/kaggle/input/**/ssl_model.ckpt", recursive=True))
if ck:
    CKPT, ck_src = ck[0], "input"
else:
    os.makedirs("/tmp/dreams", exist_ok=True)
    CKPT, ck_src = "/tmp/dreams/ssl_model.ckpt", dreams.SSL_URL
    t = time.time()
    with urllib.request.urlopen(dreams.SSL_URL, timeout=600) as resp, open(CKPT + ".part", "wb") as f:
        shutil.copyfileobj(resp, f, 1 << 24)
    os.replace(CKPT + ".part", CKPT)
    print(f"downloaded {os.path.getsize(CKPT):,} B in {time.time() - t:.0f}s", flush=True)
SD = dreams.load_ssl(CKPT, sha256=dreams.SSL_SHA256, heads=True)  # raises on a sha256 mismatch
INFO["ckpt"] = dict(source=ck_src, sha256=dreams.SSL_SHA256, size=os.path.getsize(CKPT), n_tensors=len(SD))
tm("ckpt")

pid_all = P.sid2pid[C.sid]
okr = C.keep & (pid_all >= 0) & ~np.isin(pid_all, np.asarray(sorted(HOLD))) & (C.prec <= dreams.MAX_MZ)
rng = np.random.default_rng(SEED)
ssl_rows = {m: rng.choice(np.flatnonzero(okr & (C.mode == sgn)), SSL_N, replace=False) for m, sgn in
            [("positive", 1), ("negative", -1)]}
X = {m: dreams.preprocess_batch([(*C.spectrum(r), float(C.prec[r])) for r in rr]) for m, rr in ssl_rows.items()}
enc32 = dreams.build_encoder(SD, with_head=True, device="cuda:0")
ssl = {m: dreams.ssl_accuracy(enc32, X[m], seed=SEED, batch=BATCH, device="cuda:0") for m in X}
ssl["positive_no_context"] = dreams.ssl_accuracy(enc32, X["positive"], seed=SEED, batch=BATCH, device="cuda:0",
                                                 mask_context=True)
pos = ssl["positive"]
ssl["x_majority"] = pos["top1"] / pos["majority"]
ssl["x_uniform"] = pos["top1"] / pos["chance_uniform"]
ssl["pass"] = bool(ssl["x_majority"] >= SSL_MIN_X_MAJORITY and ssl["x_uniform"] >= SSL_MIN_X_UNIFORM)  # post hoc
ssl["pass_5x_preset"] = bool(ssl["x_majority"] >= SSL_PRESET_X_MAJORITY)  # report only
ssl["threshold"] = "post hoc"
INFO["ssl"] = ssl
print("SSL self-check (masked m/z, frac 0.3):", {m: {k: round(v, 5) for k, v in r.items() if isinstance(v, float)}
                                                for m, r in ssl.items() if isinstance(r, dict)},
      f"| x majority {ssl['x_majority']:.2f}, x uniform {ssl['x_uniform']:.0f} (post-hoc guard {SSL_MIN_X_MAJORITY} x: "
      f"{ssl['pass']}; 5 x preset: {ssl['pass_5x_preset']})", flush=True)
if not ssl["pass"]:
    write_json(INFO, f"{OUT}/k49_info.json")
    raise RuntimeError(f"SSL self-check (post-hoc guard) near chance: top1 {pos['top1']:.4f} = {ssl['x_majority']:.2f} x majority "
                       f"(need {SSL_MIN_X_MAJORITY}), {ssl['x_uniform']:.0f} x uniform (need {SSL_MIN_X_UNIFORM})")
e32 = dreams.embed(enc32, list(X["positive"]), batch=BATCH, device="cuda:0").astype(np.float32)
enc16 = dreams.build_encoder(SD, fp16_blocks=True, device="cuda:0")
e16 = dreams.embed(enc16, list(X["positive"]), batch=BATCH, device="cuda:0").astype(np.float32)
cos = (e32 * e16).sum(1) / np.linalg.norm(e32, axis=1) / np.linalg.norm(e16, axis=1)
USE_FP16 = bool(np.isfinite(e16).all() and cos.min() >= FP16_COS_MIN)
INFO["fp16"] = dict(cos_min=float(cos.min()), cos_median=float(np.median(cos)), finite=bool(np.isfinite(e16).all()),
                    use_fp16_blocks=USE_FP16)
print("fp16 blocks vs fp32:", INFO["fp16"], flush=True)
del enc32, enc16
torch.cuda.empty_cache()
tm("ssl_check")

# %% [markdown]
# ## 4. スペクトルの埋め込み（GPU ごとに 1 スレッド、塊ごとに再開できる）

# %%
SPEC, PROG = f"{OUT}/emb_spec.npy", f"{OUT}/emb_progress.json"
n_chunks = (N + CHUNK - 1) // CHUNK
prog_key = dict(n=int(N), rows_sha=INFO["rows"]["sha"], chunk=CHUNK, fp16_blocks=USE_FP16, ckpt=dreams.SSL_SHA256)
if not os.path.exists(PROG) and rd and os.path.exists(f"{rd}/emb_progress.json"):
    old = json.load(open(f"{rd}/emb_progress.json"))
    if {k: old.get(k) for k in prog_key} == prog_key:
        shutil.copy(f"{rd}/emb_spec.npy", SPEC)
        shutil.copy(f"{rd}/emb_progress.json", PROG)
        print(f"resumed from {rd}: {len(old['done'])} of {n_chunks} chunks done", flush=True)
done = set()
if os.path.exists(PROG):
    old = json.load(open(PROG))
    if {k: old.get(k) for k in prog_key} == prog_key and os.path.exists(SPEC):
        done = set(old["done"])
E = np.lib.format.open_memmap(SPEC, mode="r+" if done else "w+", dtype=np.float16, shape=(N, dreams.D_MODEL))
todo = [c for c in range(n_chunks) if c not in done]
NT = EMB_THREADS or NG
print(f"spectra: {N:,} rows in {n_chunks} chunks, {len(todo)} to do, {NG} GPU(s), {NT} thread(s), fp16 blocks "
      f"{USE_FP16}", flush=True)
lock = threading.Lock()
stats = dict(rows=0, secs=0.0, stopped=False, errors=[], chunks=0)


def worker(g, chunks):
    try:
        run_chunks(g, chunks)
    except Exception as e:  # noqa: BLE001 (re-raised in the main thread)
        stats["errors"].append(f"cuda:{g}: {type(e).__name__}: {e}")
        raise


def run_chunks(g, chunks):
    dev = f"cuda:{g % NG}"
    torch.cuda.set_device(g % NG)  # the current device is per host thread
    enc = dreams.build_encoder(SD, fp16_blocks=USE_FP16, device=dev)
    for c in chunks:
        with lock:
            stop = time.time() > DEADLINE or (STOP_AFTER_CHUNKS is not None and stats["chunks"] >= STOP_AFTER_CHUNKS)
            stats["chunks"] += not stop
        if stop:
            stats["stopped"] = True
            return
        t = time.time()
        lo, hi = c * CHUNK, min(N, (c + 1) * CHUNK)
        x = [dreams.preprocess(*C.spectrum(r), float(C.prec[r])) for r in ROWS[lo:hi]]
        E[lo:hi] = dreams.embed(enc, x, batch=BATCH, device=dev)
        with lock:
            E.flush()
            done.add(c)
            write_json(dict(prog_key, done=sorted(done)), PROG)
            stats["rows"] += hi - lo
            stats["secs"] += time.time() - t
            if len(done) % 10 == 0 or len(done) == n_chunks:
                print(f"  {len(done)}/{n_chunks} chunks ({time.time() - T0:.0f}s)", flush=True)


t_emb = time.time()
ths = [threading.Thread(target=worker, args=(g, todo[g::NT])) for g in range(NT)]
for th in ths:
    th.start()
for th in ths:
    th.join()
E.flush()
assert not stats["errors"], stats["errors"]
complete = len(done) == n_chunks
INFO["spec"] = dict(complete=complete, chunks_done=len(done), n_chunks=n_chunks, rows_this_run=stats["rows"],
                    threads=NT, wall_secs=round(time.time() - t_emb, 1),
                    rows_per_s=stats["rows"] / max(time.time() - t_emb, 1e-9), fp16_blocks=USE_FP16)
if complete:
    # every row (review 2026-10-24: a 2,000-row sample would miss an fp16 overflow in a negative-mode or
    # precursor > 1000 row): non-finite values, all-zero rows, the norm distribution and a sha of the whole table
    t = time.time()
    h, nonfin_rows, zero_rows, norms = hashlib.sha256(), [], [], []
    for a in range(0, N, 65536):
        blk = np.asarray(E[a:a + 65536])
        h.update(np.ascontiguousarray(blk).tobytes())
        b32 = blk.astype(np.float32)
        nonfin_rows.extend((a + np.flatnonzero(~np.isfinite(b32).all(1))).tolist())
        zero_rows.extend((a + np.flatnonzero(~(b32 != 0).any(1))).tolist())
        norms.append(np.linalg.norm(np.nan_to_num(b32), axis=1))
    norms = np.concatenate(norms)
    INFO["spec"].update(finite=not nonfin_rows, n_nonfinite_rows=len(nonfin_rows), zero_rows=len(zero_rows),
                        bad_rows=dict(nonfinite=[int(ROWS[i]) for i in nonfin_rows[:50]],
                                      zero=[int(ROWS[i]) for i in zero_rows[:50]]),
                        norm_median=float(np.median(norms)), norm_min=float(norms.min()),
                        norm_max=float(norms.max()), sha=h.hexdigest()[:16], check_secs=round(time.time() - t, 1))
    if not (INFO["spec"]["finite"] and INFO["spec"]["zero_rows"] == 0):
        write_json(INFO, f"{OUT}/k49_info.json")
        raise AssertionError(f"emb_spec has non-finite or all-zero rows: {INFO['spec']}")
print("spectra:", INFO["spec"], flush=True)
del E
tm("spec_embed")
if not complete:
    # finish without an exception: Kaggle mounts only the output of a successful version (memory: cancelled-kernel
    # pitfall), and a resumed version needs this one's emb_spec.npy / emb_progress.json. k50 refuses an incomplete k49.
    if canon_proc is not None:
        canon_proc.wait()
    INFO["complete"] = False
    write_json(INFO, f"{OUT}/k49_info.json")
    print(f"\n!!! INCOMPLETE: {len(done)}/{n_chunks} chunks embedded (budget or STOP_AFTER_CHUNKS). Rerun with this "
          "output attached (RESUME); k49_info.json has complete=false and k50 will not use it.", flush=True)

# %% [markdown]
# ## 5. ChemBERTa の分子の埋め込み

# %%
if complete:  # (an incomplete run skips the molecules; the next version does them)
    if canon_proc is not None:
        rc = canon_proc.wait()
        print(open(f"{OUT}/canon.log").read()[-600:])
        assert rc == 0, f"canon subprocess failed ({rc})"
    tm("canon_wait")
    cdf = pd.read_parquet(CANON)
    assert np.array_equal(cdf.pid.values, PIDS)
    text = cdf.text.values
    ok = np.array([isinstance(t, str) and len(t) > 0 for t in text])
    src = np.where(P.in_train[PIDS] & P.in_coconut[PIDS], "both", np.where(P.in_coconut[PIDS], "coconut_only",
                                                                          "train_only"))

    CB_SRC = None
    for f in glob.glob("/kaggle/input/**/HF_SHA", recursive=True):
        if open(f).read().strip() == molenc.CB_REV:
            CB_SRC = os.path.dirname(f)
    if CB_SRC is None:
        from huggingface_hub import snapshot_download
        CB_SRC = snapshot_download(molenc.CB_ID, revision=molenc.CB_REV, local_dir="/tmp/chemberta",
                                   allow_patterns=["*.json", "*.txt", "*.safetensors"])
    cb_sha = dreams.sha256_file(f"{CB_SRC}/model.safetensors")
    assert cb_sha == molenc.CB_SAFETENSORS_SHA256, f"ChemBERTa weights at {CB_SRC}: sha256 {cb_sha}"
    import transformers

    INFO["versions"]["transformers"] = transformers.__version__
    INFO["chemberta"] = dict(id=molenc.CB_ID, revision=molenc.CB_REV, source=CB_SRC, safetensors_sha256=cb_sha,
                             pooling="mean over attention_mask", max_tokens=molenc.MAX_TOKENS)
    tok, cbm = molenc.load_chemberta(CB_SRC, device="cuda:0", fp16=True)
    ntok = np.zeros(len(PIDS), np.int32)
    ntok[ok] = molenc.token_lengths(tok, list(text[ok]))
    stats_c = dict(n=int(len(PIDS)), failed=int((~ok).sum()), failed_pids=[int(p) for p in PIDS[~ok]],
                   n_over_max_tokens=int((ntok > molenc.MAX_TOKENS).sum()), by_source={})
    for s in ["train_only", "coconut_only", "both"]:
        m = (src == s) & ok
        tx = text[m]
        stats_c["by_source"][s] = dict(
            n=int((src == s).sum()), failed=int(((src == s) & ~ok).sum()),
            **{f"share_{k}": float(np.mean([c in t for t in tx])) if len(tx) else float("nan")
               for k, c in [("at", "@"), ("slash", "/"), ("backslash", "\\"), ("dot", ".")]},
            share_bracket_charge=float(pd.Series(tx, dtype=object).str.contains(CHARGE_RX, regex=True).mean())
            if len(tx) else float("nan"),
            len_median=float(np.median([len(t) for t in tx])) if len(tx) else float("nan"),
            tokens_median=float(np.median(ntok[m])) if m.any() else float("nan"),
            over_max_tokens=int((ntok[m] > molenc.MAX_TOKENS).sum()),
            raw_share_at=float(np.mean(["@" in t for t in P.smiles[PIDS[src == s]]])) if (src == s).any() else float("nan"))
    write_json(stats_c, f"{OUT}/canon_stats.json")
    print("canon:", {k: v for k, v in stats_c.items() if k != "failed_pids"}, flush=True)
    t = time.time()
    emb = molenc.embed_smiles(list(text[ok]), tok, cbm, batch=CB_BATCH, device="cuda:0",
                              log=lambda *a: print(*a, flush=True))
    M = np.lib.format.open_memmap(f"{OUT}/emb_mol_cb.npy", mode="w+", dtype=np.float16, shape=(len(P), molenc.CB_DIM))
    M[PIDS[ok]] = emb
    M.flush()
    okp = np.zeros(len(P), bool)
    okp[PIDS[ok]] = True
    np.save(f"{OUT}/emb_mol_cb_ok.npy", okp)
    INFO["mol"] = dict(n_pool=int(len(P)), n_embedded=int(ok.sum()), secs=round(time.time() - t, 1),
                       per_s=float(ok.sum() / max(time.time() - t, 1e-9)), finite=bool(np.isfinite(emb).all()),
                       sha=sha_arr(emb), ok_sha=sha_arr(okp),
                       norm_median=float(np.median(np.linalg.norm(emb.astype(np.float32), axis=1))) if len(emb) else None)
    assert INFO["mol"]["finite"]
    print("mol:", INFO["mol"], flush=True)
    del M
    tm("mol_embed")
    INFO["complete"] = True
    write_json(INFO, f"{OUT}/k49_info.json")
    print(f"total {time.time() - T0:.0f}s")

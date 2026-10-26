# %% [markdown]
# # casmi26-k50-msalign
# 計画 10.2〜10.4（2026-10-24、事前登録）: k49 の凍結した埋め込み（DreaMS ssl、ChemBERTa / Morgan）の上で MSAlign 型の
# 射影の塔を 6 つの変種で学習し、問いの窓の全候補の点数と門 S、W を出す（casmi26.msalign）。
# - 変種: cb_mass、cb_formula、fp_mass、fp_formula（分子の塔 cb / fp × 負例 mass ±10 ppm / formula 同じ分子式）と、
#   近道の確かめの cb_blind、fp_blind（スペクトルの塔の入力が学習する定数、負例 mass）。設定は msalign.Cfg（10.1 の
#   公開コードの値、30,000 step、B 128 × K 128、AMP）で、最後の step のモデルを使う
# - GPU ごとに 1 つの子プロセスで 2 つずつ学習する。状態（<out>_state.pt、casmi26.ckpt）を 1,000 step ごとに書き、
#   時間の枠（BUDGET_H − SCORE_RESERVE_H）で止まった変種は、この出力を入力に付けた次の版で続きから学習する（RESUME）。
#   どれかの変種が終わっていなければ、点数と門は出さずに止める
# - 点数: k30 の窓（query_info4 の target / ppm で P.window、k47 と同じ）の全候補で msa（問いのスペクトルごとの生の
#   コサインの平均）、msa_z、msa_rank を計算し、k30 が残した (qid, pid) の行に付けて msa_<変種>_<組>.parquet に書く
#   （組 = A、B_A、R1、R2、V。B_A は A を使い回す）。問いのスペクトルは msalign.query_rows（k49 と同じ）
# - 門（10.3）: S = 同じ分子の塔の blind の B_N の窓の MRR@25 − 無作為の MRR < 0.05、かつ出どころの係数
#   （fpnet.provenance_coef を msa の点数に広げたもの、A ∪ R2 ∪ V の問いの正解の ±10 ppm の窓）>= −0.20。
#   W = B_N で msa の窓の RR が同じ塔の blind を上回る（対の差の 95% 下限 > 0）。参考に FPNet v2b の fz（k30 の列）の
#   窓の MRR を並べる。B_N = R2 ∪ V で正解が COCONUT にある問い（k48 と同じ。問いは k30 の query_info4 から取る:
#   k30 に候補が無い R2 の 5 問（ppm が NaN）は空の窓で RR 0、無作為 0、k48 / k51 と同じ 1,096 問）
# - 再開と「終わった」の判定は、重みと状態の cfg["tag"]（k49 の行の sha、表の sha、fp16 のブロック、ChemBERTa の sha）が
#   今の k49 と同じときだけ。違えば前の版のファイルを使わずに最初から学習する（2026-10-24 の確認を受けて）
# 出力: msa_<変種>.pt、msa_<変種>_state.pt、train_<変種>.log、msa_<変種>_<組>.parquet、window_mrr.csv、msa_rr.parquet、
# gates.json、gates.csv、k50_info.json。
# 入力: k49（emb_spec / emb_rows / emb_mol_cb）、k01、k05、k05b、k26（pool_formula.npz）、k30-s0〜s3、
# casmi26-mspred（audit/p3_clean_qids.csv、報告だけ）。

# %%
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from casmi26 import cache, msalign, paths, pool, subform

T0 = time.time()
OUT = "/kaggle/working"
VARIANTS = ["cb_mass", "cb_blind", "fp_mass", "fp_blind", "cb_formula", "fp_formula"]  # launch order
SETS = ["A", "R1", "R2", "V"]
SEED = 0
MAX_PER_STRUCT = 4
MAX_SPEC = 16
MAX_STEPS = 30000
WARMUP = 2000
LOG_EVERY = 500
SAVE_EVERY = 1000
BUDGET_H = 8.5
SCORE_RESERVE_H = 1.0          # kept free for scoring and gates after training
N_BOOT = 2000
PPM = 10.0                     # provenance windows (+- of the truth's mass)
TOPK = 25
BLIND_MARGIN = 0.05            # gate S: blind B_N window MRR - random MRR must stay below this
PROV_MIN = -0.20               # gate S: provenance coefficient >= this
PROV_SETS = ["A", "R2", "V"]
RESUME = None                  # a previous k50 output dir; None = look for msa_*_state.pt under /kaggle/input
STOP_AT = None                 # test only: stop every variant at this step, as if interrupted
MAX_QUERIES = None             # 手元の動作確認だけ（組ごとの先頭の問いの数）
SMOKE_EXTRA_QIDS = []          # 手元の動作確認だけ（MAX_QUERIES に加える問い、例えば候補の無い R2 の 5 問）
DEADLINE_TRAIN = T0 + (BUDGET_H - SCORE_RESERVE_H) * 3600
MOL_OF = {v: v.split("_")[0] for v in VARIANTS}
NEG_OF = {v: ("formula" if v.endswith("formula") else "mass") for v in VARIANTS}


def tm(msg):
    print(f"== {msg} ({time.time() - T0:.0f}s)", flush=True)


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def write_json(obj, path):
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else float(x))
    os.replace(tmp, path)


I49 = json.load(open(paths.find("k49_info.json")))
assert I49.get("complete"), "k49 did not finish"
SMOKE49 = bool(I49["settings"]["smoke"])
assert MAX_QUERIES or not SMOKE49, "k49 output is a smoke run"
EMB_ROWS = np.load(paths.find("emb_rows.npy"))
SPEC_PATH, CB_PATH = paths.find("emb_spec.npy"), paths.find("emb_mol_cb.npy")
CB_OK = np.load(paths.find("emb_mol_cb_ok.npy"))
assert sha_arr(EMB_ROWS) == I49["rows"]["sha"], "emb_rows.npy does not match k49_info.json"
assert MAX_QUERIES or not SMOKE_EXTRA_QIDS, "SMOKE_EXTRA_QIDS is for smoke runs only"
# identity of the k49 tables the variants are trained on (stored as cfg["tag"] in msa_<v>.pt / msa_<v>_state.pt)
TAG = dict(rows_sha=I49["rows"]["sha"], spec_sha=I49["spec"].get("sha"), fp16_blocks=I49["fp16"]["use_fp16_blocks"],
           ckpt_sha256=I49["ckpt"]["sha256"], chemberta_sha256=(I49.get("chemberta") or {}).get("safetensors_sha256"),
           mol_sha=I49["mol"].get("sha"), mol_ok_sha=I49["mol"].get("ok_sha"))
C = cache.Cache(cache.Cache.find_dir())
PD = pool.Pool.find_dir()
P = pool.Pool(PD, pubchem=False)
FT = subform.FormulaTable(subform.FormulaTable.find_dir())
Q = pd.read_csv(f"{PD}/queries.csv").set_index("qid")
HOLD = set(pd.read_csv(f"{PD}/holdout.csv").pid.values)
assert len(CB_OK) == len(P), "emb_mol_cb does not match the pool"
RMAP = msalign.row_map(EMB_ROWS, len(C.keep))
tm(f"inputs (k49 smoke {SMOKE49}, {len(EMB_ROWS):,} embedded rows, cb ok {CB_OK.sum():,}/{len(P):,})")

# %% [markdown]
# ## 1. 学習の行、負例、問い

# %%
rows, pid = msalign.train_rows(C, P, HOLD, MAX_PER_STRUCT, seed=SEED)
assert sha_arr(rows) == I49["train_rows_full"]["sha"], "train_rows differ from k49's"
if SMOKE49:
    m = RMAP[rows] >= 0
    rows, pid = rows[m], pid[m]
else:
    assert sha_arr(rows) == I49["rows"]["train_sha"]
VOCAB = msalign.make_vocab(C, rows)
ad, pol, ce = msalign.meta_of(C, rows, VOCAB)
TS = msalign.TrainSet(rows, pid, msalign.emb_index(rows, RMAP), ad, pol, ce, VOCAB)
NM = msalign.NegTable.mass(C, P, HOLD)
NF = msalign.NegTable.formula(P, FT, HOLD)
PREP, VOC = "/tmp/k50_prep.npz", "/tmp/k50_vocab.json"
np.savez(PREP, rows=TS.rows, pid=TS.pid, emb_ix=TS.emb_ix, adduct_ix=TS.adduct_ix, pol_ix=TS.pol_ix, ce=TS.ce,
         mass_order=NM.order, mass_lo=NM.lo, mass_hi=NM.hi, formula_order=NF.order, formula_lo=NF.lo,
         formula_hi=NF.hi)
json.dump(VOCAB, open(VOC, "w"))
np.save("/tmp/k50_fp.npy", P.fp)
up = np.unique(TS.pid)
INFO = dict(k49=dict(settings=I49["settings"], rows=I49["rows"], ssl={k: I49["ssl"][k] for k in ["x_majority",
                                                                                                    "x_uniform", "pass"]},
                     fp16=I49["fp16"], chemberta=I49.get("chemberta"), src_sha=I49.get("src_sha")),
            train=dict(n_rows=int(len(TS)), n_struct=int(len(up)), n_adducts=len(VOCAB),
                       cb_missing_struct=int((~CB_OK[up]).sum()),
                       neg_mass_none=float((NM.sizes(up) == 0).mean()), neg_formula_none=float((NF.sizes(up) == 0).mean())),
            settings=dict(variants=VARIANTS, max_steps=MAX_STEPS, warmup=WARMUP, log_every=LOG_EVERY,
                          save_every=SAVE_EVERY, budget_h=BUDGET_H, score_reserve_h=SCORE_RESERVE_H, n_boot=N_BOOT,
                          ppm=PPM, topk=TOPK, blind_margin=BLIND_MARGIN, prov_min=PROV_MIN, prov_sets=PROV_SETS,
                          stop_at=STOP_AT, max_queries=MAX_QUERIES, seed=SEED),
            tag=TAG, src_sha=os.environ.get("CASMI26_SHA"), timings={})
print("train:", INFO["train"], "| k49 tag:", TAG, flush=True)

infos = pd.concat([pd.read_csv(p) for p in sorted(glob.glob("/kaggle/input/**/query_info4_s*.csv", recursive=True))],
                  ignore_index=True).drop_duplicates("qid").set_index("qid")
fi = [json.load(open(p)) for p in sorted(glob.glob("/kaggle/input/**/features_info4.json", recursive=True))]
assert fi and all(f == fi[0] for f in fi), "k30 shards disagree"
feats = {s: pd.concat([pd.read_parquet(p, columns=["qid", "pid", "fz"]) for p in
                       sorted(glob.glob(f"/kaggle/input/**/feats4_{s}_s*.parquet", recursive=True))], ignore_index=True)
         .drop_duplicates(["qid", "pid"]).sort_values(["qid", "pid"], kind="mergesort").reset_index(drop=True)
         for s in SETS + ["B_A"]}
qa = Q[Q.set == "A"].reset_index().set_index("rows").qid
BA2A = Q[Q.set == "B_A"].rows.map(qa)
assert BA2A.notna().all() and BA2A.is_unique, "B_A is not a copy of A"
BA2A = BA2A.astype(int)
assert np.allclose(infos.loc[BA2A.index, ["target", "ppm"]].values, infos.loc[BA2A.values, ["target", "ppm"]].values,
                   rtol=0, atol=0), "B_A windows differ from A's"
QIDS = {}
for s in SETS:
    # the questions come from query_info4 (as k48 / k51), not from the feats rows: k30 keeps no rows for queries
    # without candidates (5 R2 queries, ppm NaN), which still count in B_N with RR 0
    q = np.sort(infos.index[infos.set == s].values)
    assert np.isin(feats[s].qid.unique(), q).all(), f"{s}: feats qids outside query_info4"
    if SMOKE49:
        q = q[np.isin(q, I49["qids"][s])]
    if MAX_QUERIES:
        q = np.union1d(q[:MAX_QUERIES], q[np.isin(q, SMOKE_EXTRA_QIDS)])
    QIDS[s] = q
q = np.sort(infos.index[infos.set == "B_A"].values)
QIDS["B_A"] = q[np.isin(BA2A.reindex(q).values, QIDS["A"])]
feats = {s: f[f.qid.isin(QIDS[s])].reset_index(drop=True) for s, f in feats.items()}


def window_of(q):
    """k30's window of query q (P.window(target, ppm), as k47); empty when k30 had no candidates (ppm NaN)."""
    t, ppm = float(infos.loc[q, "target"]), float(infos.loc[q, "ppm"])
    if not (np.isfinite(t) and np.isfinite(ppm)) or int(infos.loc[q, "n_cand"]) == 0:
        return np.zeros(0, np.int64)
    return np.asarray(P.window(t, ppm), np.int64)


QIN, WIN, TRUTH = {}, {}, {}
for s in SETS:
    QIN[s] = [msalign.query_inputs(C, msalign.query_rows(C, Q.loc[q, "rows"], MAX_SPEC), RMAP, VOCAB) for q in QIDS[s]]
    WIN[s] = [window_of(q) for q in QIDS[s]]
    TRUTH[s] = Q.loc[QIDS[s], "pid"].values.astype(np.int64)
    assert all(len(x["emb_ix"]) > 0 for x in QIN[s]), f"{s}: a query without a usable spectrum"
WIN["B_A"] = [WIN["A"][int(np.searchsorted(QIDS["A"], BA2A[q]))] for q in QIDS["B_A"]]
TRUTH["B_A"] = Q.loc[QIDS["B_A"], "pid"].values.astype(np.int64)
in_co = P.in_coconut
BN = {s: in_co[TRUTH[s]] for s in ["R2", "V"]}
clean72 = set(pd.read_csv(paths.find("p3_clean_qids.csv")).qid)
BNQ = np.concatenate([QIDS[s][BN[s]] for s in ["R2", "V"]])
INFO["queries"] = {s: int(len(v)) for s, v in QIDS.items()}
INFO["queries"]["B_N"] = int(len(BNQ))
INFO["queries"]["B_N_clean72"] = int(np.isin(BNQ, list(clean72)).sum())
INFO["windows"] = {s: dict(median=float(np.median([len(w) for w in WIN[s]])),
                           truth_in=float(np.mean([(w == t).any() for w, t in zip(WIN[s], TRUTH[s])])),
                           empty_qids=[int(q) for q, w in zip(QIDS[s], WIN[s]) if len(w) == 0])
                   for s in WIN}
print("queries:", INFO["queries"], "| windows:", INFO["windows"], flush=True)
INFO["timings"]["prep"] = round(time.time() - T0, 1)
tm("prep")

# %% [markdown]
# ## 2. 学習（GPU ごとに 1 つの子プロセス、状態から再開できる）

# %%
import torch

NG = max(1, torch.cuda.device_count())
WORKER = "/tmp/k50_worker.py"
open(WORKER, "w").write('''import json, sys, time
sys.path.insert(0, "/tmp/casmi26_src")
import numpy as np
import torch
from casmi26 import msalign
a = json.load(open(sys.argv[1]))
d = np.load(a["prep"])
vocab = json.load(open(a["vocab"]))
ts = msalign.TrainSet(d["rows"], d["pid"], d["emb_ix"], d["adduct_ix"], d["pol_ix"], d["ce"], vocab)
k = a["neg"]
neg = msalign.NegTable(d[k + "_order"], d[k + "_lo"], d[k + "_hi"], k)
mol = msalign.MolTable(np.load(a["cb"]), "cb") if a["mol"] == "cb" else msalign.MolTable(np.load(a["fp"]), "fp", 2048)
spec = None if a["blind"] else np.load(a["spec"], mmap_mode="r")
cfg = msalign.Cfg(name=a["variant"], blind=a["blind"], max_steps=a["max_steps"], warmup=a["warmup"],
                  log_every=a["log_every"], save_every=a["save_every"])
print("torch", torch.__version__, torch.cuda.get_device_name(0), flush=True)
msalign.train(spec, ts, mol, neg, a["out"], cfg, budget_secs=a["deadline"] - time.time(), seed=a["seed"],
              resume=a["resume"], stop_at=a["stop_at"], device="cuda", tag=a["tag"])
''')


def weights_step(path):
    if not os.path.exists(path):
        return None
    ck = torch.load(path, map_location="cpu", weights_only=True)
    return int(ck["step"]), ck["cfg"]


def file_tag(path):
    """cfg["tag"] of a weights (msa_<v>.pt) or state (msa_<v>_state.pt) file; 'unreadable' when it cannot be read."""
    try:
        ck = torch.load(path, map_location="cpu", weights_only=True)
        return (ck["cfg"] if "cfg" in ck else ck["extra"]["cfg"]).get("tag")
    except Exception as e:  # noqa: BLE001
        return f"unreadable ({type(e).__name__})"


def plan_variant(v):
    """-> ('done', None) | ('resume', state path) | ('new', None); copies a previous version's files in first, but
    only when they were trained on the same k49 tables (cfg["tag"] == TAG); otherwise they are not used."""
    w, st = f"{OUT}/msa_{v}.pt", f"{OUT}/msa_{v}_state.pt"
    if not os.path.exists(st):
        src = [p for p in glob.glob(f"/kaggle/input/**/msa_{v}_state.pt", recursive=True)] if RESUME is None else \
            [p for p in [f"{RESUME}/msa_{v}_state.pt"] if os.path.exists(p)]
        if src:
            t = file_tag(src[0])
            if t != TAG:
                STATUS[v]["refused"] = dict(path=src[0], tag=t)
                print(f"!!! {v}: {src[0]} was trained on other k49 tables (tag {t}); not reused", flush=True)
            else:
                shutil.copy(src[0], st)
                if os.path.exists(src[0].replace("_state.pt", ".pt")):
                    shutil.copy(src[0].replace("_state.pt", ".pt"), w)
                print(f"{v}: state copied from {os.path.dirname(src[0])}", flush=True)
    ws = weights_step(w)
    if ws and ws[0] >= MAX_STEPS and ws[1].get("max_steps") == MAX_STEPS and ws[1].get("n_rows") == len(TS) and \
            ws[1].get("tag") == TAG:
        return "done", None
    if os.path.exists(st):
        t = file_tag(st)
        if t == TAG:
            return "resume", st
        STATUS[v]["refused"] = dict(path=st, tag=t)
        print(f"!!! {v}: {st} was trained on other k49 tables (tag {t}); training from step 0 (overwrites it)",
              flush=True)
    return "new", None


STATUS = {}
pending = []
for v in VARIANTS:
    STATUS[v] = {}
    kind, st = plan_variant(v)
    STATUS[v]["start"] = kind
    if kind != "done":
        pending.append((v, st))
print("training plan:", {v: s["start"] for v, s in STATUS.items()}, f"| {NG} GPU slot(s)", flush=True)
running = {}
while pending or running:
    for g in range(NG):
        if g not in running and pending and time.time() < DEADLINE_TRAIN:
            v, st = pending.pop(0)
            a = dict(variant=v, prep=PREP, vocab=VOC, spec=SPEC_PATH, cb=CB_PATH, fp="/tmp/k50_fp.npy", mol=MOL_OF[v],
                     neg=NEG_OF[v], blind=v.endswith("blind"), max_steps=MAX_STEPS, warmup=WARMUP, log_every=LOG_EVERY,
                     save_every=SAVE_EVERY, out=f"{OUT}/msa_{v}.pt", deadline=DEADLINE_TRAIN, seed=SEED, resume=st,
                     stop_at=STOP_AT, tag=TAG)
            json.dump(a, open(f"/tmp/k50_{v}.json", "w"))
            lg = open(f"{OUT}/train_{v}.log", "a")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONUNBUFFERED="1")
            running[g] = (v, subprocess.Popen([sys.executable, WORKER, f"/tmp/k50_{v}.json"], stdout=lg,
                                              stderr=subprocess.STDOUT, env=env), time.time())
            print(f"  start {v} on GPU {g} ({'resume' if st else 'new'}) ({time.time() - T0:.0f}s)", flush=True)
    if not running:
        break  # past the deadline with variants left
    time.sleep(5)
    for g in list(running):
        v, p, t = running[g]
        if p.poll() is not None:
            STATUS[v].update(rc=p.returncode, secs=round(time.time() - t, 1))
            tail = open(f"{OUT}/train_{v}.log").read().splitlines()
            print(f"  end {v} rc {p.returncode} ({time.time() - t:.0f}s): " + " | ".join(
                l for l in tail if "step" in l)[-600:], flush=True)
            del running[g]
for v in VARIANTS:
    ws = weights_step(f"{OUT}/msa_{v}.pt")
    STATUS[v].update(step=ws[0] if ws else 0, complete=bool(ws and ws[0] >= MAX_STEPS and ws[1].get("tag") == TAG))
    if ws:
        h = torch.load(f"{OUT}/msa_{v}.pt", map_location="cpu", weights_only=True)["hist"]
        STATUS[v]["hist"] = h
        STATUS[v]["last"] = h[-1] if h else None
INFO["training"] = STATUS
INFO["timings"]["train"] = round(time.time() - T0, 1)
COMPLETE = all(STATUS[v]["complete"] for v in VARIANTS)
print("training:", {v: (s["step"], s.get("rc"), s.get("last") and round(s["last"]["loss"], 4)) for v, s in
                    STATUS.items()}, flush=True)
if not COMPLETE:
    # finish without an exception: Kaggle mounts only the output of a successful version (memory: cancelled-kernel
    # pitfall), and the resumed version needs this one's msa_<variant>_state.pt. k51 refuses an incomplete gates.json.
    INFO["complete"] = False
    write_json(INFO, f"{OUT}/k50_info.json")
    write_json(dict(complete=False, status={v: dict(step=s["step"], rc=s.get("rc")) for v, s in STATUS.items()}),
               f"{OUT}/gates.json")
    print("\n!!! INCOMPLETE: not every variant reached max_steps; rerun with this output attached (RESUME): " +
          str({v: s["step"] for v, s in STATUS.items()}) + "; no scores or gates in this version", flush=True)
tm("train")

# %% [markdown]
# ## 3. 窓の全候補の点数と、k30 の行への書き出し

# %%
if COMPLETE:
    E = np.load(SPEC_PATH, mmap_mode="r")
    MOLS = {"cb": msalign.MolTable(np.load(CB_PATH), "cb"), "fp": msalign.MolTable(P.fp, "fp", 2048)}
    RR, RAND, PROV = {}, {}, {}
    prov_q = [(s, i) for s in PROV_SETS for i in range(len(QIDS[s]))]
    for v in VARIANTS:
        t = time.time()
        M = msalign.MSAModel(f"{OUT}/msa_{v}.pt", spec_table=E, mol=MOLS[MOL_OF[v]], device="cuda:0")
        assert M.vocab == VOCAB and M.step == MAX_STEPS and file_tag(f"{OUT}/msa_{v}.pt") == TAG
        full = {}
        for s in SETS:
            sc = msalign.score_windows(M, QIN[s], WIN[s])
            RR[(v, s)] = msalign.window_rr(sc, WIN[s], TRUTH[s], k=TOPK)
            full[s] = pd.concat([pd.DataFrame({"qid": q, "pid": w, "msa": x[:, 0], "msa_z": x[:, 1], "msa_rank": x[:, 2]})
                                 for q, w, x in zip(QIDS[s], WIN[s], sc)], ignore_index=True)
        RR[(v, "B_A")] = msalign.window_rr([full["A"][full["A"].qid == BA2A[q]].msa.values for q in QIDS["B_A"]],
                                           WIN["B_A"], TRUTH["B_A"], k=TOPK)
        fa = full["A"].rename(columns={"qid": "qa"})
        for s in SETS + ["B_A"]:
            f = feats[s][["qid", "pid"]]
            if s == "B_A":
                m = f.assign(qa=BA2A.reindex(f.qid).values).merge(fa, on=["qa", "pid"], how="left").drop(columns="qa")
            else:
                m = f.merge(full[s], on=["qid", "pid"], how="left")
            assert len(m) == len(f) and m.msa.notna().all(), f"{v} {s}: k30 rows outside the recomputed window"
            m[["qid", "pid"] + msalign.MSA_FEATURES].to_parquet(f"{OUT}/msa_{v}_{s}.parquet", index=False)
        fn = msalign.msa_score_fn(M, [QIN[s][i] for s, i in prov_q])
        PROV[v] = msalign.provenance_coef(fn, P, np.array([TRUTH[s][i] for s, i in prov_q]), HOLD, ppm=PPM,
                                          n_boot=N_BOOT)
        print(f"{v}: scored ({time.time() - t:.0f}s); window MRR " +
              ", ".join(f"{s} {RR[(v, s)].mean():.4f}" for s in SETS + ["B_A"]) +
              f"; provenance {PROV[v]['est']:.4f} [{PROV[v]['lo']:.4f}, {PROV[v]['hi']:.4f}]", flush=True)
        del M
        torch.cuda.empty_cache()
    for s in SETS + ["B_A"]:
        RAND[s] = msalign.random_mrr([len(w) if (w == t).any() else 0 for w, t in zip(WIN[s], TRUTH[s])], k=TOPK)
    # FPNet v2b fz (k30 column, k30 rows: A / V full windows, R1 / R2 the kept top 300 + truth) for reference
    for s in SETS + ["B_A"]:
        f = feats[s]
        g = {q: x for q, x in f.groupby("qid")}
        e = f.iloc[:0]  # queries without k30 rows (no candidates): RR 0
        RR[("fz_v2b", s)] = msalign.window_rr([g.get(q, e).fz.values for q in QIDS[s]],
                                              [g.get(q, e).pid.values for q in QIDS[s]], TRUTH[s], k=TOPK)
    INFO["timings"]["score"] = round(time.time() - T0, 1)
    tm("score")

# %% [markdown]
# ## 4. 門 S と W（10.3）、窓の MRR の表

# %%
if COMPLETE:
    MODELS = VARIANTS + ["fz_v2b"]


    def bn(x):
        return np.concatenate([x["R2"][BN["R2"]], x["V"][BN["V"]]])


    for m in MODELS:
        RR[(m, "B_N")] = bn({s: RR[(m, s)] for s in ["R2", "V"]})
        RR[(m, "B_N_clean72")] = RR[(m, "B_N")][np.isin(BNQ, list(clean72))]
    RAND["B_N"] = bn(RAND)
    RAND["B_N_clean72"] = RAND["B_N"][np.isin(BNQ, list(clean72))]
    TSETS = ["A", "B_A", "B_N", "V", "R1", "R2", "B_N_clean72"]
    tab = pd.DataFrame([dict(model=m, set=s, n=len(RR[(m, s)]), mrr=float(RR[(m, s)].mean()) if len(RR[(m, s)]) else np.nan,
                             random=float(RAND[s].mean()) if len(RAND[s]) else np.nan,
                             **{k: msalign.paired_ci(RR[(m, s)], RAND[s], n_boot=N_BOOT)[k] for k in ["lo", "hi"]})
                        for m in MODELS for s in TSETS])
    tab.to_csv(f"{OUT}/window_mrr.csv", index=False)
    qset = {s: QIDS[s] for s in SETS + ["B_A"]}
    qset["B_N"], qset["B_N_clean72"] = BNQ, BNQ[np.isin(BNQ, list(clean72))]
    pd.concat([pd.DataFrame({"model": m, "set": s, "qid": qset[s], "rr": RR[(m, s)]}) for m in MODELS for s in TSETS],
              ignore_index=True).to_parquet(f"{OUT}/msa_rr.parquet", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(tab.pivot(index="model", columns="set", values="mrr").round(4).to_string())
    rand_bn = float(RAND["B_N"].mean())
    G = dict(complete=True, smoke=bool(SMOKE49 or MAX_QUERIES or MAX_STEPS < 30000), random_B_N=rand_bn,
             rule=dict(blind_margin=BLIND_MARGIN, prov_min=PROV_MIN, prov_sets=PROV_SETS, ppm=PPM, topk=TOPK, n_boot=N_BOOT,
                       W="paired_ci(rr_B_N(variant), rr_B_N(<mol>_blind)).lo > 0", S_blind="mrr_B_N(<mol>_blind) - "
                       "mean random_mrr_B_N < blind_margin", S_prov="provenance_coef(msa).est >= prov_min"),
             n_B_N=int(len(BNQ)), towers={}, variants={}, eligible=[])
    for t in ["cb", "fp"]:
        b = f"{t}_blind"
        ex = float(RR[(b, "B_N")].mean()) - rand_bn
        G["towers"][t] = dict(blind=b, blind_mrr_B_N=float(RR[(b, "B_N")].mean()), blind_excess=ex,
                              pass_S_blind=bool(ex < BLIND_MARGIN), blind_provenance=PROV[b])
    for v in VARIANTS:
        t = MOL_OF[v]
        rec = dict(mol=t, neg=NEG_OF[v], blind=v.endswith("blind"), step=STATUS[v]["step"], last=STATUS[v].get("last"),
                   provenance=PROV[v], mrr={s: float(RR[(v, s)].mean()) for s in TSETS})
        if not rec["blind"]:
            w = msalign.paired_ci(RR[(v, "B_N")], RR[(f"{t}_blind", "B_N")], n_boot=N_BOOT)
            rec.update(W=w, pass_S_blind=G["towers"][t]["pass_S_blind"], pass_S_prov=bool(PROV[v]["est"] >= PROV_MIN),
                       pass_W=bool(w["lo"] > 0))
            rec["eligible"] = bool(rec["pass_S_blind"] and rec["pass_S_prov"] and rec["pass_W"])
            if rec["eligible"]:
                G["eligible"].append(v)
        G["variants"][v] = rec
    G["fz_v2b"] = {s: float(RR[("fz_v2b", s)].mean()) for s in TSETS}
    write_json(G, f"{OUT}/gates.json")
    pd.DataFrame([dict(variant=v, mol=r["mol"], neg=r["neg"], blind=r["blind"], step=r["step"],
                       loss=r["last"]["loss"] if r["last"] else np.nan, acc=r["last"]["acc"] if r["last"] else np.nan,
                       temp=r["last"]["temp"] if r["last"] else np.nan,
                       **{f"mrr_{s}": r["mrr"][s] for s in TSETS}, random_B_N=rand_bn,
                       prov=r["provenance"]["est"], prov_lo=r["provenance"]["lo"], prov_hi=r["provenance"]["hi"],
                       W_d=r.get("W", {}).get("est"), W_lo=r.get("W", {}).get("lo"),
                       blind_excess=G["towers"][r["mol"]]["blind_excess"], pass_S_blind=r.get("pass_S_blind"),
                       pass_S_prov=r.get("pass_S_prov"), pass_W=r.get("pass_W"), eligible=r.get("eligible"))
                  for v, r in G["variants"].items()]).to_csv(f"{OUT}/gates.csv", index=False)
    print(pd.read_csv(f"{OUT}/gates.csv").drop(columns=[f"mrr_{s}" for s in ["R1", "B_N_clean72"]]).round(4).to_string())
    print("ELIGIBLE:", G["eligible"])
    INFO["complete"] = True
    INFO["timings"]["total"] = round(time.time() - T0, 1)
    INFO["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    INFO["torch"] = torch.__version__
    write_json(INFO, f"{OUT}/k50_info.json")
    print(f"total {time.time() - T0:.0f}s")

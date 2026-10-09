# %% [markdown]
# # casmi26-k19-gen-probe
# M5 の P0-2（2026-10-09）: 正解がライブラリにもプールにも無い分子（クラス3）の代わりのセットで、類縁体などの親から
# 反応 2 段までで正解を作れるか（再現率）、作れたら窓の候補と混ぜて何位に来るか（MRR@25）を測る。CPU。
# - 代理のセット: G_A = B_A と同じクエリ（250、timsTOF の天然物）、G_N = B_N と同じクエリ（R2 ∪ V で正解が COCONUT
#   にあるもの）。隠し方は B_A / R2 / V と同じ（正解のスペクトルはすべてのライブラリから隠す）。加えて正解の pid を
#   窓の候補と親の候補から外す。G の正解は COCONUT にある構造なので、本当のクラス3（PubChem に無い）より楽観的
# - 親: (a) 類縁体（スペクトルの類似度の上位 20、類似度 ≥ 0.3）、(b) プール（反応の組み合わせの Δ ごとに、目標の質量 − Δ
#   の窓にある構造を fz で並べた上位 5）。全体で最大 50。反応は casmi26.derive（自前、2 段まで、質量の合う組み合わせだけ）
#   (b) は窓をまたいで生の f·z で並べて上限まで取る。質量の違う窓どうしを比べるので、大きい親（除去の組み合わせ）が
#   小さい親（付加の組み合わせ）より上に来やすい（計画には無い細部。反応別の表を読むときの注意）
# - 正解との一致: すべての生成物を立体なしの正準 SMILES で（互変異性はそろわないので控えめ）。並べるのは親の順に最大 500 件
#   で、その 500 件は標準の InChIKey14 で窓の候補との重複を除く（互変異性の違いで新しい候補は少し多めに出る）
# - 点数: fz（指紋 · z）、ap4（類縁体との Tanimoto × 類似度⁴ の最大）、組み合わせ = z(fz) + z(ap4)。z と順位は
#   正解を除いた窓の候補 ∪ 生成物で計算し直す。同点は正解に不利に数える。生成物の指紋は書いたままの SMILES から
#   （窓の候補は互変異性を正準化した指紋。小さなずれがある）
# - 薄まりの目安（表示だけ）: 同じクエリで正解をプールに残したまま（B_A / B_N）、生成物を足す前と後の MRR
# 続けるかどうかの決め方（固定）: G_N か G_A のどちらかで、組み合わせの MRR@25 ≥ 0.03 かつ再現率（500 件の上限の後）≥ 0.10
# なら生成を組み込む（P1-2）。満たさなければここで止め、反応別の当たりを記録する。
# 時間の上限 3.5 時間: G_A を先に全部、次に G_N を乱数の順に処理し、時間切れならそこまでで報告する。
# 入力: k01（キャッシュ）、k05（プール、queries.csv）、k05b（質量の修正）、k15-s1（fpnet2.pt = v2b）。
# インターネット: rdkit の pip install に使う。

# %%
import subprocess
import sys

r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rdkit"], capture_output=True, text=True)
print(r.stdout[-300:], r.stderr[-300:])
import rdkit

print("rdkit", rdkit.__version__)

# %%
import glob
import json
import time

import numpy as np
import pandas as pd

from casmi26 import cache, chem, derive, fpnet, harness, pool, search, track

T0 = time.time()
OUT = "/kaggle/working"
EXP = "M5"
PPM = 10.0  # 生成物と親の窓（目標の質量の ±PPM を Da にした幅）
BUDGET_SECS = 3.5 * 3600
N_ANALOG_PARENTS = 20
MIN_SIM = 0.3
N_POOL_PER_DELTA = 5
MAX_PARENTS = 50
MAX_PER_PARENT = 100
MAX_RANKED = 500
N_BOOT = 2000
GATE_MRR, GATE_RECALL = 0.03, 0.10
EXPECT_VARIANT = "v2b"
SEARCH_CFG = search.SearchCfg()  # 手元の動作確認では窓を広げる

MODEL = fpnet.load_checked("fpnet2.pt", min_epoch=1)
res = [json.load(open(p)) for p in glob.glob("/kaggle/input/**/result_v2*.json", recursive=True)]
assert len(res) == 1 and res[0]["variant"] == EXPECT_VARIANT and res[0]["epoch"] == MODEL.epoch, \
    f"attach the k15 {EXPECT_VARIANT} output (found {[(r['variant'], r['epoch']) for r in res]})"
C = cache.Cache(cache.Cache.find_dir())
P = pool.Pool(pool.Pool.find_dir())
assert len(P) == P.n_base, "the PubChem part must not be attached"
S = search.Searcher(C, P, cfg=SEARCH_CFG, model=MODEL)
Q = pd.read_csv(f"{pool.Pool.find_dir()}/queries.csv")
Q["rows"] = Q.rows.map(lambda s: [int(x) for x in str(s).split(",")])
in_co = P.in_coconut
qa = Q[Q.set == "B_A"].assign(gset="G_A")
qn = Q[Q.set.isin(["R2", "V"]) & in_co[Q.pid.values]].assign(gset="G_N")
qn = qn.iloc[np.random.default_rng(0).permutation(len(qn))]
order = pd.concat([qa, qn], ignore_index=True)
print("queries:", order.gset.value_counts().to_dict(), "| steps", len(derive.STEPS))
DELTAS = np.array(sorted(derive.all_deltas()))
AP4 = search.COL["ap4"]


def unpack(fp):
    return np.unpackbits(np.atleast_2d(fp), axis=1)[:, :chem.FP_BITS].astype(np.float32)


def zs(x):
    if len(x) == 0:
        return x
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-9 else np.zeros_like(x)


def rank_pess(s, t):
    """1 + number of other entries scoring >= the truth (ties against the truth); s excludes the truth."""
    return 1 + int((s >= t).sum())


def ap4_of(fp_rows, af, w4):
    if len(af) == 0 or len(fp_rows) == 0:
        return np.zeros(len(fp_rows), np.float32)
    inter = fp_rows @ af.T
    tan = inter / (fp_rows.sum(1)[:, None] + af.sum(1)[None, :] - inter + 1e-9)
    return (tan * w4[None, :]).max(1)


# %% [markdown]
# ## クエリごとに親を選び、生成し、並べる

# %%
def run_one(q):
    t0 = time.time()
    rec = dict(qid=q.qid, gset=q.gset, set=q.set, qlib=C.libs[q.qlib])
    cand, X, info = S.features(harness.query_specs(C, q.rows),
                               dict(hide_lib=q.hide_lib, hide_pid=q.hide_pid, hide_pid_lib=q.hide_pid_lib))
    last, target, truth = S.last, info["target"], int(q.pid)
    z = last["z"]
    if len(cand) == 0 or z is None or not (cand == truth).any():
        rec.update(skipped=True, secs=time.time() - t0)
        return rec, []
    tol = target * PPM * 1e-6
    apid, asim = last["apid"], last["asim"]
    af = unpack(P.fp[apid]) if len(apid) else np.zeros((0, chem.FP_BITS), np.float32)
    w4 = np.clip(asim, 0, None) ** SEARCH_CFG.sim_power
    rec["cover"] = derive.n_steps(target - P.mass[apid[0]], tol) if len(apid) else None
    rec["cover"] = -1 if rec["cover"] is None else rec["cover"]

    # 親: (a) 類縁体、(b) プールの Δ ずらしの窓
    parents = []  # (pid, src)
    for j in range(min(len(apid), N_ANALOG_PARENTS)):
        if asim[j] >= MIN_SIM and apid[j] != truth:
            parents.append((int(apid[j]), "a"))
    seen = {p for p, _ in parents}
    pool_c = []
    for d in DELTAS:
        pm = target - d
        if pm < 50:
            continue
        w = P.window(pm, tol / pm * 1e6)
        w = w[(w != truth)]
        if len(w) == 0:
            continue
        s = unpack(P.fp[w]) @ z
        for k in np.argsort(-s)[:N_POOL_PER_DELTA]:
            pool_c.append((float(s[k]), int(w[k])))
    for _, p in sorted(pool_c, reverse=True):
        if len(parents) >= MAX_PARENTS:
            break
        if p not in seen:
            parents.append((p, "b"))
            seen.add(p)
    rec.update(n_par_a=sum(1 for _, s_ in parents if s_ == "a"), n_par_b=sum(1 for _, s_ in parents if s_ == "b"))

    # 生成（質量の合う組み合わせだけ）
    prods = {}  # smiles -> (parent rank, src, parent pid, seq)
    for pr, (p, src) in enumerate(parents):
        seqs = derive.sequences(target - P.mass[p], tol)
        if not seqs:
            continue
        for smi, seq in derive.generate(P.smiles[p], seqs, MAX_PER_PARENT).items():
            if smi not in prods:
                prods[smi] = (pr, src, p, seq)
    prods = {smi: v for smi, v in prods.items() if abs(derive.exact_mass(smi) - target) <= tol}
    t_smi = derive.noniso(P.smiles[truth])
    hit = prods.get(t_smi)
    rec.update(n_prod=len(prods), hit_all=hit is not None)
    hits = []
    if hit is not None:
        hits.append(dict(qid=q.qid, gset=q.gset, src=hit[1], parent=hit[2], seq="+".join(hit[3]), steps=len(hit[3])))

    # 上位 500 件を鍵でそろえ、窓の候補との重複を除く
    ranked = list(prods)[:MAX_RANKED]
    t_key = derive.ik14(P.smiles[truth])
    win_keys = {derive.ik14(P.smiles[c]) for c in cand if c != truth}
    new_smi, new_keys = [], set()
    for smi in ranked:
        k = derive.ik14(smi)
        if k is None or k in win_keys or k in new_keys:
            continue
        new_keys.add(k)
        new_smi.append((smi, k))
    t_in = [i for i, (_, k) in enumerate(new_smi) if k == t_key]
    rec.update(n_ranked=len(ranked), n_new=len(new_smi), hit_cap=bool(t_in))
    if t_in and hit is None:  # SMILES は互変異性で違うが鍵が一致した
        hit = prods[new_smi[t_in[0]][0]]
        rec["hit_all"] = True
        hits.append(dict(qid=q.qid, gset=q.gset, src=hit[1], parent=hit[2], seq="+".join(hit[3]), steps=len(hit[3])))

    # 点数: 窓の候補（正解を除く）∪ 生成物
    keep = cand != truth
    fz_w = unpack(P.fp[cand]) @ z
    ap_w = X[:, AP4]
    if new_smi:
        gf = unpack(np.stack([chem.morgan_packed(s_) for s_, _ in new_smi]))
        fz_g, ap_g = gf @ z, ap4_of(gf, af, w4)
    else:
        fz_g, ap_g = np.zeros(0, np.float32), np.zeros(0, np.float32)
    fz_all = np.r_[fz_w[keep], fz_g]
    ap_all = np.r_[ap_w[keep], ap_g]
    comb = zs(fz_all) + zs(ap_all)
    if t_in:
        i = int(keep.sum()) + t_in[0]
        others = np.ones(len(fz_all), bool)
        others[i] = False
        for nm, sc in [("fz", fz_all), ("ap4", ap_all), ("comb", comb)]:
            rec[f"r_{nm}"] = rank_pess(sc[others], sc[i])
        hits[0]["rank_comb"] = rec["r_comb"]
    else:
        for nm in ("fz", "ap4", "comb"):
            rec[f"r_{nm}"] = np.inf

    # 薄まりの目安: 正解をプールに残したまま、生成物（正解と同じ鍵は除く）を足す前と後
    ti = int(np.flatnonzero(cand == truth)[0])
    g_keep = np.array([k != t_key for _, k in new_smi], bool)
    for nm, sw, sg in [("fz", fz_w, fz_g), ("comb", None, None)]:
        if nm == "fz":
            before, after = sw, np.r_[sw, sg[g_keep]]
        else:
            before = zs(fz_w) + zs(ap_w)
            after = zs(np.r_[fz_w, fz_g[g_keep]]) + zs(np.r_[ap_w, ap_g[g_keep]])
        ob = np.ones(len(before), bool)
        ob[ti] = False
        oa = np.ones(len(after), bool)
        oa[ti] = False
        rec[f"dil_{nm}_0"] = rank_pess(before[ob], before[ti])
        rec[f"dil_{nm}_1"] = rank_pess(after[oa], after[ti])
    rec.update(n_win=len(cand), skipped=False, secs=time.time() - t0)
    return rec, hits


recs, hits = [], []
for i, q in enumerate(order.itertuples()):
    if time.time() - T0 > BUDGET_SECS:
        print(f"time budget reached after {i} queries")
        break
    r_, h_ = run_one(q)
    recs.append(r_)
    hits += h_
    if (i + 1) % 100 == 0:
        d = pd.DataFrame(recs)
        print(f"  {i + 1}/{len(order)} ({time.time() - T0:.0f}s) recall all {d.hit_all.mean():.3f}, "
              f"cap {d.hit_cap.mean():.3f}, MRR comb {harness.rr_of(d.r_comb).mean():.4f}, "
              f"secs/query mean {d.secs.mean():.2f} max {d.secs.max():.1f}, parents b {d.n_par_b.mean():.1f}, "
              f"window median {d.n_win.median():.0f} max {d.n_win.max():.0f}")
R = pd.DataFrame(recs)
R.to_parquet(f"{OUT}/gen_queries.parquet", index=False)
H = pd.DataFrame(hits, columns=["qid", "gset", "src", "parent", "seq", "steps", "rank_comb"])
H.to_parquet(f"{OUT}/gen_hits.parquet", index=False)
print(f"processed {len(R)} queries, skipped {int(R.skipped.sum())} ({time.time() - T0:.0f}s)")

# %% [markdown]
# ## 集計と判定 G（固定）

# %%
R = R[~R.skipped]
rows = []
for g, d in R.groupby("gset"):
    rec = dict(gset=g, n=len(d), recall_all=d.hit_all.mean(), recall_cap=d.hit_cap.mean(),
               n_prod_med=d.n_prod.median(), n_prod_p90=d.n_prod.quantile(0.9), n_new_med=d.n_new.median(),
               n_par_a=d.n_par_a.mean(), n_par_b=d.n_par_b.mean(), secs=d.secs.mean(),
               cover_0=(d.cover == 0).mean(), cover_1=(d.cover == 1).mean(), cover_2=(d.cover == 2).mean(),
               cover_none=(d.cover == -1).mean())
    idx = harness.boot_index(len(d), N_BOOT, seed=1)
    for nm in ("fz", "ap4", "comb"):
        rr = harness.rr_of(d[f"r_{nm}"].values)
        c = harness.ci(rr.mean(), harness.boot_reps(rr, idx))
        rec.update({f"mrr_{nm}": c["est"], f"mrr_{nm}_lo": c["lo"], f"mrr_{nm}_hi": c["hi"]})
    for nm in ("fz", "comb"):
        r0, r1 = harness.rr_of(d[f"dil_{nm}_0"].values), harness.rr_of(d[f"dil_{nm}_1"].values)
        c = harness.paired_ci(r1, r0, N_BOOT, seed=2)
        rec.update({f"dil_{nm}_before": r0.mean(), f"dil_{nm}_after": r1.mean(), f"dil_{nm}_d_lo": c["lo"],
                    f"dil_{nm}_d_hi": c["hi"]})
    rows.append(rec)
summ = pd.DataFrame(rows)
summ.to_csv(f"{OUT}/gen_probe.csv", index=False)
print(summ.round(4).T.to_string())

hh = H.merge(R[["qid", "gset", "hit_cap"]], on=["qid", "gset"], how="left")
by_rx = []
for g, d in hh.groupby("gset"):
    n_q = int((R.gset == g).sum())
    for seq, e in d.groupby("seq"):
        by_rx.append(dict(gset=g, seq=seq, hits=len(e), share=len(e) / n_q, in_cap=int(e.hit_cap.sum()),
                          src_a=int((e.src == "a").sum()), src_b=int((e.src == "b").sum()),
                          top25=int((e.rank_comb <= 25).sum())))
by_rx = pd.DataFrame(by_rx, columns=["gset", "seq", "hits", "share", "in_cap", "src_a", "src_b", "top25"])
by_rx = by_rx.sort_values(["gset", "hits"], ascending=[True, False])
by_rx.to_csv(f"{OUT}/gen_by_reaction.csv", index=False)
print("hits by reaction sequence:")
print(by_rx.head(40).round(4).to_string(index=False))

sm = summ.set_index("gset")
passed = {g: bool(sm.loc[g, "mrr_comb"] >= GATE_MRR and sm.loc[g, "recall_cap"] >= GATE_RECALL) for g in sm.index}
gate = any(passed.values())
res = dict(gate=gate, passed=passed, n_processed=len(R), budget_hit=len(R) < len(order),
           summary=sm.to_dict("index"))
json.dump(res, open(f"{OUT}/probe2.json", "w"), indent=1, default=float)
print(f"gate G: {'PASS' if gate else 'FAIL'} {passed} — comb MRR@25 >= {GATE_MRR} and recall (after the "
      f"{MAX_RANKED} cap) >= {GATE_RECALL} on G_N or G_A")
run = track.init("probe", group=EXP, config=dict(ppm=PPM, max_parents=MAX_PARENTS, max_ranked=MAX_RANKED,
                                                 n_steps=len(derive.STEPS)), online=False)
run.summary(dict(gate=gate, **{f"{g}_{k}": float(v) for g, r_ in sm.to_dict("index").items() for k, v in r_.items()
                               if isinstance(v, (int, float))}))
run.table("gen_probe", summ)
run.finish()
print(f"total {time.time() - T0:.0f}s")

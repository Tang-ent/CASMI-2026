# %% [markdown]
# # casmi26-k10-pubchem-probe
# M3 の P0: PubChem を候補プールに足す前の調査（CPU、インターネットあり）。
# 1. 分子式と質量だけの安い絞り込み（成分 1 つ、同位体標識なし、C/H/N/O/P/S/ハロゲン、150〜1,200 Da）で残る数
# 2. 標準 InChIKey14 で立体異性体をまとめ、今のプール（train ∪ COCONUT）と照らし合わせる
# 3. 注釈（文献数・名前の数・特許数・PubChemLite）と、自前の天然物らしさ分類器（COCONUT にだけある構造 vs
#    無作為な PubChem、2 分割の交差当てはめ）
# 4. chem.canonical の速さ（P1 で処理できる数の見積もり）
# 5. 絞り方の候補ごとに、増える構造の数、代理の正解集合（Q1: COCONUT にだけある構造、Q2: COCONUT に無い
#    天然物寄りライブラリの train 構造、Q3: enveda-180、参考 A）の再現率、A の窓での候補数
# 6. 決め方（結果を見る前に固定）: 上限 S 150 万 / M 500 万 / L 1,500 万。M の上限の中で Q1 と Q2 の再現率の平均が
#    最大の系統を選び（同じなら A の窓の候補数が少ない方）、S と L は同じ系統でしきい値だけを変える
# 出力: CSV / JSON の表、PubChem 側の表（新しい構造）、今のプールの構造ごとの表、分類器 2 つ。SMILES は保存しない。

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
import os
import shutil
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from casmi26 import paths, pool, pubchem

T0 = time.time()
OUT = "/kaggle/working"
W = os.cpu_count() or 4
SEED = 0
N_SAMPLE = 3_000_000  # 新しい PubChem 構造の無作為標本（分類器のスコアを付け、数と窓を見積もる）
N_POS = 300_000  # 分類器の正例の上限（負例も同じ数）
N_SPEED = 20_000  # chem.canonical の速さを測る数
BUDGETS = {"S": 1_500_000, "M": 5_000_000, "L": 15_000_000}
PPM = 10.0
NP_LIBS = ["riken", "gnps", "massbank", "mona", "msdial", "spectraverse", "pluskal_ms2", "masaryk"]
rng = np.random.default_rng(SEED)


def log(*a):
    print(*a, f"[{time.time() - T0:.0f}s]", flush=True)


DISKS = [d for d in ["/kaggle/tmp", "/tmp", "/kaggle/working"] if os.path.isdir(d)]
for d in DISKS:
    u = shutil.disk_usage(d)
    print(f"disk {d}: free {u.free / 1e9:.1f} GB of {u.total / 1e9:.1f} GB")
print("cpus", W)
# ダウンロード先は空きが最も大きいディスク。各ファイルは読み終えたら消すので、/kaggle/working でも出力には残らない
TMP = os.path.join(max(DISKS, key=lambda d: shutil.disk_usage(d).free), "pubchem_tmp")
os.makedirs(TMP, exist_ok=True)


def get(name):
    return pubchem.source(name, TMP, log=log)


def release(p, downloaded):
    if downloaded:
        os.remove(p)


# %% [markdown]
# ## 今のプールと代理の正解集合

# %%
PD = pool.Pool.find_dir()
P = pool.Pool(PD)
pp = pd.read_parquet(os.path.join(PD, "pool.parquet"), columns=["key", "smiles", "in_train", "in_coconut"])
n_pool = len(pp)
Q = pd.read_csv(os.path.join(PD, "queries.csv"), usecols=["set", "pid"])
CD = os.path.dirname(paths.find("cache_meta.json"))
libs = json.load(open(os.path.join(CD, "cache_meta.json")))["libs"]
z = np.load(os.path.join(CD, "spec_meta.npz"))
spec_sid, spec_lib = z["sid"], z["lib"]
struct_ik = pd.read_parquet(os.path.join(CD, "struct.parquet"), columns=["ik14"]).ik14.values


def pids_with_lib(names):
    ids = [libs.index(n) for n in names if n in libs]
    p = P.sid2pid[np.unique(spec_sid[np.isin(spec_lib, ids)])]
    m = np.zeros(n_pool, bool)
    m[p[p >= 0]] = True
    return m


in_tr, in_co = pp.in_train.values, pp.in_coconut.values
SETS = {"Q1": in_co & ~in_tr, "Q2": in_tr & ~in_co & pids_with_lib(NP_LIBS), "Q3": in_tr & pids_with_lib(["enveda-180"]),
        "A": np.isin(np.arange(n_pool), Q[Q.set == "A"].pid.values)}
print({k: int(v.sum()) for k, v in SETS.items()})

# 今のプールの標準 InChIKey14: プールの SMILES（互変異性体正規化後）から RDKit で計算したものと、train の inchikey14
std_key = np.array(pubchem.map_many(pubchem.std_ik14, pp.smiles.tolist(), W, chunksize=2000), dtype="S14")
comp_key = np.zeros(n_pool, "S14")
ok = P.sid2pid >= 0
comp_key[P.sid2pid[ok]] = struct_ik[ok].astype("S14")
pool_keys = np.unique(np.concatenate([std_key, comp_key, pp.key.values.astype("S14")]))
pool_keys = pool_keys[pool_keys != b""]
log(f"pool keys: {len(pool_keys):,} (std == train ik14 for {(std_key[comp_key != b''] == comp_key[comp_key != b'']).mean():.3f} of train pids)")

# %% [markdown]
# ## 1. CID-Mass: 分子式と質量の絞り込み

# %%
p, dl = get("CID-Mass.gz")
M, n_rows = pubchem.scan_mass(p, log)
release(p, dl)
N = len(M["status"])
st = M["status"]

# %% [markdown]
# ## 2. CID-InChI-Key: 成分が複数のもの（塩）を除き、立体異性体をまとめ、今のプールと照らし合わせる

# %%
p, dl = get("CID-InChI-Key.gz")
IK, multi = pubchem.scan_ik14(p, N, log)
release(p, dl)
st[(st != 255) & multi] = 1  # PubChem の分子式は塩も 1 つの式で書くので、InChI の分子式の層で判定する
del multi
stages = [("rows", n_rows)] + [(pubchem.STATUS[k], int((st == k).sum())) for k in (1, 2, 3, 4, 5, 0)]
okm = st == 0
b = M["bits"][okm]
stages += [("ok_charged", int(((b & pubchem.CHARGED) > 0).sum()))]
for name, mask in [("CHO", 0), ("+N", 1), ("+S", 2), ("+P", 4), ("+halogen", 8)]:
    stages.append((f"ok_{name}", int(((b & 15) == 0).sum()) if mask == 0 else int(((b & mask) > 0).sum())))
del b
log("formula filter:", stages)
pd.DataFrame(stages, columns=["stage", "n"]).to_csv(f"{OUT}/stage_counts.csv", index=False)  # 途中で止まっても残す
ik_all = np.unique(IK[IK != b""])
base = np.flatnonzero(okm & (IK != b""))
base = base[np.argsort((M["bits"][base] & pubchem.CHARGED) > 0, kind="stable")]  # 電荷の無い CID を代表にする
base = base[np.argsort(IK[base], kind="stable")]
keys = IK[base]
first = np.r_[True, keys[1:] != keys[:-1]]
gstart = np.flatnonzero(first)
g_key, g_cid = keys[gstart], base[gstart]
g_ncid = np.diff(np.r_[gstart, len(base)]).astype(np.int32)
g_mass = M["mass"][g_cid]
g_bits = M["bits"][g_cid]
g_new = ~np.isin(g_key, pool_keys)
n_new = int(g_new.sum())
del keys, first
stages += [("cids_ok_with_key", int(len(base))), ("groups_ik14", int(len(g_key))), ("groups_new", n_new)]
log(f"groups {len(g_key):,}, new (not in pool) {n_new:,}")
pd.DataFrame(stages, columns=["stage", "n"]).to_csv(f"{OUT}/stage_counts.csv", index=False)


def to_group(k):
    i = np.minimum(np.searchsorted(g_key, k), len(g_key) - 1)
    return np.where((k != b"") & (g_key[i] == k), i, -1)


pg_comp, pg_std = to_group(comp_key), to_group(std_key)
pool_g = np.where(pg_comp >= 0, pg_comp, pg_std)  # train の inchikey14 を優先
found = np.isin(comp_key, ik_all) | np.isin(std_key, ik_all)
proxy = pd.DataFrame([dict(set=s, n=int(m.sum()), found_in_pubchem=float(found[m].mean()),
                           base_pass=float((pool_g[m] >= 0).mean())) for s, m in SETS.items()]
                     + [dict(set="pool_train", n=int(in_tr.sum()), found_in_pubchem=float(found[in_tr].mean()),
                             base_pass=float((pool_g[in_tr] >= 0).mean()))])
print(proxy.round(4).to_string(index=False))
proxy.to_csv(f"{OUT}/proxy_sets.csv", index=False)
del ik_all

# %% [markdown]
# ## 3. 注釈: 文献数、名前の数、特許数（CID ごとに数え、まとめた構造の中の最大値）、PubChemLite

# %%
ANN = {}
for name, ncols, tag in [("CID-PMID.gz", 3, "n_pmid"), ("CID-Synonym-filtered.gz", 2, "n_syn"),
                         ("CID-Patent.gz", 2, "n_pat")]:
    p, dl = get(name)
    cnt = pubchem.count_by_cid(p, N, ncols, tag, log)
    release(p, dl)
    ANN[tag] = np.maximum.reduceat(cnt[base], gstart).astype(np.int32)
    del cnt

local_pcl = sorted(glob.glob("/kaggle/input/**/PubChemLite*.csv", recursive=True))
if local_pcl:
    p, dl = local_pcl[0], False
else:
    fname, url = pubchem.pubchemlite_url()
    p, dl = pubchem.fetch(url, os.path.join(TMP, fname), log=log), True
pcl = pubchem.read_pubchemlite(p)
release(p, dl)
gi = to_group(pcl.ik14.values.astype("S14"))
ANN["pcl_anno"] = np.full(len(g_key), -1, np.int16)
ANN["pcl_anno"][gi[gi >= 0]] = pcl.pcl_anno.values[gi >= 0]
log(f"PubChemLite: {len(pcl):,} keys, matched to groups {(gi >= 0).mean():.3f}")
del pcl, gi, base, gstart

# %% [markdown]
# ## 4. SMILES（標本と、今のプールに対応する構造の分だけ）

# %%
new_idx = np.flatnonzero(g_new)
sample = rng.choice(new_idx, size=min(N_SAMPLE, len(new_idx)), replace=False)
want = np.zeros(N, bool)
want[g_cid[sample]] = True
want[g_cid[pool_g[pool_g >= 0]]] = True
p, dl = get("CID-SMILES.gz")
s_cid, s_smi = pubchem.scan_smiles(p, want, log)
release(p, dl)
del want, M, IK
smiles_of = pd.Series(s_smi, index=s_cid)
smiles_of = smiles_of[~smiles_of.index.duplicated()]


def smiles_g(g):
    return smiles_of.reindex(g_cid[g]).values


# %% [markdown]
# ## 5. chem.canonical の速さ

# %%
sp_g = sample[:N_SPEED]
sp = smiles_g(sp_g)
okp = np.array([isinstance(s, str) for s in sp])
t = time.time()
res = pool._canon_many(list(sp[okp]), W, log, "speed")
secs = time.time() - t
good = [r for r in res if r is not None]
kchg = np.mean([r["key"].encode() != k for r, k in zip(res, g_key[sp_g[okp]]) if r is not None])
tan = pool._tanimoto_packed(np.stack([r["fp_raw"] for r in good]), np.stack([r["fp"] for r in good]))
speed = dict(n=int(okp.sum()), workers=W, secs=round(secs, 1), per_sec=round(float(okp.sum() / secs), 1),
             failed=int(sum(r is None for r in res)), key_changed=float(kchg), fp_changed=float((tan < 0.999).mean()),
             mean_mass=float(g_mass[sp_g].mean()),
             per_12h_one_kernel=int(okp.sum() / secs * 12 * 3600))
log("canonical speed:", speed)
json.dump(speed, open(f"{OUT}/canonical_speed.json", "w"), indent=1)
del res, good

# %% [markdown]
# ## 6. 天然物らしさ分類器（COCONUT にだけある構造 vs プールに無い無作為な PubChem、2 分割の交差当てはめ）

# %%
pos_g = np.unique(pool_g[SETS["Q1"] & (pool_g >= 0)])
pos_g = rng.permutation(pos_g)[:N_POS]
neg_g = sample[:len(pos_g)]
Xp, okp_ = pubchem.np_matrix(smiles_g(pos_g), W)
Xn, okn_ = pubchem.np_matrix(smiles_g(neg_g), W)
tr_g = np.r_[pos_g[okp_], neg_g[okn_]]
X = np.vstack([Xp, Xn])
y = np.r_[np.ones(len(Xp)), np.zeros(len(Xn))]
del Xp, Xn
fold = rng.integers(0, 2, len(y))
models = [pubchem.np_fit(X[fold == f], y[fold == f], seed=f) for f in (0, 1)]
oof = np.zeros(len(y))
for f in (0, 1):
    oof[fold != f] = models[f].predict(X[fold != f], num_threads=W)
del X
np_auc = pubchem.auc(y, oof)
log(f"NP classifier: {int(y.sum()):,} positives, {int((1 - y).sum()):,} negatives, out-of-fold AUC {np_auc:.4f}")
for f, mdl in enumerate(models):
    mdl.save_model(f"{OUT}/np_model_{f}.txt")

g_np = np.full(len(g_key), np.nan, np.float32)
todo = np.setdiff1d(np.unique(np.r_[sample, pool_g[pool_g >= 0]]), tr_g)
g_np[todo] = pubchem.np_score(models, smiles_g(todo), W)
g_np[tr_g] = oof  # 学習に使った構造は反対側のモデルの値
log(f"NP scores: {np.isfinite(g_np).sum():,} groups")

# %% [markdown]
# ## 7. 絞り方の候補ごとの表

# %%
G = pd.DataFrame({"n_syn": ANN["n_syn"], "n_pmid": ANN["n_pmid"], "n_pat": ANN["n_pat"],
                  "in_pcl": ANN["pcl_anno"] >= 0, "np": g_np})
FAMILIES = {
    "syn": (lambda d: d.n_syn.values.astype(float), True),
    "pmid": (lambda d: d.n_pmid.values.astype(float), True),
    "patent": (lambda d: d.n_pat.values.astype(float), True),
    "np": (lambda d: d["np"].values.astype(float), False),
    "np|pcl": (lambda d: np.where(d.in_pcl.values, np.inf, d["np"].values), False),
    "np|pmid": (lambda d: np.where(d.n_pmid.values >= 1, np.inf, d["np"].values), False),
}  # (score, exact): exact = 全部の新しい構造で計算できる。そうでなければ標本から引き伸ばす


def threshold(v, u, scale, budget):
    """Loosest t such that scale * #(s >= t) <= budget, given v = the non-NaN scores sorted in descending order and
    u = their unique values (NaN scores never pass). -inf: everything fits; NaN: infeasible (the top value alone
    is over the budget)."""
    if scale * len(v) <= budget:
        return -np.inf
    k = int(budget // scale)
    j = np.searchsorted(u, v[k], side="right")
    return u[j] if j < len(u) else np.nan


a_mass = P.mass[SETS["A"]]
pool_cnt = np.array([len(P.window(m, PPM)) for m in a_mass])


def window_counts(masses, scale):
    ms = np.sort(masses)
    tol = a_mass * PPM * 1e-6
    return scale * (np.searchsorted(ms, a_mass + tol) - np.searchsorted(ms, a_mass - tol))


rows = []
for fam, (fn, exact) in FAMILIES.items():
    pop = new_idx if exact else sample
    scale = 1.0 if exact else n_new / len(sample)
    s_all = fn(G)
    v = s_all[pop]
    v = np.sort(v[~np.isnan(v)])[::-1]
    u = np.unique(v)
    for lvl, budget in list(BUDGETS.items()) + [("all", np.inf)]:
        t = threshold(v, u, scale, budget)
        row = dict(family=fam, level=lvl, budget=budget, threshold=t, exact=exact, feasible=bool(not np.isnan(t)))
        passed = (s_all >= t) if row["feasible"] else np.zeros(len(s_all), bool)
        row["n_new"] = float(scale * passed[pop].sum())
        for sname, m in SETS.items():
            g = pool_g[m]
            row[f"recall_{sname}"] = float(np.where(g >= 0, passed[np.maximum(g, 0)], False).mean())
        row["recall_Q1Q2"] = (row["recall_Q1"] + row["recall_Q2"]) / 2
        wc = window_counts(g_mass[pop[passed[pop]]], scale)
        row.update(cand_med=float(np.median(wc)), cand_p90=float(np.quantile(wc, 0.9)),
                   total_med=float(np.median(pool_cnt + wc)), total_p90=float(np.quantile(pool_cnt + wc, 0.9)))
        rows.append(row)
front = pd.DataFrame(rows)
print(f"current pool, A windows (+-{PPM:g} ppm): median {np.median(pool_cnt):.0f}, p90 {np.quantile(pool_cnt, 0.9):.0f}")
print(front.round(4).to_string(index=False))

# %% [markdown]
# ## 決め方（冒頭のとおり、結果を見る前に固定）

# %%
fm = front[(front.level == "M") & front.feasible]
best = fm.sort_values(["recall_Q1Q2", "cand_med"], ascending=[False, True]).iloc[0]
fam = best.family
lv = front[front.family == fam].set_index("level")
levels = dict(family=fam, rule="max mean(recall Q1, Q2) at M; S and L = same family, loosest threshold within budget",
              thresholds={k: (None if not lv.loc[k, "feasible"] else float(lv.loc[k, "threshold"])) for k in BUDGETS},
              n_new={k: float(lv.loc[k, "n_new"]) for k in BUDGETS},
              recall={k: {s: float(lv.loc[k, f"recall_{s}"]) for s in SETS} for k in BUDGETS},
              cand_med={k: float(lv.loc[k, "cand_med"]) for k in BUDGETS}, exact=bool(best.exact))
print(json.dumps(levels, indent=1))

# %% [markdown]
# ## 出力

# %%
fn_best = FAMILIES[fam][0](G)
pass_lvl = {k: (fn_best >= levels["thresholds"][k]) if levels["thresholds"][k] is not None else np.zeros(len(G), bool)
            for k in BUDGETS}
front.to_csv(f"{OUT}/frontier.csv", index=False)
json.dump(levels, open(f"{OUT}/levels.json", "w"), indent=1)
q = [0.1, 0.25, 0.5, 0.75, 0.9]
np_eval = dict(auc_oof=np_auc, n_pos=int(y.sum()), n_neg=int((1 - y).sum()),
               quantiles={s: [float(x) for x in np.nanquantile(g_np[pool_g[m & (pool_g >= 0)]], q)]
                          for s, m in SETS.items()}
               | {"new_sample": [float(x) for x in np.nanquantile(g_np[sample], q)]})
json.dump(np_eval, open(f"{OUT}/np_eval.json", "w"), indent=1)
print(json.dumps(np_eval, indent=1))

gp = np.maximum(pool_g, 0)
pool_tab = pd.DataFrame({"pid": np.arange(n_pool, dtype=np.int32), "found": found, "base_pass": pool_g >= 0,
                         "cid": np.where(pool_g >= 0, g_cid[gp], -1).astype(np.int64)}
                        | {c: np.where(pool_g >= 0, G[c].values[gp], -1).astype(np.int32)
                           for c in ["n_pmid", "n_syn", "n_pat"]}
                        | {"pcl_anno": np.where(pool_g >= 0, ANN["pcl_anno"][gp], -1).astype(np.int16),
                           "np": np.where(pool_g >= 0, g_np[gp], np.nan).astype(np.float32)}
                        | {f"pass_{k}": (pool_g >= 0) & v[gp] for k, v in pass_lvl.items()}
                        | {f"set_{k}": v for k, v in SETS.items()})
pool_tab.to_parquet(f"{OUT}/pool_pubchem.parquet", index=False)

ni = new_idx
in_sample = np.zeros(len(g_key), bool)
in_sample[sample] = True
tab = pa.table({"cid": pa.array(g_cid[ni].astype(np.int64)),
                "ik14": pa.FixedSizeBinaryArray.from_buffers(pa.binary(14), len(ni), [None, pa.py_buffer(g_key[ni].tobytes())]),
                "mass": pa.array(g_mass[ni]), "bits": pa.array(g_bits[ni]), "n_cids": pa.array(g_ncid[ni]),
                "n_pmid": pa.array(ANN["n_pmid"][ni]), "n_syn": pa.array(ANN["n_syn"][ni]),
                "n_pat": pa.array(ANN["n_pat"][ni]), "pcl_anno": pa.array(ANN["pcl_anno"][ni]),
                "np": pa.array(g_np[ni]), "in_sample": pa.array(in_sample[ni])}
               | {f"pass_{k}": pa.array(v[ni]) for k, v in pass_lvl.items()})
pq.write_table(tab, f"{OUT}/pubchem_new.parquet", row_group_size=2_000_000)
for fn in sorted(os.listdir(OUT)):
    if os.path.isfile(f"{OUT}/{fn}"):
        print(f"{fn:24s} {os.path.getsize(f'{OUT}/{fn}') / 1e6:10.1f} MB")
log("done")

# %% [markdown]
# # casmi26-k26-fp2
# M6 の前処理（2026-10-12、CPU）: プールの全構造（train ∪ COCONUT、72.5 万）について、
# - 中性の分子式（k27 の分子式つきエンコーダと k33 のデコーダが使う）。train にある構造は train の分子式、それ以外は
#   pool.parquet の分子式で、電荷のある式は RDKit の Uncharger で中和する（k05b の質量の修正と同じ考え方）。
#   質量の修正後の P.mass と式の質量が合うかを確かめて数を出す
# - 豊かな指紋（k28 の目標）: Morgan r2 2048（P.fp をそのまま先頭に）、FCFP4、AtomPair、Topological Torsion、
#   RDKit path（maxPath 5）各 2048、MACCS 167、計 10,407 ビット（chem.RICH_FP_BLOCKS）
# - 部分分子式の割り当ての割合（報告だけ）: train の保持スペクトルから 20,000 件を引き、上位 30 ピークの何割に式が付くか
#   を、許容 max(10 ppm, 2 mDa)（k27 で使う）と 10 ppm だけで、ライブラリ別に出す。1 スペクトルの時間も測る
# 出力: pool_formula.npz（fid、counts、supported、keys、mass）、pool_fp2.npy（packed、n × 1301）、k26_meta.json、
# annot_rate.csv。入力: k01、k05、k05b、casmi26-wheels（rdkit。k22 / k24 と同じ版）。

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
import json
import re
import time

import numpy as np
import pandas as pd

from casmi26 import cache, chem, fpnet, pool, subform

T0 = time.time()
OUT = "/kaggle/working"
WORKERS = 4
N_ANNOT = 20_000
TOP_ANNOT = 30
C = cache.Cache(cache.Cache.find_dir())
PD = pool.Pool.find_dir()
P = pool.Pool(PD, pubchem=False)
pp = pd.read_parquet(f"{PD}/pool.parquet", columns=["formula"])
assert len(pp) == len(P)

# %% [markdown]
# ## 中性の分子式

# %%
fstr = pp.formula.values.astype(object).copy()
src = np.full(len(P), "pool", dtype=object)
ok = P.sid2pid >= 0
tf = pd.Series(C.struct.formula.values[np.flatnonzero(ok)]).groupby(P.sid2pid[ok]).first()
fstr[tf.index.values] = tf.values
src[tf.index.values] = "train"
CHG = re.compile(r"[+-]\d*$")
charged = np.array([isinstance(f, str) and bool(CHG.search(f.strip())) for f in fstr]) & (src != "train")


def unch_formula(smiles):
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if m is None:
        return None
    try:
        return CalcMolFormula(rdMolStandardize.Uncharger().uncharge(m))
    except Exception:  # noqa: BLE001
        return None


ix = np.flatnonzero(charged)
uf = pool.map_many(unch_formula, P.smiles[ix].tolist(), WORKERS) if len(ix) else []
n_unch = 0
for i, f in zip(ix, uf):
    if f is not None:
        fstr[i] = f
        src[i] = "uncharged"
        n_unch += 1
counts = np.zeros((len(P), subform.NE), np.int64)
sup = np.zeros(len(P), bool)
keys = np.empty(len(P), dtype=object)
fmass = np.full(len(P), np.nan)
for i, f in enumerate(fstr):
    v, s = subform.parse(f)
    counts[i], sup[i] = v, s
    keys[i] = subform.key_of(v) if s else "X:" + str(f).strip()
    fmass[i] = float(v @ subform.EMASS) if s else chem.formula_mass(CHG.sub("", str(f).strip()))
still_charged = np.array([isinstance(f, str) and bool(CHG.search(f.strip())) for f in fstr])
dm = np.abs(fmass - P.mass)
uk, fid = np.unique(keys.astype(str), return_inverse=True)
fid = fid.astype(np.int32)
fcounts = np.zeros((len(uk), subform.NE), np.int16)
fcounts[fid] = counts
fsup = np.zeros(len(uk), bool)
fsup[fid] = sup
fm = np.zeros(len(uk))
fm[fid] = fmass
np.savez(f"{OUT}/pool_formula.npz", fid=fid, counts=fcounts, supported=fsup, keys=uk.astype(str), mass=fm)
meta = dict(n=len(P), n_formula=len(uk), from_train=int((src == "train").sum()), charged_not_train=int(len(ix)),
            uncharged=n_unch, still_charged=int(still_charged.sum()), unsupported=int((~sup).sum()),
            mass_mismatch_1e3=int((dm > 1e-3).sum()), mass_mismatch_nan=int((~np.isfinite(dm)).sum()),
            formula_secs=round(time.time() - T0))
print("formulas:", meta, flush=True)
bad = np.flatnonzero(dm > 1e-3)[:10]
for i in bad:
    print("  mass mismatch", i, fstr[i], src[i], round(fmass[i], 5), round(P.mass[i], 5), P.smiles[i][:80])
# 窓の中の同じ分子式の候補の数（エンコーダの負例と、k27 の同じ分子式の評価の大きさの目安）
gsize = np.bincount(fid)
meta["same_formula_group_median"] = float(np.median(gsize[fid]))
meta["same_formula_group_p90"] = float(np.percentile(gsize[fid], 90))
meta["singleton_share"] = float((gsize[fid] == 1).mean())

# %% [markdown]
# ## 豊かな指紋

# %%
t1 = time.time()


def tail_packed(s):
    return np.packbits(chem.rich_fp_tail(s))


tails = pool.map_many(tail_packed, P.smiles.tolist(), WORKERS, chunksize=2000)
n_head = chem.RICH_FP_BLOCKS[0][1]
n_tail = chem.RICH_FP_BITS - n_head
fp2 = np.zeros((len(P), (chem.RICH_FP_BITS + 7) // 8), np.uint8)
empty_tail = 0
for a in range(0, len(P), 50_000):
    b = min(a + 50_000, len(P))
    head = np.unpackbits(P.fp[a:b], axis=1)[:, :n_head]
    tl = np.unpackbits(np.stack(tails[a:b]), axis=1)[:, :n_tail]
    empty_tail += int((tl.sum(1) == 0).sum())
    fp2[a:b] = np.packbits(np.concatenate([head, tl], 1), axis=1)
del tails
assert (np.unpackbits(fp2[:1000], axis=1)[:, :n_head] == np.unpackbits(P.fp[:1000], axis=1)[:, :n_head]).all()
np.save(f"{OUT}/pool_fp2.npy", fp2)
bits = np.unpackbits(fp2[np.random.default_rng(0).choice(len(P), min(20_000, len(P)), replace=False)], axis=1)[:, :chem.RICH_FP_BITS]
dens, o = {}, 0
for name, nb in chem.RICH_FP_BLOCKS:
    dens[name] = round(float(bits[:, o:o + nb].mean()), 4)
    o += nb
meta.update(rich_bits=chem.RICH_FP_BITS, rich_blocks=chem.RICH_FP_BLOCKS, empty_tail=empty_tail, block_density=dens,
            fp_secs=round(time.time() - t1), rdkit=rdkit.__version__)
print("rich fingerprints:", {k: meta[k] for k in ["rich_bits", "empty_tail", "block_density", "fp_secs"]}, flush=True)

# %% [markdown]
# ## 部分分子式の割り当ての割合（報告だけ）

# %%
t2 = time.time()
D = fpnet.TrainData(C, P, set())
rng = np.random.default_rng(0)
pick = rng.choice(len(D), min(N_ANNOT, len(D)), replace=False)
rows = []
for j in pick:
    r, p = D.rows[j], D.pid[j]
    ad = C.adducts[C.adduct[r]]
    v, s = counts[p], sup[p]
    ion, sign = subform.ion_counts(v, ad) if s else (None, 0)
    mz, it = fpnet.peak_tokens(*C.spectrum(r), TOP_ANNOT)
    t = time.time()
    _, _, ok2 = subform.annotate(mz, ion, sign, subform.PPM, subform.ABS_DA)
    dt = time.time() - t
    _, _, ok1 = subform.annotate(mz, ion, sign, subform.PPM, 0.0)
    w = it / it.sum() if len(it) and it.sum() > 0 else np.zeros(len(it))
    rows.append(dict(lib=C.libs[C.lib[r]], adduct=ad, annotated=ion is not None,
                     combos=subform.n_combos(ion) if ion is not None else 0,
                     share_floor=float(ok2.mean()) if len(ok2) else 0.0, share_ppm=float(ok1.mean()) if len(ok1) else 0.0,
                     wshare_floor=float((w * ok2).sum()), wshare_ppm=float((w * ok1).sum()), secs=dt))
an = pd.DataFrame(rows)
tab = an.groupby("lib").agg(n=("lib", "size"), annotated=("annotated", "mean"), share_floor=("share_floor", "mean"),
                            share_ppm=("share_ppm", "mean"), wshare_floor=("wshare_floor", "mean"),
                            wshare_ppm=("wshare_ppm", "mean"), ms=("secs", lambda x: 1000 * x.mean()))
tab.loc["all"] = [len(an), an.annotated.mean(), an.share_floor.mean(), an.share_ppm.mean(), an.wshare_floor.mean(),
                  an.wshare_ppm.mean(), 1000 * an.secs.mean()]
tab.to_csv(f"{OUT}/annot_rate.csv")
print(tab.round(3).to_string())
meta.update(annot_n=len(an), annot_ms_mean=float(1000 * an.secs.mean()), annot_ms_p99=float(1000 * an.secs.quantile(0.99)),
            annot_over_max_combos=float((an.combos > subform.MAX_COMBOS).mean()), annot_secs=round(time.time() - t2))
json.dump(meta, open(f"{OUT}/k26_meta.json", "w"), indent=1, default=str)
print(f"done ({time.time() - T0:.0f}s)")

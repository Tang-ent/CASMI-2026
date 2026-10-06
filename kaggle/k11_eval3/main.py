# %% [markdown]
# # casmi26-k11-eval3
# M3 の P2: PubChem 部分（段階 S′ = 文献数 ≥ 1）をつないだプールで計算した k07 v5 の特徴量を評価し、提出に使う
# スコアを選ぶ。k08_eval2 v5（M2.2）を写して変えたもの。M2.2 の提出（k09 v4）の入力である k08 を上書きしない
# ために、別のカーネルにした。
#
# PubChem にしか正解が無い評価セット（k07 の変更は要らない）: 正解のスペクトルをすべて隠したセットで、正解の構造が
# S′ を通るとき（k10 の pool_pubchem.parquet で base_pass かつ n_pmid ≥ 1）だけ正解として数え、通らなければ 0 点。
# 正解が残るときの候補の並びは元のセットと同じなので、元のセットの順位をそのまま使う。
#   B_PR = R2 のうち正解が COCONUT に無いクエリ（PubChem で足される正解に一番近い。選択に使う）
#   B_PA = B_A（正解がよく調べられた天然物で S′ を通る割合が甘い。表示のみ）、B_PC = B_C（表示のみ）
#
# 採用基準（結果を見る前に固定。2026-10-06 ユーザー確認）
# 1. S′ の特徴量で lgb_R を学習し直し、hybm:τ:w の 20 個（τ ∈ {0.5, 0.6, 0.7, 0.8} × w ∈ {0.5, 1, 2, 5, 10}）のうち、
#    A ≥ クラス1ベースラインの A − 1SE（0.9338 − 0.0127）を満たし、0.2·MRR(R1) + 0.8·MRR(R2) が最大のもの。
#    満たすものが無ければ S′ は不採用（提出ファイルは作らない）
# 2. 「PubChem なし」（M2.2 の hybm:0.7:2。B_P 系は 0）と 1 段目の勝者を J = 0.16·A + 0.18·B_A + 0.18·B_PR で比べる。
#    0.16 / 0.18 は M1 の LB から見積もったテストの内訳（クラス1 約16%、COCONUT に正解 約18%）、PubChem にしか正解が
#    無い分子も 18% と仮定する。2 段目で「PubChem なし」が勝っても S′ の提出ファイルは作り（LB で確かめるため）、
#    2 段目の結果は推奨として報告する
# 結果は wandb にオフラインで記録し、完了後に手元で tools/wandb_sync.py で送る。

# %%
import glob
import json
import time

import numpy as np
import pandas as pd

from casmi26 import harness, paths, pool, rank, search, track

T0 = time.time()
OUT = "/kaggle/working"
EXP = "M3"  # wandb の group
A_REF, A_REF_SE = 0.9338, 0.0127  # クラス1ベースライン（k02, train のみのプール）
A_MIN = A_REF - A_REF_SE
# 「PubChem なし」= M2.2 の提出（hybm:0.7:2）。出どころ: k08_eval2 v5 の results.csv（wandb run 6hp51r3j）
NONE = dict(name="none:M2.2 hybm:0.7:2", A=0.921450937950938, B_A=0.5188590473819267, C=0.8827297369297369,
            B_C=0.7191848461903257, R1=0.896693941487076, R2=0.6897606894715094, B_PR=0.0, B_PA=0.0, B_PC=0.0)
J_W = dict(A=0.16, B_A=0.18, B_PR=0.18)
SETS = ["A", "B_A", "C", "B_C", "R1", "R2"]
P_SETS = {"B_PR": "R2", "B_PA": "B_A", "B_PC": "B_C"}  # PubChem にしか正解が無いセット -> 順位を借りる元のセット
SEEDS = (0, 1, 2)
TAUS = (0.5, 0.6, 0.7, 0.8, 0.9)  # hyb:τ（参照用）
BLEND_TAUS = (0.5, 0.6, 0.7, 0.8)  # hybm:τ:w（選択の候補）
BLEND_WS = (0.5, 1, 2, 5, 10)

infos = pd.concat([pd.read_csv(p) for p in sorted(glob.glob("/kaggle/input/**/query_info_s*.csv", recursive=True))],
                  ignore_index=True).drop_duplicates("qid")
feats = {}
for s in SETS:
    paths_ = sorted(glob.glob(f"/kaggle/input/**/feats_{s}_s*.parquet", recursive=True))
    if paths_:
        feats[s] = pd.concat([pd.read_parquet(p) for p in paths_], ignore_index=True).drop_duplicates(["qid", "pid"])
infos.to_csv(f"{OUT}/query_info.csv", index=False)
PD = pool.Pool.find_dir()
truth = pd.read_csv(f"{PD}/queries.csv", usecols=["qid", "pid"]).set_index("qid").pid
N_BASE = json.load(open(f"{PD}/pool_meta.json"))["n_pool"]
print("shards found:", len(glob.glob("/kaggle/input/**/query_info_s*.csv", recursive=True)))
print(infos.groupby("set").agg(n=("qid", "size"), recall=("recall", "mean"), cand=("n_cand", "median"),
                               cand_p90=("n_cand", lambda x: x.quantile(0.9)), secs=("secs", "mean")).round(3).to_string())
for name, f in feats.items():
    nz = f.groupby("qid").fz.std().gt(0).mean()
    pc = (f.pid >= N_BASE).mean()
    print(f"{name}: {len(f):,} rows, PubChem candidates {pc:.3f} of rows, queries with non-constant fz {nz:.3f}")
    if nz < 0.5:
        raise RuntimeError(f"fz features look empty in {name}: was the model attached in k07?")
assert all((f.pid >= N_BASE).any() for f in feats.values()), "no PubChem candidates: was k05p attached in k07 v5?"

# R の切り詰め: query_info.n_cand は切り詰め前の候補数
for name in ["R1", "R2"]:
    kept = feats[name].groupby("qid").size()
    nc = infos.set_index("qid").n_cand.reindex(kept.index)
    print(f"{name}: truncated queries {(kept < nc).mean():.3f} (cap 1,000 per keep rule)")

# %% [markdown]
# ## 隠し方のチェック: B 系・R2 では正解にライブラリ由来の情報が残っていないこと

# %%
for name in ["B_A", "B_C", "R2"]:
    if name in feats:
        t = feats[name][feats[name].label == 1]
        print(name, "truth max of", {c: round(float(t[c].max()), 4) for c in
                                     ["lib_max", "lib_rows", "cand_nspec", "cand_nlib", "tan_self"]})

# %% [markdown]
# ## PubChem にしか正解が無いセットのクエリと、正解が S′ を通るか

# %%
pt = pd.read_parquet(paths.find("pool_pubchem.parquet"), columns=["pid", "base_pass", "n_pmid"]).set_index("pid")
pass_sp = (pt.base_pass & (pt.n_pmid >= 1)).reindex(np.arange(N_BASE), fill_value=False).values
in_co = pd.read_parquet(f"{PD}/pool.parquet", columns=["in_coconut"]).in_coconut.values
P_Q = {}
for pname, base in P_SETS.items():
    q = infos[infos.set == base].qid.values
    if pname == "B_PR":
        q = q[~in_co[truth.reindex(q).values]]
    P_Q[pname] = (q, pass_sp[truth.reindex(q).values])
    print(f"{pname} (from {base}): {len(q)} queries, truth passes S' {P_Q[pname][1].mean():.3f}")

# %% [markdown]
# ## ルールの評価（M2 と同じ）

# %%
def evaluate(name, score, f=None, qids=None):
    """MRR@25 etc. over the queries of set `name` (or the given qids); queries without the truth count as misses."""
    f = feats[name] if f is None else f
    r = harness.ranks_of_truth(f.qid.values, f.label.values, score, f.pid.values)
    q = infos[infos.set == name].qid.values if qids is None else qids
    return harness.metrics(r.reindex(q).fillna(np.inf).values)


def evaluate_p(pname, score):
    """A PubChem-only set: ranks of its base set, misses where the truth does not pass S'."""
    base = P_SETS[pname]
    f = feats[base]
    q, ok = P_Q[pname]
    r = harness.ranks_of_truth(f.qid.values, f.label.values, score, f.pid.values).reindex(q).fillna(np.inf).values
    return harness.metrics(np.where(ok, r, np.inf))


def evaluate_all(model, scores):
    """scores: {set: score array aligned with feats[set]} -> result rows for SETS and the PubChem-only sets."""
    out = [dict(set=s, model=model, **evaluate(s, scores[s])) for s in SETS]
    return out + [dict(set=p, model=model, **evaluate_p(p, scores[b])) for p, b in P_SETS.items()]


RULES = search.RULES + search.GATE_RULES + search.FZ_RULES
res = []
for rule in RULES:
    res += evaluate_all(f"rule:{rule}", {s: search.rule_score(feats[s][search.FEATURES].values, rule) for s in SETS})
print(f"rules done ({time.time() - T0:.0f}s)")

# %% [markdown]
# ## ランカー（R の値は正解の構造単位で分けた 5 分割の OOF、他のセットは 3 シードの平均）
# lgb_R: R1 + R2 全体で学習（M2 と同じ。特徴量は S′ のプールで計算し直したもの）

# %%
R = pd.concat([feats["R1"], feats["R2"]], ignore_index=True)
FL = list(search.FEATURES)
in_set = {n: R.qid.isin(feats[n].qid.unique()).values for n in ["R1", "R2"]}


def set_scores(models, oof_full):
    """Score arrays aligned with feats[set]: OOF for R1/R2, seed-averaged predictions for the other sets."""
    sc = {n: oof_full[in_set[n]] for n in ["R1", "R2"]}
    for n in ["A", "B_A", "C", "B_C"]:
        sc[n] = rank.predict(models, feats[n][FL].values.astype(np.float32))
    return sc


def train_ranker(mask, tag):
    rows = R[mask]
    oof_full = np.zeros(len(R))
    oof_full[mask] = rank.cv_oof(rows, FL, n_folds=5, groups=truth)
    models = [rank.fit(rows, FL, seed=s) for s in SEEDS]
    for i, mdl in enumerate(models):
        mdl.save_model(f"{OUT}/ranker_{tag}_{i}.txt")
    print(f"  ranker {tag}: {rows.qid.nunique():,} queries, {len(rows):,} rows ({time.time() - T0:.0f}s)", flush=True)
    return models, set_scores(models, oof_full)


models_lgb, S_lgb = train_ranker(np.ones(len(R), bool), "lgb_R")
res += evaluate_all("lgb_R", S_lgb)
g = np.mean([mdl.feature_importance("gain") for mdl in models_lgb], axis=0)
imp = pd.Series(g / g.sum(), index=FL, name="gain").sort_values(ascending=False)

# %% [markdown]
# ## 組み合わせの評価
# - hyb:τ（参照用）: クラス2側は lgb_R
# - hybm:τ:w（選択の候補）: クラス2側は z(lgb_R) + w·(ap4 + 0.2·lib_max)。z は分子ごと（qid ごと）の z-score

# %%
LGB_FILES = [f"ranker_lgb_R_{i}.txt" for i in range(len(SEEDS))]
HYB = {f"hyb:{tau}": dict(tau=tau) for tau in TAUS}
HYBM = {f"hybm:{tau}:{w}": dict(tau=tau, w=w) for tau in BLEND_TAUS for w in BLEND_WS}


def hyb_score(s, v):
    """Score of the hybrid variant v (dict with tau, and w for hybm) on set s, aligned with feats[s]."""
    X = feats[s][FL].values
    if "w" in v:
        return search.blend_score(X, v["tau"], S_lgb[s], v["w"], qid=feats[s].qid.values)
    return search.hybrid_score(X, v["tau"], S_lgb[s])


for hname, v in (HYB | HYBM).items():
    res += evaluate_all(hname, {s: hyb_score(s, v) for s in SETS})

res = pd.DataFrame(res)
res.to_csv(f"{OUT}/results.csv", index=False)
ALL = SETS + list(P_SETS)
piv = res.pivot(index="model", columns="set", values="mrr25")[ALL]
m = lambda s, model: NONE[s] if model == NONE["name"] else float(piv.loc[model, s])
r_obj = lambda model: 0.2 * m("R1", model) + 0.8 * m("R2", model)
j_obj = lambda model: sum(w * m(s, model) for s, w in J_W.items())
show = ["rule:lib_analog", "rule:gate:0.9", "rule:fz", "lgb_R"] + list(HYB) + list(HYBM)
print(piv.loc[show].round(4).to_string())

# %% [markdown]
# ## 見立ての確認（選択には使わない）
# 各 τ でクラス2側に回る A の分子だけを取り出し、クラス2側の順位付けを比べる。

# %%
gmax = {s: feats[s].groupby("qid").lib_grp_max.first() for s in ["A", "B_A"]}
XA = feats["A"][FL].values
diag = []
for tau in TAUS:
    routed = gmax["A"].index[gmax["A"] < tau].values
    row = dict(tau=tau, A_routed=len(routed) / len(gmax["A"]), B_A_routed=float((gmax["B_A"] < tau).mean()))
    if len(routed):
        scs = [("lib_analog", search.rule_score(XA, "lib_analog")),
               ("M1 branch", search.rule_score(XA, f"gate:{tau}")),
               ("lgb_R", S_lgb["A"])]
        if tau in BLEND_TAUS:
            scs += [(f"hybm w{w}", hyb_score("A", HYBM[f"hybm:{tau}:{w}"])) for w in BLEND_WS]
        for label, sc in scs:
            row[label] = evaluate("A", sc, qids=routed)["mrr25"]
    diag.append(row)
diag = pd.DataFrame(diag)
print("MRR@25 on the A molecules routed to the class-2 branch (lib_grp_max < tau)")
print(diag.round(3).to_string(index=False))
hyb_tab = pd.DataFrame([dict(model=h, tau=v["tau"], w=v.get("w", 0), A=m("A", h), B_A=m("B_A", h), B_C=m("B_C", h),
                             R1=m("R1", h), R2=m("R2", h), R_obj=r_obj(h), B_PR=m("B_PR", h), B_PA=m("B_PA", h),
                             B_PC=m("B_PC", h), J=j_obj(h)) for h, v in (HYB | HYBM).items()])
print(hyb_tab.round(4).to_string(index=False))

# %% [markdown]
# ## 選択（採用基準は冒頭のとおり）

# %%
ok1 = [h for h in HYBM if m("A", h) >= A_MIN]
stage1 = max(ok1, key=r_obj) if ok1 else None
print(f"stage 1 (hybm on the S' pool, A >= {A_MIN:.4f}, max R objective): {stage1} | passing: {ok1}")
cands = [NONE["name"]] + ([stage1] if stage1 else [])
for c in cands:
    print(f"  {c:24s} A {m('A', c):.4f}  B_A {m('B_A', c):.4f}  B_PR {m('B_PR', c):.4f}  J {j_obj(c):.4f}")
recommend = max(cands, key=j_obj)
print("stage 2 recommendation (max J):", recommend)
if stage1 is None:
    print("no S' variant passes the A bar: S' is not adopted, no submission variant")
variants = {"lgb_R": dict(features=FL, files=LGB_FILES)}
variants |= {h: dict(v, features=FL, files=LGB_FILES) for h, v in (HYB | HYBM).items()}
json.dump(dict(default=stage1, recommend=recommend, pubchem_level=1, variants=variants),
          open(f"{OUT}/ranker.json", "w"), indent=1)

# %%
run = track.init("eval", group=EXP, config=dict(A_REF=A_REF, A_REF_SE=A_REF_SE, none=NONE, J_W=J_W, rules=RULES,
                                                 taus=list(TAUS), blend_taus=list(BLEND_TAUS),
                                                 blend_ws=list(BLEND_WS), seeds=list(SEEDS), pubchem_level="S'"))
run.table("results", res)
run.table("hybrid", hyb_tab)
run.table("diag_routed_A", diag)
run.table("importance_lgb_R", imp.reset_index().rename(columns={"index": "feature"}))
run.summary({f"mrr25/{r.set}/{r.model}": r.mrr25 for r in res.itertuples()}
            | {f"recall/{k}": v for k, v in infos.groupby("set").recall.mean().items()}
            | {f"J/{c}": j_obj(c) for c in cands} | {"stage1": stage1, "recommend": recommend})
run.finish()
print(f"done {time.time() - T0:.0f}s")

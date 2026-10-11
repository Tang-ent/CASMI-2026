"""Metric-identity audit of real submissions (run inside the Kaggle image by tools/smoke/run.sh metricid; repo at /data).

  keys:   for every unique SMILES of the submissions, the official metric identity (copied from the scorer
          metric/casmi-mean-reciprocal-rank: RemoveStereochemistry -> default TautomerEnumerator -> InChIKey14) and our
          chem.score_key, with the RDKit installed in this container -> <out>/keys_<rdkit version>.csv
  report: reads keys_2026.03.3.csv (the scorer's RDKit) and keys_2026.09.1.csv (our kernels' RDKit) and prints, per
          submission, metric-identical duplicates inside a list (their ranks, and an upper bound of the MRR gain a
          dedup could give) and how often our key disagrees with the metric identity.

usage: metric_identity.py keys|report <out_dir> <submission.csv>...
rdkit comes from the wheel directory run.sh mounts (PIP_NO_INDEX / PIP_FIND_LINKS).
"""
import os
import subprocess
import sys
import time

MODE, OUT, SUBS = sys.argv[1], sys.argv[2], sys.argv[3:]
os.makedirs(OUT, exist_ok=True)
import pandas as pd  # noqa: E402


def lists_of(path):
    s = pd.read_csv(path)
    return s.molecule_id.tolist(), [[x.strip() for x in str(v).split(";") if x.strip()] for v in s.smiles]


# ---------------------------------------------------------------------------------------------------------- keys
_W = {}


def _init():
    sys.path.insert(0, "/data/src")
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize

    from casmi26 import chem
    RDLogger.DisableLog("rdApp.*")
    _W.update(Chem=Chem, taut=rdMolStandardize.TautomerEnumerator(), chem=chem)


def metric_ik14(smiles):
    """Verbatim logic of the scorer's canonicalize_smiles_to_ik14."""
    Chem = _W["Chem"]
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.RemoveStereochemistry(mol)
        return Chem.MolToInchiKey(_W["taut"].Canonicalize(mol))[:14]
    except (ValueError, RuntimeError, TypeError):
        return None


def one(smiles):
    t0 = time.time()
    m = metric_ik14(smiles)
    t1 = time.time()
    o = _W["chem"].score_key(smiles)
    return smiles, m, o, t1 - t0, time.time() - t1


if MODE == "keys":
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "rdkit"], check=True)
    import rdkit
    from multiprocessing import Pool
    uniq = sorted({x for p in SUBS for lst in lists_of(p)[1] for x in lst})
    print(f"rdkit {rdkit.__version__}: {len(uniq):,} unique SMILES from {len(SUBS)} submissions", flush=True)
    t0 = time.time()
    with Pool(os.cpu_count(), initializer=_init) as P:
        rows = list(P.imap_unordered(one, uniq, chunksize=16))
    k = pd.DataFrame(rows, columns=["smiles", "metric", "ours", "t_metric", "t_ours"])
    k.to_csv(f"{OUT}/keys_{rdkit.__version__}.csv", index=False)
    slow = k.nlargest(3, "t_metric")
    print(f"done {time.time() - t0:.0f}s; metric None {k.metric.isna().sum()}, ours None {k.ours.isna().sum()}; "
          f"slowest metric canonicalisation {slow.t_metric.round(1).tolist()} s")

# -------------------------------------------------------------------------------------------------------- report
if MODE == "report":
    K = {v: pd.read_csv(f"{OUT}/keys_{v}.csv").set_index("smiles") for v in ("2026.03.3", "2026.09.1")}
    k = K["2026.03.3"].join(K["2026.09.1"], rsuffix="_91")
    print(f"unique SMILES {len(k):,}")
    both = k.metric.notna() & k.ours.notna() & k.ours_91.notna()
    print(f"  ours (cap 250, no stereo strip) != metric, same RDKit 2026.03.3: {(k.ours != k.metric)[both].sum():,}")
    print(f"  ours 2026.09.1 != ours 2026.03.3 (version):                     {(k.ours_91 != k.ours)[both].sum():,}")
    print(f"  ours 2026.09.1 != metric 2026.03.3 (what the kernels see):        {(k.ours_91 != k.metric)[both].sum():,}")
    print(f"  metric function on 2026.09.1 != on 2026.03.3:                    "
          f"{(k.metric_91 != k.metric)[k.metric.notna() & k.metric_91.notna()].sum():,}")
    mid = k.metric.to_dict()
    for path in SUBS:
        ids, L = lists_of(path)
        n_ent = sum(map(len, L))
        none = sum(not isinstance(mid[x], str) for lst in L for x in lst)
        multi = sum("." in x for lst in L for x in lst)
        dup_ranks, dup_lists, n_lists, gain = [], 0, 0, []
        for lst in L:
            seen, new_rank, g, has_dup = set(), 0, 0.0, False
            for r, x in enumerate(lst, 1):
                key = mid[x]
                if not isinstance(key, str):
                    continue      # an unparsable guess never matches; a dedup would drop it too
                if key in seen:
                    dup_ranks.append(r); has_dup = True
                    continue
                seen.add(key)
                new_rank += 1
                g = max(g, 1.0 / new_rank - 1.0 / r)
            dup_lists += int(has_dup)
            n_lists += int(g > 0)
            gain.append(g)
        print(f"\n{path}: {len(L)} lists, {n_ent:,} entries; metric-unparsable {none}, multi-component (gate fails) {multi}")
        print(f"  metric-identical duplicates: {len(dup_ranks)} entries in {dup_lists} lists"
              + (f"; ranks of the later copies {sorted(dup_ranks)[:30]}" if dup_ranks else ""))
        print(f"  lists where a dedup would lift something: {n_lists}; "
              f"upper bound of the MRR gain (truth sits where it gains most) {sum(gain) / len(L):.5f}")

"""Checks of one fetched fusion-notebook version (M8, 2026-10-19; tools/push_fuse.sh).
  python tools/check_fuse.py out/m8_fuse/<MMDDHHMM> [base dir, default out/m8_fuse/base_v1]
Hard checks: the run kept the base environment (E004_PREFLIGHT: Python 3.12, RDKit 2026.03.3, T4); our engine ran
(own lists N, no OWN ENGINE FAILED, own_stats: every processed molecule, models on cuda, RDKit 2026.09.1); the E006 overlay
finished (FACTORIAL_VERIFICATION, same-run control reconstructed); submission.csv has the sample's 400 ids, 1..25 entries,
no repeated SMILES in a row.
Report: our list lengths and window sizes, rows / top-1 changed against the base version's submission (same processed
molecules), the E006 branches, our 25-entry list against the k36 v12 submission on the processed molecules."""
import json
import re
import sys

import pandas as pd

d = sys.argv[1]
base = sys.argv[2] if len(sys.argv) > 2 else "out/m8_fuse/base_v1"
txt = open(f"{d}/enveda-casmi-2026.txt").read()
ok = True


def grab(tag):
    m = re.search(rf"^{tag} (\{{.*\}})\s*$", txt, re.M)
    return json.loads(m.group(1)) if m else None


pre = grab("E004_PREFLIGHT")
print("preflight", pre)
ok &= bool(pre) and pre["python"].startswith("3.12") and pre["rdkit"] == "2026.03.3" and "T4" in pre["gpu"]
m = re.search(r"^own lists (\d+) (\d+)s BETA ([0-9.]+)", txt, re.M)
failed = "OWN ENGINE FAILED" in txt
print("own lists", m.groups() if m else None, "| OWN ENGINE FAILED (do not submit this version)" if failed else "")
ok &= bool(m) and not failed
st = json.load(open(f"{d}/own/own_stats.json"))
print("own stats", {k: st[k] for k in ("stats", "secs", "rdkit", "devices", "src_sha")})
ver = json.load(open(f"{d}/experiment_verification.json"))
processed = set(ver["processed_ids"])
ok &= st["stats"]["molecules"] == len(processed) and st["rdkit"] == "2026.09.1" and all(x == "cuda" for x in st["devices"])
# wiring: the fusion inputs saved by cell 31 carry our list for every processed molecule (the 12 visible-test molecules
# of a commit are library hits, so "rows changed vs base" can be 0 without anything being wrong)
inp = json.load(open(f"{d}/e004_candidate_inputs.json"))
n_own = sum(bool((inp.get("own") or {}).get(mid) and inp["own"][mid].get("smiles")) for mid in processed)
print(f"own lists in e004_candidate_inputs.json: {n_own}/{len(processed)}")
ok &= n_own == len(processed)
print("E006", {k: ver[k] for k in ("experiment", "window", "ice_fallback", "output_rows", "changed_rows", "changed_top1",
                                   "fallback_rows", "same_run_control_reconstructed")}, "| processed", len(processed))
ok &= ver["same_run_control_reconstructed"] and ver["output_rows"] == 400 and "FACTORIAL_VERIFICATION" in txt
sub = pd.read_csv(f"{d}/submission.csv", dtype=str, keep_default_na=False)
parts = sub.smiles.str.split(";")
dup = int((parts.map(lambda x: len(set(x))) != parts.map(len)).sum())
print(f"submission: rows {len(sub)}, ids {sub.molecule_id.nunique()}, entries {parts.map(len).min()}-{parts.map(len).max()}, "
      f"rows with repeated SMILES {dup}")
ok &= len(sub) == 400 and sub.molecule_id.is_unique and parts.map(len).between(1, 25).all() and dup == 0
own = json.load(open(f"{d}/own/own_lists.json"))
lens = [len(own.get(mid, {}).get("smiles", [])) for mid in sorted(processed)]
print("own list length per processed molecule", lens, "| window", [own.get(m, {}).get("n_window") for m in sorted(processed)])
b = pd.read_csv(f"{base}/submission.csv", dtype=str, keep_default_na=False).set_index("molecule_id").smiles.str.split(";")
a = parts.set_axis(sub.molecule_id)
pm = sorted(processed & set(b.index))
ch = sum(a[m] != b[m] for m in pm)
ch1 = sum(a[m][0] != b[m][0] for m in pm)
print(f"vs base {base}: processed rows changed {ch}/{len(pm)}, top-1 changed {ch1}/{len(pm)}")
aud = pd.read_csv(f"{d}/experiment_audit_factorial.csv")
print("E006 branches", aud[aud.processed].branch.value_counts().to_dict())
k36 = pd.read_csv("out/m7_k36/10160835/submission.csv", dtype=str).set_index("molecule_id").smiles
mine = pd.read_csv(f"{d}/own/own_submission.csv", dtype=str).set_index("molecule_id").smiles
same = sum(mine[m] == k36[m] for m in pm)
print(f"own 25-entry list == k36 v12 submission on {same}/{len(pm)} processed molecules (GPU / torch may reorder near ties)")
if ok:  # for tools/wandb_lb.py --variant-json (appends to tools/lb_points.csv)
    json.dump(dict(variant=f"fuse:v12:beta{m.group(3)}", parsed=dict(base="yuanyuan98/enveda-casmi-2026 v1 (LB 0.441)",
                   own="both_fprich_g2_grp|hybi:0.7:0.8:5+nogen", beta=float(m.group(3))), eval="fuse", local={},
                   stats=st["stats"], src_sha=st["src_sha"], rdkit=st["rdkit"]), open(f"{d}/variant.json", "w"), indent=1)
    print(f"wrote {d}/variant.json")
print("OK" if ok else "FAILED (do not submit this version)")
sys.exit(0 if ok else 1)

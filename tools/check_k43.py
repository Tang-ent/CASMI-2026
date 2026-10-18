"""Checks of one fetched k43_submit8 version before the user submits it (M7 P3, 2026-10-18).
  python tools/check_k43.py out/m7_k43/<MMDDHHMM> [<local run of the same variant, e.g. out/p3_k43_extcheck>]
Hard checks: 400 rows, unique molecule ids, 1..25 candidates per row, no duplicate SMILES in a row, ext failures 0.
Report: candidate counts, ext stats / errors, local numbers, and the agreement with a local run of the same variant
(top-1 and top-25 set overlap; GPU / torch differences may reorder near ties; the ext counts must be equal)."""
import json
import sys

import pandas as pd

d = sys.argv[1]
j = json.load(open(f"{d}/variant.json"))
s = pd.read_csv(f"{d}/submission.csv")
c = s.smiles.str.split(";")
n = c.map(len)
dup = int((c.map(lambda x: len(set(x))) != n).sum())
st = j["stats"]
print(f"{d}: {j['variant']}")
print(f"  rows {len(s)}, unique ids {s.molecule_id.nunique()}, candidates {n.min()}-{n.max()} (25: {(n == 25).sum()}),"
      f" rows with duplicate SMILES {dup}")
print("  stats", {k: v for k, v in st.items() if not k.startswith("ext_")})
print("  ext", {k: v for k, v in st.items() if k.startswith("ext_")})
print("  local", {k: round(v, 4) if isinstance(v, float) else v for k, v in j["local"].items()})
fails = st.get("ext_failed", {})
ok = len(s) == 400 and s.molecule_id.nunique() == 400 and n.min() >= 1 and n.max() <= 25 and dup == 0
ok &= all(v == 0 for v in fails.values())
if len(sys.argv) > 2:
    r = pd.read_csv(f"{sys.argv[2]}/submission.csv").set_index("molecule_id").smiles.str.split(";")
    a = c.set_axis(s.molecule_id)
    top1 = float((a.map(lambda x: x[0]) == r.loc[a.index].map(lambda x: x[0])).mean())
    ov = float(pd.Series([len(set(x) & set(y)) / max(len(x), 1) for x, y in zip(a, r.loc[a.index])]).mean())
    same = float((a == r.loc[a.index]).mean())
    print(f"  vs {sys.argv[2]}: identical rows {same:.3f}, same top-1 {top1:.3f}, top-25 set overlap {ov:.3f}")
    # the public test is part of train (every molecule takes the library-rule branch), so the rows match v12 there;
    # the ext path itself must have scored the same candidates as the local run
    rs = json.load(open(f"{sys.argv[2]}/variant.json"))["stats"]
    keys = [k for k in ("ext_molecules", "ext_with_specs", "ext_scored", "ext_entries") if k in rs and k in st]
    diff = {k: (st[k], rs[k]) for k in keys if st[k] != rs[k]}
    print(f"  ext stats vs {sys.argv[2]}: {'same' if not diff else diff}")
    ok &= not diff
print("OK" if ok else "FAILED")
sys.exit(0 if ok else 1)

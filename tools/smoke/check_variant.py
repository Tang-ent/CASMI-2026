"""M7: search.parse_variant / variant_order unit checks (hybm == blend_score order, hybb weights by band, hybi keeps
the rule's top 1 first and interleaves inside the band only). Run inside the Kaggle image with src on sys.path."""
import sys

import numpy as np

sys.path.insert(0, "/src")
from casmi26 import search as S  # noqa: E402

for n in ["both_fprich_g2|hybm:0.7:5+gen:S3:0.7:rk", "both|hybm:0.8:10+nogen", "x|hybb:0.7:0.8:5:10+gen:S2:0.7:rk",
          "x|hybi:0.7:0.8:5"]:
    print(n, S.parse_variant(n))
rng = np.random.default_rng(0)
X = rng.random((30, len(S.FEATURES)))
rk = rng.normal(size=30)
st = lambda s: np.argsort(-s, kind="mergesort")
for g in [0.65, 0.75, 0.85]:
    X[:, S.COL["lib_grp_max"]] = g
    assert (S.variant_order(X, rk, ("hybm", (0.7, 5.0))) == st(S.blend_score(X, 0.7, rk, 5.0))).all()
    o_b = S.variant_order(X, rk, ("hybb", (0.7, 0.8, 5.0, 10.0)))
    o_i = S.variant_order(X, rk, ("hybi", (0.7, 0.8, 5.0)))
    rule = st(S.rule_score(X, "lib_analog"))
    assert sorted(o_i) == list(range(30)) and sorted(o_b) == list(range(30))
    if 0.7 <= g < 0.8:
        assert o_i[0] == rule[0] and rule[1] in o_i[:4]
        assert (o_b == st(S.blend_score(X, 0.8, rk, 10.0))).all()
    elif g >= 0.8:
        assert (o_i == rule).all() and (o_b == rule).all()
    else:
        assert (o_b == st(S.blend_score(X, 0.8, rk, 5.0))).all() and (o_i == st(S.blend_score(X, 0.8, rk, 5.0))).all()
print("variant checks: PASS")

# formula_group_features: per molecule == on the stacked set; rank / gap / size by hand
fid = np.array([3, 3, 3, 7, 7, 3, 9, 9])
qid = np.array([1, 1, 1, 1, 1, 2, 2, 2])
cols = {c: rng.normal(size=8) for c in S.GRP_BASE}
cols["g2_z"][1] = np.nan
F = S.formula_group_features(cols, qid, fid)
v = np.nan_to_num(cols["fz_z"])
j = S.GRP_FEATURES.index("fz_z_frank")
assert F[0, j] == 1 + (v[[1, 2]] > v[0]).sum() and F[5, j] == 1 and F[6, j] == 1 + (v[7] > v[6])
assert np.isclose(F[3, S.GRP_FEATURES.index("fz_z_fgap")], v[3] - v[[3, 4]].max())
assert list(F[:, -1]) == [3, 3, 3, 2, 2, 1, 2, 2]
for q in (1, 2):
    m = qid == q
    assert np.array_equal(S.formula_group_features({c: x[m] for c, x in cols.items()}, None, fid[m]), F[m])
print("formula group features: PASS")

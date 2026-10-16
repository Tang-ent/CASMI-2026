"""Fit the public LB against the local yardsticks (M7: bank LB points, refit the predicted-LB weights after > 20).

    python tools/lb_fit.py [--points tools/lb_points.csv]
Report only. Models (least squares on the points with all columns present):
  fixed   LB = 0.163·A + 0.50·B_N (the selection rule's prediction since 2026-10-16, no fit; M6 rule 0.163 / 0.21 / 0.30)
  BN / BA LB = 0.163·A + s·B + g·G_N (s, g fitted; B = B_N or B_A; the A weight stays at the class-1 estimate)
  free    LB = c + a·A + s·B_N + t·B_A + g·G_N (only with enough points)
  k24+    LB = c + s·B_N + g·gen on the points since k24 v1 (same pool and generation; gen = G_N > 0), the basis of the
          2026-10-16 revision
For each: coefficients with their standard errors, RMSE and leave-one-out RMSE (the LB noise floor is about 0.02, so
fits with RMSE near it cannot be told apart). Changing the selection rule's weights is a user decision.
"""
import argparse

import numpy as np
import pandas as pd


def ols(Xm, y):
    beta, *_ = np.linalg.lstsq(Xm, y, rcond=None)
    res = y - Xm @ beta
    dof = max(len(y) - Xm.shape[1], 1)
    s2 = res @ res / dof
    cov = s2 * np.linalg.pinv(Xm.T @ Xm)
    loo = []
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        b, *_ = np.linalg.lstsq(Xm[m], y[m], rcond=None)
        loo.append(y[i] - Xm[i] @ b)
    return beta, np.sqrt(np.diag(cov)), float(np.sqrt(np.mean(res ** 2))), float(np.sqrt(np.mean(np.square(loo))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", default="tools/lb_points.csv")
    a = ap.parse_args()
    d = pd.read_csv(a.points).dropna(subset=["lb_public", "A", "B_A", "B_N", "G_N"])
    d = d[d.lb_public > 0].reset_index(drop=True)
    y = d.lb_public.values
    print(f"{len(d)} LB points ({d.date.min()} .. {d.date.max()})")
    pred = 0.163 * d.A + 0.50 * d.B_N
    d["pred_fixed"], d["resid_fixed"] = pred, y - pred
    r = y - pred  # the rule predicts differences, so its level is free: report the spread around the mean
    print(f"fixed: mean residual {r.mean():+.4f}, RMSE around it {r.std():.4f}")
    off = 0.163 * d.A.values
    for b in ["B_N", "B_A"]:
        cols = [b] + (["G_N"] if d.G_N.nunique() > 1 else [])
        beta, se, rmse, loo = ols(d[cols].values, y - off)
        print(f"{b}: " + ", ".join(f"{c} {x:.3f} ± {s:.3f}" for c, x, s in zip(cols, beta, se)) +
              f" | RMSE {rmse:.4f}, LOO RMSE {loo:.4f}")
    if len(d) >= 12:
        cols = ["A", "B_N", "B_A", "G_N"]
        Xm = np.column_stack([np.ones(len(d)), d[cols].values])
        beta, se, rmse, loo = ols(Xm, y)
        print("free: " + ", ".join(f"{c} {x:.3f} ± {s:.3f}" for c, x, s in zip(["const"] + cols, beta, se)) +
              f" | RMSE {rmse:.4f}, LOO RMSE {loo:.4f}")
    e = d[d.date >= "2026-10-11"]
    if len(e) >= 8:
        Xe = np.column_stack([np.ones(len(e)), e.B_N.values, (e.G_N > 0).values.astype(float)])
        beta, se, rmse, loo = ols(Xe, e.lb_public.values)
        print(f"k24+ ({len(e)}): " + ", ".join(f"{c} {x:.3f} ± {s:.3f}" for c, x, s in zip(["const", "B_N", "gen"], beta, se)) +
              f" | RMSE {rmse:.4f}, LOO RMSE {loo:.4f}")
    with pd.option_context("display.width", 200):
        print(d[["date", "label", "lb_public", "A", "B_A", "B_N", "G_N", "pred_fixed", "resid_fixed"]].round(4)
              .to_string(index=False))


if __name__ == "__main__":
    main()

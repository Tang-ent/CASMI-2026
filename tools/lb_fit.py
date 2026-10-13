"""Fit the public LB against the local yardsticks (M7: bank LB points, refit the predicted-LB weights after > 20).

    python tools/lb_fit.py [--points tools/lb_points.csv]
Report only. Models (least squares on the points with all columns present):
  fixed   LB = 0.163·A + 0.21·B_N + 0.30·G_N (the selection rule's prediction, no fit)
  BN / BA LB = 0.163·A + s·B + g·G_N (s, g fitted; B = B_N or B_A; the A weight stays at the class-1 estimate)
  free    LB = c + a·A + s·B_N + t·B_A + g·G_N (only with enough points)
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
    pred = 0.163 * d.A + 0.21 * d.B_N + 0.30 * d.G_N
    d["pred_fixed"], d["resid_fixed"] = pred, y - pred
    print(f"fixed: RMSE {np.sqrt(np.mean((y - pred) ** 2)):.4f}, mean residual {np.mean(y - pred):+.4f}")
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
    with pd.option_context("display.width", 200):
        print(d[["date", "label", "lb_public", "A", "B_A", "B_N", "G_N", "pred_fixed", "resid_fixed"]].round(4)
              .to_string(index=False))


if __name__ == "__main__":
    main()

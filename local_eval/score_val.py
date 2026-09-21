#!/usr/bin/env python3
"""Score a submission on the labelled val split with the OFFICIAL scorer.

Mirrors score()'s aggregation so the four-platform breakdown is visible, but
every number comes from the scorer's own _ate_traj / _ave_traj.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter"))
import kaggle_metric_tartanimu_score as K  # noqa: E402


def score_breakdown(solution: pd.DataFrame, submission: pd.DataFrame) -> tuple:
    official = K.score(solution.copy(), submission.copy(), "window_id")

    sub = submission.rename(columns={"vx": "vx_pred", "vy": "vy_pred", "vz": "vz_pred"})
    m = solution.merge(sub, on="window_id", how="left")
    per_platform = {}
    for _, g in m.groupby("traj_id", sort=False):
        g = g.sort_values("win_idx")
        ate, ave = K._ate_traj(g), K._ave_traj(g)
        if np.isfinite(ate) and np.isfinite(ave):
            per_platform.setdefault(g["platform"].iloc[0], []).append((ate, ave))

    rows = []
    for p, v in sorted(per_platform.items()):
        ate = float(np.mean([x[0] for x in v]))
        ave = float(np.mean([x[1] for x in v]))
        rows.append({"platform": p, "trajs": len(v), "ATE20": ate, "AVE": ave,
                     "score": K.W_AVE * (ave / K.AVE_REF) + K.W_ATE * (ate / K.ATE_REF)})
    table = pd.DataFrame(rows)
    return official, table


def zeros_like_solution(sol: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"window_id": sol["window_id"], "vx": 0.0, "vy": 0.0, "vz": 0.0})


def truth_like_solution(sol: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"window_id": sol["window_id"], "vx": sol["vx_gt"],
                         "vy": sol["vy_gt"], "vz": sol["vz_gt"]})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", default="local_eval/val_solution.csv")
    ap.add_argument("--submission", default=None,
                    help="CSV to score; omit to run the zero/ground-truth sanity pair")
    a = ap.parse_args()
    sol = pd.read_csv(a.solution)

    if a.submission:
        cases = [(Path(a.submission).name, pd.read_csv(a.submission))]
    else:
        cases = [("ground-truth velocities", truth_like_solution(sol)),
                 ("all-zeros", zeros_like_solution(sol))]

    for name, sub in cases:
        official, table = score_breakdown(sol, sub)
        print(f"\n=== {name} ===")
        print(f"TartanIMU Score (official scorer, val) = {official:.5f}")
        print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

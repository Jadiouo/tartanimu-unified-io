#!/usr/bin/env python3
"""Build a scorer-ready solution frame for the labelled val split.

The official scorer (starter/kaggle_metric_tartanimu_score.py) wants
    window_id, traj_id, win_idx, platform, qx, qy, qz, qw, gx, gy, gz, dt,
    vx_gt, vy_gt, vz_gt
but the released index/val_targets.csv only carries the velocities, so the
pose columns have to be recovered from the val .npz files.

Conventions used here (verified in verify_local_eval.py by feeding the ground
truth velocities back in and checking ATE20 collapses):
    gx,gy,gz : world position at the LAST frame of the window
    qx..qw   : ground-truth quaternion at the MIDDLE frame of the window
    dt       : ts[end] - ts[start] + 1/fs   (== 1.0 s for a 200-frame window)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW_SIZE = 200


def build(data_root: Path, split: str = "val", traj_ids=None) -> pd.DataFrame:
    """Scorer solution frame for a split (val by default), optionally restricted
    to traj_ids -- used for the train hold-out set."""
    index_dir = data_root / "index"
    df = (pd.read_csv(index_dir / f"{split}_windows.csv").dropna(axis=1, how="all")
          .merge(pd.read_csv(index_dir / f"{split}_targets.csv"), on="window_id"))
    df = df.rename(columns={"vx": "vx_gt", "vy": "vy_gt", "vz": "vz_gt"})
    df["traj_id"] = df["traj_id"].astype(str).str.replace(".npz", "", regex=False)
    if traj_ids is not None:
        df = df[df["traj_id"].isin(set(traj_ids))]

    paths = {p.stem: p for p in (data_root / split).rglob("*.npz")}
    missing = set(df["traj_id"]) - set(paths)
    if missing:
        raise SystemExit(f"no .npz for {len(missing)} traj_id(s), e.g. {sorted(missing)[:3]}")

    df = df.sort_values(["traj_id", "win_idx"]).reset_index(drop=True)
    pos = np.empty((len(df), 3))
    quat = np.empty((len(df), 4))
    dt = np.empty(len(df))

    for traj_id, rows in df.groupby("traj_id", sort=False):
        with np.load(paths[traj_id]) as d:
            ts, fs = d["ts"], float(d["fs"])
            starts = rows["win_idx"].to_numpy() * WINDOW_SIZE
            mids = np.minimum(starts + WINDOW_SIZE // 2, len(ts) - 1)
            ends = np.minimum(starts + WINDOW_SIZE - 1, len(ts) - 1)
            idx = rows.index.to_numpy()
            pos[idx] = d["pos"][ends]
            quat[idx] = d["quat"][mids]
            dt[idx] = ts[ends] - ts[starts] + 1.0 / fs

    df[["gx", "gy", "gz"]] = pos
    df[["qx", "qy", "qz", "qw"]] = quat
    df["dt"] = dt
    return df[["window_id", "traj_id", "win_idx", "platform",
               "qx", "qy", "qz", "qw", "gx", "gy", "gz", "dt",
               "vx_gt", "vy_gt", "vz_gt"]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root",
                    default=os.environ.get("TARTANIMU_DATA",
                                           "data/tartan-imu-challenge-iros2026"))
    ap.add_argument("--out", default="local_eval/val_solution.csv")
    a = ap.parse_args()
    sol = build(Path(a.data_root))
    sol.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(sol)} rows, {sol['traj_id'].nunique()} trajectories")
    print(sol.groupby("platform").agg(windows=("window_id", "size"),
                                      trajs=("traj_id", "nunique"),
                                      dt_mean=("dt", "mean")))

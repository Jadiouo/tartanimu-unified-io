#!/usr/bin/env python3
"""Trajectory-segment view of a split: what an ATE20-style loss needs.

The per-window loss the notebook uses is blind to how errors correlate across
windows, which is exactly what ATE20 (40% of the score) measures.  Computing an
ATE20 surrogate during training needs (a) consecutive windows from one
trajectory and (b) the ground-truth attitude to rotate body-frame velocity into
the world.  Both are available in the train .npz files.

Conventions match local_eval/build_val_solution.py, which was verified by
feeding ground-truth velocities back through the official scorer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW_SIZE = 200


def quat_to_R(q: np.ndarray) -> np.ndarray:
    """(M,4) quaternion (x, y, z, w) -> (M,3,3), same convention as the scorer."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    x, y, z, w = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], axis=1)


def attach_pose(frame: pd.DataFrame) -> dict:
    """Per-window ground-truth pose for a split already carrying file_path.

    Returns R (M,3,3), pos (M,3) at the window's last frame, dt (M,).
    """
    n = len(frame)
    pos = np.empty((n, 3), np.float64)
    quat = np.empty((n, 4), np.float64)
    dt = np.empty(n, np.float64)
    for file_path, rows in frame.groupby("file_path", sort=False):
        with np.load(file_path) as d:
            ts, fs = d["ts"], float(d["fs"])
            starts = rows["win_idx"].to_numpy() * WINDOW_SIZE
            mids = np.minimum(starts + WINDOW_SIZE // 2, len(ts) - 1)
            ends = np.minimum(starts + WINDOW_SIZE - 1, len(ts) - 1)
            i = rows.index.to_numpy()
            pos[i] = d["pos"][ends]
            quat[i] = d["quat"][mids]
            dt[i] = ts[ends] - ts[starts] + 1.0 / fs
    return {"R": quat_to_R(quat).astype(np.float32),
            "pos": pos.astype(np.float32), "dt": dt.astype(np.float32)}


def segment_starts(frame: pd.DataFrame, seg_len: int) -> np.ndarray:
    """(S, seg_len) row indices, each row = consecutive windows of one trajectory."""
    segs = []
    for _, rows in frame.groupby("traj_id", sort=False):
        order = rows.sort_values("win_idx").index.to_numpy()
        if len(order) < seg_len:
            continue
        # every possible start, so a segment boundary never becomes a fixed cut
        for s in range(len(order) - seg_len + 1):
            segs.append(order[s:s + seg_len])
    return np.stack(segs)

#!/usr/bin/env python3
"""Body-frame gravity direction as an auxiliary training target.

The accelerometer channel is dominated by gravity: the mean specific force over
a window has magnitude ~9.8, while the motion component of a 1 s window is
usually an order of magnitude smaller.  Any velocity head therefore has to
learn, implicitly, where "down" is before the residual carries information --
and it has to learn that from the same scalar loss that grades velocity.  The
ground-truth attitude in the train/val .npz files makes that sub-problem
supervisable directly: a second head predicting the body-frame gravity
direction gets a dense, well-scaled signal at every window, and forces the
trunk to keep an attitude estimate in its features.

This is a training-time-only signal.  The test .npz files carry no quat, so the
gravity head exists purely to shape the trunk and is dropped (or simply
ignored) at inference.

Conventions match unified/segments.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from unified.segments import quat_to_R

WINDOW_SIZE = 200


def gravity_dir_body(quat: np.ndarray, g_world=(0.0, 0.0, -1.0)) -> np.ndarray:
    """(M,4) quaternion (x, y, z, w) -> (M,3) unit gravity direction in the body frame.

    quat_to_R returns the body->world matrix, so world->body is its transpose:
    g_body = R.T @ g_world.  The returned vector points the way gravity *pulls*
    (down), not the way the accelerometer reads.  A resting IMU measures
    specific force, i.e. the reaction to gravity, so its mean acceleration
    vector aligns with MINUS this vector -- verified on
    train/human/human_train_0000.npz over its 200 slowest windows (speed
    <= 0.03 m/s): median angle to -gravity_dir_body 0.39 deg, median |mean acc|
    9.8055.  (Against +gravity_dir_body the same windows give 179.61 deg, which
    is the expected
    reading of this sign convention, not an inversion.)  The default
    g_world=(0,0,-1) is the world frame's down axis; the data confirms world +z
    is up.
    """
    R = quat_to_R(np.asarray(quat, np.float64))
    g = np.asarray(g_world, np.float64)
    g = g / np.linalg.norm(g)
    # einsum over the first matrix axis = multiplying by R.T
    v = np.einsum("mji,j->mi", R, g)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v.astype(np.float32)


def window_gravity_target(frame: pd.DataFrame, n_per_window: int = 1) -> np.ndarray:
    """(len(frame), n_per_window, 3) float32 body-frame gravity directions.

    Samples n_per_window frames evenly across each window; n_per_window=1 lands
    on the window's middle frame, matching attach_pose's quaternion choice.
    """
    n = len(frame)
    # sub-block centres: [0.5, 1.5, ...] * WINDOW_SIZE / n, so n=1 -> 100
    offsets = ((np.arange(n_per_window) + 0.5) * (WINDOW_SIZE / n_per_window)).astype(np.int64)
    out = np.empty((n, n_per_window, 3), np.float32)
    for file_path, rows in frame.groupby("file_path", sort=False):
        with np.load(file_path) as d:
            quat = d["quat"]
        starts = rows["win_idx"].to_numpy() * WINDOW_SIZE
        take = np.minimum(starts[:, None] + offsets[None, :], len(quat) - 1)
        g = gravity_dir_body(quat[take.ravel()])
        out[rows.index.to_numpy()] = g.reshape(len(starts), n_per_window, 3)
    return out


def cosine_loss(pred, target):
    """1 - cosine_similarity, averaged.  Shapes (..., 3); pred need not be unit."""
    import torch.nn.functional as F

    return (1.0 - F.cosine_similarity(pred, target, dim=-1, eps=1e-8)).mean()


if __name__ == "__main__":
    # Self-check: on low-dynamics windows the mean measured acceleration must
    # sit opposite the predicted gravity direction.
    from unified.data import load_index

    train = load_index()["train"]
    # reset_index because window_gravity_target, like attach_pose, writes rows
    # by frame index
    rows = train[train["traj_id"] == "human_train_0000"].reset_index(drop=True)
    with np.load(rows["file_path"].iloc[0]) as d:
        imu = d["imu"]
    starts = rows["win_idx"].to_numpy() * WINDOW_SIZE
    acc = np.stack([imu[s:s + WINDOW_SIZE, :3].mean(0) for s in starts])
    mag = np.linalg.norm(acc, axis=1)

    g = window_gravity_target(rows)[:, 0, :]
    speed = np.linalg.norm(rows[["vx", "vy", "vz"]].to_numpy(), axis=1)
    k = np.argsort(speed)[:200]
    cos = ((acc / mag[:, None]) * -g).sum(1)
    ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    print(f"windows checked      {len(k)} of {len(rows)}")
    print(f"speed of those       <= {speed[k].max():.4f} m/s")
    print(f"median angle to -g   {np.median(ang[k]):.3f} deg")
    print(f"median angle to +g   {180 - np.median(ang[k]):.3f} deg")
    print(f"median |mean acc|    {np.median(mag[k]):.4f}")

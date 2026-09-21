"""Ground-truth up direction expressed in the IMU frame -- a privileged
diagnostic input, training/validation only.

up_body = third row of the body-to-world rotation (world e_z in the label
frame); up_imu = R_ext^T up_body with R_ext the fitted IMU->label rotation.
The sign is then chosen per trajectory so that on calm frames it agrees with
the accelerometer direction, which is the convention the complementary filter
and the decomposition channels use (up = what the accelerometer reads at rest).
"""
from __future__ import annotations

import numpy as np
import torch

from unified.dense import quat_to_R_torch


def load_extrinsics(path):
    z = np.load(path)
    return {t: (R, int(q), int(s)) for t, R, q, s in zip(z["traj_id"], z["R"], z["quality"], z["sig"])}


def gt_up_frames(imu: np.ndarray, quat: np.ndarray, R_ext: np.ndarray) -> np.ndarray:
    Rb2w = quat_to_R_torch(torch.from_numpy(np.asarray(quat)).double()).numpy()
    up_body = Rb2w[:, 2, :]
    a = imu[:, :3]; an = np.linalg.norm(a, axis=1)
    calm = np.abs(an - 9.81) < 0.5
    best, best_agree = None, -2.0
    # the gyro-fitted extrinsic need not be the accelerometer's frame (the two
    # drone sources differ): try both candidates, either sign, keep the one the
    # calm-frame accelerometer direction agrees with most
    for cand in ((R_ext.T @ up_body.T).T, up_body):
        cand = cand / np.maximum(np.linalg.norm(cand, axis=1, keepdims=True), 1e-9)
        for sgn in (1.0, -1.0):
            u = sgn * cand
            agree = (a[calm] / an[calm, None] * u[calm]).sum(1).mean() if calm.sum() > 50 else 0.0
            if agree > best_agree:
                best, best_agree = u, agree
    return best.astype(np.float32)

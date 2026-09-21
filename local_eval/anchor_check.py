#!/usr/bin/env python3
"""Anchor-window audit (2026-09-20 night): for every drone recording with GT, the tilt error of
the rest-anchor 'up0' (calm-window accelerometer direction) against the GT up at that window,
under the current rule (min |mean|a|-g| + mean|w| + 0.5 std|a|) and label-free alternatives:
  cons  -- consistency rule: among the K calmest windows, the one whose gyro-integrated up
           agrees best with the accelerometer direction of the other calm windows (a true rest
           window predicts the other rest windows; a cruise window does not).
Usage: python local_eval/anchor_check.py [family: racing|all]"""
import glob, sys, os
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from unified.gtup import gt_up_frames, load_extrinsics
from unified.attnet import integrate
import torch
DATA = os.environ.get('TARTANIMU_DATA', str(ROOT / 'data/tartan-imu-challenge-iros2026'))
ext = load_extrinsics(str(ROOT / 'runs/extrinsics.npz')); torch.set_num_threads(4)
W = 200


def calm_scores(imu):
    n = len(imu) // W; acc = imu[:n * W, :3].reshape(n, W, 3); gyr = imu[:n * W, 3:6].reshape(n, W, 3)
    an = np.linalg.norm(acc, axis=2)
    return np.abs(an.mean(1) - 9.81) + np.linalg.norm(gyr, axis=2).mean(1) + 0.5 * an.std(1), acc


def gyro_up_from(imu, k0, up0):
    om = torch.from_numpy(imu[:, 3:6].astype(np.float32)); s0 = k0 * W + W // 2
    Rt = integrate(om[None], 1 / 200.0, torch.zeros(1, 3), torch.zeros(1, om.shape[0], 3), s0)[0]
    return (Rt.transpose(-1, -2) @ torch.from_numpy(up0.astype(np.float32))[:, None])[..., 0].numpy()


def consistency_anchor(imu, K=6):
    score, acc = calm_scores(imu); n = len(score)
    cand = np.argsort(score)[:min(K, n)]
    dirs = acc.mean(1); dirs = dirs / np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-9)
    best, bestv = int(cand[0]), 1e9
    for k in cand:
        up = gyro_up_from(imu, int(k), dirs[k]); others = [c for c in cand if c != k]
        # mean angular disagreement at the other calm windows' centres (weighted by their calmness)
        err = [np.degrees(np.arccos(np.clip(np.dot(up[c * W + W // 2], dirs[c]), -1, 1))) for c in others]
        w = np.array([1.0 / (1e-3 + score[c]) for c in others]); v = float((np.array(err) * w).sum() / w.sum()) if others else 0.0
        if v < bestv: best, bestv = int(k), v
    return best, bestv


fam = sys.argv[1] if len(sys.argv) > 1 else 'racing'
files = sorted(glob.glob(f'{DATA}/train/drone/*.npz')) + sorted(glob.glob(f'{DATA}/val/drone/*.npz'))
rows = []
for f in files:
    tid = Path(f).stem; num = int(tid[-4:])
    if fam == 'racing' and not (tid.startswith('drone_train') and 19 <= num <= 42): continue
    d = np.load(f); imu = d['imu'].astype(np.float32); R_ext = ext[tid][0] if tid in ext else np.eye(3)
    upg = gt_up_frames(imu, d['quat'], R_ext)
    score, acc = calm_scores(imu); k_cur = int(np.argmin(score)); k_con, v = consistency_anchor(imu)
    def tilt(k):
        a0 = acc[k].mean(0); a0 = a0 / np.linalg.norm(a0); return float(np.degrees(np.arccos(np.clip(np.dot(a0, upg[k * W + W // 2]), -1, 1))))
    rows.append((tid, k_cur, tilt(k_cur), k_con, tilt(k_con), v, len(score)))
    print(f'{tid} n {len(score):3d} | current k {k_cur:3d} tilt {tilt(k_cur):5.1f} | consistency k {k_con:3d} tilt {tilt(k_con):5.1f} (disagree {v:4.1f})', flush=True)
t_cur = np.array([r[2] for r in rows]); t_con = np.array([r[4] for r in rows])
print(f'\n{len(rows)} recordings: anchor tilt error  current mean {t_cur.mean():.2f} median {np.median(t_cur):.2f} >5deg {(t_cur > 5).sum()} | consistency mean {t_con.mean():.2f} median {np.median(t_con):.2f} >5deg {(t_con > 5).sum()}')

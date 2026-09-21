#!/usr/bin/env python3
"""Road 1 stage 3 (2026-09-20): per-recording gyro -> accelerometer frame rotation targets.
For every drone recording with GT (train+val) find the rotation R (3 params) that, applied to
the gyro, makes the rest-anchored gyro integration reproduce the accelerometer-frame GT up
(gtup.gt_up_frames). Optimised with torch autograd on top of attnet.integrate (fast), from two
inits (identity, the (-y,-x,-z) map), best kept. Writes runs/calib/gyro_frame_targets.csv:
traj, rx, ry, rz (rotvec), angle_deg, resid_deg (mean up error after), resid0_deg (before), init.
Targets with resid_deg < 8 are used by train_calnet (--frame)."""
import glob, sys, os
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from unified.attnet import integrate, so3_exp
from unified.gtup import gt_up_frames, load_extrinsics
from unified.calnet import R_MAP
DATA = os.environ.get('TARTANIMU_DATA', str(ROOT / 'data/tartan-imu-challenge-iros2026'))
ext = load_extrinsics(str(ROOT / 'runs/extrinsics.npz'))
torch.set_num_threads(4)

def rest_index(imu):
    n = len(imu) // 200; acc = imu[:n * 200, :3].reshape(n, 200, 3); gyr = imu[:n * 200, 3:6].reshape(n, 200, 3)
    an = np.linalg.norm(acc, axis=2); sc = np.abs(an.mean(1) - 9.81) + np.linalg.norm(gyr, axis=2).mean(1) + 0.5 * an.std(1)
    return int(np.argmin(sc))

def up_error(rv, om, up0, s0, up_gt):
    R = so3_exp(rv)                                                   # gyro -> acc
    Rt = integrate((om @ R.T)[None], 1 / 200.0, torch.zeros(1, 3), torch.zeros(1, om.shape[0], 3), s0)[0]
    up = (Rt.transpose(-1, -2) @ up0[:, None])[..., 0]                # up_t = R_t^T up0
    return (1 - (up * up_gt).sum(-1)).mean(), up

def fit(imu, up_gt, init):
    k0 = rest_index(imu); s0 = k0 * 200 + 100
    om = torch.from_numpy(imu[:, 3:6].astype(np.float32)); ug = torch.from_numpy(up_gt.astype(np.float32))
    a0 = imu[k0 * 200:(k0 + 1) * 200, :3].mean(0); up0 = torch.from_numpy((a0 / np.linalg.norm(a0)).astype(np.float32))
    rv = torch.tensor(init, dtype=torch.float32, requires_grad=True); opt = torch.optim.Adam([rv], lr=0.05)
    best = (1e9, None)
    for it in range(120):
        opt.zero_grad(); loss, _ = up_error(rv, om, up0, s0, ug); loss.backward(); opt.step()
        if float(loss) < best[0]: best = (float(loss), rv.detach().clone())
    with torch.no_grad():
        _, up = up_error(best[1], om, up0, s0, ug); ang = torch.rad2deg(torch.acos((up * ug).sum(-1).clamp(-1, 1)))
        _, up_id = up_error(torch.zeros(3), om, up0, s0, ug); ang0 = torch.rad2deg(torch.acos((up_id * ug).sum(-1).clamp(-1, 1)))
    return best[1].numpy(), float(ang.mean()), float(ang0.mean())

rows = []
files = sorted(glob.glob(f'{DATA}/train/drone/*.npz')) + sorted(glob.glob(f'{DATA}/val/drone/*.npz'))
only = set(sys.argv[1:])
from scipy.spatial.transform import Rotation as Rot
rv_map = Rot.from_matrix(R_MAP).as_rotvec()
for f in files:
    tid = Path(f).stem
    if only and tid not in only: continue
    d = np.load(f); imu = d['imu'].astype(np.float32); R_ext = ext[tid][0] if tid in ext else np.eye(3)
    up_gt = gt_up_frames(imu, d['quat'], R_ext)
    inits = {'identity': [0.0, 0.0, 0.0], 'map': list(rv_map), 'x180': [np.pi, 0, 0], 'y180': [0, np.pi, 0], 'z180': [0, 0, np.pi]}
    cands = {k: fit(imu, up_gt, v) for k, v in inits.items()}
    init = min(cands, key=lambda k: cands[k][1]); rv, res, res0 = cands[init]
    rows.append(dict(traj=tid, rx=rv[0], ry=rv[1], rz=rv[2], angle_deg=float(np.degrees(np.linalg.norm(rv))), resid_deg=res, resid0_deg=res0, init=init))
    print(f'{tid} angle {rows[-1]["angle_deg"]:6.1f} deg  up err {res0:6.2f} -> {res:6.2f}  ({init})', flush=True)
import pandas as pd
out = ROOT / 'runs/calib/gyro_frame_targets.csv'
if only: out = ROOT / 'runs/calib/gyro_frame_targets_partial.csv'
pd.DataFrame(rows).to_csv(out, index=False); print('wrote', out, len(rows))

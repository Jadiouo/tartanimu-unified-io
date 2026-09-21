#!/usr/bin/env python3
"""CalNet gate (learned-INS step C): GT-attitude integration oracle before / after the
predicted calibration on held-out recordings. v_w = integral of (R_b2w f' + g) from the rest
anchor (anchor velocity 0), compared with GT body velocity per 1-s window, by time since the
anchor. Gate: bad family (0019/0021/0023/0035/0041) 0-10 s AVE from 7-42 down to < 3; good
family not worse; map class correct on the other source (0101/0129...).
  python local_eval/calnet_check.py --weights runs/calnet/calnet_s42.pt [--recs ...]"""
import argparse, glob, os, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from unified.calnet import CalNet, calnet_apply_numpy
from unified.dense import quat_to_R_torch
from unified.updir import rest_reference
ap = argparse.ArgumentParser(); ap.add_argument('--weights', required=True)
ap.add_argument('--recs', nargs='*', default=['drone_train_0019', 'drone_train_0021', 'drone_train_0023', 'drone_train_0035', 'drone_train_0041',
                                              'drone_train_0022', 'drone_train_0024', 'drone_train_0037', 'drone_train_0039', 'drone_train_0101', 'drone_train_0129'])
ap.add_argument('--data', default=os.environ.get('TARTANIMU_DATA', str(ROOT / 'data/tartan-imu-challenge-iros2026')))
a = ap.parse_args(); dev = torch.device('cpu')
c = torch.load(a.weights, map_location=dev, weights_only=False); m = CalNet(**c['config']); m.load_state_dict(c['state_dict']); m.eval()
ext = np.load(ROOT / 'runs/extrinsics.npz', allow_pickle=True); E = {str(t): (R, int(s)) for t, R, s in zip(ext['traj_id'], ext['R'], ext['sig'])}
G = np.array([0, 0, -9.81])

def oracle(imu, quat, vb, R_ext, fs=200.0):
    """GT attitude, IMU accelerometer -> per-window AVE by time since the rest anchor."""
    R = quat_to_R_torch(torch.from_numpy(quat).double()).numpy()            # body->world (label frame)
    f = imu[:, :3].astype(np.float64) @ R_ext.T                              # IMU frame -> label frame
    a_w = np.einsum('nij,nj->ni', R, f) + G
    n = len(imu) // 200
    acc = imu[:n * 200, :3].reshape(n, 200, 3); gyr = imu[:n * 200, 3:6].reshape(n, 200, 3)
    an = np.linalg.norm(acc, axis=2); k0 = int(np.argmin(np.abs(an.mean(1) - 9.81) + np.linalg.norm(gyr, axis=2).mean(1) + 0.5 * an.std(1)))
    s0 = k0 * 200 + 100; dt = 1 / fs; v = np.zeros_like(a_w)
    v[s0 + 1:] = np.cumsum(a_w[s0:-1], axis=0) * dt; v[:s0] = -np.cumsum(a_w[s0 - 1::-1], axis=0)[::-1] * dt
    v_b = np.einsum('nji,nj->ni', R, v)                                      # R^T v
    vw = v_b[:n * 200].reshape(n, 200, 3).mean(1); gw = vb[:n * 200].reshape(n, 200, 3).mean(1)
    err = np.linalg.norm(vw - gw, axis=1); t = np.abs(np.arange(n) - k0)
    return [err[(t < 10)].mean(), err[(t >= 10) & (t < 20)].mean() if (t >= 10).any() else np.nan]

print(f'{"rec":18s} sig  {"raw 0-10":>8s} {"10-20":>6s} | {"cal 0-10":>8s} {"10-20":>6s} | S / b / tau / map')
for r in a.recs:
    fl = glob.glob(f'{a.data}/*/*/{r}.npz')
    if not fl: print(r, 'missing'); continue
    d = np.load(fl[0]); imu = d['imu']; R_ext, sg = E.get(r, (np.eye(3), -1))
    raw = oracle(imu, d['quat'], d['vel_body'], R_ext)
    cal, p = calnet_apply_numpy(m, imu)
    # the map class changes the gyro only; the oracle uses GT attitude, so it tests S/b/tau
    after = oracle(cal, d['quat'], d['vel_body'], R_ext)
    print(f'{r:18s} {sg:3d}  {raw[0]:8.2f} {raw[1]:6.2f} | {after[0]:8.2f} {after[1]:6.2f} | {np.round(p["S"], 3)} {np.round(p["b"], 2)} {p["tau"]:.0f} {p["map"]} frame {p.get("frame_deg", 0):.1f}deg  (true map {0 if sg == 6 else 1})')

#!/usr/bin/env python3
"""Learned-INS step A.1 gate (2026-09-19): the dead-reckoned velocity channel (updir.dr_channels,
pure gyro attitude from the rest anchor) against GT body velocity, per recording, by time since
the anchor. Gate (good family 0037/0039/0022/0024): 0-10 s AVE < 1.0, 10-20 s < 2.5.
Also checks rest_frame_gyro reproduces rest_up_gyro."""
import sys, glob
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from unified.updir import dr_channels, rest_frame_gyro, rest_up_gyro, rest_reference
DATA = ROOT / 'data/tartan-imu-challenge-iros2026'
recs = sys.argv[1:] or ['drone_train_0037', 'drone_train_0039', 'drone_train_0022', 'drone_train_0024', 'drone_train_0101']
print(f'{"rec":18s} {"anchor(s)":>9s} {"0-10s":>7s} {"10-20s":>7s} {"20s+":>7s} {"nwin":>5s} {"up err":>7s}')
for r in recs:
    d = np.load(glob.glob(f'{DATA}/*/drone/{r}.npz')[0]); imu = d['imu']; vb = d['vel_body']
    R, k0 = rest_frame_gyro(imu); ch = dr_channels(imu, R=R, k0=k0)
    rr = rest_reference(imu); u0 = rr[:3] / np.linalg.norm(rr[:3])
    up_chk = np.einsum('nji,j->ni', R, u0); up_ref = rest_up_gyro(imu)
    uperr = float(np.degrees(np.arccos(np.clip((up_chk * up_ref).sum(1), -1, 1))).max())
    n = len(imu) // 200; v = ch[:n * 200, :3].reshape(n, 200, 3).mean(1); g = vb[:n * 200].reshape(n, 200, 3).mean(1)
    t = np.abs(np.arange(n) - k0)             # windows since anchor
    err = np.linalg.norm(v - g, axis=1)
    b = [err[(t >= 0) & (t < 10)].mean(), err[(t >= 10) & (t < 20)].mean() if (t >= 10).any() else np.nan, err[t >= 20].mean() if (t >= 20).any() else np.nan]
    print(f'{r:18s} {k0:9d} {b[0]:7.2f} {b[1]:7.2f} {b[2]:7.2f} {n:5d} {uperr:7.3f}')

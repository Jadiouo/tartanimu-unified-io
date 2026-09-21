#!/usr/bin/env python3
"""AttNet gate (learned-INS step B): tilt error of the estimated up vs GT up on held-out
recordings. Gate: fast flights median < 5 deg, p90 < 10 deg; other source (0101/0129) < 15.
  python local_eval/attnet_check.py --weights runs/attnet/attnet_s42.pt [--recs ...]
Prints pure-gyro (rest_up_gyro) and slow-filter baselines alongside."""
import argparse, glob, sys, os
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from unified.attnet import AttNet
from unified.updir import rest_up_gyro, complementary_up
from unified.gtup import gt_up_frames, load_extrinsics
ap = argparse.ArgumentParser(); ap.add_argument('--weights', required=True)
ap.add_argument('--recs', nargs='*', default=['drone_train_0037', 'drone_train_0039', 'drone_train_0021', 'drone_train_0022', 'drone_train_0023', 'drone_train_0024', 'drone_train_0101', 'drone_train_0129'])
ap.add_argument('--calnet', default='')
ap.add_argument('--data', default=os.environ.get('TARTANIMU_DATA', str(ROOT / 'data/tartan-imu-challenge-iros2026')))
a = ap.parse_args(); dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
c = torch.load(a.weights, map_location=dev, weights_only=False); m = AttNet(**c['config']).to(dev); m.load_state_dict(c['state_dict']); m.eval()
ext = load_extrinsics(str(ROOT / 'runs/extrinsics.npz'))
cal = None
if a.calnet:
    from unified.calnet import CalNet, calnet_apply_numpy
    cc = torch.load(a.calnet, map_location='cpu', weights_only=False); cal = CalNet(**cc['config']); cal.load_state_dict(cc['state_dict']); cal.eval()
def ang(u, v): return np.degrees(np.arccos(np.clip((u * v).sum(1) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-9), -1, 1)))
print(f'{"rec":18s} {"att med":>8s} {"att p90":>8s} | {"gyro med":>8s} {"gyro p90":>8s} | {"slow med":>8s} {"slow p90":>8s}')
for r in a.recs:
    f = glob.glob(f'{a.data}/*/*/{r}.npz')
    if not f: print(r, 'missing'); continue
    d = np.load(f[0]); imu = d['imu'].astype(np.float32); R_ext = ext[r][0] if r in ext else np.eye(3)
    gt = gt_up_frames(imu, d['quat'], R_ext)
    if cal is not None: imu = calnet_apply_numpy(cal, imu)[0]
    with torch.no_grad(): up = m(torch.from_numpy(imu[None]).to(dev))[0][0].cpu().numpy()
    e_att = ang(up, gt); e_gy = ang(rest_up_gyro(imu), gt); e_sl = ang(complementary_up(imu, 0.001), gt)
    print(f'{r:18s} {np.median(e_att):8.2f} {np.percentile(e_att, 90):8.2f} | {np.median(e_gy):8.2f} {np.percentile(e_gy, 90):8.2f} | {np.median(e_sl):8.2f} {np.percentile(e_sl, 90):8.2f}')

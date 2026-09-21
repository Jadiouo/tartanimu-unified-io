#!/usr/bin/env python3
"""Label-free gyro-frame audit for the drone families (2026-09-20 night). For each recording:
rest-anchored pure-gyro integration of 'up' under three gyro maps (identity, -I = all three rates
negated, R_MAP = CalNet's class-1 map) and AttNet+CalNet's current up; error = angle to the
low-passed accelerometer direction on calm frames (|a| within 0.5 m/s^2 of g, |w| < 0.5),
0-20 s from the anchor. No labels. Prints per recording and per family (sig) medians."""
import glob, sys, os
from pathlib import Path
import numpy as np, torch, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from unified.attnet import integrate, AttNet
from unified.calnet import R_MAP, CalNet, _smooth
from unified.updir import derived_channels
DATA = os.environ.get('TARTANIMU_DATA', str(ROOT / 'data/tartan-imu-challenge-iros2026')); W = 200
torch.set_num_threads(4)
ext = np.load(ROOT / 'runs/extrinsics.npz', allow_pickle=True); SIG = {str(t): int(s) for t, s in zip(ext['traj_id'], ext['sig'])}


def anchor(imu):
    n = len(imu) // W; acc = imu[:n * W, :3].reshape(n, W, 3); gyr = imu[:n * W, 3:6].reshape(n, W, 3); an = np.linalg.norm(acc, axis=2)
    s = np.abs(an.mean(1) - 9.81) + np.linalg.norm(gyr, axis=2).mean(1) + 0.5 * an.std(1); k = int(np.argmin(s)); return k, acc[k].mean(0)


def up_gyro(imu, M, k0, up0):
    om = torch.from_numpy((imu[:, 3:6] @ M.T).astype(np.float32)); s0 = k0 * W + W // 2
    Rt = integrate(om[None], 1 / 200.0, torch.zeros(1, 3), torch.zeros(1, om.shape[0], 3), s0)[0]
    return (Rt.transpose(-1, -2) @ torch.from_numpy((up0 / np.linalg.norm(up0)).astype(np.float32))[:, None])[..., 0].numpy()


def err(up, imu, k0):
    a = _smooth(imu[:, :3].astype(np.float64), 25); an = np.linalg.norm(a, axis=1); w = np.linalg.norm(_smooth(imu[:, 3:6].astype(np.float64), 25), axis=1)
    t = np.abs(np.arange(len(imu)) - (k0 * W + W // 2)) / 200.0
    calm = (np.abs(an - 9.81) < 0.5) & (w < 0.5) & (t > 1.0) & (t < 20.0)
    if calm.sum() < 50: return np.nan
    u = a[calm] / an[calm, None]; return float(np.degrees(np.arccos(np.clip((u * up[calm]).sum(1), -1, 1))).mean())


if __name__ == '__main__':
    att = cal = None
    if '--sub' in sys.argv:
        c = torch.load(ROOT / 'runs/attnet/attnet_final.pt', map_location='cpu', weights_only=False); att = AttNet(**c['config']); att.load_state_dict(c['state_dict']); att.eval()
        c = torch.load(ROOT / 'runs/calnet/calnet_final.pt', map_location='cpu', weights_only=False); cal = CalNet(**c['config']); cal.load_state_dict(c['state_dict']); cal.eval()
    files = sorted(glob.glob(f'{DATA}/train/drone/*.npz')) + sorted(glob.glob(f'{DATA}/val/drone/*.npz'))
    step = int([a for a in sys.argv[1:] if a.isdigit()][0]) if any(a.isdigit() for a in sys.argv[1:]) else 1
    rows = []
    for f in files[::step]:
        tid = Path(f).stem; imu = np.load(f)['imu'].astype(np.float32); k0, up0 = anchor(imu)
        r = dict(traj=tid, sig=SIG.get(tid, -1))
        for name, M in [('I', np.eye(3)), ('negI', -np.eye(3)), ('map', R_MAP)]:
            r[name] = err(up_gyro(imu, M, k0, up0), imu, k0)
        if att is not None:
            ch = derived_channels(imu, att, cal); r['attnet'] = err(ch[:, :3], imu, k0)
        rows.append(r); print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    df = pd.DataFrame(rows); df.to_csv(ROOT / 'runs/calib/gyro_map_check.csv', index=False)
    print(df.groupby('sig')[[c for c in df.columns if c not in ('traj', 'sig')]].median().round(1)); print(df.groupby('sig').size())

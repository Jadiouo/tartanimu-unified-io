#!/usr/bin/env python3
"""Ruler 2 (2026-09-18): does a checkpoint read the drag signal on the test sprint?

  python local_eval/drag_sensitivity.py --weights unified/ckpt_<tag>_last.pt [--ids test_0028 test_0055]

For each test recording, scales the horizontal (perpendicular-to-slow-up) part of
the accelerometer by 0.5 and 1.5 and reports the predicted speed p90/max.
Reference = the recording's rest attitude (2026-09-19 correction). v6 ratio ~1.0-1.2 on
test_0028, S-fast models 1.3-1.4 on test_0055. No ground truth involved."""
import os; os.environ.pop('TARTANIMU_CACHE', None)   # temp frames share window ids: never cache them
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from unified.data import build_windows
from unified.train_v2 import Wrapped, predict
from unified.updir import N_CHANNELS, build_features, up_windows_for

ap = argparse.ArgumentParser(); ap.add_argument('--weights', required=True)
ap.add_argument('--ids', nargs='+', default=['test_0028', 'test_0055', 'test_0036', 'test_0037'])
ap.add_argument('--scales', nargs='+', type=float, default=[0.5, 1.0, 1.5])
ap.add_argument('--data', default=str(ROOT / 'data/tartan-imu-challenge-iros2026'))
a = ap.parse_args()
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
c = torch.load(a.weights, map_location=dev, weights_only=False); ar = c['args']
model = Wrapped(ar['model'], False, 0, False, ar['mixer_hidden'], ar['width'], ar['dropout'], ar['trunk'], ar['fine'], ar['moe'],
                ar['n_experts'], ar['decompose'], N_CHANNELS[ar['grav']], False, ar.get('continuous', False), ar.get('traj_film', False),
                ar.get('freq', False), ar.get('norm', 'bn'), ar.get('preint', False), ar.get('wide', False), ar.get('wide_to_gru', False),
                ar.get('wide_span', 5), ar.get('at', 'none'), ar.get('lr0', 'none'), ar.get('head', 'linear'), ar.get('drag_head', False), ar.get('fuse', False), ar.get('ekf', False), ar.get('ekf_feats', False)).to(dev)
getattr(model.net, 'adapt_tstat', lambda *a: None)(c['model'])
model.load_state_dict(c['model'], strict=True); model.eval()
att_m = cal_m = None; att_key = ''
if 'attnet' in c or 'calnet' in c:                                  # learned-INS B/C sub-modules stored in the checkpoint
    import hashlib
    if 'attnet' in c:
        from unified.attnet import AttNet
        att_m = AttNet(**c['attnet']['config']).to(dev); att_m.load_state_dict(c['attnet']['state_dict']); att_m.eval()
    if 'calnet' in c:
        from unified.calnet import CalNet
        cal_m = CalNet(**c['calnet']['config']).to(dev); cal_m.load_state_dict(c['calnet']['state_dict']); cal_m.eval()
    att_key = hashlib.md5(open(a.weights, 'rb').read()).hexdigest()[:8]
mean = torch.as_tensor(c['imu_mean'], device=dev); std = torch.as_tensor(c['imu_std'], device=dev); grav = ar['grav']; K = ar['seg_len']
from unified import updir as _updir
_updir.DR_OPTS.update(horizon=ar.get('dr_horizon', 0.0), squash=ar.get('dr_squash', 0.0))
from unified.train_v2 import EKF_OPTS
EKF_OPTS.update(dv=ar.get('ekf_dv', 'end'), conf=ar.get('ekf_conf', 0.0), horizon=ar.get('ekf_horizon', 0.0), dR=ar.get('ekf_dR', 'gyro'))
from unified.fine_context import EKF_RUNTIME
EKF_RUNTIME['mask_direct'] = bool(ar.get('ekf_mask_direct', False))
from unified.moe import TRAJ_DESC
TRAJ_DESC['version'] = ar.get('traj_desc', 1)
def norm(x, up=None): return ((build_features(x, up, grav) - mean) / std).transpose(-1, -2).contiguous(), None
tmp = ROOT / 'runs/drag_sensitivity_tmp'; tmp.mkdir(exist_ok=True)
def rest_up(d):
    imu = d['imu']; n = len(imu) // 200; acc = imu[:n * 200, :3].reshape(n, 200, 3); gyr = imu[:n * 200, 3:].reshape(n, 200, 3)
    an = np.linalg.norm(acc, axis=2); sc = np.abs(an.mean(1) - 9.81) + np.linalg.norm(gyr, axis=2).mean(1) + 0.5 * an.std(1)
    u = acc[int(np.argmin(sc))].mean(0); return u / np.linalg.norm(u)
def scale_h(d, s):
    """scale the part of the accelerometer perpendicular to the recording's REST attitude
    (2026-09-19 correction: the slow-up reference removed the DC drag/tilt component)."""
    up = rest_up(d); f = d['imu'][:, :3]; par = (f @ up)[:, None] * up[None, :]
    e = dict(d); e['imu'] = d['imu'].copy(); e['imu'][:, :3] = (par + s * (f - par)).astype(np.float32); return e
def run(name, d):
    p = tmp / f'{name}.npz'; np.savez(p, **d); n = len(d['imu']) // 200
    fr = pd.DataFrame({'window_id': np.arange(n), 'traj_id': name, 'win_idx': np.arange(n), 'file_path': str(p)})
    X = torch.from_numpy(np.asarray(build_windows(fr))).to(dev); U = torch.from_numpy(np.asarray(up_windows_for(fr, grav, att_m, cal_m, att_key, device=str(dev)))).to(dev)
    with torch.inference_mode(): v = np.asarray(predict(model, X, U, fr, ar['model'], K, norm))
    p.unlink(); return np.linalg.norm(v, axis=1)
print(f'{Path(a.weights).name}: horizontal specific-force scaling -> predicted speed (p90 / max)')
for tid in a.ids:
    d = dict(np.load(f'{a.data}/test/{tid}.npz')); out = []
    base = None
    for s in a.scales:
        sp = run(f'{tid}_{s}', scale_h(d, s)); p90 = np.percentile(sp, 90); base = p90 if s == 1.0 else base
        out.append(f'x{s}: {p90:5.2f} / {sp.max():5.2f}')
    ps = {s: np.percentile(run(f'{tid}_{s}', scale_h(d, s)), 90) for s in (1.0, 1.5)}
    print(f'  {tid}: ' + ' | '.join(out) + f'   sensitivity p90(x1.5)/p90(x1.0) = {ps[1.5] / ps[1.0]:.2f}')

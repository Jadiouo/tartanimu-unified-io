#!/usr/bin/env python3
"""Train CalNet (learned-INS step C) on released train (or train+val) recordings.
Targets: drone recordings from runs/calib/calib_targets.csv (S, b, gyro_lag_ms from GT fits)
and the extrinsic family (map class: sig 6 -> 0, else 1); car/dog/human: S=1, b=0, tau=0, map=0.
  python unified/train_calnet.py --tag s42 [--trainval] [--exclude ...] [--steps 3000]
Writes runs/calnet/calnet_<tag>.pt and runs/calnet/summaries_<tag>.npz (features cache)."""
import argparse, glob, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from unified.calnet import CalNet, summary
ap = argparse.ArgumentParser()
ap.add_argument('--tag', required=True); ap.add_argument('--trainval', action='store_true')
ap.add_argument('--exclude', nargs='*', default=[]); ap.add_argument('--steps', type=int, default=3000)
ap.add_argument('--targets', default=str(ROOT / 'runs/calib/calib_targets.csv'))
ap.add_argument('--data', default=os.environ.get('TARTANIMU_DATA', str(ROOT / 'data/tartan-imu-challenge-iros2026')))
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--frame', action='store_true', help='stage 3: 6D rotation head (targets runs/calib/gyro_frame_targets.csv, resid < 8 deg)')
a = ap.parse_args(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ext = np.load(ROOT / 'runs/extrinsics.npz', allow_pickle=True); sig = {str(t): int(s) for t, s in zip(ext['traj_id'], ext['sig'])}
tg = pd.read_csv(a.targets).set_index('traj')
splits = ['train', 'val'] if a.trainval else ['train']
X, Y, names = [], [], []
t0 = time.time()
for sp in splits:
    for f in sorted(glob.glob(f'{a.data}/{sp}/*/*.npz')):
        tid = Path(f).stem
        if tid in set(a.exclude): continue
        imu = np.load(f)['imu']
        X.append(summary(imu)); names.append(tid)
        # fitted S/b/tau targets are meaningful only for the identity family (sig 6: the GT fit
        # is in the accelerometer frame); the other families get the map target only (mask = 0)
        if tid in tg.index and sig.get(tid, 6) == 6:
            r = tg.loc[tid]; y = [r.S_x, r.S_y, r.S_z, r.b_x, r.b_y, r.b_z, r.gyro_lag_ms, 0.0, 1.0]
        elif sig.get(tid, 6) == 6:
            y = [1, 1, 1, 0, 0, 0, 0, 0.0, 1.0]                                   # car/dog/human: identity, no calibration
        else:
            y = [1, 1, 1, 0, 0, 0, 0, 1.0, 0.0]
        Y.append(y)
X = np.stack(X).astype(np.float32); Y = np.array(Y, np.float32)
Rt = np.tile(np.eye(3, dtype=np.float32), (len(X), 1, 1)); Rm = np.zeros(len(X), np.float32)
if a.frame:
    from scipy.spatial.transform import Rotation as Rot
    ft = pd.read_csv(ROOT / 'runs/calib/gyro_frame_targets.csv').set_index('traj')
    for i, n in enumerate(names):
        if n in ft.index and ft.loc[n, 'resid_deg'] < 8:
            Rt[i] = Rot.from_rotvec(ft.loc[n, ['rx', 'ry', 'rz']].values.astype(float)).as_matrix(); Rm[i] = 1.0
        elif n not in ft.index and sig.get(n, 6) == 6:
            Rm[i] = 1.0                                                                          # car/dog/human: identity target
    print(f'frame targets: {int(Rm.sum())} recordings (drones with resid < 8 deg + identity platforms)', flush=True)
print(f'{len(X)} recordings ({splits}), {int(Y[:, 7].sum())} map-class-1, drones with fitted targets {sum(n in tg.index for n in names)}  ({time.time() - t0:.0f}s features)', flush=True)
out = ROOT / 'runs/calnet'; out.mkdir(exist_ok=True, parents=True)
np.savez(out / f'summaries_{a.tag}.npz', X=X, Y=Y, names=np.array(names))
model = CalNet(frame=a.frame).to(dev)
model.mu.copy_(torch.from_numpy(X.mean(0))); model.sd.copy_(torch.from_numpy(X.std(0) + 1e-6))
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
Xt = torch.from_numpy(X).to(dev); Yt = torch.from_numpy(Y).to(dev); Rtt = torch.from_numpy(Rt).to(dev); Rmt = torch.from_numpy(Rm).to(dev)
for step in range(1, a.steps + 1):
    idx = torch.from_numpy(rng.choice(len(X), min(64, len(X)), replace=False)).to(dev)
    mo = model(Xt[idx]); S, b, tau, ml = mo[:4]; y = Yt[idx]
    mk = y[:, 8:9]
    l_frame = 0.0
    if a.frame:
        Rp = mo[4]; tr = (Rp.transpose(1, 2) @ Rtt[idx]).diagonal(dim1=1, dim2=2).sum(-1)      # trace(R^T R*)
        l_frame = (((3.0 - tr) / 2.0) * Rmt[idx]).sum() / Rmt[idx].sum().clamp(min=1)         # 1 - cos(angle)
    loss = ((S - y[:, 0:3]).abs() * mk).sum() / mk.sum().clamp(min=1) * 5 + ((b - y[:, 3:6]).abs() * mk).sum() / mk.sum().clamp(min=1) \
        + ((tau[:, 0] - y[:, 6]).abs() * mk[:, 0]).sum() / mk.sum().clamp(min=1) / 20 \
        + torch.nn.functional.binary_cross_entropy_with_logits(ml, y[:, 7]) + 2.0 * l_frame
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if step % 500 == 0 or step == 1:
        with torch.no_grad():
            mo = model(Xt); S, b, tau, ml = mo[:4]; acc = float(((ml > 0).float() == Yt[:, 7]).float().mean()); mk = Yt[:, 8] > 0
            fr = ''
            if a.frame:
                tr = (mo[4].transpose(1, 2) @ Rtt).diagonal(dim1=1, dim2=2).sum(-1); ang = torch.rad2deg(torch.acos(((tr - 1) / 2).clamp(-1, 1)))
                fr = f' frame err {float(ang[Rmt > 0].mean()):.1f} deg (n={int(Rmt.sum())})'
            print(f'step {step:5d} loss {float(loss):.4f} | S mae {float((S - Yt[:, 0:3]).abs()[mk].mean()):.4f} b mae {float((b - Yt[:, 3:6]).abs()[mk].mean()):.4f} tau mae {float((tau[:, 0] - Yt[:, 6]).abs()[mk].mean()):.1f} ms map acc {acc:.3f}  (masked n={int(mk.sum())}){fr}', flush=True)
torch.save({'state_dict': model.state_dict(), 'config': {'in_dim': 64, 'hidden': 128, 'frame': bool(a.frame)}, 'args': vars(a)}, out / f'calnet_{a.tag}.pt')
print('saved', out / f'calnet_{a.tag}.pt')

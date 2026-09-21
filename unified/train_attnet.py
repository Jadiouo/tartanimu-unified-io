#!/usr/bin/env python3
"""Train AttNet (learned-INS step B) on released train (or train+val) recordings, all platforms.
Target: GT up in the IMU frame (gtup.gt_up_frames). Loss: mean angle(up, up_GT) + 0.1 |b_g|_1
+ 0.01 |ddot|_1. 8 recordings per step, whole recordings, AdamW 1e-3.
  python unified/train_attnet.py --tag s42 [--trainval] [--exclude drone_train_0037 ...] [--steps 2000]
Writes runs/attnet/attnet_<tag>.pt (state_dict + config)."""
import argparse, glob, os, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from unified.attnet import AttNet
from unified.gtup import gt_up_frames, load_extrinsics
ap = argparse.ArgumentParser()
ap.add_argument('--tag', required=True); ap.add_argument('--trainval', action='store_true')
ap.add_argument('--exclude', nargs='*', default=[]); ap.add_argument('--steps', type=int, default=2000)
ap.add_argument('--batch', type=int, default=8); ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--seed', type=int, default=42); ap.add_argument('--max_frames', type=int, default=60000, help='crop longer recordings (random crop containing the rest window)')
ap.add_argument('--calnet', default='', help='apply this CalNet (S, b, tau, gyro-frame map) to the IMU first (learned-INS order C -> B)')
ap.add_argument('--skip_deg', type=float, default=30.0, help='drop recordings whose pure-gyro up error median exceeds this (unfixable frame)')
ap.add_argument('--limit', type=int, default=0, help='smoke: use only the first N recordings')
ap.add_argument('--data', default=os.environ.get('TARTANIMU_DATA', str(ROOT / 'data/tartan-imu-challenge-iros2026')))
a = ap.parse_args()
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
ext = load_extrinsics(os.environ.get('TARTANIMU_EXTRINSICS', str(ROOT / 'runs/extrinsics.npz')))
splits = ['train', 'val'] if a.trainval else ['train']
files = [f for sp in splits for f in sorted(glob.glob(f'{a.data}/{sp}/*/*.npz')) if Path(f).stem not in set(a.exclude)]
if a.limit: files = files[::max(1, len(files) // a.limit)][:a.limit]
cal = None
if a.calnet:
    from unified.calnet import CalNet, calnet_apply_numpy
    cc = torch.load(a.calnet, map_location='cpu', weights_only=False); cal = CalNet(**cc['config']); cal.load_state_dict(cc['state_dict']); cal.eval()
from unified.updir import rest_up_gyro
recs = []; skipped = 0
for f in files:
    d = np.load(f); imu = d['imu'].astype(np.float32); tid = Path(f).stem
    R_ext = ext[tid][0] if tid in ext else np.eye(3)
    up = gt_up_frames(imu, d['quat'], R_ext)
    if cal is not None: imu = calnet_apply_numpy(cal, imu)[0]
    if a.skip_deg > 0:
        ug = rest_up_gyro(imu); e = np.degrees(np.arccos(np.clip((ug * up).sum(1), -1, 1)))
        if np.median(e) > a.skip_deg: skipped += 1; continue
    recs.append((tid, imu, up))
print(f'{len(recs)} recordings ({splits}), excluded {len(a.exclude)}, skipped (unfixable frame) {skipped}, calnet={bool(cal)}; device {dev}', flush=True)
model = AttNet().to(dev); opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
t0 = time.time(); run = 0.0
for step in range(1, a.steps + 1):
    opt.zero_grad(set_to_none=True); tot = 0.0
    for j in rng.choice(len(recs), a.batch, replace=False):
        tid, imu, up_gt = recs[j]
        if len(imu) > a.max_frames:                       # crop, keeping the rest window inside
            k0 = int(model.rest_window(torch.from_numpy(imu[None]).to(dev))[0]); c = k0 * 200 + 100
            lo = int(np.clip(rng.integers(max(0, c - a.max_frames + 400), c - 200 + 1), 0, len(imu) - a.max_frames))
            lo = (lo // 200) * 200; imu = imu[lo:lo + a.max_frames]; up_gt = up_gt[lo:lo + a.max_frames]
        x = torch.from_numpy(imu[None]).to(dev); y = torch.from_numpy(up_gt[None]).to(dev)
        up, R, aux = model(x)
        cos = (up * y).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6)
        loss = torch.acos(cos).mean() + 0.1 * aux['b_g'].abs().mean() + 0.01 * aux['ddot'].abs().mean()
        (loss / a.batch).backward(); tot += float(loss) / a.batch
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    run = 0.98 * run + 0.02 * tot if step > 1 else tot
    if step % 50 == 0 or step == 1:
        print(f'step {step:5d}  loss {tot:.4f}  ema {run:.4f}  ({np.degrees(run):.2f} deg)  {time.time() - t0:.0f}s', flush=True)
out = ROOT / 'runs/attnet'; out.mkdir(exist_ok=True, parents=True)
torch.save({'state_dict': model.state_dict(), 'config': {'hidden': 64, 'pool': 10, 'block': 5}, 'args': vars(a)}, out / f'attnet_{a.tag}.pt')
print('saved', out / f'attnet_{a.tag}.pt')

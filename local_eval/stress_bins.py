#!/usr/bin/env python3
"""Stress-fold report: drone score plus the bins val cannot see.

  python local_eval/stress_bins.py --tag <tag> [--fold runs/stress_fold.json]

Reads local_eval/sub_hold_<tag>_last.csv (the trainer's hold-out prediction of
the fold), scores it with the official scorer, then reports
  * AVE and predicted/true speed ratio by ground-truth speed bin (compression),
  * AVE by recording class: high-rotation (gyro_mean > 1.5 rad/s), fast
    (sp_p90 > 6 m/s), rest,
so the tail groups A/B (speed) and C (rotation) are visible separately."""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT), str(ROOT / 'local_eval')]
from build_val_solution import build as build_solution
from score_val import score_breakdown
from tta_time import DATA

ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--fold', default='runs/stress_fold.json')
ap.add_argument('--csv', default=None); a = ap.parse_args()
fold = json.load(open(ROOT / a.fold)); ids = fold['held']
sub = pd.read_csv(a.csv or ROOT / f'local_eval/sub_hold_{a.tag}_last.csv')
sol = build_solution(DATA, 'train', ids)
score, table = score_breakdown(sol, sub); dr = table[table.platform == 'drone'].iloc[0]
print(f'{a.tag}: fold {len(ids)} recordings  drone score {dr.score:.4f}  AVE {dr.AVE:.4f}  ATE20 {dr.ATE20:.4f}')
m = sol.merge(sub, on='window_id')
vt = np.sqrt(m.vx_gt ** 2 + m.vy_gt ** 2 + m.vz_gt ** 2); vp = np.sqrt(m.vx ** 2 + m.vy ** 2 + m.vz ** 2)
err = np.sqrt((m.vx_gt - m.vx) ** 2 + (m.vy_gt - m.vy) ** 2 + (m.vz_gt - m.vz) ** 2)
bins = [0, 2, 4, 6, 8, 10, 99]
print('  by true speed (m/s):  n   AVE   pred/true')
for lo, hi in zip(bins[:-1], bins[1:]):
    k = (vt >= lo) & (vt < hi)
    if k.sum(): print(f'    {lo:2d}-{hi if hi < 99 else "  ":<3}   {int(k.sum()):5d} {err[k].mean():.3f}   {vp[k].mean() / vt[k].mean():.3f}')
st = pd.read_csv(ROOT / 'runs/stress_fold/drone_stats_2026-09-17.csv').set_index('traj')
m['err'] = err.values
cls = {t: ('hi-rot' if st.loc[t, 'gyro_mean'] > 1.5 else 'fast' if st.loc[t, 'sp_p90'] > 6 else 'rest') for t in ids}
m['cls'] = m.traj_id.map(cls)
print('  by recording class:   n_rec  AVE (mean of per-recording AVE)')
for c in ('hi-rot', 'fast', 'rest'):
    per = m[m.cls == c].groupby('traj_id').err.mean()
    if len(per): print(f'    {c:7s}            {len(per):3d}    {per.mean():.3f}')

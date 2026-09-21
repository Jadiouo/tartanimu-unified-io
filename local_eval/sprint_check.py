#!/usr/bin/env python3
"""Ruler 1 (2026-09-18): fold-A sprint recordings 0037-0042 per-recording AVE and the
> 8 m/s compression ratio for a stress-fold checkpoint's hold prediction.
  python local_eval/sprint_check.py --tag sf_sfast18_s42 [--tag sf_base_s42 ...]"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT), str(ROOT / 'local_eval')]
from build_val_solution import build as build_solution
from tta_time import DATA
ap = argparse.ArgumentParser(); ap.add_argument('--tag', action='append', required=True); ap.add_argument('--fold', default='runs/stress_fold.json'); a = ap.parse_args()
ids = json.load(open(ROOT / a.fold))['held']; sol = build_solution(DATA, 'train', ids)
sprint = [t for t in [f'drone_train_{i:04d}' for i in range(37, 43)] if t in ids]   # only the held-out sprint recordings
rows = []
for tag in a.tag:
    m = sol.merge(pd.read_csv(ROOT / f'local_eval/sub_hold_{tag}_last.csv'), on='window_id')
    y = m[['vx_gt', 'vy_gt', 'vz_gt']].values; yh = m[['vx', 'vy', 'vz']].values
    sy = np.linalg.norm(y, axis=1); syh = np.linalg.norm(yh, axis=1); err = np.linalg.norm(yh - y, axis=1); m['err'] = err
    per = m[m.traj_id.isin(sprint)].groupby('traj_id').err.mean()
    fast = sy > 8
    rows.append({'tag': tag, **{t[-4:]: round(per.get(t, np.nan), 2) for t in sprint}, 'sprint_mean': round(per.mean(), 2),
                 'AVE>8': round(err[fast].mean(), 2), 'pred/true>8': round(syh[fast].mean() / sy[fast].mean(), 2),
                 'radial>8': round((syh - sy)[fast].mean(), 2), 'fold_AVE': round(m.groupby('traj_id').err.mean().mean(), 3)})
print(pd.DataFrame(rows).to_string(index=False))

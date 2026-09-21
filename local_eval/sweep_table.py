#!/usr/bin/env python3
"""One-line-per-run stress-fold table (fold score, speed-bin compression, class AVE) for every
sub_hold_*_last.csv found locally or in runs/sweep_results/."""
import glob, json, os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path[:0] = [ROOT, ROOT + '/local_eval']
from build_val_solution import build as build_solution
from score_val import score_breakdown
from tta_time import DATA
ids = json.load(open(ROOT + '/runs/stress_fold.json'))['held']; sol = build_solution(DATA, 'train', ids)
st = pd.read_csv(ROOT + '/runs/stress_fold/drone_stats_2026-09-17.csv').set_index('traj')
cls = {t: ('hi-rot' if st.loc[t, 'gyro_mean'] > 1.5 else 'fast' if st.loc[t, 'sp_p90'] > 6 else 'rest') for t in ids}
files = sorted(glob.glob(ROOT + '/local_eval/sub_hold_sf_*_last.csv')) + sorted(glob.glob(ROOT + '/runs/sweep_results/sub_hold_sf_*_last.csv')) + sorted(glob.glob(ROOT + '/runs/sweep_results/m*/sub_hold_sf_*_last.csv'))
print(f"{'run':20s} {'where':5s} fold   AVE   ATE  | r6-8 r8-10 r10+ | hi-rot fast")
for f in files:
    tag = os.path.basename(f)[9:-9]; where = 'm1' if '/m1/' in f else 'm2' if '/m2/' in f else 'cloud' if 'sweep_results' in f else 'local'
    sub = pd.read_csv(f)
    if len(sub) != len(sol): continue
    _, table = score_breakdown(sol, sub); dr = table[table.platform == 'drone'].iloc[0]
    m = sol.merge(sub, on='window_id'); vt = np.sqrt(m.vx_gt**2 + m.vy_gt**2 + m.vz_gt**2); vp = np.sqrt(m.vx**2 + m.vy**2 + m.vz**2)
    m['err'] = np.sqrt((m.vx_gt - m.vx)**2 + (m.vy_gt - m.vy)**2 + (m.vz_gt - m.vz)**2); m['cls'] = m.traj_id.map(cls)
    r = lambda lo, hi: vp[(vt >= lo) & (vt < hi)].mean() / vt[(vt >= lo) & (vt < hi)].mean()
    per = m.groupby('traj_id').err.mean(); c = lambda k: per[[t for t in per.index if cls[t] == k]].mean()
    print(f"{tag:20s} {where:5s} {dr.score:.3f} {dr.AVE:.3f} {dr.ATE20:.3f} | {r(6,8):.2f} {r(8,10):.2f}  {r(10,99):.2f} | {c('hi-rot'):.3f} {c('fast'):.3f}")

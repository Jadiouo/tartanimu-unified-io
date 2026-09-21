#!/usr/bin/env python3
"""Submit a test CSV to the organisers' scoring Space through its Gradio API and
save overall / platform / per-sequence tables under runs/official/<tag>_*.csv."""
import argparse, sys, hashlib, shutil
from pathlib import Path
import pandas as pd
from gradio_client import Client, handle_file
ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument('--csv', required=True); ap.add_argument('--tag', required=True); ap.add_argument('--team', default='Lexxxxx')
a = ap.parse_args()
md5 = hashlib.md5(open(a.csv, 'rb').read()).hexdigest()
c = Client('Tartan-IMU/imu_odometry_challenge_scoring', verbose=False)
res = c.predict(handle_file(a.csv), a.team, api_name='/run')
out = ROOT / 'runs/official'; out.mkdir(exist_ok=True)
def df(x):
    if isinstance(x, dict) and 'data' in x: return pd.DataFrame(x['data'], columns=x.get('headers'))
    return None
names = ['overall', 'platform', 'sequences']
for name, x in zip(names, res[:3]):
    d = df(x)
    if d is not None: d.to_csv(out / f'{a.tag}_{name}.csv', index=False); print(f'== {name}\n{d.to_string(index=False)}')
if len(res) > 3 and isinstance(res[3], str) and Path(res[3]).exists(): shutil.copy(res[3], out / f'{a.tag}_sequences_file.csv')
if len(res) > 4: print('== message\n', str(res[4])[:600])
print('md5', md5)

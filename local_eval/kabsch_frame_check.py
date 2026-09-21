#!/usr/bin/env python3
"""advice_after_v20 §5 (2026-09-20 night): closed-form gyro -> accelerometer frame rotation per
recording, no labels. For the low-passed accelerometer direction u(t) (a world-fixed vector seen
from the body when the specific force is gravity-dominated) du/dt = -(R w) x u = [u]x R w. That is
linear in the 9 entries of R: stack rows [u]x (x) w^T, solve least squares, project on SO(3)
(SVD, det +1). Frames with |a| far from g or fast rotation are down-weighted. Compares with the
identity, the class-1 map and, where available, the autograd targets (runs/calib/gyro_frame_targets.csv)
and the GT-attitude oracle."""
import glob, sys, os
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from unified.calnet import R_MAP, _smooth
DATA = os.environ.get('TARTANIMU_DATA', str(ROOT / 'data/tartan-imu-challenge-iros2026'))


def kabsch_frame(imu, fs=200.0, n_smooth=50, iters=3):
    """(N, 6) -> R (3, 3) gyro -> accelerometer frame, residual before/after (relative)."""
    acc = imu[:, :3].astype(np.float64); gyr = imu[:, 3:6].astype(np.float64)
    u = _smooth(acc, n_smooth); un = np.linalg.norm(u, axis=1, keepdims=True); u = u / np.maximum(un, 1e-6)
    w = _smooth(gyr, n_smooth); du = np.gradient(u, axis=0) * fs
    wn = np.linalg.norm(w, axis=1)
    # weights: gravity-dominated (|a| within 30 % of g), some rotation (else no information), not too fast
    wt = np.exp(-((un[:, 0] - 9.81) / 3.0) ** 2) * (wn > 0.3) * (wn < 6.0)
    ok = wt > 0.05
    if ok.sum() < 200: return np.eye(3), np.nan, np.nan
    U = np.zeros((len(u), 3, 3)); U[:, 0, 1] = -u[:, 2]; U[:, 0, 2] = u[:, 1]; U[:, 1, 0] = u[:, 2]; U[:, 1, 2] = -u[:, 0]; U[:, 2, 0] = -u[:, 1]; U[:, 2, 1] = u[:, 0]
    # du = [u]x R w  ->  du_i = sum_jk U_ij R_jk w_k  -> A (N*3, 9) with A[(n,i),(j,k)] = U_ij w_k
    A = (U[:, :, :, None] * w[:, None, None, :]).reshape(len(u), 3, 9)
    R = np.eye(3)
    for _ in range(iters):                                                  # IRLS: down-weight outliers
        res = np.linalg.norm(du - np.einsum('nij,nj->ni', U, w @ R.T), axis=1)
        s = np.median(res[ok]) + 1e-6; wi = wt / (1 + (res / (3 * s)) ** 2)
        Aw = A[ok] * np.sqrt(wi[ok])[:, None, None]; bw = du[ok] * np.sqrt(wi[ok])[:, None]
        x, *_ = np.linalg.lstsq(Aw.reshape(-1, 9), bw.reshape(-1), rcond=None)
        M = x.reshape(3, 3); Uu, _, Vt = np.linalg.svd(M); R = Uu @ np.diag([1, 1, np.sign(np.linalg.det(Uu @ Vt))]) @ Vt
    def rel(Rc):
        r = np.linalg.norm(du - np.einsum('nij,nj->ni', U, w @ Rc.T), axis=1)
        return float((r * wt)[ok].sum() / ((np.linalg.norm(du, axis=1) * wt)[ok].sum() + 1e-9))
    return R, rel(np.eye(3)), rel(R)


if __name__ == '__main__':
    from scipy.spatial.transform import Rotation as Rot
    tg = pd.read_csv(ROOT / 'runs/calib/gyro_frame_targets.csv').set_index('traj') if (ROOT / 'runs/calib/gyro_frame_targets.csv').exists() else None
    recs = sys.argv[1:] or ['drone_train_0037', 'drone_train_0039', 'drone_train_0040', 'drone_train_0022', 'drone_train_0019', 'drone_train_0129', 'drone_train_0100', 'drone_train_0050', 'drone_train_0200', 'car_train_0005', 'human_train_0003']
    print(f'{"rec":18s} {"res id":>6s} {"res map":>7s} {"res fit":>7s} {"fit deg":>7s} {"vs map":>6s} {"vs target":>9s}')
    for r in recs:
        fl = glob.glob(f'{DATA}/*/*/{r}.npz')
        if not fl: print(r, 'missing'); continue
        imu = np.load(fl[0])['imu']; R, r0, r1 = kabsch_frame(imu)
        # residual under the class-1 map for reference
        _, _, rm = (R_MAP, None, None), None, None
        from local_eval.kabsch_frame_check import kabsch_frame as _k  # noqa (self)
        ang = np.degrees(Rot.from_matrix(R).magnitude()); dmap = np.degrees(Rot.from_matrix(R @ R_MAP.T).magnitude())
        dt = ''
        if tg is not None and r in tg.index:
            Rt = Rot.from_rotvec(tg.loc[r, ['rx', 'ry', 'rz']].values.astype(float)).as_matrix()
            dt = f'{np.degrees(Rot.from_matrix(R @ Rt.T).magnitude()):6.1f} (target resid {tg.loc[r, "resid_deg"]:.1f})'
        # residual under the map alone
        acc = imu[:, :3].astype(np.float64); gyr = imu[:, 3:6].astype(np.float64)
        print(f'{r:18s} {r0:6.2f} {"-":>7s} {r1:7.2f} {ang:7.1f} {dmap:6.1f} {dt}')

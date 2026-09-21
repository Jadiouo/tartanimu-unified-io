#!/usr/bin/env python3
"""Per-frame gravity direction in the body frame, from IMU alone.

The body frame is not inertial, so averaging the accelerometer over a segment to
find "up" only works while the platform holds its orientation: measured against
ground truth, a 20 s mean is 0.48 deg for car but 8.31 deg for human, which
turns and averages gravity across orientations. A complementary filter fixes
that by propagating the estimate with the gyroscope and correcting it towards
the accelerometer.

Its one constant is a real trade-off, and the platforms want opposite ends of it
(median error against ground-truth attitude, one trajectory each):

    alpha       0.05 (0.1 s)   0.005 (1 s)   0.002 (2.5 s)
    car             2.62           0.89          1.02
    dog             0.82           0.53          0.60
    human           3.07           0.71          0.77
    drone           0.80           3.82         10.31

A slow filter rejects motion acceleration, which is what car, dog and human
need; a fast one keeps up with attitude that changes quickly, which is what the
drone needs, and lag -- not accelerometer noise -- is what ruins the slow filter
there. Rather than pick, or route on a platform label the test split does not
carry, both are computed and handed to the network as channels: strictly more
information, no tuning, and nothing that has to name a platform.

Cost is about 5 minutes over the 16.4 M training frames, once, then cached
alongside the decoded splits.
"""
from __future__ import annotations

import numpy as np

FAST, SLOW = 0.05, 0.005


def complementary_up(imu: np.ndarray, alpha: float, fs: float = 200.0) -> np.ndarray:
    """(N, 6) IMU -> (N, 3) unit vector pointing along +gravity in the body frame.

    "Up" here is the direction the accelerometer reads at rest, i.e. the reaction
    to gravity, which is the physical convention the data follows (checked in
    unified/gravity.py: 0.39 deg against ground truth on the slowest windows).
    """
    a = np.ascontiguousarray(imu[:, 0:3], dtype=np.float64)
    w = np.ascontiguousarray(imu[:, 3:6], dtype=np.float64)
    out = np.empty((len(a), 3))
    dt = 1.0 / fs
    n0 = np.linalg.norm(a[0])
    u = a[0] / n0 if n0 > 1e-6 else np.array([0.0, 0.0, 1.0])

    for k in range(len(a)):
        if k:
            # gravity rotates the opposite way to the body over one step
            th = w[k - 1] * dt
            ang = np.sqrt(th @ th)
            if ang > 1e-9:
                ax = th / ang
                c, s = np.cos(ang), np.sin(ang)
                u = u * c - np.cross(ax, u) * s + ax * (ax @ u) * (1.0 - c)
        n = np.sqrt(a[k] @ a[k])
        if n > 1e-6:
            u = (1.0 - alpha) * u + alpha * (a[k] / n)
            u /= np.sqrt(u @ u)
        out[k] = u
    return out


ADAPT_ALPHA, ADAPT_SIGMA = 0.005, 1.0


def adaptive_up(imu: np.ndarray, alpha: float = ADAPT_ALPHA, sigma: float = ADAPT_SIGMA,
                fs: float = 200.0) -> np.ndarray:
    """Complementary filter whose accelerometer gain is gated by |a| - g.

    On a multirotor the specific force is the thrust, i.e. along the body z axis
    whatever the pitch, so the accelerometer cannot see a sustained tilt: the
    slow filter above is off by a median 20 deg on the aggressive hold-out
    flights (5 deg at alpha 0.001, but that drifts to 15 deg on calm val flights
    where the gyro bias is not corrected). Gating the correction by
    exp(-((|a| - g) / sigma)^2) trusts the accelerometer while the body is
    unaccelerated and the gyro while it is not: 7 deg on the hold-out flights,
    within 1-2 deg of the slow filter everywhere else (measured 2026-09-15).
    """
    import math
    a = np.ascontiguousarray(imu[:, 0:3], dtype=np.float64)
    w = np.ascontiguousarray(imu[:, 3:6], dtype=np.float64)
    an = np.sqrt((a * a).sum(1))
    gain = alpha * np.exp(-((an - 9.81) / sigma) ** 2)
    out = np.empty((len(a), 3))
    dt = 1.0 / fs
    ux, uy, uz = (a[0] / an[0]) if an[0] > 1e-6 else (0.0, 0.0, 1.0)
    al, wl, anl, gl = a.tolist(), w.tolist(), an.tolist(), gain.tolist()
    for k in range(len(al)):
        if k:
            tx, ty, tz = wl[k - 1]; tx *= dt; ty *= dt; tz *= dt
            ang = math.sqrt(tx * tx + ty * ty + tz * tz)
            if ang > 1e-9:
                ax, ay, az = tx / ang, ty / ang, tz / ang
                c, sn = math.cos(ang), math.sin(ang)
                d = ax * ux + ay * uy + az * uz
                cx, cy, cz = ay * uz - az * uy, az * ux - ax * uz, ax * uy - ay * ux
                ux, uy, uz = (ux * c - cx * sn + ax * d * (1 - c), uy * c - cy * sn + ay * d * (1 - c),
                              uz * c - cz * sn + az * d * (1 - c))
        n = anl[k]
        if n > 1e-6:
            g = gl[k]; x, y, z = al[k]
            ux, uy, uz = (1 - g) * ux + g * x / n, (1 - g) * uy + g * y / n, (1 - g) * uz + g * z / n
            m = math.sqrt(ux * ux + uy * uy + uz * uz); ux /= m; uy /= m; uz /= m
        out[k, 0] = ux; out[k, 1] = uy; out[k, 2] = uz
    return out


def both_up(imu: np.ndarray, fs: float = 200.0) -> np.ndarray:
    """(N, 6) -> (N, 6) = fast estimate concatenated with slow estimate."""
    return np.concatenate([complementary_up(imu, FAST, fs),
                           complementary_up(imu, SLOW, fs)], axis=1).astype(np.float32)


def decomposition(imu, up):
    """Re-express the IMU about a gravity direction, as extra channels.

    imu (..., T, 6) and up (..., T, 3) -> (..., T, 8):
      a . up            vertical specific force: gravity plus vertical motion
      a - (a . up) up   horizontal specific force: motion only
      w . up            yaw rate, the rotation gravity is invariant to
      w - (w . up) up   pitch and roll rate

    Nothing is lost -- the original six channels reconstruct from these -- but the
    split that matters physically is made explicit rather than left implicit, and
    it degrades gracefully: at a few degrees of error the parallel part is still
    mostly gravity and the perpendicular part still mostly motion.
    """
    import torch
    a, w = imu[..., 0:3], imu[..., 3:6]
    a_par = (a * up).sum(-1, keepdim=True)
    w_par = (w * up).sum(-1, keepdim=True)
    return torch.cat([a_par, a - a_par * up, w_par, w - w_par * up], dim=-1)


def _up_of_file(path):
    import numpy as np
    with np.load(path) as d:
        return both_up(d["imu"])


def up_for_files(paths):
    """{file_path: both_up(imu)} for every path, computed in parallel across
    recordings (the filter is a recursion in time, so the only parallelism is
    across files).  Each result is the same function applied to the same array
    as the serial loop, so the cached arrays are bit-identical either way.
    TARTANIMU_UP_WORKERS=1 restores the serial path."""
    import os
    from multiprocessing import get_context
    paths = list(dict.fromkeys(paths))
    n = int(os.environ.get("TARTANIMU_UP_WORKERS", "0") or 0) or min(os.cpu_count() or 1, 16)
    if n <= 1 or len(paths) < 4:
        return {p: _up_of_file(p) for p in paths}
    with get_context("fork").Pool(n) as pool:
        return dict(zip(paths, pool.map(_up_of_file, paths, chunksize=1)))


def _restgyro_windows(frame, window_size: int = 200) -> np.ndarray:
    import numpy as np
    from unified.data import cache_path

    cache = cache_path(frame, "uprestg")
    if cache is not None and cache.exists():
        return np.load(cache, mmap_mode="r")
    X = np.empty((len(frame), window_size, 3), dtype=np.float32)
    for file_path, rows in frame.groupby("file_path", sort=False):
        with np.load(file_path) as d:
            ug = rest_up_gyro(d["imu"], window_size)
        starts = rows["win_idx"].to_numpy() * window_size
        X[rows.index.to_numpy()] = np.stack([ug[s:s + window_size] for s in starts])
    if cache is not None:
        np.save(cache, X)
    return X


def build_up_windows(frame, window_size: int = 200, adapt: bool = False, rest: bool = False, restgyro: bool = False, dr: bool = False, att=None, cal=None, att_key: str = "", device="cpu") -> np.ndarray:
    """(len(frame), window_size, 6) gravity estimates, mirroring data.build_windows.

    Cached the same way and keyed the same way, so val and test pay the filter
    once per machine rather than once per run. adapt=True appends the gated
    filter as columns 6:9 (its own cache, so the existing ones stay valid).
    """
    import numpy as np
    from unified.data import cache_path

    if adapt:
        return np.concatenate([np.asarray(build_up_windows(frame, window_size)),
                               np.asarray(_adaptive_windows(frame, window_size))], axis=-1)
    if rest:                                              # columns 6:10 = rest reference (own cache)
        parts = [np.asarray(build_up_windows(frame, window_size)), np.asarray(_rest_windows(frame, window_size))]
        if dr:                                            # columns 10:27 = [rest up (3), v_DR (3), t_since (1), R 6D (6), v_F0 (3), calib dev (1)] from derived_channels
            parts.append(np.asarray(_att_windows(frame, att, cal, att_key or "gyro", window_size, device)))
            return np.concatenate(parts, axis=-1)
        if restgyro:                                      # columns 10:13 = rest up carried by the gyro
            parts.append(np.asarray(_restgyro_windows(frame, window_size)))
        if dr:                                            # columns 13:17 = v_DR (3) + t_since_anchor (1)
            parts.append(np.asarray(_dr_windows(frame, window_size)))
        return np.concatenate(parts, axis=-1)
    cache = cache_path(frame, "up")
    if cache is not None and cache.exists():
        return np.load(cache, mmap_mode="r")

    X = np.empty((len(frame), window_size, 6), dtype=np.float32)
    ups = up_for_files(frame["file_path"].tolist())
    for file_path, rows in frame.groupby("file_path", sort=False):
        up = ups[file_path]
        starts = rows["win_idx"].to_numpy() * window_size
        X[rows.index.to_numpy()] = np.stack([up[s:s + window_size] for s in starts])
    if cache is not None:
        np.save(cache, X)
    return X


N_CHANNELS = {"none": 6, "slow": 14, "fast": 14, "both": 22, "adapt": 14, "slowrest": 18, "slowrest2": 21, "slowdr": 25}


def rest_reference(imu, window_size: int = 200):
    """Per-recording rest reference (2026-09-19, single-model version of the sprint
    readout): the accelerometer mean over the calmest one-second window (|f| near g,
    low gyro, low |f| spread) and that window's calmness score. (N, 6) -> (4,) float32:
    [f_rest (3), score]. No parameters, no labels; a mount-tilt reference so that
    "tilt since rest" (the drag/tilt speed cue of a straight sprint) is separable from
    a tilted mount at rest."""
    import numpy as np
    n = len(imu) // window_size
    if n < 1:
        a = imu[:, :3].mean(0); return np.array([a[0], a[1], a[2], 9.0], np.float32)
    acc = imu[:n * window_size, :3].reshape(n, window_size, 3); gyr = imu[:n * window_size, 3:6].reshape(n, window_size, 3)
    an = np.linalg.norm(acc, axis=2); gm = np.linalg.norm(gyr, axis=2).mean(1)
    score = np.abs(an.mean(1) - 9.81) + gm + 0.5 * an.std(1)
    k = int(np.argmin(score)); f = acc[k].mean(0)
    return np.array([f[0], f[1], f[2], score[k]], np.float32)


def rest_up_gyro(imu, window_size: int = 200, fs: float = 200.0):
    """(N, 6) -> (N, 3): the rest window's up direction carried to every frame by gyro
    integration (forward and backward from the rest window). Pure attitude change since
    rest -- immune to acceleration transients, drift ~1-2 deg over tens of seconds. The
    gyro is taken in the accelerometer frame (identity mapping); for a source whose gyro
    is in another frame the channel is consistently wrong and the network learns to
    ignore it (the rest score and the raw channels are still there)."""
    import numpy as np
    from scipy.spatial.transform import Rotation as Rot
    n = len(imu) // window_size
    r = rest_reference(imu, window_size)
    k = 0
    if n >= 1:
        acc = imu[:n * window_size, :3].reshape(n, window_size, 3); gyr = imu[:n * window_size, 3:6].reshape(n, window_size, 3)
        an = np.linalg.norm(acc, axis=2); score = np.abs(an.mean(1) - 9.81) + np.linalg.norm(gyr, axis=2).mean(1) + 0.5 * an.std(1)
        k = int(np.argmin(score))
    u0 = r[:3] / max(float(np.linalg.norm(r[:3])), 1e-6)
    s0 = k * window_size + window_size // 2
    dt = 1.0 / fs; om = imu[:, 3:6].astype(np.float64)
    dR = Rot.from_rotvec(om * dt).as_matrix()                # R_{i -> i+1} increments (body frame)
    out = np.empty((len(imu), 3), np.float64); out[s0] = u0
    u = u0.copy()
    for i in range(s0 + 1, len(imu)):                       # forward: u_{i} = dR_{i-1}^T u_{i-1}
        u = dR[i - 1].T @ u; out[i] = u
    u = u0.copy()
    for i in range(s0 - 1, -1, -1):                          # backward: u_{i} = dR_{i} u_{i+1}
        u = dR[i] @ u; out[i] = u
    return out.astype(np.float32)


def rest_frame_gyro(imu, window_size: int = 200, fs: float = 200.0):
    """(N, 6) -> (R (N, 3, 3) float64, k0): the full rotation carried from the rest window
    by gyro integration (forward and backward), R[t] = R_{t -> F0}: a vector in the body
    frame at t expressed in the anchor body frame F0 (the rest window's frame). Same
    increments and anchor as rest_up_gyro: up_t = R[t]^T up_0 reproduces it (learned-INS
    step A, 2026-09-19)."""
    import numpy as np
    from scipy.spatial.transform import Rotation as Rot
    n = len(imu) // window_size
    k = 0
    if n >= 1:
        acc = imu[:n * window_size, :3].reshape(n, window_size, 3); gyr = imu[:n * window_size, 3:6].reshape(n, window_size, 3)
        an = np.linalg.norm(acc, axis=2); score = np.abs(an.mean(1) - 9.81) + np.linalg.norm(gyr, axis=2).mean(1) + 0.5 * an.std(1)
        k = int(np.argmin(score))
    s0 = k * window_size + window_size // 2
    dt = 1.0 / fs; om = imu[:, 3:6].astype(np.float64)
    dR = Rot.from_rotvec(om * dt).as_matrix()                # R_{i -> i+1} increments
    N = len(imu); R = np.empty((N, 3, 3), np.float64); R[s0] = np.eye(3)
    cur = np.eye(3)
    for i in range(s0 + 1, N):                              # forward: R_{t->F0} = dR_{s0} ... dR_{t-1}
        cur = cur @ dR[i - 1]; R[i] = cur
    cur = np.eye(3)
    for i in range(s0 - 1, -1, -1):                          # backward: inverse of the increments between t and s0
        cur = cur @ dR[i].T; R[i] = cur
    return R, k


# two_paths 2026-09-19 path 2 (car-leak stops), set by the trainer / predictor from the run's args:
#   horizon (s): v_DR zeroed beyond this time from the anchor (t_since kept); 0 = off
#   squash (m/s): v_DR -> squash * tanh(v / squash); 0 = off
DR_OPTS = {"horizon": 0.0, "squash": 0.0}


def dr_suffix():
    h, q = DR_OPTS["horizon"], DR_OPTS["squash"]
    return ("" if h <= 0 else f"_h{h:g}") + ("_q8" if q <= 0 else f"_q{q:g}")     # squash is always on (default 8)


def dr_channels(imu, window_size: int = 200, fs: float = 200.0, R=None, k0=None, up0=None, with_vf0: bool = False):
    """(N, 6) -> (N, 4) float32: v_DR (3, body frame, m/s) and t_since_anchor (1, seconds / 10,
    signed). Dead-reckoned velocity from the rest anchor (anchor velocity zero): in the anchor
    frame F0, a_F0(t) = R[t] f(t) - 9.81 up0 (f = a - g, g = -9.81 up0 in F0), v_F0 integrated
    forward and backward from the anchor window centre, v_DR(t) = R[t]^T v_F0(t). R defaults
    to rest_frame_gyro (pure gyro attitude); step B passes a learned one. Parameter-free,
    label-free, whole-recording, IMU-only (learned-INS step A, 2026-09-19)."""
    import numpy as np
    if R is None:
        R, k0 = rest_frame_gyro(imu, window_size, fs)
    N = len(imu); s0 = k0 * window_size + window_size // 2
    f = imu[:, :3].astype(np.float64)
    if up0 is None:
        a0 = f[k0 * window_size:(k0 + 1) * window_size].mean(0)   # rest specific force = +9.81 up0 in F0
        up0 = a0 / max(float(np.linalg.norm(a0)), 1e-6)
    up0 = np.asarray(up0, np.float64) / max(float(np.linalg.norm(up0)), 1e-6)
    a_F0 = np.einsum('nij,nj->ni', R, f) - 9.81 * up0
    dt = 1.0 / fs
    v = np.zeros((N, 3), np.float64)
    if s0 + 1 < N:
        v[s0 + 1:] = np.cumsum(a_F0[s0:N - 1], axis=0) * dt          # forward trapezoid-free Euler
    if s0 > 0:
        v[:s0] = -np.cumsum(a_F0[s0 - 1::-1], axis=0)[::-1] * dt      # backward
    v_dr = np.einsum('nji,nj->ni', R, v)                              # R^T v
    t_since = ((np.arange(N) - s0) / fs / 10.0)[:, None]
    # review 1.1 (2026-09-20): always bounded -- q * tanh(v / q), q = --dr_squash (default 8 m/s);
    # unbounded v_DR (car/human drift of 10^3-10^4 m/s) made the data-driven normalisation of the
    # channel ~3000 and the sprint signal invisible to the trunk
    q = DR_OPTS["squash"] if DR_OPTS["squash"] > 0 else 8.0
    v_dr = q * np.tanh(v_dr / q)
    if DR_OPTS["horizon"] > 0:
        v_dr[np.abs(np.arange(N) - s0) > DR_OPTS["horizon"] * fs] = 0.0
    out = np.concatenate([v_dr, t_since], axis=1).astype(np.float32)
    if with_vf0:
        # advice_after_v20 §2.1: the unbounded anchor-frame integral v_F0(t); its difference
        # between two window ends, rotated into the later body frame, is the exact IMU-only
        # velocity increment of that window (no tanh distortion, no single-frame attitude noise
        # entering twice; drift cancels in the difference)
        return out, v.astype(np.float32)
    return out


def _dr_windows(frame, window_size: int = 200) -> np.ndarray:
    import numpy as np
    from unified.data import cache_path

    cache = cache_path(frame, "updr" + dr_suffix())
    if cache is not None and cache.exists():
        return np.load(cache, mmap_mode="r")
    X = np.empty((len(frame), window_size, 4), dtype=np.float32)
    for file_path, rows in frame.groupby("file_path", sort=False):
        with np.load(file_path) as d:
            ch = dr_channels(d["imu"], window_size)
        starts = rows["win_idx"].to_numpy() * window_size
        X[rows.index.to_numpy()] = np.stack([ch[s:s + window_size] for s in starts])
    if cache is not None:
        np.save(cache, X)
    return X


def derived_channels(imu, att=None, cal=None, window_size: int = 200, fs: float = 200.0, device="cpu"):
    """(N, 6) raw IMU -> (N, 17) float32 = [rest up carried by the attitude (3), v_DR (3), t_since (1),
    R_{t->F0} first two columns (6), v_F0 (3), calibration deviation (1)].
    Learned-INS steps B/C: `cal` (CalNet) calibrates the IMU first (S, b, tau, gyro-frame map),
    `att` (AttNet) replaces the pure-gyro rotation and the anchor up. Both None -> identical to
    rest_up_gyro + dr_channels on the raw IMU."""
    import numpy as np
    x = imu; conf = 0.0
    if cal is not None:
        from unified.calnet import calnet_apply_numpy
        x, prm = calnet_apply_numpy(cal, imu, device=device)
        conf = calib_deviation(prm)
    if att is not None:
        from unified.attnet import att_rotation_numpy
        R, up0, k0 = att_rotation_numpy(att, x, device=device)
        ug = np.einsum('nji,j->ni', R, up0).astype(np.float32)           # up_t = R^T up0'
        dc, vf0 = dr_channels(x, window_size, fs, R=R, k0=k0, up0=up0, with_vf0=True)
    else:
        R, k0 = rest_frame_gyro(x, window_size, fs)
        ug = rest_up_gyro(x, window_size, fs)
        dc, vf0 = dr_channels(x, window_size, fs, R=R, k0=k0, with_vf0=True)
    # road 1 stage 2: the first two columns of R_{t->F0} (6D, continuous -- review 1.2: a rotation
    # vector wraps at +-pi) so the EKF can form dR_k = R_k^T R_{k-1} from the window-end frames
    r6 = np.concatenate([R[:, :, 0], R[:, :, 1]], axis=1).astype(np.float32)
    # advice_after_v20 §2 (2026-09-20 night): columns 13:16 (of this block) = v_F0 anchor-frame
    # integral for window-integrated increments, column 16 = per-recording calibration deviation
    # (constant; 0 without CalNet) for the EKF confidence mask
    cc = np.full((len(x), 1), conf, np.float32)
    return np.concatenate([ug, dc, r6, vf0, cc], axis=1).astype(np.float32)


def calib_deviation(prm) -> float:
    """CalNet output -> one non-negative scalar, in 'tolerances': max|S-1| / 0.05 + |b| / 0.5 m/s^2
    + map-class uncertainty (1 - |tanh(logit/2)|). 0 = a perfectly clean, confidently classified
    IMU; the EKF confidence mask (--ekf_conf) switches the physics increments off above a threshold."""
    import numpy as np
    S = np.asarray(prm["S"], np.float64).ravel(); b = np.asarray(prm["b"], np.float64).ravel()
    return float(np.abs(S - 1).max() / 0.05 + np.linalg.norm(b) / 0.5 + (1.0 - abs(np.tanh(float(prm.get("logit", 0.0)) / 2.0))))


def _att_windows(frame, att, cal, key: str, window_size: int = 200, device="cpu") -> np.ndarray:
    """columns 10:27 of the up stream computed with AttNet / CalNet; cached under 'upatt_<key>'."""
    import numpy as np
    from unified.data import cache_path

    cache = cache_path(frame, f"upatt_{key}" + dr_suffix() + "_r6i")
    if cache is not None and cache.exists():
        return np.load(cache, mmap_mode="r")
    X = np.empty((len(frame), window_size, 17), dtype=np.float32)
    for file_path, rows in frame.groupby("file_path", sort=False):
        with np.load(file_path) as d:
            ch = derived_channels(d["imu"], att, cal, window_size, device=device)
        starts = rows["win_idx"].to_numpy() * window_size
        X[rows.index.to_numpy()] = np.stack([ch[s:s + window_size] for s in starts])
    if cache is not None:
        np.save(cache, X)
    return X


def up_vector_groups(width: int):
    """Column starts of the 3-vector groups in an up stream of this width; the other
    columns are scalars (rest score, t_since_anchor) and must not be rotated."""
    groups = [0, 3]
    if width >= 9: groups.append(6)          # adapt filter or rest reference
    if width >= 13: groups.append(10)        # rest up carried by the gyro
    if width >= 17: groups.append(13)        # dead-reckoned velocity
    # columns 17:23 (first two columns of R_{t->F0}) and 23:26 (v_F0) are F0-frame quantities, NOT
    # body vectors: the yaw augmentation leaves them and conjugates the EKF increments instead;
    # column 26 (calibration deviation) is a scalar
    return groups


def _rest_windows(frame, window_size: int = 200) -> np.ndarray:
    import numpy as np
    from unified.data import cache_path

    cache = cache_path(frame, "uprest")
    if cache is not None and cache.exists():
        return np.load(cache, mmap_mode="r")
    X = np.empty((len(frame), window_size, 4), dtype=np.float32)
    for file_path, rows in frame.groupby("file_path", sort=False):
        with np.load(file_path) as d:
            r = rest_reference(d["imu"], window_size)
        X[rows.index.to_numpy()] = r[None, None, :]
    if cache is not None:
        np.save(cache, X)
    return X


def _adaptive_windows(frame, window_size: int = 200) -> np.ndarray:
    import numpy as np
    from unified.data import cache_path

    cache = cache_path(frame, "upa")
    if cache is not None and cache.exists():
        return np.load(cache, mmap_mode="r")
    X = np.empty((len(frame), window_size, 3), dtype=np.float32)
    for file_path, rows in frame.groupby("file_path", sort=False):
        with np.load(file_path) as d:
            up = adaptive_up(d["imu"])
        starts = rows["win_idx"].to_numpy() * window_size
        X[rows.index.to_numpy()] = np.stack([up[s:s + window_size] for s in starts])
    if cache is not None:
        np.save(cache, X)
    return X


def up_windows_for(frame, grav: str, att=None, cal=None, att_key: str = "", device="cpu"):
    """The up stream a --grav mode needs (None for 'none'); att/cal = learned-INS B/C models."""
    return None if grav == "none" else build_up_windows(frame, adapt=(grav == "adapt"), rest=grav in ("slowrest", "slowrest2", "slowdr"), restgyro=grav in ("slowrest2", "slowdr"), dr=(grav == "slowdr"), att=att, cal=cal, att_key=att_key, device=device)


def build_features(imu, up6, mode: str):
    """imu (..., T, 6) and up6 (..., T, 6) -> (..., T, N_CHANNELS[mode]).

    The raw six channels are always kept, so "none" is nested inside every other
    mode and a null result means the decomposition genuinely added nothing rather
    than that something was taken away.
    """
    import torch
    if mode == "none":
        return imu
    parts = [imu]
    if mode in ("slow", "both"):
        parts.append(decomposition(imu, up6[..., 3:6]))
    if mode in ("fast", "both"):
        parts.append(decomposition(imu, up6[..., 0:3]))
    if mode == "adapt":                                   # needs the 9-column up stream
        parts.append(decomposition(imu, up6[..., 6:9]))
    if mode in ("slowrest", "slowrest2", "slowdr"):       # 10/13/17-column up stream: slow up + rest reference
        parts.append(decomposition(imu, up6[..., 3:6]))
        parts.append(imu[..., 0:3] - up6[..., 6:9])       # tilt since rest (raw body frame)
        parts.append(up6[..., 9:10])                      # rest-window calmness score
    if mode in ("slowrest2", "slowdr"):                   # rest up carried by gyro integration (pure attitude change)
        parts.append(up6[..., 10:13])
    if mode == "slowdr":                                  # dead-reckoned velocity from the rest anchor + t_since
        parts.append(up6[..., 13:17])
    return torch.cat(parts, dim=-1)


def canonicalize_up(imu, up6):
    """Rotate each window so its (slow) up estimate points along +z.

    imu (..., T, 6), up6 (..., T, 6) -> same shapes, rotated by one rotation per
    window: the minimal rotation taking the window-mean slow up to e_z. Yaw is
    untouched (the rotation axis lies in the x-y plane). When the up is within
    ~1 deg of -z the axis is undefined; a 180-degree turn about x is used, so
    an upside-down mount lands in the same canonical pose every time. Labels
    are not changed; this is input canonicalisation only.
    """
    import torch
    u = up6[..., 3:6].mean(dim=-2)                                   # (..., 3)
    u = u / u.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    z = torch.zeros_like(u); z[..., 2] = 1.0
    axis = torch.cross(u, z, dim=-1)                                  # u x e_z
    s = axis.norm(dim=-1, keepdim=True)
    c = u[..., 2:3]                                                   # u . e_z
    degenerate = s < 1e-3
    xaxis = torch.zeros_like(u); xaxis[..., 0] = 1.0
    k = torch.where(degenerate, xaxis, axis / s.clamp(min=1e-9))
    ang = torch.where(degenerate.squeeze(-1), torch.where(c.squeeze(-1) < 0, torch.full_like(c.squeeze(-1), 3.141592653589793),
                                                          torch.zeros_like(c.squeeze(-1))),
                      torch.atan2(s.squeeze(-1), c.squeeze(-1)))
    ca, sa = torch.cos(ang)[..., None, None], torch.sin(ang)[..., None, None]
    K = torch.zeros(*k.shape[:-1], 3, 3, dtype=imu.dtype, device=imu.device)
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    eye = torch.eye(3, dtype=imu.dtype, device=imu.device)
    Q = eye + sa * K + (1 - ca) * (K @ K)                             # (..., 3, 3)

    def rot(v):                                                       # (..., T, 3)
        return torch.einsum("...ij,...tj->...ti", Q, v)
    imu_c = torch.cat([rot(imu[..., :3]), rot(imu[..., 3:6])], dim=-1)
    up_c = torch.cat([rot(up6[..., :3]), rot(up6[..., 3:6])], dim=-1)
    return imu_c, up_c


def so3_log(R):
    """(..., 3, 3) rotation -> (..., 3) rotation vector, stable near 0 and near pi.

    Away from pi: axis from the antisymmetric part, angle from the trace. Near
    pi the antisymmetric part vanishes, so the axis is taken from the symmetric
    part R + I = 2 k k^T (largest column), with its sign from the antisymmetric
    part where that is still distinguishable.
    """
    import torch
    tr = ((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1) / 2).clamp(-1, 1)
    th = torch.acos(tr)
    ax = torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0],
                      R[..., 1, 0] - R[..., 0, 1]], dim=-1)
    sn = 2 * torch.sin(th)
    small = sn.abs() < 1e-3                                            # near 0 or near pi
    # regular branch (safe denominator); near 0 the vector is ~0 anyway
    reg = ax / sn.clamp(min=1e-3)[..., None] * th[..., None]
    # near-pi branch: k from the symmetric part
    S = R + torch.eye(3, dtype=R.dtype, device=R.device)
    col = S.abs().sum(dim=-2).argmax(dim=-1)                           # (...,) most informative column
    k = torch.gather(S, -1, col[..., None, None].expand(*S.shape[:-1], 1))[..., 0]
    k = k / k.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    sign = torch.where((ax * k).sum(-1, keepdim=True) < 0, -1.0, 1.0)
    near_pi = (k * sign) * th[..., None]
    use_pi = small & (th > 1.0)
    return torch.where(use_pi[..., None], near_pi, torch.where(small[..., None], ax * 0.5, reg))


def preint_features(imu, fs: float = 200.0):
    """Gyro pre-integration inside each window (..., T, 6) -> (..., T, 9).

    R_0t: rotation from frame t back to the window's first frame, from the
    gyro alone; channels are so(3) log of R_0t (3) and the cumulative sum of
    the specific force rotated into the first frame, sum_s R_0s f_s / fs, under
    two coordinate hypotheses: the gyro's own frame (3) and the gyro mapped by
    M0 = [[0,-1,0],[-1,0,0],[0,0,-1]] (3), which is the fixed gyro->accel
    relation of the calmer drone source. Gravity is not removed: these are
    integration features, not a velocity estimate. The network chooses.
    """
    import torch
    a, w = imu[..., :3], imu[..., 3:6]
    lead = imu.shape[:-2]; T = imu.shape[-2]
    a = a.reshape(-1, T, 3).double(); w = w.reshape(-1, T, 3).double()
    M0 = torch.tensor([[0., -1., 0.], [-1., 0., 0.], [0., 0., -1.]], dtype=a.dtype, device=a.device)
    outs = []
    for W in (w, w @ M0.T):
        dt = W / fs                                                # (N, T, 3) rotation vectors
        ang = dt.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        k = dt / ang
        K = torch.zeros(*k.shape[:-1], 3, 3, dtype=a.dtype, device=a.device)
        K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
        K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
        K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
        s_, c_ = torch.sin(ang)[..., None], (1 - torch.cos(ang))[..., None]
        dR = torch.eye(3, dtype=a.dtype, device=a.device) + s_ * K + c_ * (K @ K)   # (N, T, 3, 3)
        R = torch.eye(3, dtype=a.dtype, device=a.device).expand(a.shape[0], 3, 3).clone()
        Rs = []
        for t in range(T):                                          # R_0t = R_0,t-1 @ dR_{t-1}
            Rs.append(R)
            R = R @ dR[:, t]
        Rs = torch.stack(Rs, dim=1)                                 # (N, T, 3, 3)
        fr = torch.einsum("ntij,ntj->nti", Rs, a)                   # specific force in frame 0
        dv = torch.cumsum(fr, dim=1) / fs
        if not outs:                                                # so(3) log of R_0t, gyro frame only
            outs.append(so3_log(Rs))
        outs.append(dv)
    return torch.cat(outs, dim=-1).reshape(*lead, T, 9).to(imu.dtype)

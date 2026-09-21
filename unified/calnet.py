"""CalNet (learned-INS step C, 2026-09-19): per-recording IMU calibration from a fixed
64-dim recording summary (IMU only, whole recording, no labels at inference):
  -> accelerometer scale S (3), bias b (3, m/s^2), gyro time offset tau (ms), gyro-frame map
     class logit (0 = identity, 1 = the other drone source's (-y, -x, -z) frame).
Applied before AttNet / dr_channels: acc' = (acc - b) / S, gyro' = shift(gyro, tau) @ R_map.
Targets come from released train GT (local_eval/calib_targets_2026_09_18.py fits) and the
extrinsic family; weights live in the main checkpoint ("calnet" entry)."""
import numpy as np
import torch
import torch.nn as nn

R_MAP = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])   # class 1: (-y, -x, -z)


def _smooth(x, n):
    k = np.ones(n) / n
    return np.stack([np.convolve(x[:, i], k, mode="same") for i in range(x.shape[1])], 1)


def _tilt_consistency(acc, gyr, fs=200.0):
    """relative residual of d(up)/dt = -omega x up for the low-passed (0.25 s) accelerometer
    direction under a candidate gyro frame; near 0 when the gyro frame matches the
    accelerometer frame, near 1-2 when it does not (vibration is averaged out first)."""
    n = int(0.25 * fs)
    u = _smooth(acc, n); u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-6)
    w = _smooth(gyr, n)
    du = np.gradient(u, axis=0) * fs
    pred = -np.cross(w, u)                         # world-fixed vector seen from the body: du/dt = -omega x u
    return float(np.linalg.norm(du - pred, axis=1).mean() / (np.linalg.norm(du, axis=1).mean() + 1e-6))


def summary(imu, fs: float = 200.0) -> np.ndarray:
    """(N, 6) -> (64,) float32 recording summary."""
    acc = imu[:, :3].astype(np.float64); gyr = imu[:, 3:6].astype(np.float64)
    an = np.linalg.norm(acc, axis=1); gn = np.linalg.norm(gyr, axis=1)
    calm = gn < 0.5
    if calm.sum() < 200: calm = np.argsort(gn)[:max(200, len(gn) // 20)]
    f = []
    f += list(np.percentile(an[calm], [5, 25, 50, 75, 95]))                       # 5: |f| on calm frames
    f += list(acc[calm].mean(0)); f += list(acc[calm].std(0))                     # 6: calm acc mean/std per axis
    f += list(gyr.mean(0)); f += [gn.mean(), gn.std(), an.std()]                  # 6: gyro mean, rms-ish
    # 16-bin log spectrum of |acc| in 1-100 Hz (log-spaced), averaged over 4 s blocks
    seg = int(4 * fs); nb = max(1, len(an) // seg); spec = np.zeros(seg // 2 + 1)
    for i in range(nb):
        x = an[i * seg:(i + 1) * seg] - an[i * seg:(i + 1) * seg].mean(); spec += np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(seg, 1 / fs); edges = np.geomspace(1.0, fs / 2, 17)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (freqs >= lo) & (freqs < hi); f.append(np.log1p(spec[m].mean() / nb) if m.any() else 0.0)   # 16
    f += [_tilt_consistency(acc, gyr, fs), _tilt_consistency(acc, gyr @ R_MAP.T, fs)]              # 2: frame candidates
    # gyro-acc cross-correlation lag (+-100 ms) between |omega| and |d acc/dt|
    da = np.linalg.norm(np.gradient(acc, axis=0), axis=1); x = (gn - gn.mean()) / (gn.std() + 1e-9); y = (da - da.mean()) / (da.std() + 1e-9)
    best, lag_best = -2.0, 0
    for lag in range(-20, 21, 2):
        c = np.mean(x[max(0, lag):len(x) + min(0, lag)] * y[max(0, -lag):len(y) + min(0, -lag)])
        if c > best: best, lag_best = c, lag
    f += [lag_best * 1000 / fs / 100.0, best]                                     # 2
    f += [np.log(len(imu) / fs), float(np.percentile(np.abs(an - 9.81), 90))]     # 2
    f += list(np.percentile(gn, [50, 90, 99]))                                    # 3
    out = np.zeros(64, np.float64); v = np.array(f[:64]); out[:len(v)] = v
    return out.astype(np.float32)


def rot6d_to_R(v):
    """(..., 6) -> (..., 3, 3) via Gram-Schmidt (Zhou et al. 2019); v = 0 -> identity."""
    a1 = v[..., 0:3] + torch.tensor([1.0, 0, 0], device=v.device, dtype=v.dtype); a2 = v[..., 3:6] + torch.tensor([0, 1.0, 0], device=v.device, dtype=v.dtype)
    b1 = a1 / a1.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    a2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1; b2 = a2 / a2.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)                                     # columns


class CalNet(nn.Module):
    def __init__(self, in_dim: int = 64, hidden: int = 128, frame: bool = False):
        super().__init__()
        # road 1 stage 3 (2026-09-20): `frame` adds a 6D rotation head (R_gyro->acc, Gram-Schmidt)
        # replacing the two-class map; targets from local_eval/gyro_frame_targets.py
        self.frame = frame
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 8 + (6 if frame else 0)))
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)    # start: S=1, b=0, tau=0, map=identity, R=I
        self.register_buffer("mu", torch.zeros(in_dim)); self.register_buffer("sd", torch.ones(in_dim))

    def forward(self, x):
        o = self.mlp((x - self.mu) / self.sd)
        S = 1.0 + 0.3 * torch.tanh(o[:, 0:3]); b = 2.0 * torch.tanh(o[:, 3:6]); tau = 100.0 * torch.tanh(o[:, 6:7])
        if self.frame:
            return S, b, tau, o[:, 7], rot6d_to_R(o[:, 8:14])
        return S, b, tau, o[:, 7]                                                 # map logit


def apply_calibration(imu, S, b, tau_ms, map_cls, fs: float = 200.0, R=None):
    """(N, 6) numpy, per-recording parameters -> calibrated (N, 6) float32. R (3, 3): continuous
    gyro -> accelerometer rotation (stage 3) used instead of the two-class map when given."""
    acc = (imu[:, :3].astype(np.float64) - np.asarray(b)) / np.asarray(S)
    gyr = imu[:, 3:6].astype(np.float64)
    sh = int(round(float(tau_ms) * fs / 1000.0))
    if sh:                                                                        # positive tau: gyro lags -> advance it (edge-replicated, review 2.3)
        idx = np.clip(np.arange(len(gyr)) + sh, 0, len(gyr) - 1); gyr = gyr[idx]
    if R is not None: gyr = gyr @ np.asarray(R, np.float64).T
    elif int(map_cls) == 1: gyr = gyr @ R_MAP.T
    return np.concatenate([acc, gyr], axis=1).astype(np.float32)


def calnet_apply_numpy(model, imu_np, device="cpu"):
    """The gyro-frame flip is applied only when BOTH the network (logit > 2) and the
    parameter-free frame-consistency rule (summary columns 33/34) agree: a wrong flip on an
    identity recording is catastrophic for the attitude / DR channels, a missed flip on the
    other source only leaves those channels as bad as they already are."""
    model.eval(); device = next(model.parameters()).device          # the model decides the device
    feat = summary(imu_np)
    with torch.no_grad():
        x = torch.from_numpy(feat)[None].to(device)
        out = model(x)
    S, b, tau, ml = out[:4]
    if getattr(model, "frame", False):
        # the continuous rotation is used only where the two-class rule already says "flip"
        # (network logit > 2 and the parameter-free consistency rule agree): on identity-family
        # recordings the regressed R is worse than leaving the gyro alone (dev check 2026-09-20)
        R = out[4][0].cpu().numpy(); ang = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
        flip = int(float(ml[0]) > 2.0 and feat[34] < feat[33])
        Ruse = R if (flip and ang > 3.0) else None
        return apply_calibration(imu_np, S[0].cpu().numpy(), b[0].cpu().numpy(), float(tau[0]), 0, R=Ruse), \
            {"S": S[0].cpu().numpy(), "b": b[0].cpu().numpy(), "tau": float(tau[0]), "map": int(ang > 90), "frame_deg": ang, "logit": float(ml[0])}
    flip = int(float(ml[0]) > 2.0 and feat[34] < feat[33])
    return apply_calibration(imu_np, S[0].cpu().numpy(), b[0].cpu().numpy(), float(tau[0]), flip), \
        {"S": S[0].cpu().numpy(), "b": b[0].cpu().numpy(), "tau": float(tau[0]), "map": flip, "logit": float(ml[0])}

"""AttNet (learned-INS step B, 2026-09-19): per-recording attitude from the rest anchor.

Hard-wired gyro integration from the rest window (the same anchor and increments as
updir.rest_frame_gyro) plus learned corrections from a small bidirectional GRU that reads
the whole recording at 20 Hz:
  * gyro bias b_g (3, per recording), applied to every increment
  * anchor tilt correction delta0 (3, small-angle, per recording): up0' = Exp(delta0) up0
  * a slow correction rate ddot(t) (3, 20 Hz, linearly upsampled), added to the gyro
Integration (differentiable, parallel): R_{0->t} = prod_{i<t} Exp((w_i - b_g + ddot_i) dt);
up(t) = R_{0->t}^T up0'. Increments are composed exactly inside blocks of `block` frames
(small-angle sum, block = 5 frames = 25 ms; max 0.7 deg vs the exact loop on 0101) and blocks are composed with a log-depth prefix scan.
Weights live inside the main checkpoint ("attnet" entry); hf/predict.py runs it first.
Only IMU in, no labels at inference; GT up (gtup.gt_up_frames) supervises it at training.
"""
import torch
import torch.nn as nn


def so3_exp(v):
    """(..., 3) rotation vectors -> (..., 3, 3) matrices (Rodrigues, safe at 0)."""
    th = v.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    k = v / th
    K = torch.zeros(v.shape[:-1] + (3, 3), dtype=v.dtype, device=v.device)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0] = -k[..., 2], k[..., 1], k[..., 2]
    K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -k[..., 0], -k[..., 1], k[..., 0]
    s, c = torch.sin(th)[..., None], torch.cos(th)[..., None]
    I = torch.eye(3, dtype=v.dtype, device=v.device).expand_as(K)
    return I + s * K + (1 - c) * (K @ K)


def cumprod_so3(R):
    """(B, M, 3, 3) -> (B, M, 3, 3) inclusive prefix products P_m = R_0 R_1 ... R_m
    (log-depth associative scan, differentiable)."""
    P = R.clone(); M = R.shape[1]; d = 1
    while d < M:
        Q = P.clone()
        Q[:, d:] = P[:, :-d] @ P[:, d:]
        P = Q; d *= 2
    return P


def integrate(omega, dt, b_g, ddot, s0, block: int = 5):
    """omega (B, N, 3) rad/s [padded], dt scalar, b_g (B, 3), ddot (B, N, 3), s0 anchor frame
    (int, same for the batch after alignment) -> R (B, N, 3, 3) = R_{0->t}, with R[s0] = I.
    Forward: R_t = prod_{i=s0}^{t-1} dR_i; backward: R_t = (prod_{i=t}^{s0-1} dR_i)^T."""
    B, N, _ = omega.shape
    w = (omega - b_g[:, None, :] + ddot) * dt                           # per-frame rotation vectors
    # pad to a multiple of block, exact small-angle sums inside blocks
    pad = (-N) % block
    if pad: w = torch.cat([w, torch.zeros(B, pad, 3, dtype=w.dtype, device=w.device)], 1)
    Nb = w.shape[1] // block
    wb = w.reshape(B, Nb, block, 3)
    # rotation at frame t relative to the start of its block: cumulative sum inside the block
    within = so3_exp(torch.cumsum(wb, dim=2) - wb)                      # (B, Nb, block, 3, 3): before frame t
    blk = so3_exp(wb.sum(2))                                             # (B, Nb, 3, 3): whole block
    pre = cumprod_so3(blk)                                               # P_j = blk_0 ... blk_j
    ident = torch.eye(3, dtype=w.dtype, device=w.device).expand(B, 1, 3, 3)
    pre_excl = torch.cat([ident, pre[:, :-1]], 1)                        # product of blocks before block j
    R_abs = (pre_excl[:, :, None] @ within).reshape(B, Nb * block, 3, 3)[:, :N]   # R_{start->t}
    # re-anchor at s0: R_{0->t} = R_{s0}^T R_t  (both are "start -> frame" products)
    R0 = R_abs[:, s0]                                                    # (B, 3, 3)
    return R0.transpose(1, 2)[:, None] @ R_abs


class AttNet(nn.Module):
    def __init__(self, hidden: int = 64, pool: int = 10, block: int = 5, fs: float = 200.0):
        super().__init__()
        self.pool, self.block, self.fs = pool, block, fs
        self.inp = nn.Linear(6 + 1, hidden)                                  # IMU + rest-window flag
        self.gru = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.rate = nn.Linear(2 * hidden, 3)                                 # ddot at 20 Hz
        self.rec = nn.Linear(2 * hidden, 6)                                  # [b_g, delta0] from the pooled state
        for m in (self.rate, self.rec):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)                 # start = pure gyro integration
        self.rate_scale = 0.05                                               # rad/s cap-ish scale for ddot
        self.bias_scale = 0.02                                               # rad/s
        self.tilt_scale = 0.2                                                # rad

    def rest_window(self, imu, window: int = 200):
        """calmest one-second window (updir.rest_reference rule) -> index k0, per recording."""
        B, N, _ = imu.shape; n = N // window
        acc = imu[:, :n * window, :3].reshape(B, n, window, 3); gyr = imu[:, :n * window, 3:6].reshape(B, n, window, 3)
        an = acc.norm(dim=-1); score = (an.mean(-1) - 9.81).abs() + gyr.norm(dim=-1).mean(-1) + 0.5 * an.std(-1)
        return score.argmin(dim=1)                                           # (B,)

    def forward(self, imu, k0=None, lengths=None):
        """imu (B, N, 6) float32 (padded with zeros beyond lengths) -> up (B, N, 3), R (B, N, 3, 3), aux.
        For simplicity the anchor frame s0 is taken from the FIRST recording of the batch when
        B > 1; train with B recordings aligned by rolling so that k0 coincides (train_attnet does)."""
        B, N, _ = imu.shape
        if k0 is None: k0 = self.rest_window(imu)
        s0 = int(k0[0]) * 200 + 100
        flag = torch.zeros(B, N, 1, dtype=imu.dtype, device=imu.device)
        for b in range(B): flag[b, int(k0[b]) * 200:(int(k0[b]) + 1) * 200] = 1.0
        x = torch.cat([imu, flag], -1)
        xp = x.transpose(1, 2); xp = nn.functional.avg_pool1d(xp, self.pool, ceil_mode=True).transpose(1, 2)   # (B, N/pool, 7)
        h, _ = self.gru(torch.tanh(self.inp(xp)))                            # (B, M, 2H)
        rate = torch.tanh(self.rate(h)) * self.rate_scale                    # (B, M, 3)
        pooled = h.mean(1)
        rec = self.rec(pooled)
        b_g = torch.tanh(rec[:, :3]) * self.bias_scale; delta0 = torch.tanh(rec[:, 3:]) * self.tilt_scale
        ddot = nn.functional.interpolate(rate.transpose(1, 2), size=N, mode="linear", align_corners=False).transpose(1, 2)
        R = integrate(imu[..., 3:6].float(), 1.0 / self.fs, b_g, ddot, s0, self.block)     # (B, N, 3, 3), R[s0] = I
        f0 = imu[:, int(k0[0]) * 200:(int(k0[0]) + 1) * 200, :3].mean(1)                     # anchor specific force
        up0 = f0 / f0.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        up0 = (so3_exp(delta0) @ up0[..., None])[..., 0]
        up = (R.transpose(-1, -2) @ up0[:, None, :, None])[..., 0]           # up_t = R^T up0'
        return up, R, {"b_g": b_g, "delta0": delta0, "ddot": ddot, "k0": k0}


def att_rotation_numpy(model, imu_np, device="cpu"):
    """Inference helper: (N, 6) numpy -> (R (N,3,3) float64, up0 (3,), k0) for updir.dr_channels."""
    import numpy as np
    model.eval(); device = next(model.parameters()).device          # the model decides the device
    with torch.no_grad():
        x = torch.from_numpy(np.asarray(imu_np, np.float32))[None].to(device)
        up, R, aux = model(x)
        k0 = int(aux["k0"][0]); up0 = up[0, k0 * 200 + 100]
    return R[0].double().cpu().numpy(), up0.double().cpu().numpy(), k0

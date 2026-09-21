#!/usr/bin/env python3
"""Mixture-of-experts pieces, in the form the rules allow.

    "Platform-specific internal routing is allowed, but four separately selected
     expert models are not... a single model that adapts internally (learned
     conditioning, mixture-of-experts inside one network) is allowed and
     encouraged."

So everything here is trained end to end in one optimiser step, combines its
experts by a soft weighted sum rather than picking one, and gates only on
features derived from the IMU. The released 4-head baseline is the disallowed
shape, and its weakness (val 0.4190 even with ground-truth routing, against
0.2280 for one shared head) plausibly comes from hard routing at TRAINING time
leaving each head only its own platform's quarter of the data. A soft gate keeps
every expert seeing every sample, which is the property worth preserving.

Gate entropy is returned everywhere because the failure mode that would look
exactly like "MoE does not help" is the gate collapsing onto one expert. Without
logging it the experiment cannot distinguish the two.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def gate_entropy(weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean entropy of a softmax gate, in nats. ln(K) means a uniform gate."""
    return -(weights * (weights + eps).log()).sum(-1).mean()


class ExpertHeads(nn.Module):
    """Placement A: K linear velocity heads mixed by a gate on the same feature.

    One Linear(d, 3) has to cover speeds spanning 0.001-11 m/s; letting different
    regions of feature space use a different linear map is the cheapest form the
    hypothesis can take. Implemented as a single Linear(d, K*out) so all experts
    run in one matmul.
    """

    def __init__(self, d_model: int, n_experts: int, out_dim: int = 3):
        super().__init__()
        self.n_experts, self.out_dim = n_experts, out_dim
        self.gate = nn.Linear(d_model, n_experts)
        self.experts = nn.Linear(d_model, n_experts * out_dim)

    def forward(self, f):
        w = torch.softmax(self.gate(f), dim=-1)                       # (..., K)
        y = self.experts(f).view(*f.shape[:-1], self.n_experts, self.out_dim)
        return (y * w.unsqueeze(-1)).sum(-2), self.gate(f), w


class ExpertMLP(nn.Module):
    """Placement B: K parallel feature transforms, then one shared output layer."""

    def __init__(self, d_model: int, n_experts: int, out_dim: int = 3, dropout: float = 0.0):
        super().__init__()
        self.n_experts, self.d_model = n_experts, d_model
        self.gate = nn.Linear(d_model, n_experts)
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, n_experts * d_model)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, out_dim)

    def forward(self, f):
        logits = self.gate(f)
        w = torch.softmax(logits, dim=-1)
        h = F.gelu(self.proj(self.norm(f))).view(*f.shape[:-1], self.n_experts, self.d_model)
        mixed = (h * w.unsqueeze(-1)).sum(-2)
        return self.out(self.drop(mixed)), logits, w


def segment_stats(x: torch.Tensor) -> torch.Tensor:
    """(B, K, 6, T) normalised IMU -> (B, 18) cheap whole-segment descriptors.

    The trunk-conditioning gate cannot read the mixer, which runs after the trunk,
    so it reads the raw segment instead: per-channel mean, spread, and step-to-step
    roughness. That is also the more natural signal for "what is this trajectory",
    which is the question the conditioning is meant to answer.
    """
    y = x.transpose(1, 2).reshape(x.shape[0], x.shape[2], -1)         # (B, C, K*T)
    rough = (y[..., 1:] - y[..., :-1]).abs().mean(-1)
    return torch.cat([y.mean(-1), y.std(-1), rough], dim=-1)


class FiLMGate(nn.Module):
    """Placement C': soft-mixed per-channel scale and shift for the trunk.

    A weighted sum of K expert (gamma, beta) vectors, so the trunk still runs
    once -- unlike K parallel trunk copies, which a soft mixture would force us
    to evaluate in full and would cost K times the compute in the expensive part
    of the network.
    """

    def __init__(self, in_dim: int, n_experts: int, channels: tuple[int, ...],
                 hidden: int = 64):
        super().__init__()
        self.gate = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden),
                                  nn.GELU(), nn.Linear(hidden, n_experts))
        self.gamma = nn.ParameterList(nn.Parameter(torch.zeros(n_experts, c)) for c in channels)
        self.beta = nn.ParameterList(nn.Parameter(torch.zeros(n_experts, c)) for c in channels)

    def forward(self, stats):
        logits = self.gate(stats)
        w = torch.softmax(logits, dim=-1)                             # (B, K)
        # zero-init gammas make the module start as the identity (1 + 0)
        gammas = [1.0 + w @ g for g in self.gamma]
        betas = [w @ b for b in self.beta]
        return gammas, betas, logits, w


def apply_film(h: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
               repeat: int) -> torch.Tensor:
    """h is (n, C, T); gamma/beta are (B, C) shared by all windows of a segment.

    `repeat` is an int (every segment contributes K rows in order) or a (n,)
    tensor of segment indices, one per row of h (segments of unequal length)."""
    if isinstance(repeat, int):
        g, b = gamma.repeat_interleave(repeat, dim=0), beta.repeat_interleave(repeat, dim=0)
    else:
        g, b = gamma[repeat], beta[repeat]
    return h * g.unsqueeze(-1) + b.unsqueeze(-1)


TRAJ_DESC_DIM = 35
TRAJ_DESC = {"version": 1}   # advice_after_v20 §3: 2 = yaw-invariant base statistics (set from --traj_desc, travels with the checkpoint)


def traj_descriptors(imu: torch.Tensor) -> torch.Tensor:
    if TRAJ_DESC["version"] == 2:
        return traj_descriptors_v2(imu)
    return traj_descriptors_v1(imu)


def traj_descriptors_v2(imu: torch.Tensor) -> torch.Tensor:
    """advice_after_v20 §3 (2026-09-20 night): the v1 base (per-channel mean / std / roughness of
    the six body axes) is not invariant to the yaw augmentation of the sample, so the FiLM saw a
    descriptor that disagreed with the rotated window. v2 keeps the 17 scalar extras and replaces
    the 18 base statistics by statistics of four yaw-invariant channels -- the vertical component
    and the horizontal magnitude of the accelerometer and of the gyro about the calm-frame
    gravity direction: mean / std / step roughness of [a_v, |a_h|, w_v, |w_h|] (12) plus
    percentiles 50/90/99 of |a_h| and of |a_v - 9.81| (6). Same 35 dimensions."""
    x = imu.detach().float().cpu(); N = x.shape[0]
    acc = x[:, :3]; gyr = x[:, 3:6]
    gn = gyr.norm(dim=-1); calm = gn < 0.5
    if int(calm.sum()) < 200:
        calm = torch.zeros(N, dtype=torch.bool); calm[torch.argsort(gn)[:max(200, N // 20)]] = True
    up = acc[calm].mean(0); up = up / up.norm().clamp(min=1e-6)
    a_v = acc @ up; a_h = (acc - a_v[:, None] * up).norm(dim=-1)
    w_v = gyr @ up; w_h = (gyr - w_v[:, None] * up).norm(dim=-1)
    ch = torch.stack([a_v, a_h, w_v, w_h], dim=1)                                 # (N, 4), yaw-invariant
    rough = (ch[1:] - ch[:-1]).abs().mean(0)
    base = torch.cat([ch.mean(0), ch.std(0), rough,
                      torch.quantile(a_h, torch.tensor([0.5, 0.9, 0.99])),
                      torch.quantile((a_v - 9.81).abs(), torch.tensor([0.5, 0.9, 0.99]))])   # 18
    return torch.cat([base, _traj_extras(x)]).to(imu.device, imu.dtype)


def _traj_extras(x: torch.Tensor) -> torch.Tensor:
    """the 17 scalar extras shared by v1 and v2 (x: (N, 6) float32 on cpu)."""
    N = x.shape[0]; n = max(1, N // 200)
    acc = x[:n * 200, :3].reshape(n, 200, 3); gyr = x[:n * 200, 3:6].reshape(n, 200, 3)
    an = acc.norm(dim=-1); gm = gyr.norm(dim=-1).mean(1)
    score = (an.mean(1) - 9.81).abs() + gm + 0.5 * an.std(1); k = int(torch.argmin(score))
    frest = acc[k].mean(0); rest_f = frest.norm(); rest_score = score[k]
    hf = (an - an.mean(1, keepdim=True)); hf_rms = hf.pow(2).mean().sqrt()
    seg = x[:, :3].norm(dim=-1); seg = seg[:(len(seg) // 800) * 800]
    if len(seg) >= 800:
        spec = torch.fft.rfft(seg.reshape(-1, 800) - seg.reshape(-1, 800).mean(1, keepdim=True), dim=1).abs().pow(2).mean(0)   # 0.25 Hz bins
        bins = torch.stack([torch.log1p(spec[i * 40:(i + 1) * 40].mean()) for i in range(10)])                    # 10 Hz bins to 100 Hz
    else:
        bins = torch.zeros(10)
    gq = torch.quantile(x[:, 3:6].norm(dim=-1), torch.tensor([0.5, 0.9, 0.99]))
    return torch.cat([rest_f[None], rest_score[None], hf_rms[None], bins, gq, torch.log(torch.tensor([N / 200.0]))])


def traj_descriptors_v1(imu: torch.Tensor) -> torch.Tensor:
    """(N, 6) raw IMU of one whole trajectory -> (35,) recording-level descriptors (road 4,
    2026-09-20; the first 18 are the 2026-09 originals: per-channel mean, spread, step
    roughness). Added scalars only (review: body-frame vectors would disagree with the yaw
    augmentation of the sample): rest |f| (1), rest-window score (1), high-frequency |acc| rms
    (1), 10-bin log spectrum of |acc| 0-100 Hz (10), gyro |w| percentiles 50/90/99 (3), log
    duration (1). IMU only, whole recording, no labels: legal at test time."""
    rough = (imu[1:] - imu[:-1]).abs().mean(0)
    base = torch.cat([imu.mean(0), imu.std(0), rough])
    x = imu.detach().float().cpu(); N = x.shape[0]; n = max(1, N // 200)
    acc = x[:n * 200, :3].reshape(n, 200, 3); gyr = x[:n * 200, 3:6].reshape(n, 200, 3)
    an = acc.norm(dim=-1); gm = gyr.norm(dim=-1).mean(1)
    score = (an.mean(1) - 9.81).abs() + gm + 0.5 * an.std(1); k = int(torch.argmin(score))
    frest = acc[k].mean(0); rest_f = frest.norm(); rest_score = score[k]
    hf = (an - an.mean(1, keepdim=True)); hf_rms = hf.pow(2).mean().sqrt()
    seg = x[:, :3].norm(dim=-1); seg = seg[:(len(seg) // 800) * 800]
    if len(seg) >= 800:
        spec = torch.fft.rfft(seg.reshape(-1, 800) - seg.reshape(-1, 800).mean(1, keepdim=True), dim=1).abs().pow(2).mean(0)   # 0.25 Hz bins
        bins = torch.stack([torch.log1p(spec[i * 40:(i + 1) * 40].mean()) for i in range(10)])                    # 10 Hz bins to 100 Hz
    else:
        bins = torch.zeros(10)
    gq = torch.quantile(x[:, 3:6].norm(dim=-1), torch.tensor([0.5, 0.9, 0.99]))
    extra = torch.cat([rest_f[None], rest_score[None], hf_rms[None], bins, gq, torch.log(torch.tensor([N / 200.0]))])
    return torch.cat([base, extra.to(base.device, base.dtype)])

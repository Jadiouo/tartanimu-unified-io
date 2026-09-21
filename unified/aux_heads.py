#!/usr/bin/env python3
"""Two optional, independent heads for UnifiedIMUNet.

Both attach to the 256-d shared feature produced by `UnifiedIMUNet.shared` and
are enabled one at a time by a training-script flag; neither requires the other.

PerFrameHead exists because the label file is far richer than the label we use.
The scored regression target for a window is exactly the mean of the per-frame
`vel_body` over that window's 200 frames (verified to 1.8e-15), and the .npz
files store `vel_body` at the full 200 Hz.  Collapsing 200 vectors to one throws
away 99.5% of the available supervision.  Asking the trunk to also reproduce the
within-window trajectory forces the attention pooling to keep temporal structure
that the mean alone never penalises it for discarding, while leaving the scored
quantity untouched -- the pooled target's own mean is still the scored mean, so
the auxiliary loss cannot pull the model away from the metric.

ScaleAwareHead exists because a single `nn.Linear(256, 3)` has to cover roughly
five orders of magnitude: true window speed spans about 0.001 to 11 m/s, with a
drone median of 0.908 m/s against a human median of 0.112 m/s.  An additive
head spends its output range on the fast platforms, so a fixed absolute weight
error that is invisible on a drone window is a total loss on a human one.
Factoring the prediction into a unit direction and a log-magnitude makes the
parametrisation multiplicative: a fixed step in the magnitude output is a fixed
*relative* change in speed, so the slow platforms get the same resolution as the
fast ones, and direction and speed stop competing for the same weights.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerFrameHead(nn.Module):
    """Predict the within-window velocity trajectory, (B, 256) -> (B, T, 3).

    `n_frames` (T) is the resolution at which the trajectory is supervised, and
    it trades signal against difficulty.  T = 200 uses every stored label but
    asks a window-level feature to resolve 5 ms detail that a 200-frame receptive
    field has largely averaged away, so most of the extra loss is noise.  A
    coarser T -- 20 is the default, i.e. 10 Hz -- averages the target down with
    `perframe_target`, which suppresses that noise while still distinguishing an
    accelerating window from a constant one.  T = 1 degenerates to the scored
    target and buys nothing.  Cost is linear in T: the output layer is
    256 x 3T weights.
    """

    def __init__(self, feature_dim: int = 256, n_frames: int = 20):
        super().__init__()
        self.n_frames = n_frames
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Linear(256, n_frames * 3),
        )

    def forward(self, features):
        return self.net(features).reshape(features.shape[0], self.n_frames, 3)


def perframe_target(vel_body_window, n_frames: int):
    """Average-pool per-frame ground truth, (B, 200, 3) -> (B, n_frames, 3).

    Non-overlapping equal bins, so the pooling is mean-preserving: the mean of
    the result over dim 1 is the mean of the input over dim 1, for any divisor
    `n_frames`.  In particular `perframe_target(v, 1).squeeze(1)` is the scored
    window-mean target, which is what makes the auxiliary loss consistent with
    the metric rather than a competing objective.
    """
    frames = vel_body_window.shape[-2]
    if frames % n_frames:
        raise ValueError(f"n_frames={n_frames} does not divide {frames} frames evenly")
    return vel_body_window.reshape(*vel_body_window.shape[:-2], n_frames, frames // n_frames, 3).mean(-2)


class ScaleAwareHead(nn.Module):
    """Drop-in replacement for `velocity_head`, (B, 256) -> (B, 3).

    Predicts a unit direction and a log-magnitude, then reconstructs
    `v = direction * expm1(softplus(log_mag))`.
    """

    # expm1(softplus(s)) is exp(s) written as a composition of two functions
    # that are each exact at the end that matters.  Taking the three required
    # properties in turn:
    #
    #   smooth      C-infinity and strictly increasing on all of R, so there is
    #               no kink for the optimiser to sit on, and the magnitude is
    #               positive by construction without a ReLU or an abs().
    #   exactly 0   softplus(s) underflows to exactly 0.0 for very negative s
    #               (below -102.9 in fp32, -17.3 in fp16) and expm1(0.0) is
    #               exactly 0.0, so a stationary window is representable, not
    #               merely approached.  Note d|v|/ds = |v|, so the approach is
    #               geometric; that is acceptable here because the residual it
    #               leaves behind is |v| itself.
    #   no overflow the operating band is s in [-7, +2.8] for 0.001 to 16 m/s,
    #               nowhere near the fp16 ceiling.  The clamp is a divergence
    #               guard well outside that band: it caps the reconstruction at
    #               expm1(9) = 8.1e3 < 65504, so an early-training blow-up
    #               produces a large loss rather than an inf.
    #
    # The reason to want exp(s) specifically is fact (b): a fixed step in s is a
    # fixed *relative* change in speed, so the human split at 0.1 m/s gets the
    # same output resolution as the drone split at 1 m/s instead of being
    # squeezed into the bottom 1% of a linear head's range.
    MAX_LOG_MAGNITUDE = 9.0

    def __init__(self, feature_dim: int = 256, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.direction = nn.Linear(feature_dim, 3)
        self.log_magnitude = nn.Linear(feature_dim, 1)
        # ~0.3 m/s at init, the geometric middle of the four platform medians.
        nn.init.constant_(self.log_magnitude.bias, math.log(0.3))

    def forward(self, features):
        raw = self.direction(features)
        # Smooth everywhere, unlike F.normalize's clamped norm, which has a kink
        # where the raw direction passes near the origin.
        direction = raw * torch.rsqrt(raw.square().sum(-1, keepdim=True) + self.eps ** 2)
        log_mag = self.log_magnitude(features).clamp(max=self.MAX_LOG_MAGNITUDE)
        return direction * torch.expm1(F.softplus(log_mag))


def masked_vector_huber(pred, target, beta: float = 0.25, mask=None):
    """`model.vector_huber` over an arbitrary leading shape.

    Norms are taken over the last dim, so this accepts (B, 3) as well as the
    (B, T, 3) that PerFrameHead emits, and averages over every leading position.
    `mask` is an optional broadcastable (..., ) bool or float tensor selecting
    which positions count; the result is the mean over the selected ones.
    """
    error = torch.linalg.vector_norm(pred - target, dim=-1)
    loss = torch.where(error < beta, 0.5 * error.square() / beta, error - 0.5 * beta)
    if mask is None:
        return loss.mean()
    mask = mask.to(loss.dtype)
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)

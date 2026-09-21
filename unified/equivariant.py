#!/usr/bin/env python3
"""O(2)-equivariant frame prediction, after EqNIO (arXiv 2408.06321).

Rotating the body frame about gravity, or reflecting it across a plane
containing gravity, rotates the IMU and the body-frame velocity together. The
network currently has to learn that from data. EqNIO builds it in: predict a
canonical yaw frame F from the IMU, map the measurements into it, run any
backbone, map the output back. Heading is unobservable from an IMU, so this is
the one symmetry there is no reason to spend capacity learning, and the paper
reports 14% on ATE* over RoNIN -- our own architecture family -- while converging
in 38 epochs against 100+.

Two things this file has to get right.

Gravity. The group is only O(2) once the z-axis is gravity, and EqNIO takes that
direction from an EKF, which this competition does not provide. unified/updir.py
supplies it instead: a complementary filter reaching 0.6-1.8 deg depending on
platform, against the 6.35 deg that segment averaging gave -- the number that had
me set this approach aside. EqNIO's own sensitivity study perturbs gravity by 2
to 8 deg and stabilises it with 5 deg perturbation augmentation, so 1.8 deg sits
inside its tolerance.

Angular rates. a and w transform differently -- rho_w(F) = det(F) F, so a
reflection flips w -- and mixing them linearly is therefore not equivariant. The
paper's bijection splits w into two true vectors whose cross product recovers it:

    w1 = [-wy, wx, 0],  w2 = w x w1
    v1 = sqrt(|w|) w1/|w1|,  v2 = sqrt(|w|) w2/|w2|,  w = v1 x v2

since (Ax) x (Ay) = det(A) A (x x y) turns the pair's rotation into exactly the
pseudovector rule.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def decompose_omega(w: torch.Tensor, a: torch.Tensor, eps: float = 1e-8):
    """(..., 3) angular rate -> two (..., 3) true vectors with v1 x v2 == w.

    The degenerate cases are the paper's: w1 collapses when wx = wy = 0, and the
    fallback a x w collapses in turn when a is parallel to w.
    """
    wx, wy, wz = w.unbind(-1)
    w1 = torch.stack([-wy, wx, torch.zeros_like(wz)], dim=-1)
    n1 = w1.norm(dim=-1, keepdim=True)

    alt = torch.cross(a, w, dim=-1)
    alt2 = torch.cross(w, w.new_tensor([1.0, 0.0, 0.0]).expand_as(w), dim=-1)
    w1 = torch.where(n1 > eps, w1, torch.where(alt.norm(dim=-1, keepdim=True) > eps, alt, alt2))

    w2 = torch.cross(w, w1, dim=-1)
    wn = w.norm(dim=-1, keepdim=True).clamp(min=eps).sqrt()
    v1 = wn * w1 / w1.norm(dim=-1, keepdim=True).clamp(min=eps)
    v2 = wn * w2 / w2.norm(dim=-1, keepdim=True).clamp(min=eps)
    return v1, v2


def gravity_align(up: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """(..., 3) up direction -> (..., 3, 3) rotation taking the body frame to one
    whose z-axis is up. Yaw is left arbitrary; the frame network resolves it."""
    z = up / up.norm(dim=-1, keepdim=True).clamp(min=eps)
    ref = torch.where((z[..., 0:1].abs() < 0.9), z.new_tensor([1.0, 0.0, 0.0]).expand_as(z),
                      z.new_tensor([0.0, 1.0, 0.0]).expand_as(z))
    x = torch.cross(ref, z, dim=-1)
    x = x / x.norm(dim=-1, keepdim=True).clamp(min=eps)
    y = torch.cross(z, x, dim=-1)
    return torch.stack([x, y, z], dim=-2)          # rows are the new basis


class EqLinear(nn.Module):
    """O(2)-equivariant channel mixing on 2-D vector features.

    R W v = W R v forces W = w1 * I for O(2) -- reflections kill the R90 term SO(2)
    would allow -- so an equivariant linear layer is a plain channel mix with no
    rotation of its own.
    """

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.w = nn.Parameter(torch.randn(c_in, c_out) / c_in ** 0.5)

    def forward(self, v):                          # (..., 2, C_in) -> (..., 2, C_out)
        return v @ self.w


class GatedNonlinearity(nn.Module):
    """Mixes scalars into vectors without breaking equivariance.

    Vector features may only be rescaled by an invariant, so the MLP reads their
    norms alongside the scalars and returns a gain for the vectors and an
    activation for the scalars.
    """

    def __init__(self, c_vec: int, c_scalar: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(c_vec + c_scalar, 2 * max(c_vec, c_scalar)),
                                 nn.GELU(),
                                 nn.Linear(2 * max(c_vec, c_scalar), c_vec + c_scalar))
        self.c_vec = c_vec

    def forward(self, v, s):                       # v (..., 2, C), s (..., C')
        h = self.mlp(torch.cat([v.norm(dim=-2), s], dim=-1))
        gamma, beta = h[..., :self.c_vec], h[..., self.c_vec:]
        return v * gamma.unsqueeze(-2), F.gelu(beta)


class FrameNet(nn.Module):
    """Predict a canonical yaw direction from gravity-aligned IMU.

    Equivariant by construction: rotate the input about z and the predicted
    direction rotates with it, so mapping into the frame it defines cancels the
    rotation exactly rather than approximately.
    """

    def __init__(self, width: int = 32, layers: int = 3):
        super().__init__()
        c_v, c_s = 3, 9                            # a, v1, v2 | z-parts, xy-norms, dots
        self.vin = EqLinear(c_v, width)
        self.sin = nn.Linear(c_s, width)
        self.blocks = nn.ModuleList(
            nn.ModuleList([EqLinear(width, width), nn.Linear(width, width),
                           GatedNonlinearity(width, width)]) for _ in range(layers))
        # two directions, not one: a frame built from a single direction is always
        # a rotation (det +1), and no rotation can satisfy F(rho(R)x) = R F(x) when
        # det(R) = -1. Gram-Schmidt of two equivariant vectors commutes with
        # orthogonal transforms and lets the determinant follow the data.
        self.vout = EqLinear(width, 2)

    @staticmethod
    def features(a, v1, v2):
        """-> vector (..., 2, 3) and scalar (..., 9), all O(2)-consistent."""
        xy = torch.stack([a[..., :2], v1[..., :2], v2[..., :2]], dim=-1)      # (...,2,3)
        z = torch.stack([a[..., 2], v1[..., 2], v2[..., 2]], dim=-1)         # (...,3)
        norms = xy.norm(dim=-2)                                              # (...,3)
        dots = torch.stack([(xy[..., 0] * xy[..., 1]).sum(-1),
                            (xy[..., 0] * xy[..., 2]).sum(-1),
                            (xy[..., 1] * xy[..., 2]).sum(-1)], dim=-1)      # (...,3)
        return xy, torch.cat([z, norms, dots], dim=-1)

    def forward(self, a, w):
        """a, w (B, T, 3) already gravity-aligned -> (B, 3, 3) canonical rotation."""
        v1, v2 = decompose_omega(w, a)
        v, s = self.features(a, v1, v2)
        v, s = self.vin(v), self.sin(s)
        for eq, lin, gate in self.blocks:
            v, s = gate(eq(v), lin(s))
        d = self.vout(v).mean(dim=1)                                         # (B, 2, 2)
        d1, d2 = d[..., 0], d[..., 1]
        e1 = d1 / d1.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        r = d2 - (d2 * e1).sum(-1, keepdim=True) * e1
        e2 = r / r.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        f2 = torch.stack([e1, e2], dim=-1)                                   # columns
        f3 = torch.zeros(*f2.shape[:-2], 3, 3, dtype=f2.dtype, device=f2.device)
        f3[..., :2, :2] = f2
        f3[..., 2, 2] = 1.0                        # z is already gravity, so it is fixed
        return f3


class Canonicalizer(nn.Module):
    """Map IMU into a canonical frame, and predictions back out of it.

    Two rotations compose here. R_g takes the body frame to a gravity-aligned one
    using the estimated up direction, which leaves only yaw free; F resolves that
    yaw equivariantly. A backbone runs on a' = F^T R_g a, and its body-frame
    velocity comes back as v = R_g^T F v'.

    R_g is per window -- the body turns within a segment, and using one rotation
    for all of them is what made segment-averaged gravity fail on human -- while F
    is per segment, since the canonical yaw should not jitter window to window.
    """

    def __init__(self, width: int = 32, layers: int = 3):
        super().__init__()
        self.frame = FrameNet(width, layers)

    def forward(self, imu, up):
        """imu (B, K, T, 6) and up (B, K, T, 3) -> (canonical imu, R_total (B, 3, 3)).

        R_total maps body -> canonical; a prediction v' returns as R_total^T v'.
        """
        B, K, T, _ = imu.shape
        # ONE alignment for the whole segment. Aligning each timestep by its own
        # gravity estimate leaves a yaw ambiguity that varies along the segment --
        # measured at -60, -54 and -33 degrees across three consecutive steps of one
        # rotating trajectory -- and the frame network's equivariance assumes a
        # single global transformation, so per-timestep alignment breaks it.
        # Applying one rotation to the input and undoing the same one on the output
        # keeps target and prediction in correspondence regardless.
        # The segment's up is taken at its middle frame, not averaged. Up is
        # expressed in the body frame at each instant, and a platform that pitches
        # or rolls carries it round with it; averaging across those orientations
        # gives a direction that exists at no instant. Measured against ground
        # truth on val, median error per platform:
        #                  car    dog   drone  human
        #   segment mean   0.64   1.79   5.84   7.88
        #   middle frame   0.95   1.56   5.54   1.79
        # The drone's ~5.5 deg is the estimate itself and no choice here moves it;
        # --grav_perturb is EqNIO's answer to that.
        up_seg = up[:, K // 2, T // 2]
        Rg = gravity_align(up_seg)                              # (B, 3, 3)
        a = torch.einsum("bij,bktj->bkti", Rg, imu[..., 0:3])
        w = torch.einsum("bij,bktj->bkti", Rg, imu[..., 3:6])
        # gravity_align always returns det +1, so the angular rate needs no sign flip

        Fm = self.frame(a.reshape(B, K * T, 3), w.reshape(B, K * T, 3))   # (B, 3, 3)
        Ft = Fm.transpose(-1, -2)

        # The backbone gets a, v1, v2 rather than a, w. w is a pseudovector, so it
        # is still one after canonicalisation and flips sign under a reflection --
        # the canonical signal would not be invariant. v1 and v2 are true vectors
        # carrying the same information, which is what the bijection is for.
        v1, v2 = decompose_omega(w, a)
        out = [torch.einsum("bij,bktj->bkti", Ft, x) for x in (a, v1, v2)]
        return torch.cat(out, dim=-1), torch.einsum("bij,bjl->bil", Ft, Rg)


def to_body(v_canonical, R_total):
    """(B, K, 3) prediction in the canonical frame -> body frame."""
    return torch.einsum("bji,bkj->bki", R_total, v_canonical)

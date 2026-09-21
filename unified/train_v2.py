#!/usr/bin/env python3
"""One training loop covering every variant under test, selected by flag.

  --data dense        windows at fresh random offsets off the raw trajectories
                      instead of the 81,931 fixed ones (unified/dense.py)
  --model gru|attn    mix information across consecutive windows instead of
                      predicting each second in isolation (unified/context_model.py)
  --ate_weight        add an ATE20 surrogate to the loss (unified/train_seg.py)
  --aug_mode          bias perturbation, redrawn per window or held over a segment
  --gravity_weight    supervise the body-frame gravity direction (unified/gravity.py)
  --perframe_weight   supervise the within-window velocity trajectory, not just
                      its mean (unified/aux_heads.py)
  --scale_aware       predict direction and log-magnitude instead of a raw vector

Scoring is always the official scorer on the fixed val windows, so every variant
is measured on exactly the same quantity the leaderboard reports.
"""
from __future__ import annotations

import copy
import argparse
import os
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "local_eval")]
from unified.aux_heads import PerFrameHead, ScaleAwareHead, masked_vector_huber, perframe_target
from unified.context_model import ContextIMUNet
from unified.gradnorm import GradNorm
from unified.moe import traj_descriptors
from unified.fine_context import FineContextIMUNet
from unified.data import PLATFORMS, build_windows, load_index, sample_weights
from unified.dense import DenseWindows
from unified.gravity import cosine_loss, window_gravity_target
from unified.model import UnifiedIMUNet, vector_huber
from unified.segments import attach_pose, segment_starts
from unified.updir import N_CHANNELS, build_features, build_up_windows, canonicalize_up, preint_features, up_windows_for
from unified.equivariant import Canonicalizer, to_body
try:                                   # the scorer is only needed for training-time evaluation
    from score_val import score_breakdown
except ImportError:                    # inference-only deployments (hf/predict.py) ship no scorer
    score_breakdown = None

ATE_REF = 3.1160277267
ACCEL_BIAS, GYRO_BIAS = 0.1, 0.002        # official config/unified.yaml values


class Wrapped(nn.Module):
    """Backbone plus whichever optional heads this run enables."""

    def __init__(self, model_kind: str, scale_aware: bool, perframe_n: int, gravity: bool,
                 mixer_hidden: int = 128, width: float = 1.0, dropout: float = 0.2,
                 trunk: str = "cnn", fine: int = 5, moe: str = "none",
                 n_experts: int = 4, decompose: bool = False, in_channels: int = 6,
                 eqframe: bool = False, continuous: bool = False,
                 traj_film: bool = False, freq: bool = False, norm: str = "bn", preint: bool = False,
                 wide: bool = False, wide_to_gru: bool = False, wide_span: int = 5, at: str = "none",
                 lr0: str = "none", head: str = "linear", drag: bool = False, fuse: bool = False, ekf: bool = False, ekf_feats: bool = False):
        super().__init__()
        self.kind = model_kind
        # Learned canonical frame (EqNIO): the backbone then sees a, v1, v2 in a
        # frame that is invariant to yaw and to reflections across gravity, and
        # its velocity is rotated back out. Its parameters train with the rest.
        self.canon = Canonicalizer() if eqframe else None
        if model_kind == "single":
            self.net = UnifiedIMUNet()
        elif model_kind == "fine":
            self.net = FineContextIMUNet(fine=fine, hidden=mixer_hidden, width=width,
                                         dropout=dropout, moe=moe, n_experts=n_experts,
                                         decompose=decompose, in_channels=in_channels,
                                         continuous=continuous, traj_film=traj_film,
                                         freq=freq, norm=norm, preint=preint, wide=wide,
                                         wide_to_gru=wide_to_gru, wide_span=wide_span, at=at, lr0=lr0, head=head, drag=drag, fuse=fuse, ekf=ekf, ekf_feats=ekf_feats)
        else:
            self.net = ContextIMUNet(mixer=model_kind, hidden=mixer_hidden, width=width,
                                     dropout=dropout, trunk=trunk)
        if scale_aware:
            self.net.velocity_head = ScaleAwareHead(256)
        self.perframe = PerFrameHead(256, perframe_n) if perframe_n else None
        self.gravity = nn.Linear(256, 3) if gravity else None

    def forward(self, x):
        return self.net(x)


def ate_surrogate(v, R, dt, Q):
    """RMS path error of (B,K) segments after translation alignment.

    Umeyama's rotation is dropped: R is already ground truth, so the residual
    rotation is near identity and an SVD in the inner loop is slow and unstable
    in fp16. Skipping it can only over-state the error, so this stays an upper
    bound on the scored quantity.
    """
    disp = torch.einsum("bkij,bkj->bki", R, v) * dt
    P = torch.cumsum(disp, dim=1)
    P = P - P.mean(dim=1, keepdim=True)
    Q = Q - Q.mean(dim=1, keepdim=True)
    return torch.sqrt(((P - Q) ** 2).sum(-1).mean(dim=1) + 1e-8).mean()


FUSE = "center"          # "center": nearest-chunk-centre pick; "mean": centre-weighted average


EKF_OPTS = {"dv": "end", "conf": 0.0, "horizon": 0.0, "dR": "gyro"}   # advice_after_v20 §2: increment source and confidence mask; travel with the checkpoint args


def ekf_increments(up, M=None):
    """(..., K, T, >=23) up stream -> (dv (..., K, 3), dR (..., K, 3, 3), emask (..., K, 2) = [mask, conf/10]): window-to-
    window physics increments from the window-END frames of the derived channels (v_DR columns
    13:16, first two columns of R_{t->F0} 17:23, v_F0 23:26, calibration deviation 26):
    dR_k = R_k^T R_{k-1}; dv_k = v_DR,k - dR_k v_DR,k-1 (EKF_OPTS["dv"] == "end") or
    dv_k = R_k^T (v_F0,k - v_F0,k-1) (== "int": the window-integrated increment, advice §2.1);
    k = 0: dv = 0, dR = I. mask (§2.2): 1 where the increments are usable -- calibration deviation
    < EKF_OPTS["conf"] and |t_since| < EKF_OPTS["horizon"] s (each 0 = not applied); masked
    windows get dv = 0, dR = I (propagation degrades to "carry the previous velocity").
    M (B, 3, 3): per-row body-frame relabeling of the yaw augmentation (identity rows otherwise)
    -> dR' = M dR M^T, dv' = M dv."""
    from unified.calnet import rot6d_to_R
    if up.shape[-1] < 23:
        # ablation 'carry' on a plain up stream (no learned-INS channels): no increments, no mask
        assert EKF_OPTS["dv"] == "zero" and EKF_OPTS["dR"] == "none", "EKF increments need the slowdr up stream unless --ekf_dv zero --ekf_dR none"
        K = up.shape[-3]; sh = up.shape[:-3]
        dv = torch.zeros(*sh, K, 3, device=up.device); dR = torch.eye(3, device=up.device).expand(*sh, K, 3, 3).clone()
        return dv, dR, torch.stack([torch.ones(*sh, K, device=up.device), torch.zeros(*sh, K, device=up.device)], dim=-1)
    vend = up[..., -1, 13:16].float(); r6 = up[..., -1, 17:23].float()
    R = rot6d_to_R(r6 - torch.tensor([1.0, 0, 0, 0, 1.0, 0], device=up.device))   # rot6d_to_R adds the identity offset back
    Rp = torch.cat([R[..., :1, :, :], R[..., :-1, :, :]], dim=-3)          # R_{k-1}
    dR = R.transpose(-1, -2) @ Rp
    Mb = None
    if M is not None:
        # the v_DR columns arrive already rotated by the yaw augmentation (they are a body-vector
        # group); only dR, built from the un-rotated F0-frame R columns, needs conjugating -- and it
        # must be conjugated BEFORE forming dv (second review 2026-09-20: dv was double-rotated)
        assert M.shape[0] == up.shape[0], "yaw_M rows must match the batch (no --yaw_pair pair / --ctx_split with --ekf)"
        Mb = M.float()[:, None]; dR = Mb @ dR @ Mb.transpose(-1, -2)
    if EKF_OPTS["dv"] == "int" and up.shape[-1] >= 26:
        vf = up[..., -1, 23:26].float()
        vfp = torch.cat([vf[..., :1, :], vf[..., :-1, :]], dim=-2)
        dv = (R.transpose(-1, -2) @ (vf - vfp)[..., None])[..., 0]          # un-augmented body frame
        dv = 15.0 * torch.tanh(dv / 15.0)                                      # soft bound: a > 15 m/s one-second increment is an attitude failure, not motion
        if Mb is not None:
            dv = (Mb @ dv[..., None])[..., 0]
    elif EKF_OPTS["dv"] == "zero":
        dv = torch.zeros_like(vend)                                            # ablation: no velocity increment
    else:
        vp = torch.cat([vend[..., :1, :], vend[..., :-1, :]], dim=-2)
        dv = vend - (dR @ vp[..., None])[..., 0]
    if EKF_OPTS["dR"] == "none":
        dR = torch.eye(3, device=up.device, dtype=dR.dtype).expand_as(dR).clone()   # ablation 'carry': no rotation either
    mask = torch.ones(dv.shape[:-1], device=up.device, dtype=dv.dtype)
    if EKF_OPTS["conf"] > 0 and up.shape[-1] >= 27:
        mask = mask * (up[..., -1, 26].float() < EKF_OPTS["conf"]).to(dv.dtype)
    if EKF_OPTS["horizon"] > 0:
        mask = mask * (up[..., -1, 16].float().abs() * 10.0 < EKF_OPTS["horizon"]).to(dv.dtype)
    if EKF_OPTS["conf"] > 0 or EKF_OPTS["horizon"] > 0:
        dv = dv * mask[..., None]
        dR = dR * mask[..., None, None] + torch.eye(3, device=up.device, dtype=dR.dtype) * (1 - mask)[..., None, None]
    dv[..., 0, :] = 0.0; dR[..., 0, :, :] = torch.eye(3, device=up.device, dtype=dR.dtype)
    conf = up[..., -1, 26].float() / 10.0 if up.shape[-1] >= 27 else torch.zeros_like(mask)   # per-recording calibration deviation, ~[0, 1.2]
    return dv, dR, torch.stack([mask, conf], dim=-1)


def gate_feats(up):
    """(..., T, >=17) up stream -> (..., 3) per-window gate scalars: t_since (s/10), rest score, |v_DR| (m/s)."""
    if up.shape[-1] < 17:
        return torch.zeros(*up.shape[:-2], 3, device=up.device)               # plain up stream (ablation 'carry' without learned-INS channels)
    return torch.stack([up[..., 16].float().mean(-1), up[..., 9].float().mean(-1), torch.log1p(up[..., 13:16].float().mean(-2).norm(dim=-1))], dim=-1)


@torch.inference_mode()
def predict(model, X, U, frame, kind, K, norm, batch=1024):
    """Per-window velocity for the fixed val/test windows.

    X and U are raw (N, 200, 6) tensors; norm() does whatever the run's input
    pipeline does -- gravity channels, or canonicalisation, in which case it also
    returns the rotation that takes each chunk's prediction back to the body frame.
    """
    model.eval()
    if kind == "single":
        out = []
        for i in range(0, len(X), batch):
            xb, Rt = norm(X[i:i + batch, None], U[i:i + batch, None] if U is not None else None)
            v = model.net(xb[:, 0])[0].float()
            if Rt is not None:
                v = to_body(v[:, None], Rt)[:, 0]
            out.append(v.cpu().numpy())
        return np.concatenate(out)

    # Overlapping chunks at stride K//2, each window taking the prediction from
    # the chunk whose centre it is closest to. Measured worth -0.0044 val across
    # four seeds against non-overlapping chunks, for 2x inference.
    # A trajectory shorter than K windows (6/80 val, 8/89 test, all short drone
    # flights) is padded by repeating its last window; the fine model is told the
    # true length so the padding never enters the trunk or the GRU.
    chunks, dests, offs, lens, tstats = [], [], [], [], []
    want_tstats = kind == "fine" and getattr(model.net, "tfilm", None) is not None
    stride = max(1, K // 2)
    for _, rows in frame.groupby("traj_id", sort=False):
        rows = rows.sort_values("win_idx")
        order = rows.index.to_numpy()
        # windows are frames [win_idx*200, +200): consecutive win_idx is contiguous time
        assert (np.diff(rows["win_idx"].to_numpy()) == 1).all(), "gap in trajectory windows"
        if want_tstats:                                  # whole recording, raw units
            ts = traj_descriptors(X[torch.as_tensor(order, device=X.device)].reshape(-1, X.shape[-1]))
        starts = list(range(0, max(1, len(order) - K + 1), stride))
        if starts[-1] != max(0, len(order) - K):
            starts.append(max(0, len(order) - K))
        for s in starts:
            piece = order[s:s + K]
            dests.append(piece)
            offs.append(np.arange(len(piece)))
            lens.append(len(piece))
            if want_tstats:
                tstats.append(ts)
            if len(piece) < K:
                piece = np.concatenate([piece, np.repeat(piece[-1], K - len(piece))])
            chunks.append(piece)

    chunks = np.stack(chunks)
    out = np.zeros((len(X), 3), np.float32)
    best = np.full(len(X), np.inf)
    wsum = np.zeros(len(X), np.float64)
    per_batch = max(1, batch // K)
    for i in range(0, len(chunks), per_batch):
        sel = torch.from_numpy(chunks[i:i + per_batch]).to(X.device)
        xb, Rt = norm(X[sel], U[sel] if U is not None else None)
        if kind == "fine":
            ln = torch.as_tensor(lens[i:i + per_batch], device=X.device)
            ts = torch.stack(tstats[i:i + per_batch]) if want_tstats else None
            fuse = getattr(model.net, "fuse", False); ekf = getattr(model.net, "ekf", False)
            vdr = U[sel][..., 13:16].float().mean(2) if fuse else None   # (b, K, 3) m/s
            gf = gate_feats(U[sel]) if (fuse or ekf) else None
            dv, dR, em = ekf_increments(U[sel]) if ekf else (None, None, None)
            v = model.net(xb, lengths=ln, tstats=ts, vdr=vdr, gfeat=gf, dv=dv, dR=dR, emask=em)[0].float()
        else:
            v = model.net(xb)[0].float()
        if Rt is not None:
            v = to_body(v, Rt)
        v = v.cpu().numpy()
        for j, (dest, off) in enumerate(zip(dests[i:i + per_batch], offs[i:i + per_batch])):
            d = np.abs(off - (len(dest) - 1) / 2.0)
            if FUSE == "mean":
                # every chunk contributes, weighted by closeness to its centre
                w = 1.0 - d / (len(dest) / 2.0 + 1.0)
                out[dest] += (v[j, :len(dest)] * w[:, None]).astype(np.float32)
                wsum[dest] += w
                continue
            take = d < best[dest]
            out[dest[take]] = v[j, :len(dest)][take]
            best[dest[take]] = d[take]
    if FUSE == "mean":
        out /= np.maximum(wsum, 1e-9)[:, None]
    return out


def safe_save(obj, path) -> None:
    """Write to a same-directory temp file, fsync, atomic replace, then verify the
    file loads (lane safety 2026-09-16: a timeout or disk hiccup must not leave a
    truncated checkpoint in place of a good one)."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        torch.save(obj, fh); fh.flush(); os.fsync(fh.fileno())
    torch.load(tmp, map_location="cpu", weights_only=False)      # readable, or raise before replacing
    os.replace(tmp, path)


def load_teacher(tag: str, dev):
    """A frozen expert: its network plus the input pipeline it was trained with."""
    c = torch.load(ROOT / f"unified/ckpt_{tag}.pt", map_location=dev, weights_only=False)
    ar = c["args"]
    if ar.get("eqframe"):
        raise NotImplementedError("teacher with --eqframe")
    net = Wrapped(ar["model"], False, 0, False, ar["mixer_hidden"], ar["width"], ar["dropout"],
                  ar["trunk"], ar["fine"], ar["moe"], ar["n_experts"], ar["decompose"],
                  N_CHANNELS[ar["grav"]], False, ar.get("continuous", False),
                  ar.get("traj_film", False), ar.get("freq", False), ar.get("norm", "bn"),
                  ar.get("preint", False), ar.get("wide", False), ar.get("wide_to_gru", False)).to(dev)
    getattr(net.net, "adapt_tstat", lambda *a: None)(c["model"])
    net.load_state_dict(c["model"], strict=True)
    net.eval()
    for q in net.parameters():
        q.requires_grad_(False)
    mean = torch.as_tensor(c["imu_mean"], device=dev); std = torch.as_tensor(c["imu_std"], device=dev)
    grav = ar["grav"]

    def norm(x, up):
        return ((build_features(x, up, grav) - mean) / std).transpose(-1, -2).contiguous()
    print(f"teacher {tag}: epoch {c['epoch']} val {c['score']:.4f} platforms {ar.get('platforms')}")
    return net, norm, ar["model"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["index", "dense"], default="index")
    ap.add_argument("--model", choices=["single", "gru", "attn", "fine"], default="single")
    ap.add_argument("--fine", type=int, default=5,
                    help="sub-steps per window the mixer sees (--model fine); 1 reduces to gru")
    ap.add_argument("--moe", choices=["none", "head", "mlp", "film"], default="none")
    ap.add_argument("--n_experts", type=int, default=4)
    ap.add_argument("--decompose", action="store_true",
                    help="predict a segment-level term plus a per-window deviation")
    ap.add_argument("--entropy_weight", type=float, default=0.01,
                    help="reward gate entropy; guards against collapse onto one expert")
    ap.add_argument("--gate_supervised", action="store_true",
                    help="put the platform cross-entropy on the gate logits, making the "
                         "gate infer the embodiment from IMU alone")
    ap.add_argument("--mixer_hidden", type=int, default=128)
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--trunk", choices=["cnn", "lstm"], default="cnn")
    ap.add_argument("--eqframe", action="store_true",
                    help="canonicalise into an O(2)-equivariant frame before the backbone "
                         "and rotate the prediction back (EqNIO). Input becomes a, v1, v2.")
    ap.add_argument("--grav_perturb", type=float, default=0.0,
                    help="degrees: randomly tilt the estimated up direction during training, "
                         "which EqNIO reports stabilises the frame against estimation error")
    ap.add_argument("--grav", choices=list(N_CHANNELS), default="none",
                    help="re-express the IMU about an estimated gravity direction as extra "
                         "channels; slow/fast pick the complementary filter's time constant, "
                         "both hands the network each")
    ap.add_argument("--seg_len", type=int, default=1)
    ap.add_argument("--batch_segs", type=int, default=512)
    ap.add_argument("--steps_per_epoch", type=int, default=160)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10**9,
                    help="epochs without improvement before stopping. Off by default: "
                         "OneCycleLR anneals over the full --epochs, so stopping early "
                         "leaves the run at a high learning rate. Measured on 57 full "
                         "120-epoch runs, cutting at epoch 80 costs 0.0103 -- larger than "
                         "most thresholds this project resolves.")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--ate_weight", type=float, default=0.0)
    ap.add_argument("--gravity_weight", type=float, default=0.0)
    ap.add_argument("--perframe_weight", type=float, default=0.0)
    ap.add_argument("--perframe_n", type=int, default=20)
    ap.add_argument("--scale_aware", action="store_true")
    ap.add_argument("--platform_loss_weight", type=float, default=0.05)
    ap.add_argument("--aug_mode", choices=["none", "window", "segment"], default="none")
    ap.add_argument("--ema", type=float, default=0.0,
                    help="decay of an EMA copy of the weights, evaluated and saved "
                         "separately as <tag>e (0 = off)")
    ap.add_argument("--continuous", action="store_true",
                    help="run the trunk over the whole segment as one signal (--model fine)")
    ap.add_argument("--gradnorm", type=float, default=-1.0,
                    help="GradNorm alpha over the four platform losses (<0 = off, 0 = pure "
                         "gradient balancing, 1.5 = paper default)")
    ap.add_argument("--aggr_weight", type=float, default=1.0,
                    help="sampling weight multiplier for aggressive drone trajectories "
                         "(acc_dev90 > 1.5 m/s^2, the regime the test drone flies in)")
    ap.add_argument("--yaw_aug", nargs="*", default=None,
                    help="platforms whose segments get a random rotation about the body z "
                         "axis (a different sensor mounting; exact for a yaw-symmetric quadrotor)")
    ap.add_argument("--wide", action="store_true",
                    help="add a 5-second low-frequency neighbourhood branch per window (R3)")
    ap.add_argument("--preint", action="store_true",
                    help="append 9 gyro pre-integration channels (updir.preint_features) that "
                         "bypass the trunk through a small zero-initialised branch")
    ap.add_argument("--yaw_ident", type=float, default=0.0,
                    help="probability of rotating an identity-source drone segment (IMU frame = "
                         "label frame) about its z axis by a uniform angle: a different sensor "
                         "mounting, exact for any rigid body. The label is rotated through "
                         "R_ext so a non-identity source would be handled too, but only "
                         "sig==identity segments are selected: fast lateral/backward flight "
                         "is absent from training (0%% of >3 m/s windows at azimuth 45-180 deg) "
                         "and present in the hold-out tail (0023/0024: 54%%)")
    ap.add_argument("--ctx_split", type=float, default=0.0,
                    help="context-split-v1 (2026-09-16): with this probability a segment of valid "
                         "length L >= 10 is split at m ~ U[5, L-5] into two continuous rows that "
                         "get independent GRU / neighbourhood context; every valid window stays "
                         "in exactly one row with its own label; dedicated generator")
    ap.add_argument("--acc_gain", type=float, default=0.0,
                    help="gain-v1 (2026-09-16): one positive scalar g ~ U(1-G, 1+G) per segment "
                         "multiplies the raw accelerometer (gravity included), labels and up "
                         "channel unchanged; drawn from a dedicated generator so the sampling "
                         "stream is identical to a run without it; applied after T")
    ap.add_argument("--lookahead", type=int, default=0,
                    help="Lookahead (Zhang et al. 2019) with this k over the existing AdamW: every k "
                         "SUCCESSFUL optimizer steps slow += alpha (fast - slow), fast = slow; "
                         "evaluation, checkpoints and the final model use the slow weights")
    ap.add_argument("--lookahead_alpha", type=float, default=0.5)
    ap.add_argument("--head", choices=["linear", "vb"], default="linear",
                    help="velocity readout: linear (default) or VB-H-coordinate-v1 bin expectation "
                         "(512 bins on [-20, 20] m/s, coordinate-conditioned)")
    ap.add_argument("--ext_until", type=int, default=0, help="road 4: from this epoch on sample no external recordings (competition-only finish)")
    ap.add_argument("--ext_weight", type=float, default=1.0, help="road 4: sampling weight multiplier for external recordings (traj_id contains 'neurobem' or '_ext_')")
    ap.add_argument("--att", default="", help="learned-INS step B: AttNet weights (runs/attnet/*.pt) for the attitude / DR channels; stored in the checkpoint")
    ap.add_argument("--cal", default="", help="learned-INS step C: CalNet weights (runs/calnet/*.pt) applied before the attitude / DR channels; stored in the checkpoint")
    ap.add_argument("--dr_horizon", type=float, default=0.0, help="path 2 (a): v_DR valid only within this many seconds of the rest anchor (0 = off)")
    ap.add_argument("--dr_squash", type=float, default=0.0, help="path 2 (c): v_DR -> s*tanh(v/s) with this s in m/s (0 = off)")
    ap.add_argument("--ins_drop", type=float, default=0.0, help="path 2 (b): per-sample dropout probability of the INS up-stream columns 10:17 (training only)")
    ap.add_argument("--drag_pose", action="store_true", help="road 3: drag-consistent attitude synthesis on S>1 racing-family samples (extra pitch from the recording's fitted drag slope)")
    ap.add_argument("--ekf", action="store_true", help="road 1 stage 2: learned EKF over the chunk (physics propagation from the derived channels, learned gain)")
    ap.add_argument("--ekf_sup", type=float, default=0.0, help="road 1 stage 2: BCE weight teaching the gain from which of v_direct / v_prop is closer to y")
    ap.add_argument("--ekf_dR", choices=["gyro", "none"], default="gyro", help="ablation (review 2026-09-20 night): 'none' = identity rotation between windows ('carry' when combined with --ekf_dv zero)")
    ap.add_argument("--ekf_dv", choices=["end", "int", "zero"], default="end", help="advice_after_v20 §2.1: EKF increment source -- window-end v_DR differences or the window-integrated v_F0 increment")
    ap.add_argument("--ekf_conf", type=float, default=0.0, help="§2.2: confidence mask -- increments off where the CalNet calibration deviation (up column 26) >= this (0 = off; ~2 = one tolerance each on S, b, map)")
    ap.add_argument("--ekf_horizon", type=float, default=0.0, help="§2.2: increments off beyond this many seconds from the rest anchor (0 = off)")
    ap.add_argument("--ekf_feats", action="store_true", help="§2.2/2.3: gain also reads the confidence mask and the innovation log1p|v_direct - v_prop| (own zero-init linear)")
    ap.add_argument("--ekf_mask_direct", action="store_true", help="review (2026-09-20 night): on masked windows (outside --ekf_horizon / --ekf_conf) the fused output is the direct head; the gain is learned only where the increments are informative")
    ap.add_argument("--ekf_margin", type=float, default=0.0, help="review (2026-09-20 night): the gain target trusts propagation only when it beats the direct head by this many m/s")
    ap.add_argument("--ekf_soft", type=float, default=0.0, help="§2.3: soft gain target sigmoid((e_prop - e_direct) / this) instead of the 0/1 target (0 = hard)")
    ap.add_argument("--fuse_sup", type=float, default=0.0, help="road 1 stage 1: BCE weight supervising the fuse gate with 1[|v_DR - y| < --fuse_thresh]")
    ap.add_argument("--fuse_thresh", type=float, default=1.0, help="road 1 stage 1: v_DR error (m/s) below which the gate target is 'open'")
    ap.add_argument("--fuse", action="store_true", help="learned-INS step A: gated fusion of the dead-reckoned velocity channel (--grav slowdr)")
    ap.add_argument("--fuse_direct_weight", type=float, default=0.1, help="--fuse: auxiliary Huber weight on the direct head before fusion")
    ap.add_argument("--drag_head", action="store_true", help="road 3: gated recording-level linear map from window-mean raw IMU to velocity")
    ap.add_argument("--drag_l1", type=float, default=0.0, help="L1 penalty on the drag coefficients A")
    ap.add_argument("--drag_aux", type=float, default=0.0, help="road 3b: auxiliary Huber loss making the linear drag term alone explain the velocity (drone rows)")
    ap.add_argument("--anchor_inc_weight", type=float, default=0.5, help="--lr0 anchor: weight of the increment loss")
    ap.add_argument("--anchor_mu", type=float, default=5.0, help="--lr0 anchor: solver weight of the increment consistency term")
    ap.add_argument("--lr0", choices=["none", "direct", "graph", "anchor"], default="none",
                    help="LR0-v1 label-space increment correction head (see fine_context)")
    ap.add_argument("--at", choices=["none", "static", "content"], default="none",
                    help="AT1/AT2: local residual cross-attention after the GRU (see fine_context)")
    ap.add_argument("--wide_span", type=int, default=5, choices=[1, 5],
                    help="R3-1s-control: 1 zeroes the four neighbouring windows of the --wide "
                         "branch (same capacity and filter, one second of view)")
    ap.add_argument("--wide_to_gru", action="store_true",
                    help="C1: the --wide feature also enters the fine tokens (zero-init), so "
                         "the GRU sees it; without it the branch only reaches the summary")
    ap.add_argument("--init_full", default=None,
                    help="warm start: load a train_v2 checkpoint's weights and input "
                         "normalisation; new zero-initialised modules may be missing "
                         "(matched warm-start for C0/C1, not a resume of the optimizer)")
    ap.add_argument("--tdil_aa", action="store_true",
                    help="anti-aliased time dilation for k > 1 (4x oversampled gather, 100 Hz FIR, decimate)")
    ap.add_argument("--tdil_short", choices=["keep", "cancel"], default="keep",
                    help="short trajectories under time dilation: keep k with a shorter valid "
                         "length (default, since 2026-09-14) or cancel k (the earlier sampler, v2)")
    ap.add_argument("--yaw_set", choices=["uniform", "pi", "pi_mirror"], default="uniform",
                    help="view set for --yaw_ident: 'uniform' angle in [0, 2pi); 'pi' = 180 deg only "
                         "(leaves a diagonal drag tensor invariant, so speed cues stay valid); "
                         "'pi_mirror' = {180 deg, mirror x, mirror y} (mirrors: a'=Ma, w'=-Mw, "
                         "v'=Mv, up'=M up -- exact kinematics with omega a pseudovector)")
    ap.add_argument("--yaw_pair", choices=["off", "replace", "pair"], default="off",
                    help="with --yaw_ident: the rotated view is APPENDED to the batch instead of "
                         "replacing the original. 'replace' keeps the current loss (original "
                         "selected rows get no loss, rotated rows direction-only) but both views "
                         "pass through BN -- the matched control; 'pair' gives the original view "
                         "its full vector loss and the rotated view the direction loss, 1/2 each "
                         "(reviewer proposal 2026-09-15)")
    ap.add_argument("--yaw_dironly", action="store_true",
                    help="with --yaw_ident: on the rotated segments penalise the direction of the "
                         "prediction only (it is rescaled to the label's magnitude before the "
                         "loss). A yaw is an exact frame change but not an exact flight: the "
                         "airframe's drag is not yaw-symmetric, so a rotated forward flight "
                         "carries the wrong speed cue for a real lateral one (R3yaw_s42: "
                         "hold-out 0023/24 angle 34->20 deg but speed 4.4 predicted as 6.6)")
    ap.add_argument("--init_trunk_pad", action="store_true",
                    help="init_trunk pre-trained with fewer input channels: copy its stem weights "
                         "into the first channels and zero-init the extra ones (e.g. slow -> slowrest)")
    ap.add_argument("--init_trunk_any_grav", action="store_true",
                    help="accept an --init_trunk pre-trained under another --grav of the same width")
    ap.add_argument("--data_seed", type=int, default=None,
                    help="re-seed the RNG after the model is built: same initial weights "
                         "as --seed, a different batch order / augmentation draw")
    ap.add_argument("--swa_last", type=int, default=0,
                    help="average the weights of the last N epochs (equal weights, "
                         "BN buffers included) into a second candidate <tag>w")
    ap.add_argument("--save_last", action="store_true",
                    help="also write val (and hold-out) CSVs from the LAST epoch's weights, "
                         "as sub_val_<tag>_last.csv / sub_hold_<tag>_last.csv")
    ap.add_argument("--tdil_off_last", type=int, default=0,
                    help="switch time dilation off for the last N epochs (augmentation fade-out)")
    ap.add_argument("--holdout", default=None,
                    help="runs/holdout.json: exclude these train trajectories from fine-tuning and "
                         "score them each epoch as a stress set (non-strict: pre-training saw their IMU)")
    ap.add_argument("--tdil", type=float, nargs=2, default=None, metavar=("KMIN", "KMAX"),
                    help="physics-consistent time dilation of training segments by k~U(KMIN,KMAX)")
    ap.add_argument("--tdil_platforms", nargs="+", default=["drone"])
    ap.add_argument("--tdil_prob", type=float, default=0.5)
    ap.add_argument("--strat_prob", type=float, default=0.0,
                    help="share of drone segments placed around a window from a rare speed x gyro bin")
    ap.add_argument("--tdil_fast", type=float, nargs=2, default=None, metavar=("KMIN", "KMAX"),
                    help="separate k range for the identity-extrinsic (fast) drone source")
    ap.add_argument("--sdil", type=float, nargs=2, default=None, metavar=("SMIN", "SMAX"),
                    help="translation scaling s~U(SMIN,SMAX): v'=s v, w'=w, f'=s(f-g up)+g up")
    ap.add_argument("--sdil_fast", type=float, nargs=2, default=None, metavar=("SMIN", "SMAX"),
                    help="S-fast (2026-09-18): translation scaling s~U(SMIN,SMAX) drawn only for "
                         "identity-extrinsic (racing) drone recordings; others keep s=1 (or --sdil)")
    ap.add_argument("--aug_mix", action="store_true",
                    help="with --tdil and --sdil: each augmented segment gets T or S, never both")
    ap.add_argument("--grav_canon", action="store_true",
                    help="rotate each window's IMU (and up estimates) so the slow up points "
                         "along +z before the features; labels unchanged")
    ap.add_argument("--sig_weight", type=float, default=0.0,
                    help="auxiliary CE on the fitted extrinsic signature class (8 classes, "
                         "train-time ground truth) from the segment-mean features")
    ap.add_argument("--label_frame", choices=["label", "imu"], default="label",
                    help="imu: DIAGNOSTIC ONLY -- rotate the training targets into the IMU frame "
                         "with the fitted per-trajectory extrinsic and rotate val predictions back "
                         "with the ground-truth extrinsic; no test CSV is written")
    ap.add_argument("--gyro_frame", action="store_true",
                    help="DIAGNOSTIC ONLY -- rotate the gyro into the label/accelerometer frame with "
                         "the fitted per-trajectory extrinsic (train and val); no test CSV")
    ap.add_argument("--gt_up", action="store_true",
                    help="DIAGNOSTIC ONLY -- replace the estimated up by the ground-truth up "
                         "(IMU frame, accelerometer sign) in train and val; no test CSV is written")
    ap.add_argument("--idw", type=float, default=1.0,
                    help="sampling weight multiplier for drone trajectories whose fitted extrinsic "
                         "is the identity (the aggressive, noisy source), renormalised within drone")
    ap.add_argument("--trunk_lr_mult", type=float, default=1.0,
                    help="learning-rate multiplier for the trunk (stem/stage2-4), e.g. 0.1 to "
                         "protect a pre-trained trunk while the fresh GRU and heads learn")
    ap.add_argument("--init_net", default=None,
                    help="path to a unified/pretrain_seg.py checkpoint: initialise everything "
                         "but the velocity/platform heads from it (segment-level pre-training)")
    ap.add_argument("--init_trunk", default=None,
                    help="path to a unified/pretrain.py checkpoint: initialise stem/stage2-4 "
                         "from it (self-supervised masked IMU modelling)")
    ap.add_argument("--recon_weight", type=float, default=0.0,
                    help="joint objective: keep the masked-reconstruction loss during fine-tuning "
                         "(needs an --init_trunk checkpoint that carries its decoder); applied "
                         "to a random quarter of each batch's windows with the pre-training mask")
    ap.add_argument("--sam", type=float, default=0.0,
                    help="SAM perturbation radius rho (0 = off); doubles the cost of a step")
    ap.add_argument("--rsc", type=float, default=0.0,
                    help="Representation Self-Challenging: on half the batches after warm-up, "
                         "zero this fraction of the feature channels the velocity loss is "
                         "most sensitive to, and train the head on the rest (0 = off)")
    ap.add_argument("--band_aug", type=float, default=0.0,
                    help="frequency-band randomisation: multiply IMU spectral content above "
                         "10 Hz by a smooth random gain of this amplitude, per segment (0 = off)")
    ap.add_argument("--teachers", nargs="*", default=None,
                    help="platform=tag pairs, e.g. drone=xdroneS_s42: each platform's segments "
                         "also regress onto that expert's prediction (knowledge distillation)")
    ap.add_argument("--kd_weight", type=float, default=0.5)
    ap.add_argument("--ens_teachers", nargs="*", default=None,
                    help="C2 (2026-09-15): tags of same-recipe teachers; every segment also "
                         "regresses onto the MEAN of their predictions on the same input "
                         "(the three-seed prediction mean scores 0.1704 val / 1.063 hold-out "
                         "against 0.1766 / 1.104 for one model, and weight averaging does not "
                         "keep that), weighted by --kd_weight")
    ap.add_argument("--trainval", action="store_true",
                    help="train on train+val (final-recipe runs only: the val score printed "
                         "each epoch is then on seen data and means nothing)")
    ap.add_argument("--noise_aug", type=float, default=0.0,
                    help="white noise on the raw IMU, as a fraction of each channel's std")
    ap.add_argument("--macro_loss", action="store_true",
                    help="aggregate the velocity Huber like the official AVE: mean over a "
                         "segment's windows, then over the platform's segments, then over "
                         "platforms present -- instead of one flat mean over windows")
    ap.add_argument("--time_mask", type=float, default=0.0,
                    help="fraction of training windows that get one random span of 20-60 "
                         "frames zeroed (after normalisation) -- SpecAugment-style regulariser")
    ap.add_argument("--speed_weight", type=float, default=0.0,
                    help="if > 0, per-window loss weight 1 + |target|/this (m/s), mean-normalised")
    ap.add_argument("--platforms", nargs="+", default=None,
                    help="train on these platforms only (diagnostic expert; val still all four)")
    ap.add_argument("--norm", choices=["bn", "gn"], default="bn",
                    help="trunk normalisation: BatchNorm or GroupNorm(8) (--model fine)")
    ap.add_argument("--traj_desc", type=int, default=1, choices=[1, 2], help="advice_after_v20 §3: 2 = yaw-invariant recording descriptors for --traj_film")
    ap.add_argument("--traj_film", action="store_true",
                    help="FiLM the trunk on whole-trajectory descriptors (--model fine)")
    ap.add_argument("--freq", action="store_true",
                    help="add a log-spectrum branch to each window's summary (--model fine)")
    ap.add_argument("--rel_loss", type=float, default=0.0,
                    help="if > 0, Huber on the error divided by (|target| + this) instead of "
                         "the absolute error, in m/s")
    ap.add_argument("--huber_beta_speed", type=float, default=0.0,
                    help="method 1a: per-window Huber beta = huber_beta + this * |target| (fast windows move from "
                         "median- towards mean-seeking; slow windows unchanged)")
    ap.add_argument("--speed_density_weight", type=float, default=0.0,
                    help="method 1c: window weight = inverse frequency of the target-speed bin (0-2,2-4,...,>10 m/s, "
                         "frequencies from the training targets), capped at this factor, mean-normalised")
    ap.add_argument("--logmag_weight", type=float, default=0.0,
                    help="method 1b: add this * mean (log(1+|v|) - log(1+|y|))^2 over valid windows")
    ap.add_argument("--huber_beta", type=float, default=0.25,
                    help="vector Huber transition (m/s); errors below it get gradient e/beta")
    ap.add_argument("--short_traj", choices=["mask", "drop"], default="mask",
                    help="dense sampling of trajectories shorter than seg_len: keep them "
                         "and mask the missing windows, or leave them out entirely")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--solution", default="local_eval/val_solution.csv")
    a = ap.parse_args()

    if a.model != "single" and a.seg_len < 2:
        a.seg_len = 10                                  # context needs neighbours
    need_pose = a.ate_weight > 0 or a.gravity_weight > 0
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark = os.environ.get("TARTANIMU_CUDNN_BENCHMARK", "1") != "0"   # 0 on hosts where re-benchmarking dominates
    t0 = time.time()

    idx = load_index()
    train, val, test = idx["train"], idx["val"], idx["test"]
    hold = None
    if a.holdout:
        import json
        held = set(json.load(open(ROOT / a.holdout))["held"])
        hold = train[train["traj_id"].isin(held)].reset_index(drop=True)
        train = train[~train["traj_id"].isin(held)].reset_index(drop=True)
        print(f"holdout: {hold.traj_id.nunique()} train trajectories ({len(hold)} windows) excluded "
              f"from fine-tuning and scored as stress set; {train.traj_id.nunique()} remain")
    if a.trainval:
        train = pd.concat([train, val], ignore_index=True)
        print(f"trainval: {train.traj_id.nunique()} trajectories, {len(train)} windows")
    speed_bin_w = None
    if a.speed_density_weight > 0:
        sp = np.sqrt(train["vx"].to_numpy() ** 2 + train["vy"].to_numpy() ** 2 + train["vz"].to_numpy() ** 2)
        edges = np.array([0, 2, 4, 6, 8, 10, np.inf]); cnt = np.histogram(sp, edges)[0].astype(np.float64)
        w = np.clip((cnt.sum() / len(cnt)) / np.maximum(cnt, 1), 1.0, a.speed_density_weight)   # rare bins up to cap x, common bins 1
        speed_bin_w = torch.tensor(w, dtype=torch.float32)
        print("speed density weights per bin (0-2,2-4,4-6,6-8,8-10,>10):", np.round(w, 2).tolist(), "counts", cnt.astype(int).tolist())
    Xtr = build_windows(train)
    if a.eqframe and a.grav != "none":
        raise SystemExit("--eqframe replaces the input with a, v1, v2; combine with --grav later")
    n_ch = 9 if a.eqframe else N_CHANNELS[a.grav]
    need_up = a.eqframe or a.grav != "none"
    from unified import updir as _updir
    _updir.DR_OPTS.update(horizon=a.dr_horizon, squash=a.dr_squash)
    EKF_OPTS.update(dv=a.ekf_dv, conf=a.ekf_conf, horizon=a.ekf_horizon, dR=a.ekf_dR)
    from unified.fine_context import EKF_RUNTIME
    EKF_RUNTIME["mask_direct"] = bool(a.ekf_mask_direct)
    from unified.moe import TRAJ_DESC
    TRAJ_DESC["version"] = a.traj_desc
    att_m = cal_m = None; att_key = ""
    if a.att or a.cal:
        import hashlib
        if a.att:
            from unified.attnet import AttNet
            ca = torch.load(a.att, map_location="cpu", weights_only=False); att_m = AttNet(**ca["config"]); att_m.load_state_dict(ca["state_dict"]); att_m.eval()
        if a.cal:
            from unified.calnet import CalNet
            cc = torch.load(a.cal, map_location="cpu", weights_only=False); cal_m = CalNet(**cc["config"]); cal_m.load_state_dict(cc["state_dict"]); cal_m.eval()
        att_key = hashlib.md5(b"".join(open(f, "rb").read() for f in (a.att, a.cal) if f)).hexdigest()[:8]   # content hash (review 1.3)
        print(f"learned-INS: att={Path(a.att).name or None} cal={Path(a.cal).name or None} (cache key upatt_{att_key})", flush=True)
    Utr = up_windows_for(train, a.grav, att_m, cal_m, att_key) if need_up else None

    # Statistics over the FULL channel set, from a sample of the fixed train
    # windows. Taking them from the raw six and leaving the derived channels
    # unnormalised would put the vertical specific force (about 9.8) and the
    # horizontal one (near 0) on wildly different scales.
    solution = pd.read_csv(a.solution)
    model = Wrapped(a.model, a.scale_aware, a.perframe_n if a.perframe_weight > 0 else 0,
                    a.gravity_weight > 0, a.mixer_hidden, a.width, a.dropout,
                    a.trunk, a.fine, a.moe, a.n_experts, a.decompose, n_ch,
                    a.eqframe, a.continuous, a.traj_film, a.freq, a.norm, a.preint, a.wide,
                    a.wide_to_gru, a.wide_span, a.at, a.lr0, a.head, a.drag_head, a.fuse, a.ekf, a.ekf_feats).to(dev)
    if a.lr0 == "anchor": model.net.anchor_mu = a.anchor_mu

    init_full = None
    if a.init_full:
        init_full = torch.load(ROOT / a.init_full, map_location=dev, weights_only=False)
        res = model.load_state_dict(init_full["model"], strict=False)
        assert not res.unexpected_keys, res.unexpected_keys
        new = {k.rsplit(".", 1)[0] for k in res.missing_keys}
        bad = [k for k in res.missing_keys if not (k.startswith("net.wide_tok") or k.startswith("net.at_") or k.startswith("net.lr0_") or k.startswith("net.vb_") or k.startswith("net.drag_"))]
        assert not bad, bad
        print(f"init_full: {a.init_full} (epoch {init_full.get('epoch')}, val {init_full.get('score')}); "
              f"fresh: {sorted(new) or 'none'}")
    if a.sig_weight > 0:
        model.sig_head = nn.Linear(model.net.phase.shape[1], 8).to(dev)
    if a.init_net:
        if a.init_trunk:
            raise SystemExit("--init_net and --init_trunk are alternatives")
        pre = torch.load(ROOT / a.init_net, map_location=dev, weights_only=False)
        if pre["grav"] != a.grav:
            raise SystemExit(f"init_net was pre-trained with grav={pre['grav']}, run uses {a.grav}")
        res = model.net.load_state_dict(pre["net"], strict=False)
        assert not res.unexpected_keys, res.unexpected_keys
        heads = {k.split(".")[0] for k in res.missing_keys}
        assert heads <= {"velocity_head", "platform_head", "segment_head"}, heads
        print(f"init_net: {len(pre['net'])} tensors from {a.init_net} (step {pre['step']}/{pre['total']}, "
              f"loss {pre.get('masked_mse')}); fresh: {sorted(heads)}")
    if a.recon_weight > 0 and not a.init_trunk:
        raise SystemExit("--recon_weight needs --init_trunk")
    recon_mask = None
    if a.init_trunk:
        pre = torch.load(ROOT / a.init_trunk, map_location=dev, weights_only=False)
        if pre["grav"] != a.grav:
            c_pre, c_run = N_CHANNELS[pre["grav"]], N_CHANNELS[a.grav]
            if a.init_trunk_pad and c_pre < c_run:
                trunk = dict(pre["trunk"]); w = trunk["stem.0.weight"]
                w2 = torch.zeros(w.shape[0], c_run, w.shape[2], dtype=w.dtype, device=w.device); w2[:, :c_pre] = w
                trunk["stem.0.weight"] = w2; pre["trunk"] = trunk
                print(f"init_trunk: stem padded {c_pre} -> {c_run} input channels (extra channels zero-init)")
            elif not (a.init_trunk_any_grav and c_pre == c_run):
                raise SystemExit(f"init_trunk was pre-trained with grav={pre['grav']}, run uses {a.grav}")
            else:
                print(f"init_trunk: pre-trained with grav={pre['grav']}, run uses {a.grav} (same width)")
        res = model.net.load_state_dict(pre["trunk"], strict=False)
        assert not res.unexpected_keys, res.unexpected_keys
        if a.recon_weight > 0:
            from unified.pretrain_dense import make_decoder, random_mask
            if "decoder" not in pre:
                raise SystemExit("--recon_weight needs a pre-training checkpoint with its decoder")
            model.decoder = make_decoder(model.net.stage4[-1].main[-1].num_features, n_ch).to(dev)
            model.decoder.load_state_dict(pre["decoder"])
            recon_mask = (pre.get("mask_setting", 0.3), pre.get("n_span", 3))
            print(f"joint reconstruction: weight {a.recon_weight}, mask {recon_mask}")
        print(f"init_trunk: {len(pre['trunk'])} tensors from {a.init_trunk} "
              f"(splits {pre['splits']}, masked MSE {pre.get('masked_mse')})")

    def canonical(x, up):
        """Raw (B,K,T,6) imu and (B,K,T,6) up -> ((B,K,T,9), R_total) or (x, None)."""
        if not a.eqframe:
            if a.grav_canon:
                if up is None:
                    raise SystemExit("--grav_canon needs --grav slow/both (the up estimate)")
                x, up = canonicalize_up(x, up)
            f = build_features(x, up, a.grav)
            if a.preint:
                f = torch.cat([f, preint_features(x)], dim=-1)
            return f, None
        return model.canon(x, up[..., 3:6])          # slow filter: better on 3 of 4 platforms

    # statistics over the channels the backbone actually sees. With --eqframe the
    # frame network is untrained here, so its yaw is arbitrary; that is fine,
    # because a random yaw makes the x/y statistics symmetric and gravity
    # alignment -- which sets the z statistics -- is deterministic.
    rng = np.random.default_rng(0)
    pick = rng.choice(len(Xtr), min(20000, len(Xtr)), replace=False)
    with torch.no_grad():
        xs = torch.from_numpy(np.asarray(Xtr[pick])).to(dev)
        us = torch.from_numpy(np.asarray(Utr[pick])).to(dev) if Utr is not None else None
        if a.eqframe:
            stat, _ = canonical(xs[:, None], us[:, None])
            stat = stat[:, 0]
        else:
            stat = canonical(xs, us)[0]
        raw_std = xs.reshape(-1, 6).std(dim=0)          # for --noise_aug, raw units
        mean_g = stat.mean(dim=(0, 1))
        std_g = stat.std(dim=(0, 1)).clamp(min=1e-6)
        del stat, xs, us
    if a.grav == "slowdr":                               # review 1.1: INS channels on a fixed physical scale, not data statistics
        mean_g[21:24] = 0.0; std_g[21:24] = 5.0          # v_DR / 5 m/s
        mean_g[24] = 0.0; std_g[24] = 1.0                # t_since already in s/10
    if a.grav.startswith("slowrest") or a.grav == "slowdr":
        print("imu_mean[14:25] " + " ".join(f"{float(v):.3f}" for v in mean_g[14:25]) + "\nimu_std[14:25]  " + " ".join(f"{float(v):.3f}" for v in std_g[14:25]), flush=True)
    mean, std = mean_g.cpu().numpy(), std_g.cpu().numpy()
    if init_full is not None:                            # the checkpoint's own normalisation
        mean, std = np.asarray(init_full["imu_mean"]), np.asarray(init_full["imu_std"])
        mean_g, std_g = torch.as_tensor(mean, device=dev), torch.as_tensor(std, device=dev)

    def norm(x, up=None):
        """-> (normalised channels-first, R_total or None)."""
        f, R = canonical(x, up)
        return ((f - mean_g) / std_g).transpose(-1, -2).contiguous(), R

    ext = None
    if a.label_frame == "imu" or a.gt_up or a.idw != 1.0 or a.sig_weight > 0 or a.gyro_frame or a.tdil or a.sdil or a.sdil_fast:
        from unified.gtup import load_extrinsics
        # a data root with synthetic recordings (local_eval/synth_flights) ships its
        # own extrinsics.npz (the parents' rotations for the synthetic ids)
        ext_path = Path(os.environ.get("TARTANIMU_EXTRINSICS", "") or
                        (Path(os.environ["TARTANIMU_DATA"]) / "extrinsics.npz" if os.environ.get("TARTANIMU_DATA")
                         and (Path(os.environ["TARTANIMU_DATA"]) / "extrinsics.npz").exists() else ROOT / "runs/extrinsics.npz"))
        ext = load_extrinsics(ext_path)
        print(f"extrinsics: {len(ext)} trajectories loaded from {ext_path}")

    def raw_dev(frame, X, gt=False):
        if a.gyro_frame and frame is val:
            X = np.asarray(X).copy()
            for tid, rows in frame.groupby("traj_id", sort=False):
                R = ext[tid][0]; idx = rows.index.to_numpy()
                X[idx, :, 3:6] = X[idx, :, 3:6] @ R.T
            U = torch.from_numpy(np.asarray(up_windows_for(frame, a.grav, att_m, cal_m, att_key))).to(dev) if need_up else None
            return torch.from_numpy(X).to(dev), U
        U = torch.from_numpy(np.asarray(up_windows_for(frame, a.grav, att_m, cal_m, att_key))).to(dev) if need_up else None
        if gt and need_up:
            # ground-truth up per window, IMU frame, accelerometer sign (diagnostic)
            from unified.gtup import gt_up_frames
            Ug = np.asarray(U.cpu()).copy()
            for fp, rows in frame.groupby("file_path", sort=False):
                with np.load(fp) as d:
                    imu, quat = d["imu"], d["quat"]
                tid = rows["traj_id"].iloc[0]
                u = gt_up_frames(imu, quat, ext[tid][0])
                for i, wi in zip(rows.index.to_numpy(), rows["win_idx"].to_numpy()):
                    seg = u[wi * 200:(wi + 1) * 200]
                    Ug[i, :, :3] = seg; Ug[i, :, 3:] = seg
            U = torch.from_numpy(Ug).to(dev)
        return torch.from_numpy(np.asarray(X)).to(dev), U

    Xva_raw, Uva_raw = raw_dev(val, build_windows(val), gt=a.gt_up)
    if hold is not None:
        from build_val_solution import build as build_solution
        hold_solution = build_solution(Path(os.environ.get("TARTANIMU_DATA", str(ROOT / "data/tartan-imu-challenge-iros2026"))),
                                       "train", hold["traj_id"].unique())
        Xho_raw, Uho_raw = raw_dev(hold, build_windows(hold), gt=a.gt_up)
    Xte_raw, Ute_raw = raw_dev(test, build_windows(test))

    if a.traj_film and a.data != "dense":
        raise NotImplementedError("--traj_film needs --data dense")
    if a.data == "dense":
        src = DenseWindows(train, dev, with_pose=need_pose,
                           min_windows=a.seg_len if a.short_traj == "drop" else 0,
                           platforms=a.platforms, aggr_weight=a.aggr_weight,
                           extrinsics=ext, gt_up=a.gt_up, idw=a.idw, ext_weight=a.ext_weight, gyro_frame=a.gyro_frame,
                           tdil=tuple(a.tdil) if a.tdil else None, tdil_platforms=a.tdil_platforms,
                           tdil_prob=a.tdil_prob, sdil=tuple(a.sdil) if a.sdil else None,
                           aug_mix=a.aug_mix, tdil_fast=tuple(a.tdil_fast) if a.tdil_fast else None,
                           strat_prob=a.strat_prob, adapt=(a.grav == "adapt"),
                           tdil_short=a.tdil_short, tdil_aa=a.tdil_aa,
                           sdil_fast=tuple(a.sdil_fast) if a.sdil_fast else None, rest=a.grav in ("slowrest", "slowrest2", "slowdr"), restgyro=a.grav in ("slowrest2", "slowdr"), dr=(a.grav == "slowdr"), att=att_m, cal=cal_m, ins_drop=a.ins_drop, drag_pose=a.drag_pose)
        if a.tdil or a.sdil or a.sdil_fast:
            print(f"augmentation: T={tuple(a.tdil) if a.tdil else None} Tfast={tuple(a.tdil_fast) if a.tdil_fast else None} "
                  f"S={tuple(a.sdil) if a.sdil else None} Sfast={tuple(a.sdil_fast) if a.sdil_fast else None} "
                  f"mix={a.aug_mix} on {a.tdil_platforms} p={a.tdil_prob}")
        if a.idw != 1.0:
            print(f"idw {a.idw}: {src.n_ident} identity-extrinsic drone trajectories upweighted")
        if a.aggr_weight != 1.0:
            print(f"aggr_weight {a.aggr_weight}: {src.n_aggr} aggressive drone trajectories upweighted")
        if a.traj_film:
            with torch.no_grad():
                model.net.tstat_mean.copy_(src.traj_stats.mean(0))
                model.net.tstat_std.copy_(src.traj_stats.std(0).clamp(min=1e-6))
        if src.n_dropped:
            print(f"short_traj=drop: {src.n_dropped} trajectories shorter than {a.seg_len} "
                  f"windows excluded from sampling")
        del Xtr, Utr
    else:
        Xtr_g = torch.from_numpy(np.asarray(Xtr)).to(dev)
        Utr_g = torch.from_numpy(np.asarray(Utr)).to(dev) if need_up else None
        ytr = torch.from_numpy(train[["vx", "vy", "vz"]].to_numpy(np.float32)).to(dev)
        ptr = torch.from_numpy(train["platform_id"].to_numpy(np.int64)).to(dev)
        segs = torch.from_numpy(segment_starts(train, a.seg_len).astype(np.int64)).to(dev)
        wseg = torch.as_tensor(sample_weights(train)[segs[:, 0].cpu().numpy()],
                               dtype=torch.double, device=dev)
        pose = attach_pose(train) if need_pose else None
        if need_pose:
            R_all = torch.from_numpy(pose["R"]).to(dev)
            pos_all = torch.from_numpy(pose["pos"]).to(dev)
    gtar = None
    if a.gravity_weight > 0 and a.data == "index":
        gtar = torch.from_numpy(window_gravity_target(train, 1)[:, 0]).to(dev)

    trunk_names = {"stem", "stage2", "stage3", "stage4"}
    trunk_p = [q for n_, q in model.named_parameters() if n_.split(".")[1:2] and n_.split(".")[1] in trunk_names]
    other_p = [q for n_, q in model.named_parameters() if not (n_.split(".")[1:2] and n_.split(".")[1] in trunk_names)]
    groups = [{"params": other_p, "lr": a.lr}, {"params": trunk_p, "lr": a.lr * a.trunk_lr_mult}]
    opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=a.weight_decay)
    la = None
    if a.lookahead > 0:
        if a.sam > 0 or a.ema > 0 or a.swa_last > 0:
            raise SystemExit("--lookahead: not combined with --sam / --ema / --swa_last")
        la_params = [q for g in opt.param_groups for q in g["params"]]
        la = {"params": la_params, "slow": [q.detach().clone() for q in la_params],
              "ok": 0, "att": 0, "sync": 0, "skipped": 0}
        print(f"lookahead: k={a.lookahead} alpha={a.lookahead_alpha} over {len(la_params)} tensors "
              f"(slow copy taken after model/optimizer setup)")
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.lr * a.trunk_lr_mult],
                                                epochs=a.epochs, steps_per_epoch=a.steps_per_epoch,
                                                pct_start=0.1)
    if a.trunk_lr_mult != 1.0:
        print(f"trunk lr x{a.trunk_lr_mult}: {sum(q.numel() for q in trunk_p)} trunk params, "
              f"{sum(q.numel() for q in other_p)} others")
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda" and a.sam <= 0)
    amp_dtype = torch.bfloat16 if a.sam > 0 else torch.float16
    yaw_gen = None
    if a.yaw_ident > 0:
        yaw_gen = torch.Generator(device=dev); yaw_gen.manual_seed(a.seed + 300007)
    ctx_gen = None
    if a.ctx_split > 0:
        if a.data != "dense" or a.model != "fine":
            raise SystemExit("--ctx_split needs --data dense --model fine")
        for bad_flag in ("teachers", "ens_teachers"):
            if getattr(a, bad_flag): raise SystemExit(f"--ctx_split is not supported with --{bad_flag}")
        if a.perframe_weight > 0 or a.sig_weight > 0 or a.macro_loss or a.yaw_pair != "off" or a.rsc > 0 or a.recon_weight > 0:
            raise SystemExit("--ctx_split: unsupported combination (perframe/sig/macro_loss/yaw_pair/rsc/recon)")
        ctx_gen = torch.Generator(device=dev); ctx_gen.manual_seed(a.seed + 200003)
    gain_gen = None
    if a.acc_gain > 0:
        gain_gen = torch.Generator(device=dev); gain_gen.manual_seed(a.seed + 100003)
    ema = None
    swa = None
    if a.swa_last > 0:
        # running mean of every float tensor in state_dict (weights and BN stats);
        # integer buffers (num_batches_tracked) are copied from the last epoch
        swa = {"n": 0, "sum": {k: torch.zeros_like(v, dtype=torch.float32)
                               for k, v in model.state_dict().items() if v.is_floating_point()}}
    if a.ema > 0:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        if a.eqframe:
            # norm() closes over model.canon; the EMA copy would be canonicalised
            # by the live frame network, not its own averaged one
            raise NotImplementedError("--ema with --eqframe")
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(a.ema), use_buffers=True)

    B, K = a.batch_segs, a.seg_len
    print(f"data={a.data} model={a.model} grav={a.grav}({n_ch}ch) seg_len={K} "
          f"batch={B}x{K}={B*K} windows | "
          f"ate {a.ate_weight} grav {a.gravity_weight} pf {a.perframe_weight} "
          f"scale {a.scale_aware} aug {a.aug_mode} | {dev}   ({time.time()-t0:.0f}s setup)\n")

    teachers = None
    if a.teachers:
        if a.data != "dense" or a.model != "fine":
            raise NotImplementedError("--teachers needs --data dense --model fine")
        teachers = {}
        for item in a.teachers:
            plat, tag = item.split("=")
            try:
                teachers[PLATFORMS.index(plat)] = load_teacher(tag, dev)
            except (FileNotFoundError, RuntimeError, KeyError) as exc:
                print(f"teacher {tag} not usable ({exc}); {plat} trains on ground truth only")
        if not teachers:
            print("no usable teachers: running WITHOUT distillation")
            teachers = None
    ens = None
    if a.ens_teachers:
        if a.data != "dense" or a.model != "fine":
            raise NotImplementedError("--ens_teachers needs --data dense --model fine")
        ens = [load_teacher(t, dev) for t in a.ens_teachers]
    gn = None
    if a.gradnorm >= 0:
        if a.model != "fine":
            raise NotImplementedError("--gradnorm needs --model fine")
        gn = GradNorm(4, a.gradnorm, model.net.mixer_norm.weight)

    history, best, waiting = [], float("inf"), 0
    ckpt = ROOT / f"unified/ckpt_{a.tag}.pt"
    best_ema = float("inf")
    ckpt_ema = ROOT / f"unified/ckpt_{ema_tag(a.tag)}.pt"

    def back_to_label(pred, frame):
        if a.label_frame != "imu":
            return pred
        out = pred.copy()
        for tid, rows in frame.groupby("traj_id", sort=False):
            R = ext[tid][0]; idx = rows.index.to_numpy()
            out[idx] = pred[idx] @ R.T                      # v_label = R v_imu
        return out

    def evaluate_hold(net):
        pred = predict(net, Xho_raw, Uho_raw, hold, a.model, K, norm)
        sub = pd.DataFrame({"window_id": hold["window_id"], "vx": pred[:, 0], "vy": pred[:, 1], "vz": pred[:, 2]})
        _, table = score_breakdown(hold_solution, sub)
        r = table[table.platform == "drone"].iloc[0]
        return float(r.score), float(r.AVE), float(r.ATE20)

    def evaluate(net):
        pred = back_to_label(predict(net, Xva_raw, Uva_raw, val, a.model, K, norm), val)
        sub = pd.DataFrame({"window_id": val["window_id"], "vx": pred[:, 0],
                            "vy": pred[:, 1], "vz": pred[:, 2]})
        score, table = score_breakdown(solution, sub)
        return score, {r.platform: r.score for r in table.itertuples()}

    def write_submissions(net, tag):
        diagnostic = a.label_frame == "imu" or a.gt_up or a.gyro_frame
        splits = (("val", (Xva_raw, Uva_raw), val),) if diagnostic else \
            (("val", (Xva_raw, Uva_raw), val), ("test", (Xte_raw, Ute_raw), test))
        if diagnostic:
            print("DIAGNOSTIC run (ground truth used at validation): no test CSV written")
        for name, (Xg, Ug), frame in splits:
            p = predict(net, Xg, Ug, frame, a.model, K, norm)
            if name == "val":
                p = back_to_label(p, frame)
            out = ROOT / f"local_eval/sub_{name}_{tag}.csv"
            pd.DataFrame({"window_id": frame["window_id"], "vx": p[:, 0], "vy": p[:, 1],
                          "vz": p[:, 2]}).to_csv(out, index=False)
            print(f"wrote {out.relative_to(ROOT)}  rows={len(p)}")
        print(f"total {time.time()-t0:.0f}s")

    if a.data_seed is not None:
        torch.manual_seed(a.data_seed); torch.cuda.manual_seed_all(a.data_seed)
    for epoch in range(1, a.epochs + 1):
        if a.ext_until and epoch == a.ext_until and a.data == "dense":
            src.drop_ext(); print(f"ext_until: external recordings dropped from sampling at epoch {epoch}", flush=True)
        model.train()
        if a.tdil_off_last and a.data == "dense" and src.tdil is not None and epoch > a.epochs - a.tdil_off_last:
            if src.tdil != (1.0, 1.0):
                print(f"epoch {epoch}: time dilation off for the remaining epochs")
            src.tdil = (1.0, 1.0); src.tdil_fast = None
        tot = {"ave": 0.0, "ate": 0.0, "grav": 0.0, "pf": 0.0, "Hgate": 0.0, "wdrone": 0.0,
               "kd": 0.0, "rec": 0.0, "sig": 0.0, "vb_oor": 0.0}
        for _ in range(a.steps_per_epoch):
            B = a.batch_segs                                  # --yaw_pair may have grown it
            if a.data == "dense":
                b = src.sample(B, K)
                x, y, pl, vf = b["imu"], b["target"], b["platform"], b["vel_frames"]
                lens = b["lengths"] if a.model == "fine" else None
                ts = b["tstats"] if a.traj_film else None
                if a.label_frame == "imu":
                    # v_imu = R_ext^T v_label, one R per segment (per trajectory)
                    y = torch.einsum("bji,bkj->bki", b["R_ext"], y)
                    if vf is not None:
                        vf = torch.einsum("bji,bkfj->bkfi", b["R_ext"], vf)
                xu = b["up"] if need_up else None
                yaw_M = None                                  # per-row body relabeling of the yaw augmentation (EKF increments)
                if gain_gen is not None:
                    # accelerometer scale factor: x (B, K, T, 6), gravity scales with it,
                    # gyro / up / labels untouched (the slow-up filter is direction-only)
                    g_ = 1.0 + a.acc_gain * (2 * torch.rand(B, device=dev, generator=gain_gen, dtype=x.dtype) - 1)
                    x = torch.cat([x[..., :3] * g_[:, None, None, None], x[..., 3:]], dim=-1)
                pl = pl[:, None].expand(-1, K)
                Rb, Qb = (b.get("R"), b.get("pos")) if need_pose else (None, None)
                # g_body = R^T @ (0,0,-1) = -R[..., 2, :]; checked against
                # gravity.gravity_dir_body to 1e-7
                gb = -Rb[..., 2, :] if a.gravity_weight > 0 else None
            else:
                rows = segs[torch.multinomial(wseg, B, replacement=True)]
                x = Xtr_g[rows]; y = ytr[rows]; pl = ptr[rows]; vf = None; lens = None; ts = None
                xu = Utr_g[rows] if need_up else None
                Rb, Qb = (R_all[rows], pos_all[rows]) if need_pose else (None, None)
                gb = gtar[rows] if gtar is not None else None

            kd_target = None
            if teachers is not None:
                # each platform's segments get their own expert's prediction on the
                # same raw segment, same valid length; computed before any augmentation
                kd_target = torch.zeros(B, K, 3, device=dev)
                kd_has = torch.zeros(B, dtype=torch.bool, device=dev)   # rows with a teacher
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16,
                                                     enabled=dev.type == "cuda"):
                    for i, (tnet, tnorm, _) in teachers.items():
                        rows = (pl[:, 0] == i).nonzero(as_tuple=True)[0]
                        if len(rows) == 0:
                            continue
                        tv = tnet.net(tnorm(x[rows], b["up"][rows]),
                                      lengths=lens[rows] if lens is not None else None)[0]
                        kd_target[rows] = tv.float()
                        kd_has[rows] = True
            if ens is not None:
                # the teachers' mean on this very segment (after T, before the
                # later augmentations, which rotate/perturb the target alongside)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16,
                                                     enabled=dev.type == "cuda"):
                    kd_target = torch.stack([
                        tnet.net(tnorm(x, b["up"]), lengths=lens)[0].float() for tnet, tnorm, _ in ens]).mean(0)
                kd_has = torch.ones(B, dtype=torch.bool, device=dev)
            if a.noise_aug > 0:
                x = x + torch.randn_like(x) * (a.noise_aug * raw_std)
            if a.band_aug > 0:
                # frequency-space randomisation: content above 10 Hz gets a smooth
                # random gain, one draw per segment, so a vibration signature is no
                # longer a fixed fingerprint of the recording; DC and low bands untouched
                spec = torch.fft.rfft(x.float(), dim=2)                 # (B,K,101,6)
                nb = spec.shape[2]
                knots = 1 + a.band_aug * (torch.rand(B, 1, 8, 6, device=dev) * 2 - 1)
                gain = F.interpolate(knots.permute(0, 1, 3, 2).reshape(B, 6, 8), size=nb,
                                     mode="linear", align_corners=True)   # (B,6,101)
                gain = gain.permute(0, 2, 1)[:, None]                      # (B,1,101,6)
                f = torch.arange(nb, device=dev)                           # 1 Hz bins
                gain = torch.where((f >= 10)[None, None, :, None], gain, torch.ones_like(gain))
                x = torch.fft.irfft(spec * gain, n=x.shape[2], dim=2).to(x.dtype)
            if a.yaw_aug is not None and a.data == "dense":
                # a different mounting yaw for the whole segment: rotate a, w, the
                # up estimates and the target velocity about body z by the same angle
                sel = torch.zeros(B, dtype=torch.bool, device=dev)
                for q in a.yaw_aug:
                    sel |= pl[:, 0] == PLATFORMS.index(q)
                th = torch.rand(B, device=dev) * 2 * math.pi * sel
                cth, sth = torch.cos(th)[:, None, None], torch.sin(th)[:, None, None]
                def rz(v):                                    # (..., 3) about z
                    x0, y0, z0 = v[..., 0], v[..., 1], v[..., 2]
                    c = cth.reshape(B, *([1] * (v.dim() - 2))); s_ = sth.reshape(B, *([1] * (v.dim() - 2)))
                    return torch.stack([c * x0 - s_ * y0, s_ * x0 + c * y0, z0], dim=-1)
                x = torch.cat([rz(x[..., :3]), rz(x[..., 3:6])], dim=-1)
                if xu is not None:
                    xu = torch.cat([rz(xu[..., :3]), rz(xu[..., 3:6])], dim=-1)
                y = rz(y)
                if vf is not None:
                    vf = rz(vf)
            yaw_sel = None; row_w = None
            if a.yaw_ident > 0 and a.data == "dense":
                # the selection and the view draws come from a dedicated generator so the
                # base segment / T sampling stream is identical to a run without yaw
                sel = (b["sig"] == 6) & (pl[:, 0] == PLATFORMS.index("drone")) & \
                      (torch.rand(B, device=dev, generator=yaw_gen) < a.yaw_ident)
                if a.yaw_set == "uniform":
                    th = torch.rand(B, device=dev, generator=yaw_gen) * 2 * math.pi * sel
                    c_, s_ = torch.cos(th), torch.sin(th)
                    Rz = torch.zeros(B, 3, 3, device=dev, dtype=x.dtype)
                    Rz[:, 0, 0] = c_; Rz[:, 0, 1] = -s_; Rz[:, 1, 0] = s_; Rz[:, 1, 1] = c_; Rz[:, 2, 2] = 1
                    Rw = Rz                                   # gyro rotates like a vector
                else:
                    # categorical view: 180-degree yaw, or a mirror of the x / y axis; a
                    # mirror is an improper frame change: polar vectors (a, v, up) by M,
                    # the angular rate (pseudovector) by det(M) M
                    n_views = 3 if a.yaw_set == "pi_mirror" else 1
                    choice = torch.randint(0, n_views, (B,), device=dev, generator=yaw_gen)
                    diag = torch.ones(B, 3, device=dev, dtype=x.dtype)
                    diag[choice == 0, 0] = -1; diag[choice == 0, 1] = -1              # R_z(pi)
                    diag[choice == 1, 0] = -1                                         # mirror x
                    diag[choice == 2, 1] = -1                                         # mirror y
                    diag[~sel] = 1.0
                    Rz = torch.diag_embed(diag)
                    Rw = torch.diag_embed(diag * diag.prod(dim=1, keepdim=True))      # det(M) M
                def rot(v, R):                                # (B, ..., 3) by (B, 3, 3)
                    return (v.reshape(B, -1, 3) @ R.transpose(1, 2)).reshape(v.shape)
                Re = b["R_ext"].to(x.dtype)                   # IMU -> label frame
                Rl = Re @ Rz @ Re.transpose(1, 2)
                yaw_M = Rz
                x_r = torch.cat([rot(x[..., :3], Rz), rot(x[..., 3:6], Rw)], dim=-1)
                # rotate the 3-vector groups of the up stream; a trailing non-vector
                # column (slowrest: the rest-window score) passes through unchanged
                if xu is not None:                            # rotate the 3-vector groups only (updir.up_vector_groups)
                    from unified.updir import up_vector_groups
                    vg = set(up_vector_groups(xu.shape[-1])); cols = []; i = 0
                    while i < xu.shape[-1]:
                        if i in vg: cols.append(rot(xu[..., i:i + 3], Rw if i == 17 else Rz)); i += 3
                        else: cols.append(xu[..., i:i + 1]); i += 1
                    xu_r = torch.cat(cols, dim=-1)
                else:
                    xu_r = None
                y_r = rot(y, Rl)
                vf_r = rot(vf, Rl) if vf is not None else None
                kd_r = rot(kd_target, Rl) if kd_target is not None else None
                if a.yaw_pair == "off":
                    x, xu, y, vf, kd_target = x_r, xu_r, y_r, vf_r, kd_r
                    yaw_sel = sel
                elif bool(sel.any()):
                    # append the rotated view of the selected rows; row weights decide
                    # which view carries which loss (see forward_loss)
                    idx = sel.nonzero(as_tuple=True)[0]
                    x = torch.cat([x, x_r[idx]]); y = torch.cat([y, y_r[idx]])
                    if xu is not None: xu = torch.cat([xu, xu_r[idx]])
                    if vf is not None: vf = torch.cat([vf, vf_r[idx]])
                    if kd_target is not None:
                        kd_target = torch.cat([kd_target, kd_r[idx]]); kd_has = torch.cat([kd_has, kd_has[idx]])
                    pl = torch.cat([pl, pl[idx]])
                    if lens is not None: lens = torch.cat([lens, lens[idx]])
                    if ts is not None: ts = torch.cat([ts, ts[idx]])
                    n_sel = len(idx)
                    yaw_sel = torch.cat([torch.zeros(B, dtype=torch.bool, device=dev),
                                         torch.ones(n_sel, dtype=torch.bool, device=dev)])
                    row_w = torch.ones(B + n_sel, device=dev)
                    if a.yaw_pair == "replace":
                        row_w[idx] = 0.0
                    else:
                        row_w[idx] = 0.5; row_w[B:] = 0.5
                    B = B + n_sel
            if a.aug_mode != "none":
                shape = (B, 1, 1, 6) if a.aug_mode == "segment" else (B, K, 1, 6)
                scale = torch.tensor([ACCEL_BIAS] * 3 + [GYRO_BIAS] * 3, device=dev)
                x = x + (torch.rand(shape, device=dev) * 2 - 1) * scale

            if a.grav_perturb > 0 and xu is not None:
                # tilt the up estimate by a random small rotation so the frame
                # network learns to tolerate the filter's few degrees of error
                th = torch.deg2rad(torch.rand(B, 1, 1, 1, device=dev) * a.grav_perturb)
                ax = torch.randn(B, 1, 1, 3, device=dev)
                ax = ax / ax.norm(dim=-1, keepdim=True)
                c, s_ = torch.cos(th), torch.sin(th)
                u = xu[..., 3:6]
                u = u * c + torch.cross(ax.expand_as(u), u, dim=-1) * s_ \
                    + ax * (ax * u).sum(-1, keepdim=True) * (1 - c)
                xu = torch.cat([xu[..., :3], u], dim=-1)
            if ctx_gen is not None and lens is not None:
                # context-split-v1: selected rows become two continuous pieces; the
                # second piece is shifted to start at window 0 (as an inference chunk
                # would) and padded by repeating its last valid window (masked)
                Lr = lens
                sel = (Lr >= 10) & (torch.rand(B, device=dev, generator=ctx_gen) < a.ctx_split)
                idx = sel.nonzero(as_tuple=True)[0]
                if len(idx) > 0:
                    Li = Lr[idx]
                    m = 5 + (torch.rand(len(idx), device=dev, generator=ctx_gen) * (Li - 9).to(torch.float32)).long()  # [5, L-5]
                    m = torch.minimum(m, Li - 5)
                    pos = torch.minimum(m[:, None] + torch.arange(K, device=dev)[None, :], (Li - 1)[:, None])  # (n, K)
                    def take(t):
                        if t is None: return None
                        src = t[idx]                                            # (n, K, ...)
                        g = pos.reshape(len(idx), K, *([1] * (t.dim() - 2))).expand(-1, -1, *t.shape[2:])
                        return torch.cat([t, torch.gather(src, 1, g)], dim=0)
                    x = take(x); y = take(y); xu = take(xu); vf = take(vf)
                    pl = take(pl)
                    if kd_target is not None: raise SystemExit("--ctx_split with a KD target")
                    lens = torch.cat([lens, Li - m]); lens[idx] = m
                    if ts is not None: ts = torch.cat([ts, ts[idx]])
                    if yaw_sel is not None: yaw_sel = torch.cat([yaw_sel, yaw_sel[idx]])
                    if row_w is not None: row_w = torch.cat([row_w, row_w[idx]])
                    B = B + len(idx)
            def forward_loss(rec=True, y=y):
                """One forward pass and its loss on the current batch; rec=False for a
                repeated pass (SAM) whose numbers should not go into the epoch totals.
                y is bound as a default so --rel_loss can rescale it locally."""
                xb, Rt = norm(x, xu)                             # (B,K,n_ch,200)
                if a.time_mask > 0:
                    # a memorised phase is useless if a random piece of the window is
                    # gone; the trunk must read the rest of the second instead
                    hit = torch.rand(B, K, device=dev) < a.time_mask
                    span = torch.randint(20, 61, (B, K, 1), device=dev)
                    start = (torch.rand(B, K, 1, device=dev) * (200 - span)).long()
                    t = torch.arange(200, device=dev)[None, None, :]
                    m = ((t >= start) & (t < start + span)) & hit[..., None]
                    xb = xb.masked_fill(m[:, :, None, :], 0.0)
                # learned-INS step A: un-normalised window-mean v_DR (m/s) from the (possibly
                # rotated / dilated) up stream, columns 13:16
                vdr_b = xu[..., 13:16].float().mean(2) if (a.fuse and xu is not None) else None
                gf_b = gate_feats(xu) if ((a.fuse or a.ekf) and xu is not None) else None
                dv_b, dR_b, em_b = ekf_increments(xu, yaw_M) if (a.ekf and xu is not None) else (None, None, None)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=amp_dtype, enabled=dev.type == "cuda"):
                    if a.model == "single":
                        feat, v, logits = model.net.features_and_heads(xb.reshape(B * K, n_ch, 200))  # noqa: E501
                        v = v.view(B, K, 3); logits = logits.view(B * K, -1)
                    elif lens is not None:
                        feat, v, logits = model.net.features_and_heads(xb, lens, ts, vdr=vdr_b, gfeat=gf_b, dv=dv_b, dR=dR_b, emask=em_b)
                        logits = logits.reshape(B * K, -1)
                    else:
                        feat, v, logits = model.net.features_and_heads(xb, vdr=vdr_b, gfeat=gf_b, dv=dv_b, dR=dR_b, emask=em_b)
                        logits = logits.reshape(B * K, -1)
                    if Rt is not None:
                        v = to_body(v.reshape(B, K, 3).float(), Rt).to(v.dtype)
                    # padded windows of short trajectories carry no label
                    ok = (torch.arange(K, device=dev)[None, :] < lens[:, None]).reshape(-1) \
                        if lens is not None else slice(None)
                    if a.head == "vb" and rec:
                        # VB support audit: valid targets outside [-R, R] (never clipped)
                        okv = ok if lens is not None else slice(None)
                        tot["vb_oor"] += float((y.reshape(-1, 3)[okv].abs() > model.net.vb_range).any(-1).sum())
                    if a.yaw_dironly and yaw_sel is not None and bool(yaw_sel.any()):
                        v3 = v.reshape(B, K, 3)
                        v_dir = v3 / (v3.norm(dim=-1, keepdim=True) + 1e-6) * y.norm(dim=-1, keepdim=True).to(v3.dtype)
                        v = torch.where(yaw_sel[:, None, None], v_dir, v3).reshape(v.shape)
                    if a.rel_loss > 0:
                        # error in units of the target's own magnitude (floored), so a
                        # 0.1 m/s miss on a hovering drone counts like a 1 m/s miss on a car
                        inv = 1.0 / (y.norm(dim=-1, keepdim=True) + a.rel_loss)
                        v, y = v * inv.to(v.dtype), y * inv
                    if gn is None and a.macro_loss:
                        e = torch.linalg.vector_norm(v.reshape(B, K, 3).float() - y, dim=-1)
                        b_ = a.huber_beta
                        hub = torch.where(e < b_, 0.5 * e.square() / b_, e - 0.5 * b_)   # (B,K)
                        okm = ok.reshape(B, K).float() if lens is not None else torch.ones_like(hub)
                        seg = (hub * okm).sum(1) / okm.sum(1).clamp(min=1)               # (B,)
                        plat0 = pl[:, 0]
                        parts = [seg[plat0 == i].mean() for i in range(4) if bool((plat0 == i).any())]
                        l_ave = torch.stack(parts).mean()
                    elif gn is None and a.speed_weight > 0:
                        vv, yy = v.reshape(-1, 3)[ok], y.reshape(-1, 3)[ok]
                        wt = 1.0 + yy.norm(dim=-1) / a.speed_weight
                        wt = wt / wt.mean()
                        e = torch.linalg.vector_norm(vv.float() - yy, dim=1); b_ = a.huber_beta
                        l_ave = (wt * torch.where(e < b_, 0.5 * e.square() / b_, e - 0.5 * b_)).mean()
                    elif gn is None and row_w is not None:
                        e = torch.linalg.vector_norm(v.reshape(B, K, 3).float() - y, dim=-1)
                        b_ = a.huber_beta
                        hub = torch.where(e < b_, 0.5 * e.square() / b_, e - 0.5 * b_)   # (B,K)
                        okm = ok.reshape(B, K).float() if lens is not None else torch.ones_like(hub)
                        w = okm * row_w[:, None]
                        l_ave = (hub * w).sum() / w.sum().clamp(min=1)
                    elif gn is None and (a.huber_beta_speed > 0 or speed_bin_w is not None):
                        # method 1a/1c: per-window beta grows with the target speed and/or the
                        # window is weighted by the inverse frequency of its speed bin
                        vv, yy = v.reshape(-1, 3)[ok].float(), y.reshape(-1, 3)[ok]
                        sp_ = yy.norm(dim=-1); e = torch.linalg.vector_norm(vv - yy, dim=1)
                        b_ = a.huber_beta + a.huber_beta_speed * sp_
                        hub = torch.where(e < b_, 0.5 * e.square() / b_, e - 0.5 * b_)
                        if speed_bin_w is not None:
                            bi = torch.bucketize(sp_, torch.tensor([2., 4., 6., 8., 10.], device=sp_.device))
                            wt = speed_bin_w.to(sp_.device)[bi]; wt = wt / wt.mean()
                            l_ave = (wt * hub).mean()
                        else:
                            l_ave = hub.mean()
                    elif gn is None:
                        l_ave = vector_huber(v.reshape(-1, 3)[ok], y.reshape(-1, 3)[ok], a.huber_beta)
                    else:
                        vv, yy, pp = v.reshape(-1, 3)[ok].float(), y.reshape(-1, 3)[ok], pl.reshape(-1)[ok]
                        per_pl = torch.stack([
                            vector_huber(vv[pp == i], yy[pp == i], a.huber_beta) if bool((pp == i).any())
                            else vv.new_tensor(float("nan")) for i in range(4)])
                        l_ave = gn.weighted_total(per_pl) / 4          # mean over platforms
                        if rec: tot["wdrone"] += float(gn.w[2])
                    if (a.rsc > 0 and rec and epoch > 10 and a.model == "fine"
                            and gn is None and not a.macro_loss and a.speed_weight <= 0
                            and torch.rand(()) < 0.5):
                        # Representation Self-Challenging (Huang et al. 2020), regression
                        # form: find the feature channels the velocity loss leans on most,
                        # silence them, and train the head on what is left
                        g = torch.autograd.grad(l_ave, feat, retain_graph=True)[0]
                        imp = (g * feat).abs().reshape(-1, feat.shape[-1]).mean(0)
                        top = imp.topk(max(1, int(a.rsc * feat.shape[-1]))).indices
                        keep = torch.ones(feat.shape[-1], device=dev); keep[top] = 0
                        v = model.net.velocity_head(feat * keep)
                        if Rt is not None:
                            v = to_body(v.reshape(B, K, 3).float(), Rt).to(v.dtype)
                        l_ave = vector_huber(v.reshape(-1, 3)[ok], y.reshape(-1, 3)[ok], a.huber_beta)
                    if a.lr0 == "anchor" and gn is None and not a.macro_loss and a.speed_weight <= 0:
                        # method 2: anchor loss on the direct head p, increment loss on e vs y_k - y_{k-1}
                        p_, e_ = model.net._anchor
                        p3, e3, y3 = p_.reshape(B, K, 3).float(), e_.reshape(B, K, 3).float(), y.reshape(B, K, 3)
                        okm = ok.reshape(B, K) if lens is not None else torch.ones(B, K, dtype=torch.bool, device=dev)
                        l_anchor = vector_huber(p3[okm], y3[okm], a.huber_beta)
                        pair = okm[:, 1:] & okm[:, :-1]
                        dy = (y3[:, 1:] - y3[:, :-1])[pair]; de = e3[:, 1:][pair]
                        l_inc = vector_huber(de, dy, a.huber_beta) if len(dy) else l_ave * 0
                        l_ave = 0.5 * l_ave + 0.5 * l_anchor + a.anchor_inc_weight * l_inc
                        if rec: tot["rec"] += float(l_inc)
                    if a.fuse and a.fuse_direct_weight > 0 and "fuse_vdirect" in model.net.aux:
                        vd = model.net.aux["fuse_vdirect"].reshape(B, K, 3).float(); y3 = y.reshape(B, K, 3)
                        okm = ok.reshape(B, K) if lens is not None else torch.ones(B, K, dtype=torch.bool, device=dev)
                        l_ave = l_ave + a.fuse_direct_weight * vector_huber(vd[okm], y3[okm], a.huber_beta)
                        if rec: tot["fgate"] = tot.get("fgate", 0.0) + float(model.net.aux["fuse_gate"].detach()[pl == 2].mean()) if bool((pl == 2).any()) else tot.get("fgate", 0.0)
                    if a.ekf and "ekf_vdirect" in model.net.aux:
                        # direct head keeps its own loss; gain supervised by which estimate is closer to y
                        y3 = y.reshape(B, K, 3); vd = model.net.aux["ekf_vdirect"].reshape(B, K, 3).float(); vp = model.net.aux["ekf_vprop"].reshape(B, K, 3).float()
                        okm = ok.reshape(B, K) if lens is not None else torch.ones(B, K, dtype=torch.bool, device=dev)
                        l_ave = l_ave + a.fuse_direct_weight * vector_huber(vd[okm], y3[okm], a.huber_beta)
                        if a.ekf_sup > 0:
                            ed = (vd.detach() - y3).norm(dim=-1); ep_ = (vp.detach() - y3).norm(dim=-1)
                            tgt = (ed < ep_ + a.ekf_margin).float()                    # 1 = trust the direct head (propagation must win by --ekf_margin m/s)
                            # §2.3: soft target -- how much closer the direct head is, not just which
                            tsoft = torch.sigmoid((ep_ + a.ekf_margin - ed) / a.ekf_soft) if a.ekf_soft > 0 else tgt
                            lg = model.net.aux["ekf_logit"].reshape(B, K, 3)
                            okb = okm & (em_b[..., 0] > 0.5) if (a.ekf_mask_direct and em_b is not None) else okm   # mask->direct: learn the gain only where it acts
                            if bool(okb.any()):
                                l_ave = l_ave + a.ekf_sup * torch.nn.functional.binary_cross_entropy_with_logits(lg[okb], tsoft[okb][:, None].expand(-1, 3))
                            if rec:
                                tot["vprop_p99"] = tot.get("vprop_p99", 0.0) + float(torch.quantile(ep_[okm].detach().float(), 0.99))
                                Kg = model.net.aux["ekf_gain"].detach().reshape(B, K, 3).mean(-1)
                                pb = okm & (tgt == 0)
                                tot["kprop"] = tot.get("kprop", 0.0) + (float(Kg[pb].mean()) if bool(pb.any()) else 0.0)
                                tot["pbetter"] = tot.get("pbetter", 0.0) + float(pb.float().sum() / okm.float().sum().clamp(min=1))
                                if em_b is not None and (a.ekf_conf > 0 or a.ekf_horizon > 0):
                                    # review note 2 (2026-09-20 night): masked windows propagate "carry the previous
                                    # fused velocity", a strong smoother -- report physics (unmasked) separately
                                    um = okm & (em_b[..., 0] > 0.5); mm = okm & (em_b[..., 0] <= 0.5); um[:, 0] = False; mm[:, 0] = False
                                    tot["pbetter_u"] = tot.get("pbetter_u", 0.0) + float((pb & um).float().sum() / um.float().sum().clamp(min=1))
                                    tot["pbetter_m"] = tot.get("pbetter_m", 0.0) + float((pb & mm).float().sum() / mm.float().sum().clamp(min=1))
                                    tot["kprop_u"] = tot.get("kprop_u", 0.0) + (float(Kg[pb & um].mean()) if bool((pb & um).any()) else 0.0)
                                    tot["mask_frac"] = tot.get("mask_frac", 0.0) + float(um.float().sum() / (um.float().sum() + mm.float().sum()).clamp(min=1))
                    if a.fuse and a.fuse_sup > 0 and vdr_b is not None and "fuse_logit" in model.net.aux:
                        # road 1 stage 1: teach the gate WHEN v_DR is trustworthy
                        y3 = y.reshape(B, K, 3); r = (vdr_b - y3).norm(dim=-1)                       # (B, K) v_DR error, m/s
                        tgt = (r < a.fuse_thresh).float()
                        okm = ok.reshape(B, K) if lens is not None else torch.ones(B, K, dtype=torch.bool, device=dev)
                        lg = model.net.aux["fuse_logit"].reshape(B, K, 3)
                        bce = torch.nn.functional.binary_cross_entropy_with_logits(lg[okm], tgt[okm][:, None].expand(-1, 3))
                        l_ave = l_ave + a.fuse_sup * bce
                        if rec:
                            g = model.net.aux["fuse_gate"].detach().reshape(B, K, 3).mean(-1)
                            tr = okm & (tgt > 0); un = okm & (tgt == 0)
                            tot["gopen"] = tot.get("gopen", 0.0) + (float(g[tr].mean()) if bool(tr.any()) else 0.0)
                            tot["gclose"] = tot.get("gclose", 0.0) + (float(g[un].mean()) if bool(un.any()) else 0.0)
                            tot["gtrust"] = tot.get("gtrust", 0.0) + float(tr.float().sum() / okm.float().sum().clamp(min=1))
                    if a.drag_head and a.drag_l1 > 0:
                        l_ave = l_ave + a.drag_l1 * model.net.aux["drag_A"].abs().mean()
                    if a.drag_head and a.drag_aux > 0:
                        lin = model.net.aux["drag_lin"].reshape(B, K, 3).float(); y3 = y.reshape(B, K, 3)
                        okm = ok.reshape(B, K) if lens is not None else torch.ones(B, K, dtype=torch.bool, device=dev)
                        okd = okm & (pl == 2)                                     # drone windows only
                        if bool(okd.any()):
                            l_ave = l_ave + a.drag_aux * vector_huber(lin[okd], y3[okd], a.huber_beta)
                    if a.logmag_weight > 0:
                        # method 1b: magnitude in log space, insensitive to direction, no new parameters
                        vv, yy = v.reshape(-1, 3)[ok].float(), y.reshape(-1, 3)[ok]
                        l_mag = (torch.log1p(vv.norm(dim=-1)) - torch.log1p(yy.norm(dim=-1))).square().mean()
                        l_ave = l_ave + a.logmag_weight * l_mag
                        if rec: tot["rec"] += float(l_mag)
                    l_pl = F.cross_entropy(logits[ok], pl.reshape(-1)[ok])
                    loss = l_ave + a.platform_loss_weight * l_pl
                    if a.sig_weight > 0 and lens is not None:
                        okm = ok.reshape(B, K).float()
                        pooled = (feat.reshape(B, K, -1) * okm[..., None]).sum(1) / okm.sum(1, keepdim=True).clamp(min=1)
                        sig_t = b["sig"]
                        has = sig_t >= 0
                        if bool(has.any()):
                            l_sig = F.cross_entropy(model.sig_head(pooled[has]).float(), sig_t[has])
                            loss = loss + a.sig_weight * l_sig
                            if rec: tot["sig"] += float(l_sig)
                    if a.recon_weight > 0:
                        # masked reconstruction on a quarter of the windows, same mask
                        # generator as pre-training; the velocity task keeps its clean input
                        xw = xb.reshape(B * K, n_ch, 200)
                        if lens is not None:
                            xw = xw[ok]
                        sel = torch.randperm(xw.shape[0], device=dev)[:max(1, xw.shape[0] // 4)]
                        xw = xw[sel].detach()
                        mrec = random_mask(xw.shape[0], 200, recon_mask[0], recon_mask[1], dev)
                        rec_ = model.decoder(model.net.trunk_seq(xw.masked_fill(mrec[:, None, :], 0.0)))
                        l_rec = ((rec_.float() - xw.float()) ** 2)[mrec[:, None, :].expand_as(xw)].mean()
                        loss = loss + a.recon_weight * l_rec
                        if rec: tot["rec"] += float(l_rec)
                    if kd_target is not None:
                        # only windows whose platform has a teacher; a platform without
                        # one is trained on ground truth alone (review 2026-09-12: the
                        # first version pulled teacherless platforms toward zero)
                        okk = kd_has[:, None].expand(B, K).reshape(-1)
                        okk = okk & ok if lens is not None else okk
                        if bool(okk.any()):
                            l_kd = vector_huber(v.reshape(-1, 3)[okk].float(),
                                                kd_target.reshape(-1, 3)[okk], a.huber_beta)
                            loss = loss + a.kd_weight * l_kd
                            if rec: tot["kd"] += float(l_kd)
                    aux = getattr(model.net, "aux", {})
                    if "gate_entropy" in aux:
                        if rec: tot["Hgate"] += float(aux["gate_entropy"])
                        # reward entropy: a gate that collapses onto one expert would
                        # look identical to the mixture simply not helping
                        loss = loss - a.entropy_weight * aux["gate_entropy"]
                        if a.gate_supervised:
                            gl = aux["gate_logits"]
                            tgt = pl if gl.dim() == 3 else pl[:, 0]   # film gates per segment
                            loss = loss + a.platform_loss_weight * F.cross_entropy(
                                gl.reshape(-1, gl.shape[-1]), tgt.reshape(-1))
                    fflat = feat.reshape(B * K, -1)[ok]
                    if a.perframe_weight > 0 and vf is not None:
                        pf = model.perframe(fflat)
                        tgt = perframe_target(vf.reshape(B * K, 200, 3)[ok], a.perframe_n)
                        l_pf = masked_vector_huber(pf, tgt)
                        loss = loss + a.perframe_weight * l_pf
                        if rec: tot["pf"] += float(l_pf)
                    if a.gravity_weight > 0 and gb is not None:
                        l_g = cosine_loss(model.gravity(fflat), gb.reshape(B * K, 3)[ok])
                        loss = loss + a.gravity_weight * l_g
                        if rec: tot["grav"] += float(l_g)
                if a.ate_weight > 0 and a.tdil:
                    raise NotImplementedError("ate_weight under time dilation")
                if a.ate_weight > 0:                             # fp32: fp16 cumsum drifts
                    if lens is not None and bool((lens < K).any()):
                        raise NotImplementedError("ate_weight with padded segments; use --short_traj drop")
                    l_ate = ate_surrogate(v.float(), Rb, 1.0, Qb) / ATE_REF
                    loss = loss + a.ate_weight * l_ate
                    if rec: tot["ate"] += float(l_ate)
                if rec: tot["ave"] += float(l_ave)
                return loss
            loss = forward_loss()

            if a.sam > 0:
                # Sharpness-Aware Minimisation (Foret et al. 2021): step to the
                # worst nearby weights, take the gradient there, apply it here.
                # bf16 autocast, no GradScaler: two unscale_() calls per update
                # is not something the scaler allows.
                loss.backward()
                with torch.no_grad():
                    ps = [q for q in model.parameters() if q.grad is not None]
                    gnorm = torch.norm(torch.stack([q.grad.norm() for q in ps])) + 1e-12
                    eps = [a.sam * q.grad / gnorm for q in ps]
                    for q, e in zip(ps, eps):
                        q.add_(e)
                opt.zero_grad(set_to_none=True)
                loss2 = forward_loss(rec=False)
                loss2.backward()
                with torch.no_grad():
                    for q, e in zip(ps, eps):
                        q.sub_(e)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step()
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scale_before = scaler.get_scale() if scaler.is_enabled() else 1.0
                scaler.step(opt); scaler.update(); sched.step()
                if la is not None:
                    la["att"] += 1
                    if scaler.is_enabled() and scaler.get_scale() < scale_before:
                        la["skipped"] += 1                     # inf/nan step: optimizer did not update
                    else:
                        la["ok"] += 1
                        if la["ok"] % a.lookahead == 0:
                            with torch.no_grad():
                                for q, sl in zip(la["params"], la["slow"]):
                                    sl.add_(q.detach() - sl, alpha=a.lookahead_alpha)
                                    q.copy_(sl)
                            la["sync"] += 1
            if ema is not None:
                ema.update_parameters(model)

        if swa is not None and epoch > a.epochs - a.swa_last:
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if k in swa["sum"]: swa["sum"][k] += v.float()
            swa["n"] += 1

        if la is not None:
            # evaluate / save the SLOW weights; the fast weights (and the optimizer
            # moments, untouched) are restored before the next epoch's steps
            la["fast_backup"] = [q.detach().clone() for q in la["params"]]
            with torch.no_grad():
                for q, sl in zip(la["params"], la["slow"]): q.copy_(sl)
        score, per = evaluate(model)
        # a single-platform expert is selected on its own platform, not on the
        # three it never trained on (found by review 2026-09-12: xdrone's saved
        # weights were epoch 28 / drone 0.505, its best drone epoch 36 / 0.463)
        select = per[a.platforms[0]] if (a.platforms and len(a.platforms) == 1) else score
        if a.trainval:
            # val is in the training set: fixed length, the last epoch's weights
            # are the model; the printed val score is on seen data
            select = -float(epoch)
        n = a.steps_per_epoch
        row = {"epoch": epoch, "score": score, **{k: v / n for k, v in tot.items()}, **per}
        if hold is not None:
            hs, hav, hat = evaluate_hold(model)
            row.update({"hold_score": hs, "hold_ave": hav, "hold_ate": hat})
        star = ""
        if select < best - 1e-4:
            best, waiting, star = select, 0, "  *"
            safe_save({"model": model.state_dict(), "imu_mean": mean, "imu_std": std,
                        "epoch": epoch, "score": score, "select": select, "args": vars(a), **({"attnet": {"state_dict": att_m.state_dict(), "config": ca["config"]}} if att_m is not None else {}), **({"calnet": {"state_dict": cal_m.state_dict(), "config": cc["config"]}} if cal_m is not None else {})}, ckpt)
        else:
            waiting += 1
        line = f"Score {score:.4f}   " + "  ".join(f"{p} {per[p]:.3f}" for p in PLATFORMS)
        if hold is not None:
            line += f"   | hold {row['hold_score']:.3f} (ave {row['hold_ave']:.3f})"
        if ema is not None:
            # the averaged weights are a second candidate model, scored and kept
            # on their own; until 2026-09-11 they were updated and never read
            ema_score, ema_per = evaluate(ema.module)
            row.update({"ema_score": ema_score, **{f"ema_{k}": v for k, v in ema_per.items()}})
            if ema_score < best_ema - 1e-4:
                best_ema = ema_score
                safe_save({"model": ema.module.state_dict(), "imu_mean": mean, "imu_std": std,
                            "epoch": epoch, "score": ema_score, "args": vars(a), **({"attnet": {"state_dict": att_m.state_dict(), "config": ca["config"]}} if att_m is not None else {}), **({"calnet": {"state_dict": cal_m.state_dict(), "config": cc["config"]}} if cal_m is not None else {})}, ckpt_ema)
                star += " e*"
            line += f"   | ema {ema_score:.4f}"
        history.append(row)
        # periodic on-disk state (lane safety 2026-09-16): history every epoch,
        # the current weights every 5 epochs when --save_last is on
        pd.DataFrame(history).to_csv(ROOT / f"unified/history_{a.tag}.csv", index=False)
        if a.save_last and epoch % 5 == 0:
            safe_save({"model": model.state_dict(), "imu_mean": mean, "imu_std": std, "epoch": epoch,
                        "score": score, "args": vars(a), "periodic": True, **({"attnet": {"state_dict": att_m.state_dict(), "config": ca["config"]}} if att_m is not None else {}), **({"calnet": {"state_dict": cal_m.state_dict(), "config": cc["config"]}} if cal_m is not None else {})}, ROOT / f"unified/ckpt_{a.tag}_last.pt")
        extra = " ".join(f"{k} {v/n:.3f}" for k, v in tot.items() if v)
        print(f"ep{epoch:3d}  {extra}  {line}{star}")
        if la is not None:
            print(f"      lookahead: attempted {la['att']} ok {la['ok']} skipped {la['skipped']} syncs {la['sync']}")
        if waiting >= a.patience:
            print("early stop"); break
        stop_at = os.environ.get("TARTANIMU_STOP_AT")
        if (ROOT / "runs/lanes/STOP").exists() or (stop_at and time.strftime("%Y-%m-%d %H:%M") >= stop_at):
            # lane safety (2026-09-16): save what exists and leave at an epoch boundary
            safe_save({"model": model.state_dict(), "imu_mean": mean, "imu_std": std, "epoch": epoch,
                        "score": score, "args": vars(a), "stopped": True, **({"attnet": {"state_dict": att_m.state_dict(), "config": ca["config"]}} if att_m is not None else {}), **({"calnet": {"state_dict": cal_m.state_dict(), "config": cc["config"]}} if cal_m is not None else {})}, ROOT / f"unified/ckpt_{a.tag}_stopped.pt")
            pd.DataFrame(history).to_csv(ROOT / f"unified/history_{a.tag}.csv", index=False)
            print(f"STOPPED at epoch {epoch}/{a.epochs} (STOP flag or TARTANIMU_STOP_AT={stop_at}); "
                  f"ckpt_{a.tag}_stopped.pt written, no submissions", flush=True)
            return
        if la is not None:
            with torch.no_grad():
                for q, fb in zip(la["params"], la["fast_backup"]): q.copy_(fb)
            del la["fast_backup"]

    if la is not None:
        with torch.no_grad():
            for q, sl in zip(la["params"], la["slow"]): q.copy_(sl)    # the final model is the slow copy
    if a.save_last:
        pl_ = predict(model, Xva_raw, Uva_raw, val, a.model, K, norm)
        pl_ = back_to_label(pl_, val)
        pd.DataFrame({"window_id": val["window_id"], "vx": pl_[:, 0], "vy": pl_[:, 1], "vz": pl_[:, 2]}
                     ).to_csv(ROOT / f"local_eval/sub_val_{a.tag}_last.csv", index=False)
        if hold is not None:
            ph_ = predict(model, Xho_raw, Uho_raw, hold, a.model, K, norm)
            pd.DataFrame({"window_id": hold["window_id"], "vx": ph_[:, 0], "vy": ph_[:, 1], "vz": ph_[:, 2]}
                         ).to_csv(ROOT / f"local_eval/sub_hold_{a.tag}_last.csv", index=False)
        safe_save({"model": model.state_dict(), "imu_mean": mean, "imu_std": std, "epoch": epoch,
                    "score": score, "args": vars(a), **({"attnet": {"state_dict": att_m.state_dict(), "config": ca["config"]}} if att_m is not None else {}), **({"calnet": {"state_dict": cal_m.state_dict(), "config": cc["config"]}} if cal_m is not None else {})}, ROOT / f"unified/ckpt_{a.tag}_last.pt")
        print("wrote last-epoch CSVs and ckpt_<tag>_last.pt")
    if swa is not None:
        # the last-N average is a second candidate next to the last epoch; on
        # the hold-out it is a paired comparison inside one run
        sd = {k: (swa["sum"][k] / swa["n"]).to(v.dtype) if k in swa["sum"] else v.clone()
              for k, v in model.state_dict().items()}
        swa_net = copy.deepcopy(model); swa_net.load_state_dict(sd)
        ws, wper = evaluate(swa_net)
        line = f"swa({a.swa_last}) Score {ws:.4f}   " + "  ".join(f"{p} {wper[p]:.3f}" for p in PLATFORMS)
        if hold is not None:
            hs, hav, hat = evaluate_hold(swa_net)
            line += f"   | hold {hs:.3f} (ave {hav:.3f})"
            ph_ = predict(swa_net, Xho_raw, Uho_raw, hold, a.model, K, norm)
            pd.DataFrame({"window_id": hold["window_id"], "vx": ph_[:, 0], "vy": ph_[:, 1], "vz": ph_[:, 2]}
                         ).to_csv(ROOT / f"local_eval/sub_hold_{swa_tag(a.tag)}.csv", index=False)
        print(line)
        ckpt_swa = ROOT / f"unified/ckpt_{swa_tag(a.tag)}.pt"
        safe_save({"model": swa_net.state_dict(), "imu_mean": mean, "imu_std": std, "epoch": a.epochs,
                    "score": ws, "args": vars(a), **({"attnet": {"state_dict": att_m.state_dict(), "config": ca["config"]}} if att_m is not None else {}), **({"calnet": {"state_dict": cal_m.state_dict(), "config": cc["config"]}} if cal_m is not None else {})}, ckpt_swa)
        history.append({"epoch": -a.swa_last, "score": ws, **wper,
                        **({"hold_score": hs, "hold_ave": hav, "hold_ate": hat} if hold is not None else {})})
    pd.DataFrame(history).to_csv(ROOT / f"unified/history_{a.tag}.csv", index=False)
    print(f"\nbest Score {best:.4f}" + (f"   ema {best_ema:.4f}" if ema is not None else ""))
    finals = [(model, ckpt, a.tag)]
    if ema is not None:
        finals.append((ema.module, ckpt_ema, ema_tag(a.tag)))
    if swa is not None:
        finals.append((swa_net, ckpt_swa, swa_tag(a.tag)))
    for net, path, tag in finals:
        net.load_state_dict(torch.load(path, map_location=dev, weights_only=False)["model"])
        write_submissions(net, tag)


def swa_tag(tag: str) -> str:
    """R3_s42 -> R3w_s42: the last-N weight average as its own condition."""
    stem, _, seed = tag.rpartition("_s")
    return f"{stem}w_s{seed}" if seed.isdigit() else f"{tag}w"


def ema_tag(tag: str) -> str:
    """f5_s42 -> f5e_s42: the EMA copy is its own condition for report.py."""
    stem, _, seed = tag.rpartition("_s")
    return f"{stem}e_s{seed}" if seed.isdigit() else f"{tag}e"


if __name__ == "__main__":
    main()

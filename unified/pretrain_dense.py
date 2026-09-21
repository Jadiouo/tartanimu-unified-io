#!/usr/bin/env python3
"""Masked IMU modelling at scale: windows drawn at random offsets from the
continuous training streams (every offset is a fresh window, as in the
supervised dense sampler), for a fixed wall-clock or update budget, with
resumable checkpoints and intermediate snapshots so a long run can be
evaluated early, mid and late.

Everything the fine-tune or a joint objective needs is saved: trunk, decoder,
optimiser, scheduler, scaler, step, the normalisation statistics, the mask
setting and its realised fraction, the splits and seeds.  The normalisation
statistics are computed exactly as train_v2 computes them (rng 0, 20000 fixed
training windows) so the trunk transfers unchanged.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from unified.data import WINDOW_SIZE, build_windows, load_index  # noqa: E402
from unified.dense import DenseWindows  # noqa: E402
from unified.fine_context import FineContextIMUNet  # noqa: E402
from unified.updir import N_CHANNELS, build_features, build_up_windows  # noqa: E402


def make_decoder(c_trunk: int, c_out: int) -> nn.Module:
    return nn.Sequential(nn.Upsample(scale_factor=8, mode="linear", align_corners=False),
                         nn.Conv1d(c_trunk, 96, 5, padding=2), nn.GELU(), nn.Conv1d(96, c_out, 1))


def train_stats(idx, grav, dev):
    tr = idx["train"]
    Xtr = torch.from_numpy(np.asarray(build_windows(tr)))
    Utr = torch.from_numpy(np.asarray(build_up_windows(tr)))
    rng = np.random.default_rng(0)
    pick = rng.choice(len(Xtr), min(20000, len(Xtr)), replace=False)
    stat = build_features(Xtr[pick], Utr[pick], grav)
    return stat.mean(dim=(0, 1)).to(dev), stat.std(dim=(0, 1)).clamp(min=1e-6).to(dev)


def random_mask(B, T, frac, n_span, dev):
    span = int(frac * T / n_span)
    m = torch.zeros(B, T, dtype=torch.bool, device=dev)
    tt = torch.arange(T, device=dev)[None]
    for _ in range(n_span):
        s0 = torch.randint(0, T - span, (B, 1), device=dev)
        m |= (tt >= s0) & (tt < s0 + span)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train"])
    ap.add_argument("--grav", default="slow")
    ap.add_argument("--minutes", type=float, default=120.0, help="wall-clock budget")
    ap.add_argument("--updates", type=int, default=0, help="or a fixed update budget (0 = use minutes)")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mask_frac", type=float, default=0.3)
    ap.add_argument("--n_span", type=int, default=3)
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--snapshots", type=int, default=3, help="intermediate saves, evenly spaced")
    ap.add_argument("--objective", choices=["masked", "jepa", "both"], default="masked",
                    help="masked: reconstruct masked input spans (default). jepa: predict the EMA "
                         "target-encoder features of the NEXT window from the two preceding windows "
                         "(latent prediction, no reconstruction). both: sum of the two losses.")
    ap.add_argument("--jepa_ema", type=float, default=0.996)
    ap.add_argument("--tdil", type=float, nargs=2, default=None,
                    help="time-dilate the pre-training windows too (drone, p=0.5)")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--out", required=True, help="path stem; snapshots get _k suffixes")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    idx = load_index()
    if a.splits == ["train"]:
        frame = idx["train"]
    elif sorted(a.splits) == ["train", "val"]:
        import pandas as pd
        frame = pd.concat([idx["train"], idx["val"]], ignore_index=True)   # final-recipe runs only
    else:
        raise SystemExit("dense pre-training decodes train or train+val streams only")
    ext = None
    if a.tdil:
        from unified.gtup import load_extrinsics
        ext = load_extrinsics(ROOT / "runs/extrinsics.npz")
    src = DenseWindows(frame, dev, platform_balanced=False, with_pose=False, extrinsics=ext,
                       tdil=tuple(a.tdil) if a.tdil else None)
    print(f"{src.n_traj} trajectories, {src.total_frames} frames", flush=True)
    mean, std = train_stats(idx, a.grav, dev)
    C, T = N_CHANNELS[a.grav], WINDOW_SIZE

    net = FineContextIMUNet(fine=5, in_channels=C, width=a.width).to(dev)
    c4 = net.stage4[-1].main[-1].num_features
    dec = make_decoder(c4, C).to(dev)
    params = [q for m in (net.stem, net.stage2, net.stage3, net.stage4, dec) for q in m.parameters()]
    if a.objective != "masked":
        # JEPA: target encoder = EMA copy of the trunk (no grad); predictor maps the
        # pooled context features (two windows) to the pooled target features (next window)
        import copy
        tgt = copy.deepcopy(net).to(dev).eval()
        for q in tgt.parameters():
            q.requires_grad_(False)
        pool = lambda h: torch.cat([h.float().mean(-1), h.float().std(-1, unbiased=False)], 1)   # (B, 2*c4), fp32
        pred = nn.Sequential(nn.Linear(4 * c4, 512), nn.GELU(), nn.Linear(512, 2 * c4)).to(dev)
        params += list(pred.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    # a wall-clock budget needs a step count for the schedule: measure 20 steps first
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    gen = torch.Generator(device=dev); gen.manual_seed(a.seed)

    n_win = 1 if a.objective == "masked" else 3

    def batch():
        b = src.sample(a.batch, n_win, generator=gen)
        x = build_features(b["imu"].flatten(0, 1), b["up"].flatten(0, 1), a.grav)   # (B*n_win, T, C)
        xb = ((x - mean) / std).transpose(1, 2).contiguous().view(a.batch, n_win, C, T)
        xb.full_rows = (b["lengths"] >= n_win) if n_win > 1 else None       # segments with 3 real windows
        return xb

    def step_fn(xb):
        full = getattr(xb, "full_rows", None)
        loss = 0.0
        if a.objective != "jepa":
            x1 = xb[:, -1]                                                  # one window per row
            m = random_mask(x1.shape[0], T, a.mask_frac, a.n_span, dev)
            xin = x1.masked_fill(m[:, None, :], 0.0)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                rec = dec(net.trunk_seq(xin))
                loss = loss + ((rec.float() - x1) ** 2)[m[:, None, :].expand_as(x1)].mean()
        else:
            m = torch.zeros(xb.shape[0], T, dtype=torch.bool, device=dev)
        if a.objective != "masked":
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                h0, h1 = net.trunk_seq(xb[:, 0]), net.trunk_seq(xb[:, 1])
                with torch.no_grad():
                    h2 = tgt.trunk_seq(xb[:, 2])
            # pooling, target standardisation and the predictor in float32 (fp16 variance overflows)
            # bounded formulation (v2): L2-normalised pooled features on both sides and a
            # cosine loss, so a drifting feature scale cannot blow the loss up (v1 with
            # batch-standardised targets skipped 73 % of steps as non-finite)
            ctx = torch.cat([F.normalize(pool(h0), dim=1), F.normalize(pool(h1), dim=1)], 1)
            with torch.no_grad():
                z = F.normalize(pool(h2), dim=1)
            err = 2 - 2 * (F.normalize(pred(ctx), dim=1) * z).sum(1)
            loss = loss + (err * full).sum() / full.sum().clamp(min=1)
        if not torch.isfinite(loss):                                        # skip a non-finite step, never backprop it
            opt.zero_grad(set_to_none=True); step_fn.skipped = getattr(step_fn, "skipped", 0) + 1
            return float("nan"), float(m.float().mean())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt); scaler.update()
        if a.objective != "masked":
            with torch.no_grad():
                for q_t, q in zip(tgt.parameters(), net.parameters()):
                    q_t.mul_(a.jepa_ema).add_(q.detach(), alpha=1 - a.jepa_ema)
        return float(loss.detach()), float(m.float().mean())

    if a.updates:
        total = a.updates
    else:
        tw = time.time()
        for _ in range(20):
            step_fn(batch())
        per = (time.time() - tw) / 20
        total = int(a.minutes * 60 / per)
        print(f"{per*1000:.0f} ms/update -> {total} updates in {a.minutes} min", flush=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.05)
    step, mask_seen, loss_run = 0, 0.0, []
    if a.resume:
        c = torch.load(a.resume, map_location=dev, weights_only=False)
        net.load_state_dict(c["trunk"], strict=False); dec.load_state_dict(c["decoder"])
        opt.load_state_dict(c["optimizer"]); sched.load_state_dict(c["scheduler"])
        scaler.load_state_dict(c["scaler"]); step = c["step"]
        print(f"resumed at step {step}", flush=True)

    def save(path, tag):
        trunk = {k: v for k, v in net.state_dict().items()
                 if k.split(".")[0] in ("stem", "stage2", "stage3", "stage4")}
        torch.save({"trunk": trunk, "decoder": dec.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "scaler": scaler.state_dict(), "step": step,
                    "total": total, "imu_mean": mean.cpu(), "imu_std": std.cpu(),
                    "splits": a.splits, "grav": a.grav, "width": a.width, "seed": a.seed,
                    "sampling": "dense", "epochs": None, "updates": step,
                    "mask_setting": a.mask_frac, "n_span": a.n_span, "tdil": a.tdil, "objective": a.objective,
                    "mask_actual": mask_seen / max(step, 1),
                    "masked_mse": float(np.mean(loss_run[-200:])) if loss_run else None,
                    "tag": tag, "elapsed_s": time.time() - t0}, path)
        print(f"saved {tag} at step {step}/{total}  masked MSE {np.mean(loss_run[-200:]):.4f}  "
              f"({(time.time()-t0)/60:.0f} min) -> {Path(path).name}", flush=True)

    snaps = {int(total * k / (a.snapshots + 1)) for k in range(1, a.snapshots + 1)}
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    while step < total:
        l, mf = step_fn(batch()); sched.step(); step += 1
        if l == l: loss_run.append(l)                                       # NaN steps are skipped, not averaged
        mask_seen += mf
        if step % 500 == 0:
            print(f"step {step}/{total}  masked MSE {np.mean(loss_run[-500:]):.4f}  "
                  f"lr {sched.get_last_lr()[0]:.2e}  ({(time.time()-t0)/60:.0f} min)", flush=True)
        if step in snaps:
            k = sorted(snaps).index(step) + 1
            save(out.with_name(out.stem + f"_snap{k}.pt"), f"snap{k}")
    save(out, "final")
    print(f"actual mask fraction {mask_seen/step:.3f} (setting {a.mask_frac}); non-finite steps skipped: {getattr(step_fn, 'skipped', 0)}")


if __name__ == "__main__":
    main()

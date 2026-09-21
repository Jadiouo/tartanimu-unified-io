#!/usr/bin/env python3
"""Continuous-offset window sampling straight off the raw trajectories.

`index/train_windows.csv` enumerates only the 81,931 NON-OVERLAPPING windows, but
the .npz files hold 22.8 h of continuous IMU (16.4 M frames) together with
per-frame `vel_body`, and the scored target for a window is exactly the mean of
`vel_body` over its 200 frames -- verified to 1.8e-15.  Any offset is therefore a
legitimate training example with an exact label, and the model stops being able
to memorise the 82k fixed window phases.  Measured train/val Huber gap on the
fixed windows is about 6x, so phase memorisation is the binding constraint.

Materialising every stride-20 window would cost 3.9 GB; the raw streams cost
590 MB, so the trajectories are held whole on the GPU and windows are gathered
on the fly at a fresh random offset every batch.
"""
from __future__ import annotations

import numpy as np
import torch

from unified.data import PLATFORM_TO_ID, WINDOW_SIZE, cache_path, load_index
from unified.moe import traj_descriptors
from unified.updir import both_up


def quat_to_R_torch(q: torch.Tensor) -> torch.Tensor:
    """(..., 4) quaternion (x, y, z, w) -> (..., 3, 3); mirrors segments.quat_to_R."""
    q = q / q.norm(dim=-1, keepdim=True)
    x, y, z, w = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], dim=-2)


class DenseWindows:
    """Raw train trajectories on device, sampled at arbitrary offsets."""

    def __init__(self, frame, device, platform_balanced: bool = True,
                 with_pose: bool = False, min_windows: int = 0, platforms=None,
                 aggr_weight: float = 1.0, aggr_thresh: float = 1.5,
                 extrinsics=None, gt_up: bool = False, idw: float = 1.0, ext_weight: float = 1.0,
                 gyro_frame: bool = False, tdil=None, tdil_platforms=("drone",),
                 tdil_prob: float = 0.5, sdil=None, aug_mix: bool = False, tdil_fast=None,
                 strat_prob: float = 0.0, adapt: bool = False, tdil_short: str = "keep",
                 tdil_aa: bool = False, sdil_fast=None, rest: bool = False, restgyro: bool = False, dr: bool = False, att=None, cal=None, ins_drop: float = 0.0, drag_pose: bool = False):
        """tdil=(kmin, kmax): physics-consistent time dilation of a segment by a
        factor k drawn uniformly, applied with probability tdil_prob to the
        listed platforms. The recording is resampled at k times the rate; the
        gyro and the velocity scale by k, the specific force by k^2 about the
        gravity reaction g*up (ground-truth up in the accelerometer's frame),
        which does not scale. Training only."""
        """extrinsics: {traj_id: (R 3x3 IMU->label, quality, sig)} from
        local_eval/extrinsics.py. gt_up replaces both up estimates by the
        ground-truth up in the IMU frame (needs with_pose). idw > 1 upweights
        drone trajectories whose fitted extrinsic is the identity."""
        """min_windows > 0 drops trajectories shorter than that many windows from
        the sampling weights entirely (the "exclude" policy); 0 keeps them and
        sample() returns their valid length for the model to mask."""
        if (gt_up or tdil is not None or sdil is not None) and not with_pose:
            with_pose = True
        self.tdil, self.tdil_prob = tdil, tdil_prob
        # sdil=(smin, smax): translation scaling -- time and attitude unchanged,
        # v' = s v, w' = w, f' = s (f - g up) + g up. aug_mix: each eligible
        # segment gets either T or S (never both), half of the augmented share each.
        self.sdil, self.aug_mix = sdil, aug_mix
        # tdil_fast=(kmin, kmax): a separate k range for drone trajectories of the
        # identity-extrinsic source (the fast, noisy one); needs extrinsics
        self.tdil_fast = tdil_fast
        # sdil_fast=(smin, smax) (S-fast, 2026-09-18): translation scaling drawn ONLY for
        # identity-extrinsic (racing) drone recordings; other recordings keep s = 1
        # (or the global --sdil range if given). Drag-consistent: f-g scales with v.
        self.sdil_fast = sdil_fast
        # "cancel": the sampler before 2026-09-14 -- a trajectory that cannot give
        # K windows at rate k is not dilated (identity-source 40 s flights then
        # only ever saw k < 1). Kept as an option: on the official test split the
        # two most extreme sequences (#29/#56, 17 s) score 4.6/4.5 AVE with it
        # and 5.0-5.4 with "keep" (v3/v4/v5/v6), while the body prefers "keep".
        self.tdil_short = tdil_short
        # tdil_aa (D1a, 2026-09-15): for k > 1 the IMU is gathered at 4x oversampling,
        # low-passed at the output Nyquist (100 Hz in dilated time) with a 63-tap
        # Hamming FIR and decimated, so original content above 100/k Hz does not
        # fold back. CPU gate: plain lerp at k=2 puts 8.5 (5-20 Hz band energy)
        # where real fast flights have 1.8; anti-aliased 0.9.
        self.tdil_aa = tdil_aa
        if tdil_aa:
            n = 63; t = torch.arange(n, dtype=torch.float64) - (n - 1) / 2
            fc = 0.125                                           # 100 Hz of 800 Hz (cycles/sample)
            h = 2 * fc * torch.sinc(2 * fc * t) * torch.hamming_window(n, periodic=False, dtype=torch.float64)
            self.aa_kernel = (h / h.sum()).to(torch.float32).to(device)[None, None, :]
        # strat_prob > 0: that share of drone segments is placed around a window
        # drawn from a RARE motion bin (speed quintile x gyro tercile, bins with
        # fewer windows than the median), recording chosen uniformly within the
        # bin so no single flight dominates. The rest keeps the usual draw.
        self.strat_prob = strat_prob
        if (sdil is not None or sdil_fast is not None) and tdil is None:
            tdil = (1.0, 1.0)                   # route through the dilated sampler
            self.tdil = tdil
        self.tdil_plats = [PLATFORM_TO_ID[q] for q in tdil_platforms]
        raw = self._decode(frame, with_pose)
        self.device, self.with_pose = device, with_pose
        from pathlib import Path as _P
        self.traj_ids = [_P(fp).stem for fp, _ in frame.groupby("file_path", sort=False)]
        self.imu = torch.from_numpy(raw["imu"]).to(device)
        n_tr = len(raw["starts"])
        R_ext = np.tile(np.eye(3, dtype=np.float32), (n_tr, 1, 1))
        self.sig = np.full(n_tr, -1)
        if extrinsics is not None:
            for i, t in enumerate(self.traj_ids):
                if t in extrinsics:
                    R_ext[i] = extrinsics[t][0]; self.sig[i] = extrinsics[t][2]
        self.R_ext = torch.from_numpy(R_ext).to(device)
        if gyro_frame:
            # rotate the gyro into the label/accelerometer frame with the fitted
            # extrinsic: omega' = R_ext omega, per trajectory (diagnostic, train side)
            imu = raw["imu"]
            for i, (s0, n) in enumerate(zip(raw["starts"], raw["lengths"])):
                imu[s0:s0 + n, 3:6] = imu[s0:s0 + n, 3:6] @ R_ext[i].T
            raw["imu"] = imu
            self.imu = torch.from_numpy(raw["imu"]).to(device)
            # the up estimates (accelerometer-driven complementary filter) are kept
            # as they are; recomputing them is a 16M-frame Python loop
        if tdil is not None or sdil is not None:
            from unified.gtup import gt_up_frames
            gu = np.zeros((len(raw["imu"]), 3), np.float32)
            for i, (s0, n) in enumerate(zip(raw["starts"], raw["lengths"])):
                gu[s0:s0 + n] = gt_up_frames(raw["imu"][s0:s0 + n], raw["quat"][s0:s0 + n], R_ext[i])
            self.gt_up = torch.from_numpy(gu).to(device)
        if gt_up:
            from unified.gtup import gt_up_frames
            up = raw["up"].copy()
            for i, (s0, n) in enumerate(zip(raw["starts"], raw["lengths"])):
                u = gt_up_frames(raw["imu"][s0:s0 + n], raw["quat"][s0:s0 + n], R_ext[i])
                up[s0:s0 + n, :3] = u; up[s0:s0 + n, 3:] = u
            raw["up"] = up
        if adapt:
            # gated-gain filter as columns 6:9 of the up stream (updir.adaptive_up);
            # its own cache so the dense caches stay valid
            raw["up"] = np.concatenate([raw["up"], self._adaptive(frame, raw)], axis=1)
        if rest:
            # per-recording rest reference as columns 6:10 (updir.rest_reference), constant in time
            from unified.updir import rest_reference
            rr = np.zeros((len(raw["imu"]), 4), np.float32)
            for s0, n in zip(raw["starts"], raw["lengths"]):
                rr[s0:s0 + n] = rest_reference(raw["imu"][s0:s0 + n])[None, :]
            raw["up"] = np.concatenate([raw["up"], rr], axis=1)
            if dr:
                # columns 10:27 = [rest up, v_DR, t_since, R 6D, v_F0, calib dev] from derived_channels (AttNet / CalNet optional)
                from unified.updir import derived_channels
                dc = np.zeros((len(raw["imu"]), 17), np.float32)
                for s0, n in zip(raw["starts"], raw["lengths"]):
                    dc[s0:s0 + n] = derived_channels(raw["imu"][s0:s0 + n], att, cal, device=str(device))
                raw["up"] = np.concatenate([raw["up"], dc], axis=1)
                restgyro = dr = False
            if restgyro:
                from unified.updir import rest_up_gyro
                ug = np.zeros((len(raw["imu"]), 3), np.float32)
                for s0, n in zip(raw["starts"], raw["lengths"]):
                    ug[s0:s0 + n] = rest_up_gyro(raw["imu"][s0:s0 + n])
                raw["up"] = np.concatenate([raw["up"], ug], axis=1)
            if dr:
                # columns 13:17 = dead-reckoned velocity from the rest anchor (3) + t_since (1)
                from unified.updir import dr_channels
                dc = np.zeros((len(raw["imu"]), 4), np.float32)
                for s0, n in zip(raw["starts"], raw["lengths"]):
                    dc[s0:s0 + n] = dr_channels(raw["imu"][s0:s0 + n])
                raw["up"] = np.concatenate([raw["up"], dc], axis=1)
        self.dr = raw["up"].shape[1] >= 17                 # v_DR columns present (scaled like the labels)
        self.ins_drop = ins_drop                            # path 2 (b): per-sample INS-channel dropout (columns 10:17), training only
        # road 3 (beyond-deadline plan, 2026-09-20): drag-consistent attitude synthesis for S > 1
        # samples of the racing family. Per recording a (m/s per m/s^2): |v_h| = a * |f_h - f_h,rest|
        # fitted on GT (training time only); NaN = no fit -> no extra rotation for that recording.
        self.drag_pose = drag_pose
        self.drag_a = torch.full((n_tr,), float("nan"), dtype=torch.float32, device=device)
        if drag_pose:
            for i, (s0, n) in enumerate(zip(raw["starts"], raw["lengths"])):
                if self.sig[i] != 6 or raw["plats"][i] != PLATFORM_TO_ID["drone"]: continue
                im = raw["imu"][s0:s0 + n]; vv = raw["vel"][s0:s0 + n]; nw = n // WINDOW_SIZE
                if nw < 5: continue
                acc = im[:nw * WINDOW_SIZE, :3].reshape(nw, WINDOW_SIZE, 3); gyr = im[:nw * WINDOW_SIZE, 3:6].reshape(nw, WINDOW_SIZE, 3)
                an = np.linalg.norm(acc, axis=2); k0 = int(np.argmin(np.abs(an.mean(1) - 9.81) + np.linalg.norm(gyr, axis=2).mean(1) + 0.5 * an.std(1)))
                fr = acc[k0].mean(0); up = fr / max(np.linalg.norm(fr), 1e-6)
                fm = acc.mean(1); fh = fm - (fm @ up)[:, None] * up; fhr = fr - (fr @ up) * up
                tilt = np.linalg.norm(fh - fhr, axis=1); vm = vv[:nw * WINDOW_SIZE].reshape(nw, WINDOW_SIZE, 3).mean(1)
                sp = np.linalg.norm(vm - (vm @ up)[:, None] * up, axis=1); mv = sp > 1.0
                if mv.sum() >= 10:
                    a_fit = float(np.polyfit(tilt[mv], sp[mv], 1)[0])
                    if 0.5 < a_fit < 10: self.drag_a[i] = a_fit
            print(f"drag_pose: slope fitted for {int(torch.isfinite(self.drag_a).sum())} racing recordings", flush=True)
        self.up = torch.from_numpy(raw["up"]).to(device)
        if strat_prob > 0:
            self._build_strata(raw)
        self.vel = torch.from_numpy(raw["vel"]).to(device)
        if with_pose:
            self.quat = torch.from_numpy(raw["quat"]).to(device)
            self.pos = torch.from_numpy(raw["pos"]).to(device)

        starts, lengths, plats = raw["starts"], raw["lengths"], raw["plats"]
        self.n_traj = len(starts)
        assert lengths.min() >= WINDOW_SIZE, "a trajectory shorter than one window"
        # last legal window start inside each trajectory; a window never straddles two
        self.lo = torch.from_numpy(starts).to(device)
        self.span = torch.from_numpy(np.maximum(lengths - WINDOW_SIZE, 0)).to(device)
        self.n_windows = torch.from_numpy(lengths // WINDOW_SIZE).to(device)
        self.platform = torch.from_numpy(plats).to(device)

        keep = (lengths // WINDOW_SIZE) >= max(1, min_windows)
        if platforms is not None:                      # single-platform expert
            keep &= np.isin(plats, [PLATFORM_TO_ID[q] for q in platforms])
        self.n_dropped = int((~keep).sum())
        if platform_balanced:
            counts = np.bincount(plats[keep], minlength=len(PLATFORM_TO_ID)).astype(np.float64)
            w = 1.0 / np.maximum(counts[plats], 1)      # each platform equally likely,
        else:                                           # each trajectory equally likely within it
            w = lengths.astype(np.float64)
        w = w * keep
        # aggressiveness of each trajectory: 90th percentile of | |a| - g |. The
        # test drone flights sit at ~1.9 m/s^2 by this measure against 0.66 on
        # val, so aggr_weight > 1 tilts drone sampling toward that regime.
        acc = torch.linalg.norm(self.imu[:, :3], dim=1)
        dev = (acc - 9.81).abs()
        self.aggr = np.array([float(torch.quantile(dev[int(s):int(s) + int(n)], 0.9))
                              for s, n in zip(starts, lengths)])
        if aggr_weight != 1.0:
            hot = (plats == PLATFORM_TO_ID["drone"]) & (self.aggr > aggr_thresh)
            is_drone = plats == PLATFORM_TO_ID["drone"]
            before = w[is_drone].sum()
            w = w * np.where(hot, aggr_weight, 1.0)
            # renormalise so the drone's share of sampling is unchanged: only the
            # aggressive/calm split inside the platform moves (the first version
            # also raised the drone from 25% to 34% of all sampling)
            w[is_drone] *= before / w[is_drone].sum()
            self.n_aggr = int(hot.sum())
        if idw != 1.0:
            is_drone = plats == PLATFORM_TO_ID["drone"]
            ident = is_drone & (self.sig == 6)
            before = w[is_drone].sum(); w = w * np.where(ident, idw, 1.0)
            w[is_drone] *= before / w[is_drone].sum()
            self.n_ident = int(ident.sum())
        if ext_weight != 1.0:
            # external recordings (road 4): scale their share within the drone platform
            is_drone = plats == PLATFORM_TO_ID["drone"]
            is_ext = np.array([("neurobem" in t) or ("_ext_" in t) for t in self.traj_ids])
            before = w[is_drone].sum(); w = w * np.where(is_ext, ext_weight, 1.0)
            w[is_drone] *= before / w[is_drone].sum()
            self.n_ext = int(is_ext.sum())
        self.traj_w = torch.as_tensor(w / w.sum(), dtype=torch.double, device=device)
        # --ext_until: the same weights with the external recordings removed (drone
        # share renormalised), swapped in by drop_ext() for a competition-only finish
        is_ext = np.array([("neurobem" in t) or ("_ext_" in t) for t in self.traj_ids])
        is_drone = plats == PLATFORM_TO_ID["drone"]
        w0 = w * (~is_ext)
        if is_ext.any() and w0[is_drone].sum() > 0:
            w0[is_drone] *= w[is_drone].sum() / w0[is_drone].sum()
        self.traj_w_noext = torch.as_tensor(w0 / w0.sum(), dtype=torch.double, device=device)

        self.arange = torch.arange(WINDOW_SIZE, device=device)
        self.total_frames = int(lengths.sum())
        # whole-trajectory descriptors for --traj_film, in raw units
        self.traj_stats = torch.stack([
            traj_descriptors(self.imu[int(s):int(s) + int(n)]) for s, n in zip(starts, lengths)])

    def drop_ext(self):
        """--ext_until: competition-only sampling from here on."""
        self.traj_w = self.traj_w_noext

    @staticmethod
    def _adaptive(frame, raw) -> np.ndarray:
        from unified.updir import adaptive_up
        cache = cache_path(frame, "densea")
        if cache is not None and cache.exists():
            return np.load(cache)
        out = np.concatenate([adaptive_up(raw["imu"][s0:s0 + n]).astype(np.float32)
                              for s0, n in zip(raw["starts"], raw["lengths"])])
        if cache is not None:
            np.save(cache, out)
        return out

    @staticmethod
    def _decode(frame, with_pose: bool) -> dict:
        """Concatenated raw streams for every trajectory, memoised on disk.

        Decoding 395 .npz files costs seconds against a warm page cache but
        minutes against Kaggle's network-mounted input, and every run pays it
        again; caching the concatenated arrays turns that into one cost per node.
        """
        cache = cache_path(frame, f"dense{int(with_pose)}")
        cache = cache.with_suffix(".npz") if cache is not None else None
        if cache is not None and cache.exists():
            with np.load(cache) as z:
                return {k: z[k] for k in z.files}

        keys = ("imu", "vel_body") + (("quat", "pos") if with_pose else ())
        parts: dict = {k: [] for k in keys}
        starts, lengths, plats = [], [], []
        offset = 0
        ups = []
        from unified.updir import up_for_files
        up_all = up_for_files(frame["file_path"].tolist())      # parallel across recordings
        for file_path, rows in frame.groupby("file_path", sort=False):
            with np.load(file_path) as d:
                cols = {k: d[k] for k in keys}
            n = min(len(v) for v in cols.values())
            ups.append(up_all[file_path][:n])
            for k, v in cols.items():
                parts[k].append(v[:n])
            starts.append(offset)
            lengths.append(n)
            plats.append(PLATFORM_TO_ID[rows["platform"].iloc[0]])
            offset += n

        out = {"imu": np.concatenate(parts["imu"]),
               "up": np.concatenate(ups),
               "vel": np.concatenate(parts["vel_body"]),
               "starts": np.asarray(starts, np.int64),
               "lengths": np.asarray(lengths, np.int64),
               "plats": np.asarray(plats, np.int64)}
        if with_pose:
            out["quat"] = np.concatenate(parts["quat"])
            out["pos"] = np.concatenate(parts["pos"])
        if cache is not None:
            np.savez(cache, **out)
        return out

    def sample(self, batch_size: int, seg_len: int = 1, generator=None) -> dict:
        """A batch of windows drawn at fresh random offsets.

        Returns a dict so callers take only what they need:
          imu         (B, seg_len, WINDOW_SIZE, 6) raw physical units
          target      (B, seg_len, 3) window-mean body-frame velocity, exact
          platform    (B,)
          vel_frames  (B, seg_len, WINDOW_SIZE, 3) per-frame velocity
          R, pos      ground-truth attitude / position per window, if with_pose
          lengths     (B,) how many of the seg_len windows are real
          tstats      (B, 18) whole-trajectory descriptors (moe.traj_descriptors)

        seg_len > 1 draws consecutive windows for a trajectory-level loss; the
        whole run stays inside one trajectory.  A trajectory shorter than seg_len
        windows contributes only as many as it has, the rest of the segment is a
        repeat of its last window and must be masked with `lengths`.  (Until
        2026-09-11 the clamp on the start let the gather run past the end into
        the next trajectory: 37 drone runs, 3.2% of 20 s segments.)
        """
        t = torch.multinomial(self.traj_w, batch_size, replacement=True, generator=generator)
        if self.tdil is not None:
            return self._sample_dilated(t, seg_len, generator)
        L = self.n_windows[t].clamp(max=seg_len)               # valid windows
        span = self.span[t] - (L - 1) * WINDOW_SIZE             # last legal segment start
        u = torch.rand(batch_size, device=self.device, generator=generator)
        base = self.lo[t] + torch.minimum((u * (span + 1).double()).long(), span)

        step = torch.arange(seg_len, device=self.device) * WINDOW_SIZE
        step = torch.minimum(step[None, :], (L[:, None] - 1) * WINDOW_SIZE)   # (B, seg_len)
        idx = base[:, None, None] + step[:, :, None] + self.arange[None, None, :]
        vel = self.vel[idx]                                   # (B, seg_len, 200, 3)
        out = {"imu": self.imu[idx], "up": self._ins_dropout(self.up[idx], generator), "target": vel.mean(dim=2),
               "platform": self.platform[t], "vel_frames": vel, "lengths": L,
               "tstats": self.traj_stats[t], "traj": t, "R_ext": self.R_ext[t],
               "sig": torch.from_numpy(self.sig).to(self.device)[t]}
        if self.with_pose:
            # same convention as local_eval/build_val_solution.py, verified by
            # feeding ground-truth velocities through the official scorer:
            # attitude at the window's middle frame, position at its last frame
            mid = base[:, None] + step + WINDOW_SIZE // 2
            end = base[:, None] + step + WINDOW_SIZE - 1
            out["R"] = quat_to_R_torch(self.quat[mid])
            out["pos"] = self.pos[end]
        return out


def _build_strata(self, raw):
    import numpy as _np
    starts, lengths, plats = raw["starts"], raw["lengths"], raw["plats"]
    vel = raw["vel"]; gy = _np.linalg.norm(raw["imu"][:, 3:6], axis=1)
    rows = []                                             # (traj, frame) of drone windows
    for i, (s0, n) in enumerate(zip(starts, lengths)):
        if plats[i] != PLATFORM_TO_ID["drone"]:
            continue
        for w0 in range(0, n - WINDOW_SIZE + 1, WINDOW_SIZE):
            f0 = s0 + w0
            rows.append((i, f0, _np.linalg.norm(vel[f0:f0 + WINDOW_SIZE], axis=1).mean(), gy[f0:f0 + WINDOW_SIZE].mean()))
    arr = _np.array(rows, dtype=_np.float64)
    sp, gr = arr[:, 2], arr[:, 3]
    sb = _np.digitize(sp, _np.quantile(sp, [0.2, 0.4, 0.6, 0.8])); gb = _np.digitize(gr, _np.quantile(gr, [1 / 3, 2 / 3]))
    bins = sb * 3 + gb
    counts = _np.bincount(bins, minlength=15); rare = _np.where(counts < _np.median(counts[counts > 0]))[0]
    self.strata = []
    for b in rare:
        sel = arr[bins == b]
        by_traj = {}
        for tr, f0 in zip(sel[:, 0].astype(int), sel[:, 1].astype(int)):
            by_traj.setdefault(tr, []).append(f0)
        self.strata.append({tr: _np.array(v) for tr, v in by_traj.items()})
    self.strat_bins = rare; self.strat_counts = counts
    print(f"strata: {len(rare)} rare bins of 15 (counts {counts.tolist()}), "
          f"{sum(len(d) for d in self.strata)} (bin, recording) cells")


def _strat_pick(self, n, generator):
    """n stratified picks -> (traj idx, frame) arrays; bin uniform, recording uniform, window uniform."""
    import numpy as _np
    if generator is None:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    else:
        seed = int(torch.randint(0, 2**31 - 1, (1,), generator=generator, device=generator.device).item())
    rng = _np.random.default_rng(seed)
    tr = _np.empty(n, dtype=_np.int64); fr = _np.empty(n, dtype=_np.int64)
    for j in range(n):
        d = self.strata[rng.integers(len(self.strata))]
        t = list(d)[rng.integers(len(d))]
        tr[j] = t; fr[j] = d[t][rng.integers(len(d[t]))]
    return tr, fr


DenseWindows._build_strata = _build_strata
DenseWindows._strat_pick = _strat_pick


def _ins_dropout(self, up6, generator):
    """zero the INS up-stream columns 10:23 (rest up, v_DR, t_since, R columns) of a whole sample with probability ins_drop."""
    if getattr(self, "ins_drop", 0.0) <= 0 or up6.shape[-1] < 17:
        return up6
    keep = (torch.rand(up6.shape[0], device=up6.device, generator=generator) >= self.ins_drop).to(up6.dtype)
    shape = (up6.shape[0],) + (1,) * (up6.dim() - 1)
    return torch.cat([up6[..., :10], up6[..., 10:23] * keep.view(shape), up6[..., 23:]], dim=-1)


DenseWindows._ins_dropout = _ins_dropout


def _lerp_gather(x, idx_f):
    """x (N, C); idx_f (...,) fractional frame indices -> (..., C) linear interpolation."""
    i0 = idx_f.floor().long().clamp(max=x.shape[0] - 2)
    w = (idx_f - i0.to(idx_f.dtype)).unsqueeze(-1).to(x.dtype)
    return x[i0] * (1 - w) + x[i0 + 1] * w


def _sample_dilated(self, t, seg_len, generator):
    dev = self.device
    B = len(t)
    kmin, kmax = self.tdil
    k = torch.ones(B, device=dev, dtype=torch.float64)
    eligible = torch.zeros(B, dtype=torch.bool, device=dev)
    for pid in self.tdil_plats:
        eligible |= self.platform[t] == pid
    draw = torch.rand(B, device=dev, generator=generator, dtype=torch.float64)
    u_k = torch.rand(B, device=dev, generator=generator, dtype=torch.float64)
    kd = kmin + (kmax - kmin) * u_k
    if self.tdil_fast is not None:
        fmin, fmax = self.tdil_fast
        fast = torch.from_numpy(self.sig == 6).to(dev)[t]
        kd = torch.where(fast, fmin + (fmax - fmin) * u_k, kd)
    sc = torch.ones(B, device=dev, dtype=torch.float64)           # translation scale s
    aug = eligible & (draw < self.tdil_prob)
    if self.sdil is not None or self.sdil_fast is not None:
        u_s = torch.rand(B, device=dev, generator=generator, dtype=torch.float64)
        has_s = torch.zeros(B, dtype=torch.bool, device=dev)       # rows with an S range
        sd = torch.ones(B, device=dev, dtype=torch.float64)
        if self.sdil is not None:
            smin, smax = self.sdil
            sd = smin + (smax - smin) * u_s; has_s[:] = True
        if self.sdil_fast is not None:
            fmin, fmax = self.sdil_fast
            fast = torch.from_numpy(self.sig == 6).to(dev)[t]
            sd = torch.where(fast, fmin + (fmax - fmin) * u_s, sd); has_s |= fast
        if self.aug_mix:                                          # T or S, half each (S only where a range exists)
            pick_s = (torch.rand(B, device=dev, generator=generator, dtype=torch.float64) < 0.5) & has_s
            k = torch.where(aug & ~pick_s, kd, k)
            sc = torch.where(aug & pick_s, sd, sc)
        else:
            sc = torch.where(aug & has_s, sd, sc)
            if self.tdil != (1.0, 1.0):
                k = torch.where(aug, kd, k)
    else:
        k = torch.where(aug, kd, k)
    lengths = self.n_windows[t] * WINDOW_SIZE               # usable frames (whole windows)
    # a trajectory too short for K windows at rate k contributes as many
    # dilated windows as it has (valid-length masking), keeping its k
    L = (lengths.double() / (k * WINDOW_SIZE)).floor().long().clamp(max=seg_len)
    if self.tdil_short == "cancel":
        k = torch.where(L < seg_len, torch.ones_like(k), k)
        L = (lengths.double() / (k * WINDOW_SIZE)).floor().long().clamp(max=seg_len)
    k = torch.where(L < 1, torch.ones_like(k), k)             # (cannot happen for N >= 200/k)
    L = (lengths.double() / (k * WINDOW_SIZE)).floor().long().clamp(min=1, max=seg_len)
    span = (lengths - (L.double() * k * WINDOW_SIZE).ceil().long() - 1).clamp(min=0)
    u = torch.rand(B, device=dev, generator=generator)
    base = self.lo[t] + torch.minimum((u.double() * (span + 1).double()).long(), span)
    if self.strat_prob > 0:
        drone = self.platform[t] == PLATFORM_TO_ID["drone"]
        pick = drone & (torch.rand(B, device=dev, generator=generator) < self.strat_prob)
        n_pick = int(pick.sum())
        if n_pick:
            tr, fr = self._strat_pick(n_pick, generator)
            tr = torch.from_numpy(tr).to(dev); fr = torch.from_numpy(fr).to(dev)
            idx = pick.nonzero(as_tuple=True)[0]
            t = t.clone(); t[idx] = tr
            lengths = self.n_windows[t] * WINDOW_SIZE
            L = (lengths.double() / (k * WINDOW_SIZE)).floor().long().clamp(min=1, max=seg_len)
            span = (lengths - (L.double() * k * WINDOW_SIZE).ceil().long() - 1).clamp(min=0)
            # place the segment so the chosen window lands inside it, at a random position
            off = (torch.rand(n_pick, device=dev, generator=generator).double() * (L[idx].double() - 1) * k[idx] * WINDOW_SIZE).long()
            b2 = (fr - off - self.lo[t[idx]]).clamp(min=0)
            base = base.clone(); base[idx] = self.lo[t[idx]] + torch.minimum(b2, span[idx])
    step = torch.arange(seg_len, device=dev)[None, :].clamp(max=(L[:, None] - 1))   # (B, K)
    frames = torch.arange(WINDOW_SIZE, device=dev)
    idx_f = base[:, None, None].double() + k[:, None, None] * (step[:, :, None] * WINDOW_SIZE + frames[None, None, :]).double()
    imu = _lerp_gather(self.imu, idx_f)                                # (B, K, T, 6)
    if self.tdil_aa and bool((k > 1.02).any()):
        # continuous 4x-oversampled grid over the whole segment plus a 31-sample
        # halo on each side, taken from the SAME recording (indices clamped to its
        # own [lo, hi], i.e. replicate padding at a true recording boundary), then
        # a 'valid' convolution: constants are preserved exactly, there is no
        # zero-padding edge, and a valid sample never depends on K or on the
        # repeated windows of a short trajectory (review 2026-09-15 22:35)
        Bn, Kn, Tn = idx_f.shape
        halo = self.aa_kernel.shape[-1] // 2
        j = torch.arange(-halo, Kn * Tn * 4 + halo, device=dev, dtype=torch.float64) / 4.0
        grid = base[:, None].double() + k[:, None] * j[None, :]              # (B, 4KT + 2 halo)
        lo_t = self.lo[t].double()[:, None]; hi_t = lo_t + (self.n_windows[t] * WINDOW_SIZE).double()[:, None] - 1
        grid = torch.minimum(torch.maximum(grid, lo_t), hi_t)
        raw = _lerp_gather(self.imu, grid[:, None, :])[:, 0]              # (B, 4KT + 2 halo, 6)
        sig_ = raw.permute(0, 2, 1).reshape(Bn * 6, 1, -1)
        lp = torch.nn.functional.conv1d(sig_, self.aa_kernel)[:, :, ::4]  # valid conv -> 4KT, then decimate
        aa = lp.reshape(Bn, 6, Kn, Tn).permute(0, 2, 3, 1)
        imu = torch.where((k > 1.02)[:, None, None, None], aa, imu)
    vel = _lerp_gather(self.vel, idx_f)
    up6 = _lerp_gather(self.up, idx_f)
    gu = _lerp_gather(self.gt_up, idx_f)
    gu = gu / gu.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    kk = k.to(imu.dtype)[:, None, None, None]
    ss = sc.to(imu.dtype)[:, None, None, None]
    a = imu[..., :3]; w = imu[..., 3:6]
    a = (kk * kk * ss) * (a - 9.81 * gu) + 9.81 * gu           # time k^2, translation s
    imu = torch.cat([a, kk * w], dim=-1)                        # gyro: time only
    vel = kk * ss * vel
    if getattr(self, "drag_pose", False):
        # road 3: an S-scaled straight flight needs the extra pitch that balances the larger drag.
        # theta(v) = atan(|v_h| / (a g)); dtheta = theta(s|v_h|) - theta(|v_h|); axis = up x h_hat;
        # IMU, up channels and labels are rotated by R_d^T (body frame), gyro gets + d(dtheta)/dt axis.
        from unified.attnet import so3_exp
        a_rec = self.drag_a[t]                                              # (B,)
        act = (sc > 1.0) & torch.isfinite(a_rec.double())
        if bool(act.any()):
            aa = torch.nan_to_num(a_rec.to(imu.dtype), nan=1.0).clamp(min=0.5)[:, None, None]   # NaN slope rows are masked by `act` (NaN * 0 = NaN otherwise)
            v0 = vel / ss.clamp(min=1e-6)                                    # (B, K, T, 3) pre-S velocity (already x k)
            vh = v0 - (v0 * gu).sum(-1, keepdim=True) * gu; spd = vh.norm(dim=-1)   # horizontal speed
            th0 = torch.atan(spd / (aa * 9.81)); th1 = torch.atan(spd * ss[..., 0] / (aa * 9.81))
            dth = (th1 - th0) * act.to(imu.dtype)[:, None, None]
            hhat = vh / spd.clamp(min=1e-3)[..., None]
            axis = torch.cross(gu, hhat, dim=-1); axis = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            rv = dth[..., None] * axis; Rd = so3_exp(rv.float())              # (B, K, T, 3, 3)
            RdT = Rd.transpose(-1, -2)
            rot = lambda v: (RdT @ v.float()[..., None])[..., 0].to(imu.dtype)
            om_d = torch.gradient(dth, dim=-1)[0] * 200.0                    # (B, K, T): the augmented sample plays at 200 Hz
            assert not getattr(self, "dr", False), "--drag_pose is not implemented for --grav slowdr (v_DR / R columns would need rotating)"
            imu = torch.cat([rot(imu[..., :3]), rot(imu[..., 3:6]) + (om_d[..., None] * axis).to(imu.dtype)], dim=-1)
            vel = rot(vel); gu = rot(gu)
            from unified.updir import up_vector_groups
            cols = []; i = 0; W = up6.shape[-1]
            while i < W:
                if i in set(up_vector_groups(W)) and i < 13: cols.append(rot(up6[..., i:i + 3])); i += 3
                else: cols.append(up6[..., i:i + 1]); i += 1
            up6 = torch.cat(cols, dim=-1)
    if getattr(self, "dr", False):                              # v_DR scales like the labels (T: x k, S: x s)
        # t_since in compressed time (review 2.1); v_F0 (23:26) is velocity-like and scales like v_DR
        up6 = torch.cat([up6[..., :13], up6[..., 13:16] * kk * ss, up6[..., 16:17] / kk, up6[..., 17:23], up6[..., 23:26] * kk * ss, up6[..., 26:]], dim=-1)
    up6 = self._ins_dropout(up6, generator)
    out = {"imu": imu, "up": up6, "target": vel.mean(dim=2), "platform": self.platform[t],
           "vel_frames": vel, "lengths": L, "tstats": self.traj_stats[t], "traj": t,
           "R_ext": self.R_ext[t], "sig": torch.from_numpy(self.sig).to(dev)[t],
           "k": k.to(imu.dtype), "s": sc.to(imu.dtype)}
    if self.with_pose:
        out["R"] = None; out["pos"] = None                          # not supported under dilation
    return out


DenseWindows._sample_dilated = _sample_dilated


def build(device, platform_balanced: bool = True, with_pose: bool = False,
          min_windows: int = 0) -> DenseWindows:
    return DenseWindows(load_index()["train"], device, platform_balanced, with_pose, min_windows)

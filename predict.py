#!/usr/bin/env python3
"""Frozen-weights inference for the TartanIMU Challenge (IROS 2026).

    python predict.py --data /path/to/tartan-imu-challenge-iros2026 --out submission.csv

Reads index/test_windows.csv and test/*.npz, runs the single unified model with
one shared set of weights (weights/tartanimu_a3v20_s42.pt) on every test window,
and writes window_id,vx,vy,vz. No internet, no test ground truth, no platform
label. Per recording, two frozen sub-modules stored in the same checkpoint
pre-process the raw IMU (CalNet: accelerometer scale / bias / time constant and
gyro-axis map; AttNet: rest-anchored gyro attitude) and give the learned-INS
input channels (bounded dead-reckoned velocity, time since the rest anchor,
rotation to the anchor frame) next to a complementary-filter gravity estimate;
the main network then predicts each 200-frame window's body velocity through a
direct head and a learned-gain recursion over the window increments. Inference
is deterministic; no test-time adaptation, augmentation or ensembling.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from unified.data import build_windows                        # noqa: E402
from unified.train_v2 import Wrapped, predict                  # noqa: E402
from unified.updir import N_CHANNELS, build_features, up_windows_for  # noqa: E402


def load_split(root: Path, split: str) -> pd.DataFrame:
    """Only this split's window index and its .npz files are read (no targets,
    no platform labels, no other split): index/<split>_windows.csv gives
    window_id, traj_id, win_idx; <split>/**/<traj_id>.npz holds the IMU."""
    frame = pd.read_csv(root / "index" / f"{split}_windows.csv").dropna(axis=1, how="all")
    frame = frame[["window_id", "traj_id", "win_idx"]].reset_index(drop=True)
    frame["traj_id"] = frame["traj_id"].astype(str).str.replace(".npz", "", regex=False)
    paths = {p.stem: str(p) for p in (root / split).rglob("*.npz")}
    frame["file_path"] = frame["traj_id"].map(paths)
    missing = frame["file_path"].isna().sum()
    if missing:
        raise SystemExit(f"{split}: {missing} windows have no .npz under {root / split}")
    return frame


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="competition data root (index/, test/)")
    ap.add_argument("--weights", default=str(HERE / "weights/tartanimu_a3v20_s42.pt"))
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    t0 = time.time()
    dev = torch.device(a.device)

    c = torch.load(a.weights, map_location=dev, weights_only=False)
    ar = c["args"]
    model = Wrapped(ar["model"], False, 0, False, ar["mixer_hidden"], ar["width"], ar["dropout"],
                    ar["trunk"], ar["fine"], ar["moe"], ar["n_experts"], ar["decompose"],
                    N_CHANNELS[ar["grav"]], False, ar.get("continuous", False), ar.get("traj_film", False),
                    ar.get("freq", False), ar.get("norm", "bn"), ar.get("preint", False),
                    ar.get("wide", False), ar.get("wide_to_gru", False), ar.get("wide_span", 5),
                    ar.get("at", "none"), ar.get("lr0", "none"), ar.get("head", "linear"), ar.get("drag_head", False),
                    ar.get("fuse", False), ar.get("ekf", False)).to(dev)
    getattr(model.net, "adapt_tstat", lambda *a: None)(c["model"])
    model.load_state_dict(c["model"], strict=True)
    model.eval()
    mean = torch.as_tensor(c["imu_mean"], device=dev); std = torch.as_tensor(c["imu_std"], device=dev)
    grav = ar["grav"]; K = ar["seg_len"]
    from unified import updir as _updir
    _updir.DR_OPTS.update(horizon=ar.get("dr_horizon", 0.0), squash=ar.get("dr_squash", 0.0))   # path-2 channel options travel with the checkpoint

    def norm(x, up=None):
        f = build_features(x, up, grav)
        return ((f - mean) / std).transpose(-1, -2).contiguous(), None

    frame = load_split(Path(a.data), a.split)
    print(f"{a.split}: {len(frame)} windows, {frame.traj_id.nunique()} trajectories; "
          f"model epoch {c['epoch']} ({sum(p.numel() for p in model.parameters())} params) on {dev}", flush=True)
    X = torch.from_numpy(np.asarray(build_windows(frame))).to(dev)
    att_m = cal_m = None; att_key = ""
    if "attnet" in c or "calnet" in c:                       # learned-INS B/C: sub-modules stored in the same checkpoint
        import hashlib
        if "attnet" in c:
            from unified.attnet import AttNet
            att_m = AttNet(**c["attnet"]["config"]).to(dev); att_m.load_state_dict(c["attnet"]["state_dict"]); att_m.eval()
        if "calnet" in c:
            from unified.calnet import CalNet
            cal_m = CalNet(**c["calnet"]["config"]).to(dev); cal_m.load_state_dict(c["calnet"]["state_dict"]); cal_m.eval()
        att_key = hashlib.md5(open(a.weights, "rb").read()).hexdigest()[:8]     # content hash of the checkpoint
        print(f"learned-INS sub-modules: attnet={att_m is not None} calnet={cal_m is not None}", flush=True)
    U = torch.from_numpy(np.asarray(up_windows_for(frame, grav, att_m, cal_m, att_key, device=str(dev)))).to(dev) if grav != "none" else None
    print(f"windows + gravity estimate ready ({time.time() - t0:.0f}s)", flush=True)
    with torch.inference_mode():
        p = predict(model, X, U, frame, ar["model"], K, norm)
    sub = pd.DataFrame({"window_id": frame["window_id"].to_numpy(), "vx": p[:, 0], "vy": p[:, 1], "vz": p[:, 2]})
    sub.to_csv(a.out, index=False)
    peak = f", peak GPU alloc {torch.cuda.max_memory_allocated() / 2**30:.2f} GB" if dev.type == "cuda" else ""
    print(f"wrote {a.out}: {len(sub)} rows, finite={np.isfinite(p).all()}  ({time.time() - t0:.0f}s total{peak})")


if __name__ == "__main__":
    main()

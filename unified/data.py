#!/usr/bin/env python3
"""Load the challenge splits into memory as (N, 200, 6) float32 windows.

Ported from sample_code/ts-tcmpio-unified-temporal-resnet.ipynb (cells 1-2, 11),
with the Kaggle-mounted paths replaced by the local data root.  The whole train
split is 82k windows = 0.39 GB, so everything is held in RAM (and later on the
GPU) rather than streamed.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW_SIZE = 200
PLATFORM_TO_ID = {"car": 0, "dog": 1, "drone": 2, "human": 3}
PLATFORMS = tuple(PLATFORM_TO_ID)
# Kaggle mounts the competition at /kaggle/input/<slug>; set TARTANIMU_DATA there
DATA_ROOT = Path(os.environ.get("TARTANIMU_DATA", "data/tartan-imu-challenge-iros2026"))
CACHE_SCHEMA = 2        # 2: DenseWindows carries the gravity-direction stream


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).dropna(axis=1, how="all")


def load_index(data_root: Path = DATA_ROOT) -> dict:
    idx = data_root / "index"
    train = _read(idx / "train_windows.csv").merge(_read(idx / "train_targets.csv"), on="window_id")
    val = _read(idx / "val_windows.csv").merge(_read(idx / "val_targets.csv"), on="window_id")
    test = _read(idx / "test_windows.csv")
    train["platform_id"] = train["platform"].map(PLATFORM_TO_ID)
    val["platform_id"] = val["platform"].map(PLATFORM_TO_ID)

    paths = {split: {p.stem: str(p) for p in (data_root / split).rglob("*.npz")}
             for split in ("train", "val", "test")}
    out = {}
    for split, frame in (("train", train), ("val", val), ("test", test)):
        frame = frame.reset_index(drop=True)
        frame["traj_id"] = frame["traj_id"].astype(str).str.replace(".npz", "", regex=False)
        frame["file_path"] = frame["traj_id"].map(paths[split])
        if frame["file_path"].isna().any():
            raise SystemExit(f"{split}: {frame['file_path'].isna().sum()} windows have no .npz")
        out[split] = frame
    return out


def cache_path(frame: pd.DataFrame, kind: str) -> Path | None:
    """Where to memoise a decoded split, or None when caching is off.

    Keyed by the frame's own contents, so no call site has to name its split and
    a mismatched cache cannot be picked up. Set TARTANIMU_CACHE to a writable,
    node-local directory: on Kaggle the competition mounts over the network with
    no page cache, and every run re-decoding 395 .npz files dominates the job.
    """
    root = os.environ.get("TARTANIMU_CACHE")
    if not root:
        return None
    Path(root).mkdir(parents=True, exist_ok=True)
    # CACHE_SCHEMA must change whenever the cached arrays gain or lose a field,
    # or a stale cache is silently reused with the wrong contents
    key = (f"{kind}_v{CACHE_SCHEMA}_{len(frame)}"
           f"_{frame['window_id'].iloc[0]}_{frame['window_id'].iloc[-1]}")
    return Path(root) / f"{key}.npy"


def build_windows(frame: pd.DataFrame) -> np.ndarray:
    """(len(frame), 200, 6) float32, row i matching frame.iloc[i]."""
    cache = cache_path(frame, "win")
    if cache is not None and cache.exists():
        return np.load(cache, mmap_mode="r")
    X = np.empty((len(frame), WINDOW_SIZE, 6), dtype=np.float32)
    for file_path, rows in frame.groupby("file_path", sort=False):
        with np.load(file_path) as d:
            imu = d["imu"]
        starts = rows["win_idx"].to_numpy() * WINDOW_SIZE
        X[rows.index.to_numpy()] = np.stack([imu[s:s + WINDOW_SIZE] for s in starts])
    if cache is not None:
        np.save(cache, X)
    return X


def sample_weights(train: pd.DataFrame) -> np.ndarray:
    """Notebook's WeightedRandomSampler weights.

    1 / (windows in this trajectory * trajectories in this platform) makes every
    platform equally likely and every trajectory within a platform equally
    likely -- which is exactly how the competition macro-averages.
    """
    traj_windows = train.groupby("traj_id")["window_id"].transform("size")
    plat_trajs = train.groupby("platform")["traj_id"].transform("nunique")
    return (1.0 / (traj_windows * plat_trajs)).to_numpy(np.float64)

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class WindowedDataBundle:
    x: np.ndarray
    y: np.ndarray
    obs_mask: np.ndarray
    target_mask: np.ndarray
    dt: np.ndarray


class TelemetryWindowDataset(Dataset):
    def __init__(self, bundle: WindowedDataBundle):
        self.bundle = bundle

    def __len__(self):
        return len(self.bundle.x)

    def __getitem__(self, idx):
        return {
            'x': torch.tensor(self.bundle.x[idx], dtype=torch.float32),
            'y': torch.tensor(self.bundle.y[idx], dtype=torch.float32),
            'obs_mask': torch.tensor(self.bundle.obs_mask[idx], dtype=torch.float32),
            'target_mask': torch.tensor(self.bundle.target_mask[idx], dtype=torch.float32),
            'dt': torch.tensor(self.bundle.dt[idx], dtype=torch.float32),
        }


def create_irregular_windows(
    timestamps: np.ndarray,
    values: np.ndarray,
    window_size: int,
    stride: int,
    mask_rate: float,
    min_observed_ratio: float,
    seed: int = 42,
) -> WindowedDataBundle:
    rng = np.random.default_rng(seed)
    x_list, y_list, obs_mask_list, target_mask_list, dt_list = [], [], [], [], []
    T, N = values.shape
    if T < window_size:
        raise ValueError('Time series shorter than window_size.')

    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        y = values[start:end].copy()
        window_times = timestamps[start:end]
        dt = np.diff(window_times, prepend=window_times[0])
        dt[0] = np.median(np.diff(timestamps)) if len(timestamps) > 1 else 1.0

        valid_mask = ~np.isnan(y)
        random_mask = rng.random(size=y.shape) > mask_rate
        obs_mask = (valid_mask & random_mask).astype(np.float32)
        if obs_mask.mean() < min_observed_ratio:
            continue

        x = y.copy()
        x[obs_mask == 0] = 0.0
        target_mask = valid_mask.astype(np.float32)

        y = np.nan_to_num(y, nan=0.0)
        x_list.append(x)
        y_list.append(y)
        obs_mask_list.append(obs_mask)
        target_mask_list.append(target_mask)
        dt_list.append(dt[:, None].repeat(N, axis=1))

    return WindowedDataBundle(
        x=np.asarray(x_list, dtype=np.float32),
        y=np.asarray(y_list, dtype=np.float32),
        obs_mask=np.asarray(obs_mask_list, dtype=np.float32),
        target_mask=np.asarray(target_mask_list, dtype=np.float32),
        dt=np.asarray(dt_list, dtype=np.float32),
    )


def split_bundle(bundle: WindowedDataBundle, val_ratio: float, test_ratio: float):
    n = len(bundle.x)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    n_train = n - n_val - n_test
    idx1 = n_train
    idx2 = n_train + n_val

    def slice_bundle(s, e):
        return WindowedDataBundle(
            x=bundle.x[s:e],
            y=bundle.y[s:e],
            obs_mask=bundle.obs_mask[s:e],
            target_mask=bundle.target_mask[s:e],
            dt=bundle.dt[s:e],
        )

    return slice_bundle(0, idx1), slice_bundle(idx1, idx2), slice_bundle(idx2, n)

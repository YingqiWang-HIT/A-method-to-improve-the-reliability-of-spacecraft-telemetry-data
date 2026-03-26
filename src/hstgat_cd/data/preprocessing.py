from __future__ import annotations
import numpy as np
import pandas as pd


class ZScoreNormalizer:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, x: np.ndarray):
        self.mean_ = np.nanmean(x, axis=0)
        self.std_ = np.nanstd(x, axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        return self

    def transform(self, x: np.ndarray):
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x: np.ndarray):
        return x * self.std_ + self.mean_


def load_csv_timeseries(path: str, timestamp_col: str = 'timestamp'):
    df = pd.read_csv(path)
    if timestamp_col not in df.columns:
        raise ValueError(f'Missing timestamp column: {timestamp_col}')
    timestamps = df[timestamp_col].to_numpy(dtype=float)
    feature_cols = [c for c in df.columns if c != timestamp_col]
    values = df[feature_cols].to_numpy(dtype=float)
    return timestamps, values, feature_cols

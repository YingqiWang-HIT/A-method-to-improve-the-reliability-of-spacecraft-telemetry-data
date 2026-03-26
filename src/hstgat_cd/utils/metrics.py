from __future__ import annotations
import numpy as np


def masked_mae(pred, target, mask):
    denom = np.maximum(mask.sum(), 1)
    return np.abs((pred - target) * mask).sum() / denom


def masked_rmse(pred, target, mask):
    denom = np.maximum(mask.sum(), 1)
    return np.sqrt((((pred - target) ** 2) * mask).sum() / denom)


def masked_mape(pred, target, mask, eps=1e-6):
    denom = np.maximum(mask.sum(), 1)
    return (np.abs((pred - target) / (np.abs(target) + eps)) * mask).sum() / denom

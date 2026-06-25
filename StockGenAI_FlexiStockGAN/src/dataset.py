from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset


def make_windows(df, feature_cols, target_col="Close", window_size=30, horizon=5):
    X = df[feature_cols].values.astype("float32")
    y = df[target_col].values.astype("float32")
    xs, ys = [], []
    for i in range(len(df) - window_size - horizon + 1):
        xs.append(X[i:i + window_size])
        ys.append(y[i + window_size:i + window_size + horizon])
    return np.asarray(xs, dtype="float32"), np.asarray(ys, dtype="float32")


class StockWindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

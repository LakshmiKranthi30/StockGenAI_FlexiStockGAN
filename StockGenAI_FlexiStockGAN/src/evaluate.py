from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import entropy


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def directional_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim == 2:
        true_delta = y_true[:, -1] - y_true[:, 0]
        pred_delta = y_pred[:, -1] - y_pred[:, 0]
    else:
        true_delta = np.diff(y_true)
        pred_delta = np.diff(y_pred)
    return float(np.mean(np.sign(true_delta) == np.sign(pred_delta)) * 100.0)


def kl_divergence_returns(y_true, y_pred, bins=50):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    rt = np.diff(np.log(np.maximum(y_true, 1e-8)))
    rp = np.diff(np.log(np.maximum(y_pred, 1e-8)))
    lo, hi = np.nanpercentile(np.concatenate([rt, rp]), [1, 99])
    p, edges = np.histogram(rt, bins=bins, range=(lo, hi), density=True)
    q, _ = np.histogram(rp, bins=edges, density=True)
    p = p + 1e-9
    q = q + 1e-9
    p = p / p.sum()
    q = q / q.sum()
    return float(entropy(p, q))


def compute_metrics(y_true, y_pred):
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "DA_percent": directional_accuracy(y_true, y_pred),
        "KL_Divergence": kl_divergence_returns(y_true, y_pred),
    }


def save_metrics(metrics: dict, path):
    pd.DataFrame([metrics]).to_csv(path, index=False)

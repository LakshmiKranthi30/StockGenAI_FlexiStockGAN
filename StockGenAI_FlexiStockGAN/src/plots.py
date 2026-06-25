from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_actual_vs_pred(y_true, y_pred, path, title="Actual vs Predicted Close Price"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(np.asarray(y_true)[:, -1], label="Actual")
    plt.plot(np.asarray(y_pred)[:, -1], label="Predicted")
    plt.title(title)
    plt.xlabel("Test Window")
    plt.ylabel("Close Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_error_distribution(y_true, y_pred, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    err = (np.asarray(y_pred) - np.asarray(y_true)).reshape(-1)
    plt.figure(figsize=(8, 5))
    plt.hist(err, bins=40)
    plt.title("Forecast Error Distribution")
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_return_distribution(y_true, y_pred, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    yt = np.asarray(y_true).reshape(-1)
    yp = np.asarray(y_pred).reshape(-1)
    rt = np.diff(np.log(np.maximum(yt, 1e-8)))
    rp = np.diff(np.log(np.maximum(yp, 1e-8)))
    plt.figure(figsize=(8, 5))
    plt.hist(rt, bins=40, alpha=0.5, label="Real Returns", density=True)
    plt.hist(rp, bins=40, alpha=0.5, label="Generated Returns", density=True)
    plt.title("Real vs Generated Return Distribution")
    plt.xlabel("Log Return")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_training_history(history, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(history)
    if df.empty:
        return
    plt.figure(figsize=(10, 5))
    for col in ["g_loss", "d_loss", "val_rmse"]:
        if col in df.columns:
            plt.plot(df["epoch"], df[col], label=col)
    plt.title("FlexiStockGAN Training Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_model_comparison(metrics_df, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if metrics_df.empty:
        return
    plt.figure(figsize=(9, 5))
    plt.bar(metrics_df["Model"], metrics_df["RMSE"])
    plt.title("Model Comparison by RMSE")
    plt.xlabel("Model")
    plt.ylabel("RMSE")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

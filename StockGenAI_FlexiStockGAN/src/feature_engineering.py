from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
from pathlib import Path


def ema(s, span):
    return s.ewm(span=span, adjust=False).mean()


def rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))


def macd(close, fast=12, slow=26, signal=9):
    m = ema(close, fast) - ema(close, slow)
    sig = ema(m, signal)
    return m, sig, m - sig


def atr(df, period=14):
    h_l = df["High"] - df["Low"]
    h_pc = (df["High"] - df["Close"].shift()).abs()
    l_pc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def obv(close, volume):
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Return"] = out["Close"].pct_change()
    out["LogReturn"] = np.log(out["Close"] / out["Close"].shift(1))
    for w in [5, 10, 20, 30]:
        out[f"EMA_{w}"] = ema(out["Close"], w)
        out[f"SMA_{w}"] = out["Close"].rolling(w).mean()
    out["RSI_14"] = rsi(out["Close"], 14)
    out["MACD"], out["MACD_signal"], out["MACD_hist"] = macd(out["Close"])
    mid = out["Close"].rolling(20).mean()
    std = out["Close"].rolling(20).std()
    out["BB_mid"] = mid
    out["BB_upper"] = mid + 2 * std
    out["BB_lower"] = mid - 2 * std
    out["BB_width"] = (out["BB_upper"] - out["BB_lower"]) / (mid + 1e-12)
    out["ATR_14"] = atr(out, 14)
    out["OBV"] = obv(out["Close"], out["Volume"])
    out["Volatility_10"] = out["Return"].rolling(10).std()
    out["Volatility_20"] = out["Return"].rolling(20).std()
    out["HighLowSpread"] = (out["High"] - out["Low"]) / (out["Close"] + 1e-12)
    out["OpenCloseSpread"] = (out["Open"] - out["Close"]) / (out["Close"] + 1e-12)
    return out


def chronological_split(df, train_ratio=0.70, val_ratio=0.15):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()


def winsorize_fit(train_df, numeric_cols, lower=0.01, upper=0.99):
    return {c: (train_df[c].quantile(lower), train_df[c].quantile(upper)) for c in numeric_cols}


def winsorize_apply(df, limits):
    out = df.copy()
    for c, (lo, hi) in limits.items():
        out[c] = out[c].clip(lo, hi)
    return out


def fit_transform_scaler(train_df, val_df, test_df, feature_cols, out_dir=None):
    scaler = MinMaxScaler()
    train_scaled = train_df.copy(); val_scaled = val_df.copy(); test_scaled = test_df.copy()
    train_scaled[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_scaled[feature_cols] = scaler.transform(val_df[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test_df[feature_cols])
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, Path(out_dir) / "feature_scaler.joblib")
    return train_scaled, val_scaled, test_scaled, scaler


def build_fused_features(stock_df, index_df=None, sentiment_df=None, ablation="none"):
    df = compute_technical_indicators(stock_df)
    if index_df is not None and ablation != "no_indices":
        df = df.join(index_df, how="left")
    if sentiment_df is not None and ablation != "no_sentiment":
        df = df.join(sentiment_df[["Sentiment"]], how="left")
    if "Sentiment" not in df.columns:
        df["Sentiment"] = 0.0
    df["Sentiment"] = df["Sentiment"].ffill().fillna(0.0)
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()
    if ablation == "no_technical":
        keep = ["Open", "High", "Low", "Close", "Volume"]
        keep += [c for c in ["SP500", "NASDAQ", "DJI", "VIX", "Sentiment"] if c in df.columns]
        df = df[keep]
    return df

from __future__ import annotations
import numpy as np
import pandas as pd


def generate_synthetic_stock(start="2010-01-01", end="2022-12-31", seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n = len(dates)
    drift = 0.00035
    vol = 0.015
    returns = drift + vol * rng.standard_normal(n)
    close = 100 * np.exp(np.cumsum(returns))
    open_ = close * (1 + rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.007, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.007, n)))
    volume = rng.integers(1_000_000, 8_000_000, n)
    df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=dates)
    df.index.name = "Date"
    return df


def load_stock_data(ticker: str, start: str, end: str, demo: bool = False) -> pd.DataFrame:
    if demo:
        return generate_synthetic_stock(start, end)
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        if df.empty:
            raise ValueError("No rows returned by yfinance")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        df = df[keep].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "Date"
        return df.dropna()
    except Exception as exc:
        print(f"[WARN] yfinance failed ({exc}). Falling back to synthetic demo data.")
        return generate_synthetic_stock(start, end)


def load_index_data(start: str, end: str, demo: bool = False) -> pd.DataFrame:
    dates = pd.bdate_range(start, end)
    if demo:
        rng = np.random.default_rng(7)
        return pd.DataFrame({
            "SP500": 3000 * np.exp(np.cumsum(0.0002 + 0.010 * rng.standard_normal(len(dates)))),
            "NASDAQ": 9000 * np.exp(np.cumsum(0.00025 + 0.012 * rng.standard_normal(len(dates)))),
            "DJI": 25000 * np.exp(np.cumsum(0.00018 + 0.009 * rng.standard_normal(len(dates)))),
            "VIX": np.clip(20 + 6 * rng.standard_normal(len(dates)), 8, 80),
        }, index=dates)
    try:
        import yfinance as yf
        symbols = {"^GSPC": "SP500", "^IXIC": "NASDAQ", "^DJI": "DJI", "^VIX": "VIX"}
        frames = []
        for sym, name in symbols.items():
            s = yf.download(sym, start=start, end=end, progress=False, auto_adjust=False)
            if not s.empty:
                if isinstance(s.columns, pd.MultiIndex):
                    s.columns = [c[0] for c in s.columns]
                frames.append(s[["Close"]].rename(columns={"Close": name}))
        if not frames:
            raise ValueError("No index data downloaded")
        df = pd.concat(frames, axis=1)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception as exc:
        print(f"[WARN] index download failed ({exc}). Using synthetic indices.")
        return load_index_data(start, end, demo=True)


def load_news_csv(path: str) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=["date", "headline", "summary", "source"])
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError("news_csv must contain a 'date' column")
    if "headline" not in df.columns:
        df["headline"] = ""
    if "summary" not in df.columns:
        df["summary"] = ""
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df

from __future__ import annotations
import numpy as np
import pandas as pd


def synthetic_sentiment(index, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    vals = 0.15 * np.sin(np.linspace(0, 20, len(index))) + 0.15 * rng.standard_normal(len(index))
    vals = np.clip(vals, -1, 1)
    return pd.DataFrame({"Sentiment": vals}, index=index)


def vader_score(text: str) -> float:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        return analyzer.polarity_scores(text or "")["compound"]
    except Exception:
        # deterministic light fallback lexicon
        pos = {"gain", "rise", "beat", "growth", "strong", "positive", "profit", "surge", "upgrade"}
        neg = {"fall", "drop", "miss", "loss", "weak", "negative", "decline", "lawsuit", "downgrade"}
        words = str(text).lower().split()
        score = sum(w in pos for w in words) - sum(w in neg for w in words)
        return float(np.tanh(score / 3))


def aggregate_daily_sentiment(news_df: pd.DataFrame, trading_index, mode="vader") -> pd.DataFrame:
    if news_df is None or news_df.empty:
        return pd.DataFrame({"Sentiment": 0.0}, index=trading_index)
    df = news_df.copy()
    df["text"] = df.get("headline", "").fillna("").astype(str) + " " + df.get("summary", "").fillna("").astype(str)
    # FinBERT can be added externally; VADER is default because it is light and reproducible.
    df["score"] = df["text"].apply(vader_score)
    df["date_only"] = pd.to_datetime(df["date"]).dt.normalize()
    daily = df.groupby("date_only")["score"].mean().rename("Sentiment").to_frame()
    daily.index = pd.to_datetime(daily.index)
    aligned = daily.reindex(pd.to_datetime(trading_index).normalize()).ffill().fillna(0.0)
    aligned.index = trading_index
    return aligned

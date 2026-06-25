from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from src.config import Config
from src.utils import set_seed, ensure_dirs, safe_name, inverse_close_from_scaled
from src.data_loader import load_stock_data, load_index_data, load_news_csv
from src.sentiment_extraction import aggregate_daily_sentiment, synthetic_sentiment
from src.feature_engineering import build_fused_features, chronological_split, winsorize_fit, winsorize_apply, fit_transform_scaler
from src.dataset import make_windows
from src.train_gan import train_flexistockgan, evaluate_generator
from src.evaluate import compute_metrics
from src.baselines import train_svr_baseline, train_rnn_baseline
from src.plots import plot_actual_vs_pred, plot_error_distribution, plot_return_distribution, plot_training_history, plot_model_comparison


def parse_args():
    p = argparse.ArgumentParser(description="StockGenAI / FlexiStockGAN implementation")
    p.add_argument("--ticker", default="AAPL")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default="2022-12-31")
    p.add_argument("--window", type=int, default=30)
    p.add_argument("--horizon", type=int, default=5, choices=[1,5,10,20])
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--demo", action="store_true", help="Use synthetic data; runs offline without downloads.")
    p.add_argument("--news_csv", default="", help="Optional CSV with columns: date, headline, summary")
    p.add_argument("--use_sentiment", type=str, default="true")
    p.add_argument("--use_indices", type=str, default="true")
    p.add_argument("--ablation", default="none", choices=["none", "no_sentiment", "no_technical", "no_indices", "generator_only"])
    p.add_argument("--run_baselines", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--critic_steps", type=int, default=1, help="Use 5 for final WGAN-GP experiments; 1 is faster for demo.")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def str2bool(x):
    return str(x).lower() in ["1", "true", "yes", "y"]


def main():
    args = parse_args()
    cfg = Config(
        ticker=args.ticker, start=args.start, end=args.end, window_size=args.window, horizon=args.horizon,
        epochs=args.epochs, batch_size=args.batch_size, demo=args.demo, news_csv=args.news_csv,
        use_sentiment=str2bool(args.use_sentiment), use_indices=str2bool(args.use_indices),
        ablation=args.ablation, seed=args.seed, device=args.device
    )
    cfg.critic_steps = args.critic_steps
    set_seed(cfg.seed)
    out = ensure_dirs(cfg.output_dir)
    print(f"Running StockGenAI for {cfg.ticker}, horizon={cfg.horizon}, device={cfg.effective_device}")

    stock = load_stock_data(cfg.ticker, cfg.start, cfg.end, cfg.demo)
    indices = load_index_data(cfg.start, cfg.end, cfg.demo) if cfg.use_indices else None
    news_df = load_news_csv(cfg.news_csv)
    if cfg.use_sentiment:
        if news_df.empty:
            sentiment = synthetic_sentiment(stock.index, cfg.seed)
        else:
            sentiment = aggregate_daily_sentiment(news_df, stock.index)
    else:
        sentiment = None

    fused = build_fused_features(stock, indices, sentiment, cfg.ablation)
    feature_cols = list(fused.columns)
    target_col = "Close"
    if target_col not in fused.columns:
        raise ValueError("Close column missing after feature construction.")

    train_df, val_df, test_df = chronological_split(fused, cfg.train_ratio, cfg.val_ratio)
    numeric_cols = feature_cols
    limits = winsorize_fit(train_df, numeric_cols)
    train_df = winsorize_apply(train_df, limits)
    val_df = winsorize_apply(val_df, limits)
    test_df = winsorize_apply(test_df, limits)
    close_min, close_max = float(train_df[target_col].min()), float(train_df[target_col].max())
    train_df, val_df, test_df, scaler = fit_transform_scaler(train_df, val_df, test_df, feature_cols, out / "models")

    X_train, y_train = make_windows(train_df, feature_cols, target_col, cfg.window_size, cfg.horizon)
    X_val, y_val = make_windows(val_df, feature_cols, target_col, cfg.window_size, cfg.horizon)
    X_test, y_test = make_windows(test_df, feature_cols, target_col, cfg.window_size, cfg.horizon)
    if min(len(X_train), len(X_val), len(X_test)) <= 0:
        raise ValueError("Not enough data for selected window/horizon. Use a longer date range.")
    input_dim = len(feature_cols)
    print(f"Feature count={input_dim}; train/val/test windows={len(X_train)}/{len(X_val)}/{len(X_test)}")

    G, D, history = train_flexistockgan(X_train, y_train, X_val, y_val, input_dim, cfg.horizon, cfg, out / "models")
    _, pred_scaled = evaluate_generator(G, X_test, y_test, torch.device(cfg.effective_device), cfg.noise_dim)
    y_true = inverse_close_from_scaled(y_test, close_min, close_max)
    y_pred = inverse_close_from_scaled(pred_scaled, close_min, close_max)
    metrics = compute_metrics(y_true, y_pred)
    metrics.update({"Model": "FlexiStockGAN", "Ticker": cfg.ticker, "Horizon": cfg.horizon, "Ablation": cfg.ablation})
    all_metrics = [metrics]

    pred_df = pd.DataFrame(y_pred, columns=[f"pred_t+{i+1}" for i in range(cfg.horizon)])
    for i in range(cfg.horizon):
        pred_df[f"true_t+{i+1}"] = y_true[:, i]
    pred_path = out / "predictions" / f"{safe_name(cfg.ticker)}_h{cfg.horizon}_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    if args.run_baselines:
        print("Training SVR baseline...")
        svr_pred = inverse_close_from_scaled(train_svr_baseline(X_train, y_train, X_test), close_min, close_max)
        m = compute_metrics(y_true, svr_pred); m.update({"Model": "SVR", "Ticker": cfg.ticker, "Horizon": cfg.horizon, "Ablation": cfg.ablation}); all_metrics.append(m)
        print("Training GRU baseline...")
        gru_pred = inverse_close_from_scaled(train_rnn_baseline(X_train, y_train, X_val, y_val, X_test, input_dim, cfg.horizon, "gru", cfg.hidden_dim, min(40,cfg.epochs), cfg.batch_size, device=cfg.effective_device), close_min, close_max)
        m = compute_metrics(y_true, gru_pred); m.update({"Model": "GRU", "Ticker": cfg.ticker, "Horizon": cfg.horizon, "Ablation": cfg.ablation}); all_metrics.append(m)
        print("Training LSTM baseline...")
        lstm_pred = inverse_close_from_scaled(train_rnn_baseline(X_train, y_train, X_val, y_val, X_test, input_dim, cfg.horizon, "lstm", cfg.hidden_dim, min(40,cfg.epochs), cfg.batch_size, device=cfg.effective_device), close_min, close_max)
        m = compute_metrics(y_true, lstm_pred); m.update({"Model": "LSTM", "Ticker": cfg.ticker, "Horizon": cfg.horizon, "Ablation": cfg.ablation}); all_metrics.append(m)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = out / "metrics" / f"{safe_name(cfg.ticker)}_h{cfg.horizon}_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    pd.DataFrame(history).to_csv(out / "metrics" / f"{safe_name(cfg.ticker)}_h{cfg.horizon}_training_history.csv", index=False)

    plot_actual_vs_pred(y_true, y_pred, out / "figures" / f"{safe_name(cfg.ticker)}_h{cfg.horizon}_actual_vs_pred.png")
    plot_error_distribution(y_true, y_pred, out / "figures" / f"{safe_name(cfg.ticker)}_h{cfg.horizon}_error_distribution.png")
    plot_return_distribution(y_true, y_pred, out / "figures" / f"{safe_name(cfg.ticker)}_h{cfg.horizon}_return_distribution.png")
    plot_training_history(history, out / "figures" / f"{safe_name(cfg.ticker)}_h{cfg.horizon}_training_curves.png")
    plot_model_comparison(metrics_df, out / "figures" / f"{safe_name(cfg.ticker)}_h{cfg.horizon}_model_comparison.png")

    print("\nCompleted.")
    print(metrics_df.to_string(index=False))
    print(f"Predictions: {pred_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Figures: {out / 'figures'}")


if __name__ == "__main__":
    main()

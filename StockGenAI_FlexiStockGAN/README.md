# StockGenAI / FlexiStockGAN

Complete runnable implementation of **FlexiStockGAN**, a multi-source adversarial deep learning framework for stock price forecasting.

The project implements:

- Yahoo Finance / synthetic stock data loading
- Market index feature loading
- Technical indicator engineering
- Optional news sentiment fusion from CSV
- Chronological train/validation/test split
- Training-only winsorization and Min-Max scaling to avoid data leakage
- Sliding-window forecasting dataset
- GRU generator + 1D-CNN WGAN-GP critic
- Optional generator-only ablation
- SVR, GRU, and LSTM baselines
- RMSE, MAE, Directional Accuracy, and KL Divergence
- Prediction CSVs, metric CSVs, trained model files, and figures

## 1. Install

```bash
pip install -r requirements.txt
```

## 2. Quick Offline Demo

Runs without internet using synthetic stock and index data:

```bash
python main.py --demo --ticker AAPL --horizon 5 --epochs 5 --critic_steps 1
```

## 3. Real Data Run

Uses `yfinance` to download OHLCV and market index data:

```bash
python main.py --ticker AAPL --start 2010-01-01 --end 2022-12-31 --horizon 5 --epochs 80 --critic_steps 5
```

## 4. Run with Baselines

```bash
python main.py --ticker AAPL --horizon 5 --epochs 80 --critic_steps 5 --run_baselines
```

## 5. Multi-Horizon Runs

```bash
python main.py --ticker AAPL --horizon 1 --epochs 80
python main.py --ticker AAPL --horizon 5 --epochs 80
python main.py --ticker AAPL --horizon 10 --epochs 80
python main.py --ticker AAPL --horizon 20 --epochs 80
```

## 6. Ablation Studies

```bash
python main.py --ticker AAPL --horizon 5 --ablation no_sentiment
python main.py --ticker AAPL --horizon 5 --ablation no_technical
python main.py --ticker AAPL --horizon 5 --ablation no_indices
python main.py --ticker AAPL --horizon 5 --ablation generator_only
```

## 7. News Sentiment CSV Format

Use a CSV with the following columns:

```csv
date,headline,summary,source
2020-01-03,Apple stock rises after strong demand,Analysts report strong iPhone demand,Yahoo
2020-01-04,Market volatility increases,Investors respond to macro uncertainty,News
```

Run:

```bash
python main.py --ticker AAPL --news_csv data/news/news.csv --horizon 5
```

If no news file is provided, the code uses a deterministic synthetic sentiment sequence for reproducible demonstration.

## 8. Outputs

After execution, results are saved in:

```text
outputs/models/       trained PyTorch model and scaler
outputs/predictions/  predicted and true values
outputs/metrics/      metric CSV and training history
outputs/figures/      actual-vs-predicted, error, return-distribution, and loss plots
```

## 9. Important Methodological Decisions

- WGAN-GP uses a critic with linear output, not sigmoid.
- The generator loss combines adversarial loss and MAE supervision for better forecasting accuracy.
- Scaling and winsorization parameters are fitted only on training data.
- News sentiment is aligned by trading date and forward-filled if absent.
- Chronological splitting is used to preserve temporal integrity.

## 10. Suggested SCI-Quality Experimental Protocol

Run five independent seeds and all horizons:

```bash
for seed in 1 2 3 4 5; do
  for h in 1 5 10 20; do
    python main.py --ticker AAPL --horizon $h --epochs 300 --seed $seed --critic_steps 5 --run_baselines
  done
done
```

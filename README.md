# StockGenAI: FlexiStockGAN for Multi-Source Stock Price Forecasting

StockGenAI is a complete Python implementation of **FlexiStockGAN**, a multi-source adversarial deep learning framework for stock price forecasting.

The framework integrates:

* Historical stock prices
* Technical indicators
* Optional market index features
* Optional financial news sentiment

to generate realistic multi-step stock price forecasts using a **GRU-based Generator** and a **1D-CNN WGAN-GP Critic**.

The implementation is designed for **research reproducibility, experimentation, and extension toward standard financial forecasting studies.**

---

# Project Overview

Financial stock price forecasting is a highly nonlinear time-series learning problem influenced by:

* Historical market behavior
* Technical trading indicators
* Market-wide conditions
* Investor sentiment

Traditional forecasting models generally produce deterministic point estimates and often fail to model uncertainty during volatile market conditions.

StockGenAI addresses this limitation through a **GAN-based forecasting framework** where:

* A GRU Generator learns temporal dependencies from fused financial features.
* A CNN Critic evaluates whether generated future price sequences resemble real market behavior.
* WGAN-GP optimization stabilizes adversarial training.
* A supervised forecasting loss improves numerical prediction accuracy.

---

# Key Features

* Yahoo Finance stock data loading using `yfinance`
* Offline demo mode (no Internet required)
* Technical indicator generation
* Optional sentiment feature integration
* Chronological train/validation/test split
* Leakage-safe preprocessing
* Sliding window sequence generation
* GRU Generator
* 1D CNN WGAN-GP Critic
* Multi-step stock forecasting
* Baseline models:

  * SVR
  * LSTM
  * GRU
* Ablation study support
* Evaluation metrics:

  * RMSE
  * MAE
  * Directional Accuracy
  * KL Divergence
* Automatic result saving
* Publication-ready figures

---

# Repository Structure

```text
StockGenAI_FlexiStockGAN/
│
├── main.py
├── requirements.txt
├── README.md
├── how_to_run.txt
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── sentiment_extraction.py
│   ├── dataset.py
│   ├── models.py
│   ├── train_gan.py
│   ├── evaluate.py
│   ├── baselines.py
│   ├── plots.py
│   └── utils.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── news/
│
└── outputs/
    ├── models/
    ├── predictions/
    ├── metrics/
    └── figures/
```

---

# Methodology

The complete forecasting pipeline consists of five stages.

## Stage 1 – Data Collection

Historical OHLCV stock data are collected from Yahoo Finance.

Optional inputs include:

* Market index data
* Financial news sentiment

---

## Stage 2 – Feature Engineering

Technical indicators are computed, including:

* Daily Returns
* Moving Averages
* RSI
* MACD
* Bollinger Bands
* Historical Volatility
* Volume Indicators

Daily sentiment scores are aligned with trading dates.

---

## Stage 3 – Data Fusion

The following features are merged into one feature matrix:

* OHLCV
* Technical Indicators
* Market Index Features
* Sentiment Features

Data preprocessing includes:

* Missing value handling
* Training-only normalization
* Data leakage prevention

---

## Stage 4 – GAN Forecasting

The GRU Generator predicts future Close-price sequences.

The CNN Critic distinguishes:

* Real future price sequences
* Generated future price sequences

Training is performed using:

* Wasserstein GAN
* Gradient Penalty (WGAN-GP)

---

## Stage 5 – Evaluation

Predictions are evaluated using:

* RMSE
* MAE
* Directional Accuracy
* KL Divergence

Figures and CSV files are automatically generated.

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/your-username/StockGenAI_FlexiStockGAN.git

cd StockGenAI_FlexiStockGAN
```

---

## 2. Create Virtual Environment

Windows

```bash
python -m venv venv

venv\Scripts\activate
```

Linux/macOS

```bash
python -m venv venv

source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Quick Demo

```bash
python main.py \
--demo \
--ticker AAPL \
--horizon 5 \
--epochs 5 \
--critic_steps 1
```

This generates synthetic stock-like data for a quick demonstration.

---

# Full Research Experiment

```bash
python main.py \
--ticker AAPL \
--start 2010-01-01 \
--end 2022-12-31 \
--horizon 5 \
--epochs 80 \
--critic_steps 5 \
--run_baselines
```

Example:

```bash
python main.py \
--ticker MSFT \
--start 2010-01-01 \
--end 2022-12-31 \
--horizon 10 \
--epochs 100 \
--critic_steps 5 \
--run_baselines
```

---

# Command Line Arguments

| Argument          | Description           | Default    |
| ----------------- | --------------------- | ---------- |
| `--ticker`        | Stock ticker          | AAPL       |
| `--start`         | Start date            | 2010-01-01 |
| `--end`           | End date              | 2022-12-31 |
| `--window`        | Input sequence length | 30         |
| `--horizon`       | Forecast horizon      | 5          |
| `--epochs`        | Training epochs       | 50         |
| `--batch_size`    | Batch size            | 32         |
| `--critic_steps`  | Critic updates        | 5          |
| `--demo`          | Offline demo mode     | Disabled   |
| `--run_baselines` | Evaluate baselines    | Disabled   |
| `--ablation`      | Ablation mode         | None       |

---

# Supported Forecast Horizons

* 1-Day Ahead
* 5-Day Ahead
* 10-Day Ahead
* 20-Day Ahead

Example:

```bash
python main.py --ticker AAPL --horizon 20 --epochs 100
```

---

# Ablation Study

Examples

```bash
python main.py --ticker AAPL --ablation no_sentiment
```

```bash
python main.py --ticker AAPL --ablation no_technical
```

```bash
python main.py --ticker AAPL --ablation generator_only
```

| Mode           | Description                 |
| -------------- | --------------------------- |
| no_sentiment   | Remove sentiment features   |
| no_technical   | Remove technical indicators |
| generator_only | Disable adversarial critic  |

---

# Model Architecture

## Generator

```text
Input Sequence
      │
      ▼
GRU Layer
      │
      ▼
GRU Layer
      │
      ▼
Dense Layer
      │
      ▼
Dropout
      │
      ▼
Forecast Output
```

---

## Critic

```text
Real / Generated Sequence
          │
          ▼
Conv1D
          │
          ▼
LeakyReLU
          │
          ▼
Conv1D
          │
          ▼
LeakyReLU
          │
          ▼
Conv1D
          │
          ▼
LeakyReLU
          │
          ▼
Flatten
          │
          ▼
Dense
          │
          ▼
Dense
          │
          ▼
Linear Critic Score
```

---

# Evaluation Metrics

## RMSE

Measures overall prediction error while penalizing large deviations.

---

## MAE

Measures the average absolute forecasting error.

---

## Directional Accuracy

Measures whether the model correctly predicts market movement direction.

---

## KL Divergence

Measures similarity between generated and real return distributions.

---

# Output Files

```text
outputs/

├── models/
│   ├── generator.pt
│   └── critic.pt
│
├── predictions/
│   └── predictions.csv
│
├── metrics/
│   └── metrics.csv
│
└── figures/
    ├── actual_vs_predicted.png
    ├── error_distribution.png
    ├── real_vs_generated_distribution.png
    └── training_losses.png
```

---

# Custom Sentiment Data

Store CSV files inside

```text
data/news/
```

Example

```csv
date,headline,summary,sentiment

2020-01-03,Apple rises after strong demand,Positive earnings outlook,0.64

2020-01-04,Market uncertainty increases,Investors remain cautious,-0.32
```

Missing trading-day sentiment values may be:

* Forward-filled
* Assigned a neutral value of **0**

---

# Reproducibility

Recommended settings

* Random Seeds

```
42
52
62
72
82
```

* Chronological split
* Window size = 30
* Forecast horizons = 1,5,10,20
* Critic updates = 5
* Gradient penalty = 10

---

# Recommended Experimental Protocol

1. Evaluate multiple stock tickers.
2. Compare 1, 5, 10, and 20-day forecasts.
3. Compare against:

   * SVR
   * LSTM
   * GRU
   * Transformer
4. Conduct ablation studies.
5. Report:

   * RMSE
   * MAE
   * Directional Accuracy
   * KL Divergence
6. Plot:

   * Actual vs Predicted
   * Distribution Comparison
7. Repeat experiments across multiple random seeds.

---

# Limitations

* Intended for academic and research purposes.
* Stock forecasting is inherently uncertain.
* Predictions should **not** be interpreted as investment or financial advice.
* Performance depends on market conditions, feature quality, selected stocks, and random initialization.

---

# Future Work

* FinBERT Sentiment Extraction
* FRED Macroeconomic Features
* Transformer Generator
* Temporal Fusion Transformer
* N-BEATS
* Multi-stock Portfolio Forecasting
* Probabilistic Confidence Intervals
* Trading Backtesting
* SHAP Explainability
* Attention Visualization

---
---

# DOI and Citation

This repository has been archived on **Zenodo** to ensure long-term preservation and reproducibility.

**DOI:** https://doi.org/10.5281/zenodo.20846812

You can cite this software using the DOI above.

---

# Citation

If you use this repository in your research, please cite both the software repository and the archived Zenodo release.

## BibTeX

```bibtex
@software{LakshmiKranthi2026,
  author       = {Lakshmi Kranthi},
  title        = {StockGenAI: FlexiStockGAN for Multi-Source Stock Price Forecasting},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {v1},
  doi          = {10.5281/zenodo.20846812},
  url          = {https://doi.org/10.5281/zenodo.20846812}
}
```

---

# Repository

GitHub Repository

https://github.com/LakshmiKranthi30/StockGenAI_FlexiStockGAN

Archived Release (Zenodo)

https://doi.org/10.5281/zenodo.20846812

---
# Disclaimer

This repository is provided exclusively for educational and research purposes.

It does **not** constitute financial, trading, or investment advice.

Users are responsible for independently validating all experimental results before applying them in real-world financial scenarios.

---

# License

This project may be released under the **MIT License** (or another suitable open-source license selected by the repository owner).

# Representation Learning for Financial Time Series

This project investigates whether learned latent representations of financial time series improve volatility-regime classification compared to classical handcrafted statistical features.

The work is conducted as part of a Bachelor’s thesis in Computer Science and Engineering at KTH Royal Institute of Technology.

---

## Research Question

To what extent can latent representations learned from rolling windows of daily financial returns using:

- Principal Component Analysis (PCA)  
- Autoencoders (AE)  
- Variational Autoencoders (VAE)  

improve classification of volatility regimes compared to classical feature-based approaches across different financial markets?

---

## Project Scope

The project focuses strictly on evaluating representation quality for downstream classification — not price prediction or trading.

Key constraints:

- Binary volatility regime classification (high vs low volatility)
- Two markets: US index proxy and Swedish index proxy
- CPU-feasible dataset and models
- Emphasis on evaluation design, robustness, and reproducibility

---

## Motivation

Financial time series are noisy, non-stationary, and difficult to model using traditional feature engineering alone. Representation learning methods offer a way to compress rolling return windows into compact embeddings that may capture underlying structure more effectively.

Rather than attempting to predict returns, this project evaluates whether learned embeddings provide more informative features for regime classification than classical handcrafted statistics.

The work emphasizes:

- Representation quality  
- Fair benchmarking against classical features  
- Cross-market robustness  
- Applied machine learning methodology  

---

## Method Overview

### Data Processing

1. Download daily adjusted closing prices for selected equity indices.
2. Compute log returns.
3. Compute rolling volatility.
4. Construct rolling windows (e.g., 60 trading days) of:
   - Returns  
   - Absolute returns (volatility proxy)

Each window forms the input sample for representation learning.

### Representation Methods

Each rolling window is mapped to a fixed-size embedding using:

- PCA (linear baseline)
- Autoencoder (nonlinear learned compression)
- Variational Autoencoder (probabilistic latent representation)

Embedding dimension typically ranges from 8–32.

### Baseline Features

Handcrafted statistical features are computed per window:

- Rolling mean  
- Rolling standard deviation  
- Momentum  

These serve as a classical benchmark for comparison.

### Downstream Evaluation

Embeddings and baseline features are evaluated using:

- Logistic regression classifier for volatility regime classification

Volatility regimes are defined via rolling volatility thresholding (e.g., above/below median).

---

## Evaluation Metrics

### Regime Classification

- Accuracy  
- F1-score  
- Balanced accuracy (if class imbalance occurs)

### Reliability

- Multiple random seeds for AE/VAE training  
- Comparison across two markets (US vs Sweden)  
- Statistical comparison where appropriate  

---

## Dataset

- Daily adjusted close prices
- US broad market index proxy (e.g., S&P 500 ETF)
- Swedish index proxy (e.g., OMXS30)
- Optional volatility proxy (VIX)

All data is sourced from publicly available providers (e.g., Yahoo Finance, Stooq).  
No proprietary or personal data is used.

---

## Reproducing Results

### 1. Set up environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run experiments

```bash
python src/run_experiment.py --market OMXS
python src/run_experiment.py --market SPX
```

Results are appended to `results/tables/metrics.csv`.

### 3. Generate tables and plots

Open and run all cells in `analysis.ipynb`. This produces:

- `results/plots/method_comparison.pdf` — bar chart comparing all methods
- `results/plots/dim_vs_perf.pdf` — line plots of performance vs embedding dimension
- `results/tables/method_comparison.tex/.csv`
- `results/tables/dim_ablation.tex/.csv`
- `results/tables/cross_market.tex/.csv`

---

## Repository Structure

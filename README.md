# Representation Learning for Financial Time Series

This project investigates whether learned latent representations of financial time series improve volatility regime classification and regime change detection compared to classical feature engineering methods.

The work is conducted as part of a Bachelors's thesis at KTH Royal Institute of Technology.

---

## Research Objective

We evaluate whether embeddings learned from rolling windows of daily financial returns using:

- Principal Component Analysis (PCA)
- Autoencoders (AE)
- Variational Autoencoders (VAE)

improve downstream performance on:

- Volatility regime classification  
- Regime change detection  

compared to handcrafted statistical features and supervised baselines.

---

## Motivation

Financial time series are noisy and non-stationary. Rather than predicting price direction, this project focuses on:

- Learning compact representations of return windows  
- Quantitatively evaluating representation quality  
- Testing transferability across markets  

The emphasis is on evaluation design, robustness, and applied machine learning methodology.

---

## Method Overview

1. Download daily adjusted closing prices for selected equity indices.
2. Compute log returns and rolling volatility.
3. Construct rolling windows (e.g., 60 trading days).
4. Learn embeddings using PCA, Autoencoder, and VAE.
5. Evaluate embeddings via:
   - Logistic regression for regime classification
   - Change-point detection in embedding space
6. Compare results against classical handcrafted feature baselines.

---

## Dataset

- Daily adjusted close prices
- US broad index proxy (e.g., S&P 500)
- Swedish index proxy (e.g., OMXS30)
- Optional volatility proxy (VIX)

Data is sourced from publicly available CSV providers.

---

## Evaluation Metrics

**Regime Classification**
- Accuracy
- F1-score
- Balanced accuracy (if needed)

**Change Detection**
- Precision
- Recall
- F1-score
- Detection delay
- False alarm rate

---

## Repository Structure

data/
src/
notebooks/
results/
report/


- `data/` – Raw and processed datasets  
- `src/` – Model implementations and evaluation code  
- `notebooks/` – Exploratory analysis  
- `results/` – Saved metrics and figures  
- `report/` – Thesis materials

---

## Key Focus

- Representation learning for time series
- Benchmarking and evaluation
- Cross-market generalization
- Applied ML engineering practices
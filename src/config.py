# src/config.py
"""
Central configuration for the financial time series representation learning project.

All hyperparameters and paths should be defined here so experiments are:
- Reproducible
- Deterministic
- Easy to modify
"""

from __future__ import annotations

from pathlib import Path
import random
import numpy as np

# ============================================================
# Repository paths
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
MODELS_DIR = RESULTS_DIR / "models"
PLOTS_DIR = RESULTS_DIR / "plots"

METRICS_PATH = TABLES_DIR / "metrics.csv"


# ============================================================
# Data processing parameters
# ============================================================

WINDOW_LENGTH = 60          # Rolling window length (L)
VOL_WINDOW = 20             # Rolling volatility window (w)

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

VOL_THRESHOLD_METHOD = "median"   # Currently using median (quantile=0.5)


# ============================================================
# Representation learning parameters
# ============================================================

EMBEDDING_DIMS = [8, 16, 32]

# PCA
PCA_WHITEN = False

# Autoencoder
AE_HIDDEN_DIM = 64
AE_EPOCHS = 100
AE_BATCH_SIZE = 128
AE_LEARNING_RATE = 1e-3
AE_PATIENCE = 10

# VAE
BETA_VAE = 1.0


# ============================================================
# Classifier parameters
# ============================================================

LOGREG_C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
LOGREG_MAX_ITER = 1000


# ============================================================
# Reproducibility
# ============================================================

RANDOM_SEED = 42
AE_SEEDS = [42, 123, 456]


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """
    Set all relevant random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Optional: If PyTorch is used later
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ============================================================
# Utility
# ============================================================

def ensure_directories() -> None:
    """
    Create results directories if they do not exist.
    """
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
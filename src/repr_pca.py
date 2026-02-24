# src/repr_pca.py
"""
PCA-based representations for volatility regime classification.

Pipeline:
1) Load train/val/test windows via datasets.create_dataset()
2) Flatten windows (N, 60, 2) -> (N, 120)
3) Standardize inputs for PCA using train stats only
4) Fit PCA on train only, transform train/val/test
5) For each embedding dim in config.EMBEDDING_DIMS:
   - Train + tune Logistic Regression via train_classifier.train_and_evaluate_logreg()
   - Append metrics to results/tables/metrics.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import (
    EMBEDDING_DIMS,
    METRICS_PATH,
    PCA_WHITEN,
    ensure_directories,
)
from data_processing import ProcessingConfig
from datasets import SplitConfig, create_dataset
from train_classifier import ClassifierConfig, train_and_evaluate_logreg


PathLike = Union[str, Path]


@dataclass(frozen=True)
class PCAConfig:
    """
    Configuration for PCA representations.
    """
    embedding_dims: Sequence[int] = tuple(EMBEDDING_DIMS)
    whiten: bool = PCA_WHITEN
    # Use same random_state as classifier for consistency
    random_state: int = 42
    standardize_input: bool = True  # standardize flattened windows before PCA


def _flatten_windows(X: np.ndarray) -> np.ndarray:
    """
    Flatten windows (N, L, C) -> (N, L*C).
    """
    if X.ndim != 3:
        raise ValueError(f"Expected X shape (N, L, C). Got {X.shape}")
    N, L, C = X.shape
    return X.reshape(N, L * C)


def _standardize_for_pca(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[StandardScaler]]:
    """
    Standardize inputs for PCA (fit on train only).
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_val_s, X_test_s, scaler


def append_metrics_row(path: Path, row: Dict[str, object]) -> None:
    """
    Append one experiment row to metrics.csv, creating the file with header if needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df_row = pd.DataFrame([row])
    if path.exists():
        df_row.to_csv(path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(path, mode="w", header=True, index=False)


def run_pca(
    data_path: PathLike,
    market_name: str,
    processing_cfg: ProcessingConfig = ProcessingConfig(),
    split_cfg: SplitConfig = SplitConfig(),
    pca_cfg: PCAConfig = PCAConfig(),
    clf_cfg: ClassifierConfig = ClassifierConfig(),
    save_metrics: bool = True,
) -> Dict[str, object]:
    """
    Run PCA-based representation + logistic regression for one market file.

    Returns:
        Dict with per-dim results and dataset stats.
    """
    ensure_directories()

    # 1) Load dataset (windows + splits)
    ds = create_dataset(
        data_path=Path(data_path),
        processing_cfg=processing_cfg,
        split_cfg=split_cfg,
        label_threshold=None,  # compute threshold on train only
    )

    X_train, y_train = ds["X_train"], ds["y_train"]
    X_val, y_val = ds["X_val"], ds["y_val"]
    X_test, y_test = ds["X_test"], ds["y_test"]

    # 2) Flatten windows
    F_train = _flatten_windows(X_train)
    F_val = _flatten_windows(X_val)
    F_test = _flatten_windows(X_test)

    # 3) Standardize for PCA (train-only stats)
    if pca_cfg.standardize_input:
        F_train_s, F_val_s, F_test_s, _ = _standardize_for_pca(F_train, F_val, F_test)
    else:
        F_train_s, F_val_s, F_test_s = F_train, F_val, F_test

    results_by_dim: Dict[int, Dict[str, object]] = {}

    # 4) Loop over embedding dimensions
    for d in pca_cfg.embedding_dims:
        if d <= 0 or d > F_train_s.shape[1]:
            raise ValueError(f"Invalid embedding dim {d} for input dim {F_train_s.shape[1]}")

        # Fit PCA on train only
        pca = PCA(
            n_components=int(d),
            whiten=bool(pca_cfg.whiten),
            random_state=int(pca_cfg.random_state),
        )
        Z_train = pca.fit_transform(F_train_s)
        Z_val = pca.transform(F_val_s)
        Z_test = pca.transform(F_test_s)

        # 5) Train/evaluate logistic regression on PCA embeddings
        res = train_and_evaluate_logreg(
            X_train=Z_train,
            y_train=y_train,
            X_val=Z_val,
            y_val=y_val,
            X_test=Z_test,
            y_test=y_test,
            cfg=clf_cfg,
            tune_metric="f1",
            standardize=True,  # standardize embeddings again for classifier, consistent with baseline
        )

        out = {
            "market": market_name,
            "method": "pca",
            "embedding_dim": int(d),
            "input_dim": int(F_train_s.shape[1]),
            "threshold": float(ds["threshold"]),
            "best_C": float(res["best_C"]),
            "val_metrics_best": res["val_metrics_best"],
            "test_metrics": res["test_metrics"],
            "dataset_stats": ds["stats"],
        }
        results_by_dim[int(d)] = out

        if save_metrics:
            row = {
                "market": market_name,
                "method": "pca",
                "embedding_dim": int(d),
                "n_features": int(d),
                "threshold": float(ds["threshold"]),
                "best_C": float(res["best_C"]),
                "val_accuracy": float(res["val_metrics_best"]["accuracy"]),
                "val_f1": float(res["val_metrics_best"]["f1"]),
                "val_balanced_accuracy": float(res["val_metrics_best"]["balanced_accuracy"]),
                "test_accuracy": float(res["test_metrics"]["accuracy"]),
                "test_f1": float(res["test_metrics"]["f1"]),
                "test_balanced_accuracy": float(res["test_metrics"]["balanced_accuracy"]),
            }
            append_metrics_row(METRICS_PATH, row)

    return {
        "market": market_name,
        "method": "pca",
        "results_by_dim": results_by_dim,
        "dataset_stats": ds["stats"],
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="PCA representations + logistic regression.")
    parser.add_argument("--input", type=str, required=True, help="Path to raw data file, e.g. data/omxs.txt")
    parser.add_argument("--market", type=str, required=True, help="Market name tag, e.g. OMXS or SPX")
    parser.add_argument("--no-save", action="store_true", help="Do not append results to metrics.csv")
    parser.add_argument(
        "--embedding-dims",
        type=int,
        nargs="+",
        default=EMBEDDING_DIMS,
        help="Embedding dimensions for PCA (e.g. 8 16 32)",
    )

    parser.add_argument("--window-len", type=int, default=60)
    parser.add_argument("--vol-window", type=int, default=20)
    parser.add_argument("--train-split", type=float, default=0.70)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.15)

    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--no-whiten", action="store_true", help="Disable PCA whitening.")

    args = parser.parse_args()

    processing_cfg = ProcessingConfig(window_len=args.window_len, vol_window=args.vol_window)
    split_cfg = SplitConfig(train_split=args.train_split, val_split=args.val_split, test_split=args.test_split)
    pca_cfg = PCAConfig(
        embedding_dims=tuple(args.embedding_dims),
        whiten=not args.no_whiten,
        random_state=args.random_state,
    )
    clf_cfg = ClassifierConfig(max_iter=args.max_iter, random_state=args.random_state)

    res = run_pca(
        data_path=args.input,
        market_name=args.market,
        processing_cfg=processing_cfg,
        split_cfg=split_cfg,
        pca_cfg=pca_cfg,
        clf_cfg=clf_cfg,
        save_metrics=not args.no_save,
    )

    print(json.dumps(res, indent=2))
# src/train_classifier.py
"""
Utilities for training and evaluating downstream classifiers (Logistic Regression)
in a leakage-safe, time-series setup.

Design goals:
- One implementation shared by baseline/PCA/AE/VAE
- Tune hyperparameters on validation only
- Report consistent metrics (Accuracy, F1, Balanced Accuracy)
- Reproducible via fixed random_state
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ClassifierConfig:
    c_grid: Tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    max_iter: int = 1000
    random_state: int = 42
    # For imbalanced labels, this is often helpful. Keep None for now; set to "balanced" if needed.
    class_weight: Optional[str] = None


def _validate_shapes(X: np.ndarray, y: np.ndarray, name: str) -> None:
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2D (n_samples, n_features). Got shape {X.shape}.")
    if y.ndim != 1:
        raise ValueError(f"{name} labels must be 1D (n_samples,). Got shape {y.shape}.")
    if len(X) != len(y):
        raise ValueError(f"{name}: X and y length mismatch: {len(X)} vs {len(y)}")
    if not np.isin(y, [0, 1]).all():
        raise ValueError(f"{name}: y must be binary 0/1. Got unique values: {np.unique(y)}")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Consistent metric set for volatility regime classification.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def standardize_train_val_test(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit StandardScaler on train only, transform train/val/test.
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_val_s, X_test_s, scaler


def fit_logreg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float,
    cfg: ClassifierConfig,
) -> LogisticRegression:
    """
    Fit a logistic regression model.
    """
    model = LogisticRegression(
        C=float(C),
        max_iter=int(cfg.max_iter),
        random_state=int(cfg.random_state),
        class_weight=cfg.class_weight,
        solver="lbfgs",
    )
    model.fit(X_train, y_train)
    return model


def tune_logreg_C_on_val(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: ClassifierConfig,
    metric: str = "f1",
) -> Tuple[float, Dict[float, Dict[str, float]]]:
    """
    Tune C by training on train and selecting best performance on val.

    Args:
        metric: one of {"accuracy", "f1", "balanced_accuracy"}

    Returns:
        best_C, val_results_by_C
    """
    if metric not in {"accuracy", "f1", "balanced_accuracy"}:
        raise ValueError(f"Unknown metric '{metric}'. Choose from accuracy, f1, balanced_accuracy.")

    results: Dict[float, Dict[str, float]] = {}
    best_C: Optional[float] = None
    best_score = -np.inf

    for C in cfg.c_grid:
        model = fit_logreg(X_train, y_train, C=C, cfg=cfg)
        y_val_pred = model.predict(X_val)
        m = compute_metrics(y_val, y_val_pred)
        results[float(C)] = m

        score = m[metric]
        if score > best_score:
            best_score = score
            best_C = float(C)

    assert best_C is not None
    return best_C, results


def train_and_evaluate_logreg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: ClassifierConfig = ClassifierConfig(),
    tune_metric: str = "f1",
    standardize: bool = True,
) -> Dict[str, object]:
    """
    Full pipeline:
    - validate
    - (optional) standardize using train only
    - tune C on validation
    - refit on train+val with best C
    - evaluate on test

    Returns:
        Dict with best_C, val_metrics_by_C, val_metrics_best, test_metrics, scaler (if used)
    """
    _validate_shapes(X_train, y_train, "train")
    _validate_shapes(X_val, y_val, "val")
    _validate_shapes(X_test, y_test, "test")

    if standardize:
        X_train_s, X_val_s, X_test_s, scaler = standardize_train_val_test(X_train, X_val, X_test)
    else:
        X_train_s, X_val_s, X_test_s = X_train, X_val, X_test
        scaler = None

    # Tune C on validation
    best_C, val_results_by_C = tune_logreg_C_on_val(
        X_train_s, y_train, X_val_s, y_val, cfg=cfg, metric=tune_metric
    )

    # Refit on train+val with best C
    X_trainval = np.vstack([X_train_s, X_val_s])
    y_trainval = np.concatenate([y_train, y_val])

    final_model = fit_logreg(X_trainval, y_trainval, C=best_C, cfg=cfg)

    # Metrics
    y_val_pred_best = final_model.predict(X_val_s)
    y_test_pred = final_model.predict(X_test_s)

    val_metrics_best = compute_metrics(y_val, y_val_pred_best)
    test_metrics = compute_metrics(y_test, y_test_pred)

    return {
        "best_C": float(best_C),
        "val_metrics_by_C": val_results_by_C,
        "val_metrics_best": val_metrics_best,
        "test_metrics": test_metrics,
        "scaler": scaler,
        "model": final_model,
    }
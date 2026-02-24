# src/repr_ae.py
"""
Autoencoder (AE) representations for volatility regime classification.

Pipeline:
1) Load train/val/test windows via datasets.create_dataset()
2) Flatten windows (N, 60, 2) -> (N, 120)
3) Train MLP autoencoder on train windows only, early stopping on val recon loss
4) Extract latent embeddings (encoder output) for train/val/test
5) For each latent dim and seed:
   - Train + tune Logistic Regression via train_classifier.train_and_evaluate_logreg()
   - Append metrics to results/tables/metrics.csv
   - Save AE weights to results/models/
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

from config import (
    AE_BATCH_SIZE,
    AE_EPOCHS,
    AE_HIDDEN_DIM,
    AE_LEARNING_RATE,
    AE_PATIENCE,
    AE_SEEDS,
    EMBEDDING_DIMS,
    METRICS_PATH,
    MODELS_DIR,
    ensure_directories,
    set_global_seed,
)
from data_processing import ProcessingConfig
from datasets import SplitConfig, create_dataset
from train_classifier import ClassifierConfig, train_and_evaluate_logreg


PathLike = Union[str, Path]


# ======================================================================
# Model
# ======================================================================

class MLPAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


# ======================================================================
# Helpers
# ======================================================================

def _flatten_windows(X: np.ndarray) -> np.ndarray:
    """
    Flatten windows (N, L, C) -> (N, L*C).
    """
    if X.ndim != 3:
        raise ValueError(f"Expected X shape (N, L, C). Got {X.shape}")
    N, L, C = X.shape
    return X.reshape(N, L * C)


def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(x.astype(np.float32)).to(device)


def _make_dataloaders(
    X_train: np.ndarray,
    X_val: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Tuple[DataLoader, DataLoader]:
    x_train_t = _to_tensor(X_train, device)
    x_val_t = _to_tensor(X_val, device)

    train_ds = TensorDataset(x_train_t)
    val_ds = TensorDataset(x_val_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def _train_ae_one_run(
    X_train: np.ndarray,
    X_val: np.ndarray,
    input_dim: int,
    hidden_dim: int,
    latent_dim: int,
    max_epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    seed: int,
    device: torch.device,
) -> Tuple[MLPAutoencoder, int, List[float], List[float]]:
    """
    Train AE on train windows only, early stopping on val recon loss.
    """
    set_global_seed(seed)

    train_loader, val_loader = _make_dataloaders(X_train, X_val, batch_size, device)

    model = MLPAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch = -1
    best_state: Optional[Dict[str, torch.Tensor]] = None

    train_losses: List[float] = []
    val_losses: List[float] = []
    no_improve_epochs = 0

    for epoch in range(1, max_epochs + 1):
        # Train
        model.train()
        epoch_train_loss = 0.0
        n_train_batches = 0

        for (x_batch,) in train_loader:
            optimizer.zero_grad()
            x_hat, _ = model(x_batch)
            loss = criterion(x_hat, x_batch)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            n_train_batches += 1

        epoch_train_loss /= max(n_train_batches, 1)

        # Validation
        model.eval()
        epoch_val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for (x_batch,) in val_loader:
                x_hat, _ = model(x_batch)
                loss = criterion(x_hat, x_batch)
                epoch_val_loss += loss.item()
                n_val_batches += 1

        epoch_val_loss /= max(n_val_batches, 1)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        # Early stopping on val loss
        if epoch_val_loss < best_val_loss - 1e-6:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_epoch, train_losses, val_losses


def _encode_full(
    model: MLPAutoencoder,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    """
    Run encoder on full dataset X (N, D) -> Z (N, latent_dim).
    """
    model.eval()
    Z_list: List[np.ndarray] = []
    with torch.no_grad():
        x_t = _to_tensor(X, device)
        N = x_t.shape[0]
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            z = model.encode(x_t[start:end])
            Z_list.append(z.cpu().numpy())
    return np.concatenate(Z_list, axis=0)


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


# ======================================================================
# Main AE runner
# ======================================================================

@dataclass(frozen=True)
class AEConfig:
    """
    Configuration for AE representations.
    """
    hidden_dim: int = AE_HIDDEN_DIM
    latent_dims: Sequence[int] = tuple(EMBEDDING_DIMS)
    batch_size: int = AE_BATCH_SIZE
    learning_rate: float = AE_LEARNING_RATE
    max_epochs: int = AE_EPOCHS
    patience: int = AE_PATIENCE
    seeds: Sequence[int] = tuple(AE_SEEDS)
    device: str = "cpu"  # keep CPU for reproducibility


def run_ae(
    data_path: PathLike,
    market_name: str,
    processing_cfg: ProcessingConfig = ProcessingConfig(),
    split_cfg: SplitConfig = SplitConfig(),
    ae_cfg: AEConfig = AEConfig(),
    clf_cfg: ClassifierConfig = ClassifierConfig(),
    save_metrics: bool = True,
    save_models: bool = True,
) -> Dict[str, object]:
    """
    Run AE-based representation + logistic regression for one market file.

    Returns:
        Dict with per-dim+seed results and dataset stats.
    """
    ensure_directories()
    device = torch.device(ae_cfg.device)

    # 1) Load dataset
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

    input_dim = F_train.shape[1]

    results: Dict[int, Dict[int, Dict[str, object]]] = {}

    for d in ae_cfg.latent_dims:
        if d <= 0 or d > input_dim:
            raise ValueError(f"Invalid latent dim {d} for input dim {input_dim}")

        results[d] = {}

        for seed in ae_cfg.seeds:
            # 3) Train AE for this latent dim and seed
            model, best_epoch, train_losses, val_losses = _train_ae_one_run(
                X_train=F_train,
                X_val=F_val,
                input_dim=input_dim,
                hidden_dim=ae_cfg.hidden_dim,
                latent_dim=int(d),
                max_epochs=ae_cfg.max_epochs,
                batch_size=ae_cfg.batch_size,
                lr=ae_cfg.learning_rate,
                patience=ae_cfg.patience,
                seed=seed,
                device=device,
            )

            # 4) Extract embeddings
            Z_train = _encode_full(model, F_train, device=device)
            Z_val = _encode_full(model, F_val, device=device)
            Z_test = _encode_full(model, F_test, device=device)

            # 5) Train/evaluate logistic regression on embeddings
            res_clf = train_and_evaluate_logreg(
                X_train=Z_train,
                y_train=y_train,
                X_val=Z_val,
                y_val=y_val,
                X_test=Z_test,
                y_test=y_test,
                cfg=clf_cfg,
                tune_metric="f1",
                standardize=True,  # standardize embeddings for classifier
            )

            result_entry = {
                "market": market_name,
                "method": "ae_mlp",
                "embedding_dim": int(d),
                "input_dim": int(input_dim),
                "threshold": float(ds["threshold"]),
                "seed": int(seed),
                "best_epoch": int(best_epoch),
                "train_losses": train_losses,
                "val_losses": val_losses,
                "best_C": float(res_clf["best_C"]),
                "val_metrics_best": res_clf["val_metrics_best"],
                "test_metrics": res_clf["test_metrics"],
                "dataset_stats": ds["stats"],
            }
            results[d][seed] = result_entry

            # Save metrics row
            if save_metrics:
                row = {
                    "market": market_name,
                    "method": "ae_mlp",
                    "embedding_dim": int(d),
                    "seed": int(seed),
                    "n_features": int(d),
                    "threshold": float(ds["threshold"]),
                    "best_C": float(res_clf["best_C"]),
                    "val_accuracy": float(res_clf["val_metrics_best"]["accuracy"]),
                    "val_f1": float(res_clf["val_metrics_best"]["f1"]),
                    "val_balanced_accuracy": float(res_clf["val_metrics_best"]["balanced_accuracy"]),
                    "test_accuracy": float(res_clf["test_metrics"]["accuracy"]),
                    "test_f1": float(res_clf["test_metrics"]["f1"]),
                    "test_balanced_accuracy": float(res_clf["test_metrics"]["balanced_accuracy"]),
                }
                append_metrics_row(METRICS_PATH, row)

            # Save model weights
            if save_models:
                MODELS_DIR.mkdir(parents=True, exist_ok=True)
                model_path = MODELS_DIR / f"ae_mlp_{market_name}_L{processing_cfg.window_len}_d{int(d)}_seed{int(seed)}.pt"
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "input_dim": input_dim,
                        "hidden_dim": ae_cfg.hidden_dim,
                        "latent_dim": int(d),
                        "seed": int(seed),
                        "best_epoch": int(best_epoch),
                        "train_losses": train_losses,
                        "val_losses": val_losses,
                    },
                    model_path,
                )

    return {
        "market": market_name,
        "method": "ae_mlp",
        "results_by_dim_and_seed": results,
        "dataset_stats": ds["stats"],
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="MLP Autoencoder representations + logistic regression.")
    parser.add_argument("--input", type=str, required=True, help="Path to raw data file, e.g. data/omxs.txt")
    parser.add_argument("--market", type=str, required=True, help="Market name tag, e.g. OMXS or SPX")
    parser.add_argument("--no-save", action="store_true", help="Do not append results to metrics.csv")
    parser.add_argument("--no-save-models", action="store_true", help="Do not save AE model weights")

    parser.add_argument("--window-len", type=int, default=60)
    parser.add_argument("--vol-window", type=int, default=20)
    parser.add_argument("--train-split", type=float, default=0.70)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.15)

    parser.add_argument("--latent-dims", type=int, nargs="+", default=EMBEDDING_DIMS)
    parser.add_argument("--hidden-dim", type=int, default=AE_HIDDEN_DIM)
    parser.add_argument("--batch-size", type=int, default=AE_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=AE_LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=AE_EPOCHS)
    parser.add_argument("--patience", type=int, default=AE_PATIENCE)
    parser.add_argument("--seeds", type=int, nargs="+", default=AE_SEEDS)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()

    processing_cfg = ProcessingConfig(window_len=args.window_len, vol_window=args.vol_window)
    split_cfg = SplitConfig(train_split=args.train_split, val_split=args.val_split, test_split=args.test_split)
    ae_cfg = AEConfig(
        hidden_dim=args.hidden_dim,
        latent_dims=tuple(args.latent_dims),
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_epochs=args.epochs,
        patience=args.patience,
        seeds=tuple(args.seeds),
        device=args.device,
    )
    clf_cfg = ClassifierConfig(max_iter=args.max_iter, random_state=args.random_state)

    res = run_ae(
        data_path=args.input,
        market_name=args.market,
        processing_cfg=processing_cfg,
        split_cfg=split_cfg,
        ae_cfg=ae_cfg,
        clf_cfg=clf_cfg,
        save_metrics=not args.no_save,
        save_models=not args.no_save_models,
    )

    print(json.dumps(res, indent=2))
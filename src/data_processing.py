# src/data_processing.py
"""
Load SPX/OMXS raw files (Stooq-like CSV-in-.txt), clean, compute returns/volatility,
and (optionally) build fixed-length windows for representation learning.

Expected raw header (your files):
<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>

Outputs:
- processed time series CSV (date-indexed)
- optional windows NPZ with X (n, L, 2), y (n,), end_dates (n,), threshold (float)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"


@dataclass(frozen=True)
class ProcessingConfig:
    window_len: int = 60          # L = 60 trading days
    vol_window: int = 20          # w = 20 days rolling std
    feature_cols: Tuple[str, ...] = ("log_return", "abs_log_return")
    label_col: str = "rolling_vol"
    label_quantile: float = 0.5   # median threshold by default (high-vol if >= threshold)


def _normalize_columns(cols: Iterable[str]) -> list[str]:
    out = []
    for c in cols:
        c = c.strip()
        if c.startswith("<") and c.endswith(">"):
            c = c[1:-1]
        out.append(c.strip().lower())
    return out


def load_raw_market_file(path: Path) -> pd.DataFrame:
    """
    Reads the raw .txt file (comma-separated), parses date, converts numeric columns,
    sorts ascending, and drops invalid rows.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find file: {path}")

    df = pd.read_csv(path, sep=",", engine="python")
    df.columns = _normalize_columns(df.columns)

    required = {"date", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {missing} in file: {path}")

    # Parse dates (YYYYMMDD)
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")

    # Convert price columns if present
    for col in ("open", "high", "low", "close", "vol", "openint"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean
    df = df.dropna(subset=["date", "close"]).copy()
    df = df.sort_values("date").reset_index(drop=True)

    # Must be positive for log-returns
    df = df[df["close"] > 0].copy()

    return df


def filter_date_range(
    df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    start_date/end_date: 'YYYY-MM-DD' (inclusive bounds).
    """
    out = df.copy()
    if start_date is not None:
        out = out[out["date"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        out = out[out["date"] <= pd.to_datetime(end_date)]
    return out.reset_index(drop=True)


def add_returns_and_volatility(df: pd.DataFrame, vol_window: int = 20) -> pd.DataFrame:
    """
    Adds:
      - log_return = log(close_t) - log(close_{t-1})
      - abs_log_return = |log_return|
      - rolling_vol = rolling std of log_return over `vol_window`
    """
    out = df.copy()

    out["log_return"] = np.log(out["close"]).diff()
    out["abs_log_return"] = out["log_return"].abs()
    out["rolling_vol"] = out["log_return"].rolling(vol_window).std()

    # Drop first row (diff produced NaN)
    out = out.dropna(subset=["log_return"]).reset_index(drop=True)
    return out


def build_windows(
    df: pd.DataFrame,
    cfg: ProcessingConfig,
    label_threshold: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Builds rolling windows X of shape (n, L, 2) using cfg.feature_cols.
    Labels y are binary high-vol regimes based on cfg.label_col at window end date.

    If label_threshold is None, it is computed as quantile(cfg.label_quantile) on the
    available label values (after window/NaN filtering).
    """
    L = cfg.window_len
    feat_cols = cfg.feature_cols
    label_col = cfg.label_col

    if any(c not in df.columns for c in feat_cols):
        missing = [c for c in feat_cols if c not in df.columns]
        raise ValueError(f"Missing feature columns in df: {missing}")

    if label_col not in df.columns:
        raise ValueError(f"Missing label column in df: {label_col}")

    feats = df.loc[:, feat_cols].to_numpy(dtype=float)
    labels_cont = df[label_col].to_numpy(dtype=float)
    dates = df["date"].to_numpy()

    n_total = len(df)
    if n_total < L:
        raise ValueError(f"Not enough rows ({n_total}) to build windows of length {L}")

    valid_end_indices: list[int] = []
    for i in range(L - 1, n_total):
        window = feats[i - L + 1 : i + 1, :]
        if not np.isfinite(window).all():
            continue
        if not np.isfinite(labels_cont[i]):
            continue
        valid_end_indices.append(i)

    if not valid_end_indices:
        raise ValueError("No valid windows found (check NaNs/volatility window/date filtering).")

    end_idx = np.array(valid_end_indices, dtype=int)

    X = np.stack([feats[i - L + 1 : i + 1, :] for i in end_idx], axis=0)
    y_cont = labels_cont[end_idx]
    end_dates = dates[end_idx]

    if label_threshold is None:
        label_threshold = float(np.quantile(y_cont, cfg.label_quantile))

    y = (y_cont >= label_threshold).astype(np.int64)

    return X, y, end_dates.astype("datetime64[ns]"), float(label_threshold)


def save_processed_timeseries(df: pd.DataFrame, name: str) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"{name}_processed.csv"
    df.to_csv(out_path, index=False)
    return out_path


def save_windows_npz(
    X: np.ndarray,
    y: np.ndarray,
    end_dates: np.ndarray,
    threshold: float,
    name: str,
    cfg: ProcessingConfig,
) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"{name}_windows_L{cfg.window_len}_W{cfg.vol_window}.npz"
    np.savez_compressed(
        out_path,
        X=X,
        y=y,
        end_dates=end_dates.astype("datetime64[ns]").astype("int64"),  # store as int64 ns
        threshold=np.array([threshold], dtype=float),
        feature_cols=np.array(cfg.feature_cols, dtype="U"),
        label_col=np.array([cfg.label_col], dtype="U"),
        window_len=np.array([cfg.window_len], dtype=int),
        vol_window=np.array([cfg.vol_window], dtype=int),
        label_quantile=np.array([cfg.label_quantile], dtype=float),
    )
    return out_path


def process_market(
    input_path: Path,
    name: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    make_windows: bool = True,
    cfg: ProcessingConfig = ProcessingConfig(),
    label_threshold: Optional[float] = None,
) -> dict:
    """
    End-to-end: load -> filter -> add returns/vol -> save CSV -> optional windows NPZ.
    Returns paths + key stats.
    """
    df = load_raw_market_file(input_path)
    df = filter_date_range(df, start_date=start_date, end_date=end_date)
    df = add_returns_and_volatility(df, vol_window=cfg.vol_window)

    csv_path = save_processed_timeseries(df, name=name)

    out = {
        "name": name,
        "rows": len(df),
        "csv_path": str(csv_path),
    }

    if make_windows:
        X, y, end_dates, thr = build_windows(df, cfg=cfg, label_threshold=label_threshold)
        npz_path = save_windows_npz(X, y, end_dates, thr, name=name, cfg=cfg)
        out.update(
            {
                "windows": int(X.shape[0]),
                "X_shape": tuple(X.shape),
                "y_mean": float(y.mean()),
                "threshold": float(thr),
                "npz_path": str(npz_path),
            }
        )

    return out


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Process SPX/OMXS raw files into returns/vol + windows.")
    parser.add_argument("--input", type=str, required=True, help="Path to raw file, e.g. data/spx.txt")
    parser.add_argument("--name", type=str, required=True, help="Name tag, e.g. spx or omxs")
    parser.add_argument("--start-date", type=str, default=None, help="YYYY-MM-DD (inclusive)")
    parser.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD (inclusive)")
    parser.add_argument("--no-windows", action="store_true", help="Only save processed CSV, no NPZ windows.")
    parser.add_argument("--window-len", type=int, default=60)
    parser.add_argument("--vol-window", type=int, default=20)
    parser.add_argument("--label-quantile", type=float, default=0.5, help="Quantile for high-vol threshold.")
    parser.add_argument("--label-threshold", type=float, default=None, help="Override threshold (float).")

    args = parser.parse_args()

    cfg = ProcessingConfig(
        window_len=args.window_len,
        vol_window=args.vol_window,
        label_quantile=args.label_quantile,
    )

    result = process_market(
        input_path=Path(args.input),
        name=args.name,
        start_date=args.start_date,
        end_date=args.end_date,
        make_windows=not args.no_windows,
        cfg=cfg,
        label_threshold=args.label_threshold,
    )

    print(json.dumps(result, indent=2))
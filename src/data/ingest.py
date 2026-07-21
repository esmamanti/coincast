from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = ROOT / "data_raw"
PROCESSED_DATA_DIR = ROOT / "data_processed"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def fetch_binance_klines(symbol: str, interval: str = "1h", limit: int = 500) -> pd.DataFrame:
    """Placeholder ingest function for a future Binance API integration."""
    raise NotImplementedError("Binance API integration is not wired yet")


def save_raw_parquet(df: pd.DataFrame, symbol: str, interval: str = "1h") -> Path:
    ensure_directory(RAW_DATA_DIR)
    path = RAW_DATA_DIR / f"{symbol}_{interval}.parquet"
    df.to_parquet(path, index=False)
    return path


def load_raw_parquet(symbol: str, interval: str = "1h") -> Optional[pd.DataFrame]:
    path = RAW_DATA_DIR / f"{symbol}_{interval}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def save_processed_parquet(df: pd.DataFrame, symbol: str, interval: str = "1h") -> Path:
    ensure_directory(PROCESSED_DATA_DIR)
    path = PROCESSED_DATA_DIR / f"{symbol}_{interval}.parquet"
    df.to_parquet(path, index=False)
    return path


def load_processed_parquet(symbol: str, interval: str = "1h") -> Optional[pd.DataFrame]:
    path = PROCESSED_DATA_DIR / f"{symbol}_{interval}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def clean_and_resample(df: pd.DataFrame, interval: str = "1h") -> pd.DataFrame:
    clean_df = df.copy()
    if "open_time" in clean_df.columns:
        clean_df["open_time"] = pd.to_datetime(clean_df["open_time"])
        clean_df = clean_df.sort_values("open_time").drop_duplicates("open_time")
        clean_df = clean_df.set_index("open_time")

    if "close" not in clean_df.columns:
        raise KeyError("Expected a 'close' column in the incoming OHLCV dataframe")

    temperature = None
    if interval == "1h":
        temperature = "1h"
    elif interval == "4h":
        temperature = "4h"
    elif interval == "1d":
        temperature = "1d"

    if temperature is None:
        raise ValueError(f"Unsupported interval: {interval}")

    resampled = clean_df.resample(temperature).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    resampled = resampled.reset_index()
    return resampled


def ingest_symbol(symbol: str, interval: str = "1h", force_refresh: bool = False) -> tuple[Path, Path]:
    """Minimal ingest pipeline: save raw parquet, clean/resample, and save processed parquet."""
    raw_df = load_raw_parquet(symbol, interval)
    if raw_df is None or force_refresh:
        raw_df = fetch_binance_klines(symbol, interval=interval, limit=1000)
        raw_path = save_raw_parquet(raw_df, symbol, interval)
    else:
        raw_path = RAW_DATA_DIR / f"{symbol}_{interval}.parquet"

    processed_df = clean_and_resample(raw_df, interval=interval)
    processed_path = save_processed_parquet(processed_df, symbol, interval)
    return raw_path, processed_path

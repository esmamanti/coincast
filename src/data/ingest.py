from __future__ import annotations

import re
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


SUPPORTED_INTERVALS = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"}
SYMBOL_PATTERN = re.compile(r"^[A-Z0-9]{5,20}$")


def normalize_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper()
    if not SYMBOL_PATTERN.fullmatch(normalized):
        raise ValueError("Symbol must contain only 5-20 uppercase letters or digits")
    return normalized


def fetch_binance_klines(
    symbol: str,
    interval: str = "1h",
    limit: int = 500,
    closed_only: bool = True,
) -> pd.DataFrame:
    """Fetch recent public spot candles from Binance without API credentials."""
    from binance.client import Client

    normalized_symbol = normalize_symbol(symbol)
    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(f"Unsupported interval: {interval}")
    if limit < 2 or limit > 1000:
        raise ValueError("Kline limit must be between 2 and 1000")

    client = Client()
    market_source = "binance_spot_closed_candles"
    try:
        rows = client.get_klines(symbol=normalized_symbol, interval=interval, limit=limit)
    except Exception as exc:
        if getattr(exc, "code", None) != -1121:
            raise
        rows = client.futures_klines(symbol=normalized_symbol, interval=interval, limit=limit)
        market_source = "binance_futures_closed_candles"
    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])

    frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    frame["close_time"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True)
    numeric_columns = ["open", "high", "low", "close", "volume"]
    frame[numeric_columns] = frame[numeric_columns].astype(float)
    if closed_only:
        frame = frame.loc[frame["close_time"] <= pd.Timestamp.now(tz="UTC")]

    result = (
        frame[["open_time", *numeric_columns]]
        .sort_values("open_time")
        .drop_duplicates(subset="open_time", keep="last")
        .reset_index(drop=True)
    )
    result.attrs["market_source"] = market_source
    return result


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

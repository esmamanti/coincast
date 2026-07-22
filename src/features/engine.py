from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import pandas_ta as ta


def infer_sentiment_score(headline: str) -> float:
    if not isinstance(headline, str) or not headline.strip():
        return 0.0
    text = headline.lower()
    positive = {"surge", "breakout", "bull", "buy", "gain", "rally", "adoption", "approval", "profit"}
    negative = {"drop", "crash", "bear", "sell", "fall", "panic", "hack", "loss", "scam"}
    positive_hits = sum(word in text for word in positive)
    negative_hits = sum(word in text for word in negative)
    if positive_hits + negative_hits == 0:
        return 0.0
    return float(np.clip((positive_hits - negative_hits) / (positive_hits + negative_hits), -1, 1))


def _sentiment_series(
    frame: pd.DataFrame,
    sentiment_scores: Optional[Sequence[float]],
    news_headlines: Optional[Sequence[str]],
) -> pd.Series:
    if sentiment_scores is not None:
        return pd.Series(sentiment_scores, index=frame.index, dtype=float)
    if news_headlines is not None:
        return pd.Series([infer_sentiment_score(item) for item in news_headlines], index=frame.index, dtype=float)
    return pd.Series(0.0, index=frame.index, dtype=float)


def add_features(
    df: pd.DataFrame,
    sentiment_scores: Optional[Sequence[float]] = None,
    news_headlines: Optional[Sequence[str]] = None,
    include_targets: bool = True,
) -> pd.DataFrame:
    frame = df.copy().sort_index()
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    frame["rsi_14"] = ta.rsi(frame["close"], length=14)
    frame["rsi_7"] = ta.rsi(frame["close"], length=7)
    frame["rsi_21"] = ta.rsi(frame["close"], length=21)
    frame = pd.concat([frame, ta.macd(frame["close"]), ta.bbands(frame["close"], length=20)], axis=1)
    frame["sma_20"] = ta.sma(frame["close"], length=20)
    frame["ema_12"] = ta.ema(frame["close"], length=12)
    frame["ema_26"] = ta.ema(frame["close"], length=26)
    frame["ema_diff"] = frame["ema_12"] - frame["ema_26"]
    frame["close_vs_sma20"] = frame["close"] - frame["sma_20"]
    frame["roc_12"] = ta.roc(frame["close"], length=12)
    frame["mom_5"] = ta.mom(frame["close"], length=5)
    frame["willr_14"] = ta.willr(frame["high"], frame["low"], frame["close"], length=14)
    frame["adx_14"] = ta.adx(frame["high"], frame["low"], frame["close"], length=14)["ADX_14"]
    frame = pd.concat([frame, ta.stoch(frame["high"], frame["low"], frame["close"])], axis=1)
    frame["obv"] = ta.obv(frame["close"], frame["volume"])
    frame["atr"] = ta.atr(frame["high"], frame["low"], frame["close"], length=14)

    sentiment = _sentiment_series(frame, sentiment_scores, news_headlines)
    frame["news_sentiment_score"] = sentiment.to_numpy()
    frame["sentiment_lag_1"] = frame["news_sentiment_score"].shift(1)
    for lag in [1, 2, 3, 4, 5]:
        frame[f"close_lag_{lag}"] = frame["close"].shift(lag)
    frame["close_diff_1"] = frame["close"] - frame["close"].shift(1)
    frame["close_diff_2"] = frame["close"] - frame["close"].shift(2)
    frame["rolling_mean_7"] = frame["close"].rolling(7).mean()
    frame["rolling_std_7"] = frame["close"].rolling(7).std()
    frame["rolling_mean_30"] = frame["close"].rolling(30).mean()
    frame["volume_mean_7"] = frame["volume"].rolling(7).mean()
    frame["volume_mean_30"] = frame["volume"].rolling(30).mean()

    if include_targets:
        frame["target_return"] = frame["close"].pct_change().shift(-1)
        frame["target"] = frame["close"].shift(-1)
    return frame.dropna()


def build_feature_matrix(
    df: pd.DataFrame,
    sentiment_scores: Optional[Sequence[float]] = None,
    news_headlines: Optional[Sequence[str]] = None,
    include_targets: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    enriched = add_features(df, sentiment_scores, news_headlines, include_targets=include_targets)
    excluded = {"open", "high", "low", "close", "target", "target_return"}
    return enriched, [column for column in enriched.columns if column not in excluded]


def process_all_coins(data_dir: str = "data", output_dir: str = "data_processed", sentiment_scores_by_symbol=None) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for file_name in sorted(os.listdir(data_dir)):
        if not file_name.endswith(".csv"):
            continue
        symbol = file_name.removesuffix(".csv")
        frame = pd.read_csv(os.path.join(data_dir, file_name), parse_dates=["open_time"]).set_index("open_time")
        scores = sentiment_scores_by_symbol.get(symbol) if sentiment_scores_by_symbol else None
        enriched = add_features(frame, sentiment_scores=scores)
        enriched.to_csv(os.path.join(output_dir, f"{symbol}_features.csv"))

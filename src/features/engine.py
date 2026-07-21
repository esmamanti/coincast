from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import pandas_ta as ta


def add_features(df: pd.DataFrame, sentiment_scores: Optional[Sequence[float]] = None) -> pd.DataFrame:
    feature_frame = df.copy()

    required_columns = {"open", "high", "low", "close", "volume"}
    missing = required_columns - set(feature_frame.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    feature_frame = feature_frame.sort_index()

    feature_frame["rsi_14"] = ta.rsi(feature_frame["close"], length=14)
    feature_frame["rsi_7"] = ta.rsi(feature_frame["close"], length=7)

    macd_result = ta.macd(feature_frame["close"])
    if isinstance(macd_result, pd.DataFrame):
        feature_frame["macd"] = macd_result.iloc[:, 0]
        feature_frame["macd_signal"] = macd_result.iloc[:, 1] if macd_result.shape[1] > 1 else 0.0
    else:
        feature_frame["macd"] = macd_result
        feature_frame["macd_signal"] = 0.0

    bb_result = ta.bbands(feature_frame["close"], length=20)
    if isinstance(bb_result, pd.DataFrame):
        feature_frame["bb_upper"] = bb_result.iloc[:, 0]
        feature_frame["bb_lower"] = bb_result.iloc[:, 2] if bb_result.shape[1] > 2 else 0.0
    else:
        feature_frame["bb_upper"] = 0.0
        feature_frame["bb_lower"] = 0.0
    feature_frame["sma_20"] = ta.sma(feature_frame["close"], length=20)
    feature_frame["ema_12"] = ta.ema(feature_frame["close"], length=12)
    feature_frame["ema_26"] = ta.ema(feature_frame["close"], length=26)
    feature_frame["ema_diff"] = feature_frame["ema_12"] - feature_frame["ema_26"]
    feature_frame["atr"] = ta.atr(feature_frame["high"], feature_frame["low"], feature_frame["close"], length=14)
    feature_frame["obv"] = ta.obv(feature_frame["close"], feature_frame["volume"])

    for lag in [1, 2, 3, 5]:
        feature_frame[f"close_lag_{lag}"] = feature_frame["close"].shift(lag)
        feature_frame[f"return_lag_{lag}"] = feature_frame["close"].pct_change().shift(lag)

    feature_frame["rolling_mean_7"] = feature_frame["close"].rolling(7).mean()
    feature_frame["rolling_std_7"] = feature_frame["close"].rolling(7).std()
    feature_frame["rolling_mean_30"] = feature_frame["close"].rolling(30).mean()

    if sentiment_scores is not None:
        feature_frame["news_sentiment_score"] = np.array(sentiment_scores, dtype=float)
    else:
        feature_frame["news_sentiment_score"] = 0.0

    feature_frame["hour_sin"] = np.sin(2 * np.pi * feature_frame.index.hour / 24)
    feature_frame["hour_cos"] = np.cos(2 * np.pi * feature_frame.index.hour / 24)
    feature_frame["day_of_week_sin"] = np.sin(2 * np.pi * feature_frame.index.dayofweek / 7)
    feature_frame["day_of_week_cos"] = np.cos(2 * np.pi * feature_frame.index.dayofweek / 7)

    feature_frame["target_return"] = feature_frame["close"].pct_change().shift(-1)
    feature_frame["target"] = feature_frame["close"].shift(-1)

    feature_frame = feature_frame.dropna()
    return feature_frame


def build_feature_matrix(df: pd.DataFrame, sentiment_scores: Optional[Sequence[float]] = None) -> tuple[pd.DataFrame, list[str]]:
    enriched = add_features(df, sentiment_scores=sentiment_scores)
    feature_columns = [
        col for col in enriched.columns
        if col not in {"open", "high", "low", "close", "target", "target_return"}
    ]
    return enriched, feature_columns

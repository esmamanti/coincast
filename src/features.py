import os
from typing import Optional, Sequence

import pandas as pd
import pandas_ta as ta


def infer_sentiment_score(headline: str) -> float:
    """Very lightweight sentiment heuristic for news headlines."""
    if not isinstance(headline, str) or not headline.strip():
        return 0.0

    text = headline.lower()
    positive_words = {
        "surge", "breakout", "bull", "buy", "gain", "up", "rally", "boost", "adoption", "launch",
        "approval", "positive", "win", "strong", "profit"
    }
    negative_words = {
        "drop", "crash", "bear", "sell", "fall", "down", "panic", "hack", "warning", "negative",
        "loss", "weak", "reject", "delay", "regulation", "scam"
    }

    positive_hits = sum(1 for word in positive_words if word in text)
    negative_hits = sum(1 for word in negative_words if word in text)

    if positive_hits == 0 and negative_hits == 0:
        return 0.0

    score = (positive_hits - negative_hits) / max(positive_hits + negative_hits, 1)
    return max(-1.0, min(1.0, score))


def _build_sentiment_series(df: pd.DataFrame, sentiment_scores: Optional[Sequence[float]] = None, news_headlines: Optional[Sequence[str]] = None) -> pd.Series:
    if sentiment_scores is not None:
        return pd.Series(sentiment_scores, index=df.index, dtype=float)

    if news_headlines is not None:
        scores = [infer_sentiment_score(item) for item in news_headlines]
        return pd.Series(scores, index=df.index, dtype=float)

    return pd.Series(0.0, index=df.index, dtype=float)


def add_features(df: pd.DataFrame, sentiment_scores: Optional[Sequence[float]] = None, news_headlines: Optional[Sequence[str]] = None) -> pd.DataFrame:
    df = df.copy()

    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    # Teknik indikatörler
    df["rsi_14"] = ta.rsi(df["close"], length=14)
    df["rsi_7"] = ta.rsi(df["close"], length=7)
    df["rsi_21"] = ta.rsi(df["close"], length=21)

    macd = ta.macd(df["close"])
    df = pd.concat([df, macd], axis=1)

    bbands = ta.bbands(df["close"], length=20)
    df = pd.concat([df, bbands], axis=1)

    df["sma_20"] = ta.sma(df["close"], length=20)
    df["ema_12"] = ta.ema(df["close"], length=12)
    df["ema_26"] = ta.ema(df["close"], length=26)
    df["ema_diff"] = df["ema_12"] - df["ema_26"]
    df["close_vs_sma20"] = df["close"] - df["sma_20"]

    df["roc_12"] = ta.roc(df["close"], length=12)
    df["mom_5"] = ta.mom(df["close"], length=5)
    df["willr_14"] = ta.willr(df["high"], df["low"], df["close"], length=14)
    df["adx_14"] = ta.adx(df["high"], df["low"], df["close"], length=14)["ADX_14"]

    stoch = ta.stoch(df["high"], df["low"], df["close"])
    df = pd.concat([df, stoch], axis=1)

    df["obv"] = ta.obv(df["close"], df["volume"])
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    # Duygu bilgisi
    sentiment_series = _build_sentiment_series(df, sentiment_scores=sentiment_scores, news_headlines=news_headlines)
    df["news_sentiment_score"] = sentiment_series.to_numpy()
    df["sentiment_lag_1"] = df["news_sentiment_score"].shift(1)

    # Lag features (geçmiş fiyat bilgisi)
    for lag in [1, 2, 3, 4, 5]:
        df[f"close_lag_{lag}"] = df["close"].shift(lag)
    df["close_diff_1"] = df["close"] - df["close"].shift(1)
    df["close_diff_2"] = df["close"] - df["close"].shift(2)

    # Rolling istatistikler
    df["rolling_mean_7"] = df["close"].rolling(7).mean()
    df["rolling_std_7"] = df["close"].rolling(7).std()
    df["rolling_mean_30"] = df["close"].rolling(30).mean()
    df["volume_mean_7"] = df["volume"].rolling(7).mean()
    df["volume_mean_30"] = df["volume"].rolling(30).mean()

    # Hedef değişken: bir sonraki mumun getirisi (% değişim), ham fiyat DEĞİL.
    df["target_return"] = df["close"].pct_change().shift(-1)
    df["target"] = df["close"].shift(-1)

    df.dropna(inplace=True)
    return df


def process_all_coins(data_dir="data", output_dir="data_processed", sentiment_scores_by_symbol=None):
    os.makedirs(output_dir, exist_ok=True)
    for file in sorted(os.listdir(data_dir)):
        if not file.endswith(".csv"):
            continue
        symbol = file.replace(".csv", "")
        df = pd.read_csv(f"{data_dir}/{file}", parse_dates=["open_time"])
        df.set_index("open_time", inplace=True)
        try:
            sentiment_scores = None
            if sentiment_scores_by_symbol is not None:
                sentiment_scores = sentiment_scores_by_symbol.get(symbol)
            df_feat = add_features(df, sentiment_scores=sentiment_scores)
            df_feat.to_csv(f"{output_dir}/{symbol}_features.csv")
            print(f"OK: {symbol} -> {df_feat.shape[0]} satır, {df_feat.shape[1]} kolon")
        except Exception as e:
            print(f"HATA: {symbol} atlandı -> {e}")


if __name__ == "__main__":
    process_all_coins()
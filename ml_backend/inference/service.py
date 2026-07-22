from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import joblib
import pandas as pd

from src.data.ingest import fetch_binance_klines, normalize_symbol
from src.features import add_features


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR = ROOT / "models_saved"
SUPPORTED_HORIZONS = {1, 4, 24}


class ModelNotFoundError(FileNotFoundError):
    pass


class StaleMarketDataError(RuntimeError):
    pass


class InferenceService:
    def __init__(
        self,
        models_dir: str | Path | None = None,
        market_fetcher: Callable[..., pd.DataFrame] | None = None,
        enforce_freshness: bool = True,
    ) -> None:
        self.models_dir = Path(models_dir or DEFAULT_MODELS_DIR)
        self.market_fetcher = market_fetcher or fetch_binance_klines
        self.enforce_freshness = enforce_freshness
        self._artifacts: dict[tuple[str, int], tuple[object, dict]] = {}
        self._latest_market_frames: dict[tuple[str, str], pd.DataFrame] = {}

    def _artifact_paths(self, symbol: str, horizon: int) -> tuple[Path, Path]:
        stem = f"xgb_{symbol.lower()}_h{horizon}"
        return (
            self.models_dir / f"{stem}_return.pkl",
            self.models_dir / f"{stem}_metadata.json",
        )

    def _load_artifacts(self, symbol: str, horizon: int) -> tuple[object, dict]:
        key = (symbol, horizon)
        if key in self._artifacts:
            return self._artifacts[key]

        model_path, metadata_path = self._artifact_paths(symbol, horizon)
        if not model_path.exists() or not metadata_path.exists():
            raise ModelNotFoundError(
                f"No trained model for {symbol} horizon={horizon}. "
                "Run src/train_validated_models.py first."
            )

        model = joblib.load(model_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("symbol") != symbol or int(metadata.get("horizon", -1)) != horizon:
            raise ValueError(f"Model metadata does not match {symbol} horizon={horizon}")
        self._artifacts[key] = (model, metadata)
        return model, metadata

    @staticmethod
    def _interval_delta(interval: str) -> timedelta:
        if interval == "1h":
            return timedelta(hours=1)
        raise ValueError(f"Unsupported inference interval: {interval}")

    @staticmethod
    def _confidence_interval(predicted_return: float, metadata: dict) -> dict[str, float]:
        lower = predicted_return + float(metadata["residual_q05"])
        upper = predicted_return + float(metadata["residual_q95"])
        return {"lower": round(lower, 8), "upper": round(upper, 8)}

    def predict(self, symbol: str, horizon: int = 1) -> dict:
        normalized_symbol = normalize_symbol(symbol)
        if horizon not in SUPPORTED_HORIZONS:
            raise ValueError(f"Supported horizons are {sorted(SUPPORTED_HORIZONS)}")

        model, metadata = self._load_artifacts(normalized_symbol, horizon)
        interval = str(metadata.get("interval", "1h"))
        candles = self.market_fetcher(normalized_symbol, interval=interval, limit=300)
        if candles.empty or len(candles) < 60:
            raise RuntimeError(f"Not enough closed market candles for {normalized_symbol}")

        market_source = candles.attrs.get("market_source", "binance_closed_candles")
        candles = candles.sort_values("open_time").drop_duplicates("open_time", keep="last")
        self._latest_market_frames[(normalized_symbol, interval)] = candles.copy()
        feature_input = candles.set_index("open_time")
        features = add_features(feature_input, include_targets=False)
        feature_columns = list(metadata["feature_columns"])
        missing = sorted(set(feature_columns) - set(features.columns))
        if missing:
            raise RuntimeError(f"Live feature pipeline is missing columns: {missing}")

        latest = features.iloc[[-1]]
        model_input = latest[feature_columns]
        predicted_return = float(model.predict(model_input)[0])
        current_price = float(latest["close"].iloc[0])
        confidence_interval = self._confidence_interval(predicted_return, metadata)
        predicted_price = current_price * (1.0 + predicted_return)
        predicted_price_interval = {
            "lower": current_price * (1.0 + confidence_interval["lower"]),
            "upper": current_price * (1.0 + confidence_interval["upper"]),
        }
        candle_close_time = pd.Timestamp(latest.index[-1]).to_pydatetime() + self._interval_delta(interval)
        if candle_close_time.tzinfo is None:
            candle_close_time = candle_close_time.replace(tzinfo=timezone.utc)
        data_age_seconds = max(0.0, (datetime.now(timezone.utc) - candle_close_time).total_seconds())
        if self.enforce_freshness and data_age_seconds > self._interval_delta(interval).total_seconds() * 2.5:
            raise StaleMarketDataError(
                f"Latest closed candle for {normalized_symbol} is stale ({data_age_seconds:.0f}s old)"
            )

        return {
            "symbol": normalized_symbol,
            "horizon": horizon,
            "interval": interval,
            "current_price": current_price,
            "predicted_price": predicted_price,
            "predicted_price_interval": predicted_price_interval,
            "predicted_return": predicted_return,
            "confidence_interval": confidence_interval,
            "mini_chart": [float(value) for value in candles["close"].tail(12).tolist()],
            "data_timestamp": candle_close_time.isoformat(),
            "data_age_seconds": round(data_age_seconds, 3),
            "model_id": metadata["model_id"],
            "model_verified": bool(metadata.get("verified", False)),
            "signal_threshold": float(metadata["signal_threshold"]),
            "quality_metrics": metadata.get("test", {}),
            "source": f"{market_source}+xgboost",
        }

    def latest_market_frame(self, symbol: str, interval: str = "1h") -> pd.DataFrame | None:
        frame = self._latest_market_frames.get((normalize_symbol(symbol), interval))
        return None if frame is None else frame.copy()

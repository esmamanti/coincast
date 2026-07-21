from __future__ import annotations

import joblib
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = ROOT / "models_saved" / "xgb_ethusdt_return.pkl"
DEFAULT_DATA_PATH = ROOT / "data_processed" / "ETHUSDT_features.csv"


class InferenceService:
    def __init__(self, model_path: str | Path | None = None, data_path: str | Path | None = None) -> None:
        self.model_path = Path(model_path or DEFAULT_MODEL_PATH)
        self.data_path = Path(data_path or DEFAULT_DATA_PATH)
        self.model = None
        self.feature_frame = None
        self.feature_columns = []
        self.active_symbol = None
        self.load()

    def _resolve_artifacts(self, symbol: str) -> tuple[Path, Path]:
        normalized_symbol = symbol.upper()
        candidate_model = ROOT / "models_saved" / f"xgb_{normalized_symbol.lower()}_return.pkl"
        candidate_data = ROOT / "data_processed" / f"{normalized_symbol}_features.csv"

        if candidate_model.exists() and candidate_data.exists():
            return candidate_model, candidate_data

        return self.model_path, self.data_path

    def load(self, symbol: str | None = None) -> None:
        target_symbol = (symbol or "ETHUSDT").upper()
        model_path, data_path = self._resolve_artifacts(target_symbol)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        if not data_path.exists():
            raise FileNotFoundError(f"Feature data not found at {data_path}")

        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.model = joblib.load(self.model_path)
        self.feature_frame = pd.read_csv(self.data_path, parse_dates=["open_time"]).set_index("open_time")
        self.feature_columns = [
            col for col in self.feature_frame.columns
            if col not in {"open", "high", "low", "close", "target", "target_return"}
        ]
        self.active_symbol = target_symbol

    def _confidence_interval(self, predicted_return: float) -> dict[str, float]:
        width = max(0.01, abs(predicted_return) * 0.25 + 0.01)
        return {
            "lower": round(predicted_return - width, 6),
            "upper": round(predicted_return + width, 6),
        }

    def predict(self, symbol: str, horizon: int = 1) -> dict:
        if self.model is None or self.feature_frame is None or self.active_symbol != symbol.upper():
            self.load(symbol)

        latest_row = self.feature_frame.iloc[[-1]][self.feature_columns]
        predicted_return = float(self.model.predict(latest_row)[0])
        return {
            "symbol": symbol.upper(),
            "horizon": horizon,
            "predicted_return": predicted_return,
            "confidence_interval": self._confidence_interval(predicted_return),
            "source": "xgb_model",
        }

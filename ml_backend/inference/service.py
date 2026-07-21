from __future__ import annotations

import joblib
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models_saved" / "xgb_ethusdt_return.pkl"
DATA_PATH = ROOT / "data_processed" / "ETHUSDT_features.csv"


class InferenceService:
    def __init__(self, model_path: str | Path | None = None, data_path: str | Path | None = None) -> None:
        self.model_path = Path(model_path or MODEL_PATH)
        self.data_path = Path(data_path or DATA_PATH)
        self.model = None
        self.feature_frame = None
        self.feature_columns = []
        self.load()

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        if not self.data_path.exists():
            raise FileNotFoundError(f"Feature data not found at {self.data_path}")

        self.model = joblib.load(self.model_path)
        self.feature_frame = pd.read_csv(self.data_path, parse_dates=["open_time"]).set_index("open_time")
        self.feature_columns = [
            col for col in self.feature_frame.columns
            if col not in {"open", "high", "low", "close", "target", "target_return"}
        ]

    def predict(self, symbol: str, horizon: int = 1) -> dict:
        latest_row = self.feature_frame.iloc[[-1]][self.feature_columns]
        predicted_return = float(self.model.predict(latest_row)[0])
        return {
            "symbol": symbol.upper(),
            "horizon": horizon,
            "predicted_return": predicted_return,
            "source": "xgb_model",
        }

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml_backend.api.routes import CACHE, inference_service
from ml_backend.main import app


def recent_candles(rows: int = 120) -> pd.DataFrame:
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    index = pd.date_range(end=end, periods=rows, freq="h")
    trend = np.linspace(100.0, 112.0, rows)
    wave = np.sin(np.arange(rows) / 4.0)
    close = trend + wave
    return pd.DataFrame(
        {
            "open_time": index,
            "open": close - 0.2,
            "high": close + 0.8,
            "low": close - 0.8,
            "close": close,
            "volume": 1000 + np.arange(rows) * 3,
        }
    )


class BackendPredictTests(unittest.TestCase):
    def setUp(self):
        CACHE.clear()
        self.frame = recent_candles()
        self.original_fetcher = inference_service.market_fetcher
        inference_service.market_fetcher = lambda *args, **kwargs: self.frame.copy()
        self.client = TestClient(app)

    def tearDown(self):
        inference_service.market_fetcher = self.original_fetcher
        CACHE.clear()

    def test_predict_endpoint_uses_symbol_and_horizon_specific_model(self):
        response = self.client.post("/predict", json={"symbol": "BTCUSDT", "horizon": 4})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["symbol"], "BTCUSDT")
        self.assertEqual(payload["horizon"], 4)
        self.assertIn("btcusdt-h4", payload["model_id"])
        self.assertIn("data_timestamp", payload)
        self.assertIn("model_verified", payload)
        self.assertAlmostEqual(
            payload["predicted_price"],
            payload["current_price"] * (1 + payload["predicted_return"]),
            places=6,
        )
        self.assertIn("lower", payload["predicted_price_interval"])
        self.assertIn("upper", payload["predicted_price_interval"])

    def test_predict_endpoint_includes_recent_mini_chart(self):
        response = self.client.post("/predict", json={"symbol": "BTCUSDT", "horizon": 1})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["mini_chart"], [float(value) for value in self.frame["close"].tail(12)])

    def test_invalid_horizon_is_rejected(self):
        response = self.client.post("/predict", json={"symbol": "BTCUSDT", "horizon": 2})
        self.assertEqual(response.status_code, 422)

    def test_all_performance_endpoint_returns_every_coin(self):
        response = self.client.get("/performance/all?horizon=1&limit=1")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["horizon"], 1)
        self.assertEqual(len(payload["coins"]), 13)


if __name__ == "__main__":
    unittest.main()

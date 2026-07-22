import sys
import unittest
import uuid
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.trading.prediction_tracker import PredictionTracker


class PredictionTrackerTests(unittest.TestCase):
    def setUp(self):
        self.db_path = ROOT / "tests" / f"prediction_{uuid.uuid4().hex}.sqlite3"
        self.tracker = PredictionTracker(self.db_path)

    def tearDown(self):
        self.db_path.unlink(missing_ok=True)

    def test_forecast_is_deduplicated_and_scored_after_target_candle(self):
        prediction = {
            "symbol": "BTCUSDT",
            "horizon": 1,
            "interval": "1h",
            "model_id": "test-model",
            "data_timestamp": "2026-07-21T10:00:00+00:00",
            "current_price": 100.0,
            "predicted_price": 102.0,
            "predicted_return": 0.02,
            "predicted_price_interval": {"lower": 99.0, "upper": 104.0},
        }
        self.assertTrue(self.tracker.record(prediction, "BUY"))
        self.assertFalse(self.tracker.record(prediction, "BUY"))

        pending = self.tracker.performance("BTCUSDT", 1)
        self.assertEqual(pending["total_predictions"], 1)
        self.assertEqual(pending["pending_predictions"], 1)

        candles = pd.DataFrame(
            {
                "open_time": pd.to_datetime(
                    ["2026-07-21T09:00:00Z", "2026-07-21T10:00:00Z"], utc=True
                ),
                "close": [100.0, 103.0],
            }
        )
        self.assertEqual(self.tracker.resolve_with_candles("BTCUSDT", "1h", candles), 1)

        result = self.tracker.performance("BTCUSDT", 1)
        self.assertEqual(result["resolved_predictions"], 1)
        self.assertEqual(result["direction_accuracy"], 1.0)
        self.assertEqual(result["mae"], 1.0)
        self.assertEqual(result["naive_mae"], 3.0)
        self.assertEqual(result["price_improvement_ratio"], 3.0)
        self.assertEqual(result["interval_coverage"], 1.0)
        self.assertEqual(result["recent"][0]["actual_price"], 103.0)


if __name__ == "__main__":
    unittest.main()

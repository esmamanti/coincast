import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.market_config import COINCAST_HORIZONS, COINCAST_SYMBOLS
from src.run_forecast_tracker import run_tracking_cycle


class FakeTrackingService:
    def __init__(self):
        self.calls = []

    def track_prediction(self, symbol: str, horizon: int) -> dict:
        self.calls.append((symbol, horizon))
        return {
            "action": "HOLD",
            "performance": {"resolved_predictions": 0, "pending_predictions": 1},
        }


class ForecastTrackerTests(unittest.TestCase):
    def test_cycle_tracks_every_coin_and_horizon(self):
        service = FakeTrackingService()
        summary = run_tracking_cycle(service, COINCAST_SYMBOLS, COINCAST_HORIZONS)
        self.assertEqual(summary["attempted"], 39)
        self.assertEqual(summary["succeeded"], 39)
        self.assertEqual(len(service.calls), 39)
        self.assertEqual(set(symbol for symbol, _ in service.calls), set(COINCAST_SYMBOLS))
        self.assertEqual(set(horizon for _, horizon in service.calls), set(COINCAST_HORIZONS))


if __name__ == "__main__":
    unittest.main()

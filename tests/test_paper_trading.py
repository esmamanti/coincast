import sys
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.trading.paper_broker import PaperBroker
from src.trading.prediction_tracker import PredictionTracker
from src.trading.service import TradingService


class FakeInference:
    def __init__(self):
        self.predicted_return = 0.01

    def predict(self, symbol: str, horizon: int = 1) -> dict:
        return {
            "symbol": symbol,
            "horizon": horizon,
            "interval": "1h",
            "current_price": 100.0,
            "predicted_price": 101.0,
            "predicted_price_interval": {"lower": 99.0, "upper": 102.0},
            "predicted_return": self.predicted_return,
            "confidence_interval": {"lower": -0.01, "upper": 0.02},
            "mini_chart": [99.0, 100.0],
            "data_timestamp": "2026-07-21T10:00:00+00:00",
            "data_age_seconds": 0.0,
            "model_id": "test-model",
            "model_verified": False,
            "signal_threshold": 0.003,
            "quality_metrics": {},
            "source": "test",
        }

    def latest_market_frame(self, symbol: str, interval: str = "1h"):
        return None


class FakeNotifier:
    def __init__(self):
        self.messages = []

    def send_trade_report(self, subject: str, message: str) -> list[dict]:
        self.messages.append((subject, message))
        return [{"channel": "test", "sent": True}]


class PaperTradingTests(unittest.TestCase):
    def setUp(self):
        self.db_path = ROOT / "tests" / f"paper_{uuid.uuid4().hex}.sqlite3"
        self.tracking_db_path = ROOT / "tests" / f"tracking_{uuid.uuid4().hex}.sqlite3"
        self.broker = PaperBroker(self.db_path, initial_cash=10_000)
        self.inference = FakeInference()
        self.notifier = FakeNotifier()
        self.tracker = PredictionTracker(self.tracking_db_path)
        self.service = TradingService(
            inference=self.inference,
            broker=self.broker,
            notifier=self.notifier,
            tracker=self.tracker,
        )

    def tearDown(self):
        self.db_path.unlink(missing_ok=True)
        self.tracking_db_path.unlink(missing_ok=True)

    def test_buy_then_sell_cycle_persists_trades_and_reports(self):
        buy = self.service.run_paper_cycle("BTCUSDT", horizon=1)
        self.assertEqual(buy["status"], "EXECUTED")
        self.assertEqual(buy["action"], "BUY")
        self.assertEqual(len(buy["account"]["positions"]), 1)

        self.inference.predicted_return = -0.01
        sell = self.service.run_paper_cycle("BTCUSDT", horizon=1)
        self.assertEqual(sell["status"], "EXECUTED")
        self.assertEqual(sell["action"], "SELL")
        self.assertEqual(len(sell["account"]["positions"]), 0)
        self.assertEqual(len(self.broker.recent_trades()), 2)
        self.assertEqual(len(self.notifier.messages), 2)

    def test_second_buy_is_blocked_by_position_limit(self):
        self.service.run_paper_cycle("BTCUSDT", horizon=1)
        second = self.service.run_paper_cycle("BTCUSDT", horizon=1)
        self.assertEqual(second["status"], "NO_TRADE")
        self.assertIn("pozisyon", second["reason"].lower())


if __name__ == "__main__":
    unittest.main()

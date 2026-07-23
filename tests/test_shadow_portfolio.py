import sys
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.trading.shadow_portfolio import ShadowPortfolioBroker


class ShadowPortfolioTests(unittest.TestCase):
    def setUp(self):
        self.db_path = ROOT / "tests" / f"shadow_portfolio_{uuid.uuid4().hex}.sqlite3"
        self.broker = ShadowPortfolioBroker(self.db_path, initial_cash=10_000, position_fraction=0.10)

    def tearDown(self):
        self.db_path.unlink(missing_ok=True)

    @staticmethod
    def prediction(timestamp: str, price: float, predicted_return: float) -> dict:
        return {
            "symbol": "BTCUSDT",
            "horizon": 1,
            "current_price": price,
            "predicted_return": predicted_return,
            "data_timestamp": timestamp,
            "model_id": "test-model",
        }

    def test_buy_mark_sell_cycle_is_independent_and_cost_aware(self):
        buy_prediction = self.prediction("2026-07-21T10:00:00+00:00", 100.0, 0.01)
        buy = self.broker.process_signal(buy_prediction, "BUY")
        self.assertTrue(buy["processed"])
        self.assertEqual(buy["outcome"], "OPENED")
        self.assertEqual(buy["portfolio"]["trade_count"], 1)
        self.assertIsNotNone(buy["portfolio"]["open_position"])
        self.assertLess(buy["portfolio"]["net_return"], 0)

        duplicate = self.broker.process_signal(buy_prediction, "BUY")
        self.assertFalse(duplicate["processed"])
        self.assertEqual(duplicate["portfolio"]["trade_count"], 1)

        hold = self.broker.process_signal(
            self.prediction("2026-07-21T11:00:00+00:00", 110.0, 0.0), "HOLD"
        )
        self.assertGreater(hold["portfolio"]["net_return"], 0)

        sell = self.broker.process_signal(
            self.prediction("2026-07-21T12:00:00+00:00", 110.0, -0.01), "SELL"
        )
        portfolio = sell["portfolio"]
        self.assertEqual(sell["outcome"], "CLOSED")
        self.assertEqual(portfolio["trade_count"], 2)
        self.assertEqual(portfolio["closed_trades"], 1)
        self.assertEqual(portfolio["winning_trades"], 1)
        self.assertEqual(portfolio["win_rate"], 1.0)
        self.assertGreater(portfolio["net_return"], 0)
        self.assertIsNone(portfolio["open_position"])

    def test_each_horizon_has_a_separate_account(self):
        prediction = self.prediction("2026-07-21T10:00:00+00:00", 100.0, 0.01)
        self.broker.process_signal(prediction, "BUY")
        h4 = self.broker.snapshot("BTCUSDT", 4)
        self.assertEqual(h4["equity"], 10_000)
        self.assertEqual(h4["trade_count"], 0)


if __name__ == "__main__":
    unittest.main()

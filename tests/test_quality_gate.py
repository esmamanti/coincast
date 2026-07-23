import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.trading.quality_gate import ModelQualityGate


class QualityGateTests(unittest.TestCase):
    def setUp(self):
        self.gate = ModelQualityGate()

    @staticmethod
    def performance(resolved: int, direction: float = 0.56, improvement: float = 1.06) -> dict:
        return {
            "resolved_predictions": resolved,
            "direction_accuracy": direction,
            "price_improvement_ratio": improvement,
        }

    @staticmethod
    def shadow(closed: int = 30, net_return: float = 0.02, drawdown: float = -0.05) -> dict:
        return {"closed_trades": closed, "net_return": net_return, "max_drawdown": drawdown}

    def test_insufficient_data_has_priority(self):
        result = self.gate.evaluate(self.performance(119), self.shadow())
        self.assertEqual(result["status"], "INSUFFICIENT_DATA")
        self.assertAlmostEqual(result["sample_progress"], 119 / 200)

    def test_candidate_requires_every_gate(self):
        result = self.gate.evaluate(self.performance(200), self.shadow())
        self.assertEqual(result["status"], "CANDIDATE")
        self.assertTrue(all(result["checks"].values()))

    def test_failed_metric_rejects_mature_strategy(self):
        result = self.gate.evaluate(
            self.performance(200, direction=0.49, improvement=0.95),
            self.shadow(net_return=-0.01, drawdown=-0.12),
        )
        self.assertEqual(result["status"], "REJECTED")
        self.assertIn("direction_accuracy", result["failed_checks"])
        self.assertIn("max_drawdown", result["failed_checks"])

    def test_mature_strategy_can_still_need_more_trades(self):
        result = self.gate.evaluate(self.performance(200), self.shadow(closed=4))
        self.assertEqual(result["status"], "INSUFFICIENT_TRADES")


if __name__ == "__main__":
    unittest.main()

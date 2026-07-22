import sys
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.trading.daily_report import DailyReportService
from src.trading.paper_broker import PaperBroker
from src.trading.prediction_tracker import PredictionTracker


class FakeChannel:
    configured = True


class FakeNotifier:
    def __init__(self):
        self.channels = [FakeChannel()]
        self.messages = []

    def send_trade_report(self, subject: str, message: str) -> list[dict]:
        self.messages.append((subject, message))
        return [{"channel": "test", "sent": True}]


class DailyReportTests(unittest.TestCase):
    def setUp(self):
        token = uuid.uuid4().hex
        self.paper_path = ROOT / "tests" / f"report_paper_{token}.sqlite3"
        self.tracking_path = ROOT / "tests" / f"report_tracking_{token}.sqlite3"
        self.notifier = FakeNotifier()
        self.reporter = DailyReportService(
            PaperBroker(self.paper_path), PredictionTracker(self.tracking_path), self.notifier
        )

    def tearDown(self):
        self.paper_path.unlink(missing_ok=True)
        self.tracking_path.unlink(missing_ok=True)

    def test_builds_safe_paper_summary_without_sending(self):
        result = self.reporter.build(datetime(2026, 7, 21, 12, tzinfo=timezone.utc))
        self.assertIn("CoinCast günlük paper raporu", result["report"])
        self.assertIn("Gerçek para işlemi kapalıdır", result["report"])
        self.assertEqual(len(result["horizons"]), 3)
        self.assertEqual(self.notifier.messages, [])

    def test_send_uses_configured_notifier(self):
        result = self.reporter.send(datetime(2026, 7, 21, 12, tzinfo=timezone.utc))
        self.assertEqual(result["notifications"][0]["sent"], True)
        self.assertEqual(len(self.notifier.messages), 1)


if __name__ == "__main__":
    unittest.main()

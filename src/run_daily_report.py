from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from src.trading.daily_report import DailyReportService
from src.trading.service import ROOT, TradingService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and send the CoinCast daily paper report")
    parser.add_argument("--preview", action="store_true", help="Generate the report without sending notifications")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(ROOT / ".env")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    trading = TradingService()
    reporter = DailyReportService(trading.broker, trading.tracker, trading.notifier)
    result = reporter.build() if args.preview else reporter.send()

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "daily_report_latest.txt").write_text(result["report"], encoding="utf-8")
    timestamp = datetime.fromisoformat(result["generated_at"]).strftime("%Y%m%dT%H%M%SZ")
    (results_dir / f"daily_report_{timestamp}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(result["report"])
    if "notifications" in result:
        print(json.dumps(result["notifications"], ensure_ascii=False))


if __name__ == "__main__":
    main()

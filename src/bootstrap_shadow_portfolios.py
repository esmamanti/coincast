from __future__ import annotations

import json

from dotenv import load_dotenv

from src.market_config import COINCAST_HORIZONS, COINCAST_SYMBOLS
from src.trading.service import ROOT, TradingService


def main() -> None:
    load_dotenv(ROOT / ".env")
    service = TradingService()
    summary = service.backfill_shadow_portfolios(COINCAST_SYMBOLS, COINCAST_HORIZONS)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

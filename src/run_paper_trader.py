from __future__ import annotations

import argparse
import logging
import time

from dotenv import load_dotenv

from src.trading.service import TradingService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CoinCast automatic paper-trading cycles")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--horizon", type=int, choices=[1, 4, 24], default=1)
    parser.add_argument("--every-seconds", type=int, default=3600)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    service = TradingService()
    while True:
        for symbol in args.symbols:
            try:
                result = service.run_paper_cycle(symbol, horizon=args.horizon)
                logging.info(result["report"])
            except Exception:
                logging.exception("Paper cycle failed for %s", symbol)
        if args.once:
            return
        time.sleep(max(60, args.every_seconds))


if __name__ == "__main__":
    main()

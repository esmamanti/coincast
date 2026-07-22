from __future__ import annotations

import argparse
import ctypes
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from src.market_config import COINCAST_HORIZONS, COINCAST_SYMBOLS
from src.trading.service import TradingService


LOGGER = logging.getLogger("coincast_forecast_tracker")
ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track and score forecasts for every CoinCast coin")
    parser.add_argument("--symbols", nargs="+", default=COINCAST_SYMBOLS)
    parser.add_argument("--horizons", nargs="+", type=int, choices=COINCAST_HORIZONS, default=COINCAST_HORIZONS)
    parser.add_argument("--every-seconds", type=int, default=3600)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--pid-file", default=str(ROOT / "data" / "forecast_tracker.pid"))
    return parser.parse_args()


def run_tracking_cycle(service: TradingService, symbols: list[str], horizons: list[int]) -> dict:
    succeeded = 0
    failures: list[dict[str, object]] = []
    for symbol in symbols:
        for horizon in horizons:
            try:
                result = service.track_prediction(symbol, horizon=horizon)
                performance = result["performance"]
                LOGGER.info(
                    "%s h=%s action=%s resolved=%s pending=%s",
                    symbol,
                    horizon,
                    result["action"],
                    performance["resolved_predictions"],
                    performance["pending_predictions"],
                )
                succeeded += 1
            except Exception as exc:
                LOGGER.exception("Forecast tracking failed for %s h=%s", symbol, horizon)
                failures.append({"symbol": symbol, "horizon": horizon, "error": str(exc)})
    return {"attempted": len(symbols) * len(horizons), "succeeded": succeeded, "failures": failures}


def main() -> None:
    args = parse_args()
    load_dotenv(ROOT / ".env")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    service = TradingService()
    pid_path = None if args.once else Path(args.pid_file)
    if pid_path is not None:
        _claim_pid_file(pid_path)
    try:
        while True:
            started = time.monotonic()
            summary = run_tracking_cycle(service, args.symbols, args.horizons)
            LOGGER.info("Tracking cycle complete: %s", summary)
            if args.once:
                return
            elapsed = time.monotonic() - started
            time.sleep(max(60.0, float(args.every_seconds) - elapsed))
    finally:
        if pid_path is not None and pid_path.exists() and pid_path.read_text(encoding="utf-8").strip() == str(os.getpid()):
            pid_path.unlink(missing_ok=True)


def _claim_pid_file(pid_path: Path) -> None:
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    if pid_path.exists():
        try:
            existing_pid = int(pid_path.read_text(encoding="utf-8").strip())
        except ValueError:
            pid_path.unlink(missing_ok=True)
        else:
            if _process_alive(existing_pid):
                raise SystemExit(f"Forecast tracker is already running with PID {existing_pid}")
            pid_path.unlink(missing_ok=True)
    pid_path.write_text(str(os.getpid()), encoding="utf-8")


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


if __name__ == "__main__":
    main()

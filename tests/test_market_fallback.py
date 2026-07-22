import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.ingest import fetch_binance_klines


def kline(open_time: int, close: str) -> list:
    return [
        open_time,
        close,
        close,
        close,
        close,
        "1",
        open_time + 3_599_999,
        "1",
        1,
        "1",
        "1",
        "0",
    ]


class InvalidSpotSymbol(Exception):
    code = -1121


class FakeClient:
    def get_klines(self, **kwargs):
        raise InvalidSpotSymbol("invalid spot symbol")

    def futures_klines(self, **kwargs):
        return [kline(1_700_000_000_000, "100"), kline(1_700_003_600_000, "101")]


class MarketFallbackTests(unittest.TestCase):
    def test_invalid_spot_symbol_falls_back_to_futures_candles(self):
        with patch("binance.client.Client", return_value=FakeClient()):
            frame = fetch_binance_klines("HYPEUSDT", interval="1h", limit=2, closed_only=False)
        self.assertEqual(frame["close"].tolist(), [100.0, 101.0])
        self.assertEqual(frame.attrs["market_source"], "binance_futures_closed_candles")


if __name__ == "__main__":
    unittest.main()

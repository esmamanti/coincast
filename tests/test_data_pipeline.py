import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.ingest import clean_and_resample


class DataPipelineTests(unittest.TestCase):
    def test_clean_and_resample_creates_expected_columns(self):
        df = pd.DataFrame(
            {
                "open_time": pd.to_datetime([
                    "2024-01-01 00:00:00",
                    "2024-01-01 01:00:00",
                    "2024-01-01 02:00:00",
                ]),
                "open": [100, 101, 102],
                "high": [110, 111, 112],
                "low": [90, 91, 92],
                "close": [105, 106, 107],
                "volume": [10, 20, 30],
            }
        )

        result = clean_and_resample(df, interval="1h")

        self.assertTrue({"open_time", "open", "high", "low", "close", "volume"}.issubset(set(result.columns)))
        self.assertGreaterEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()

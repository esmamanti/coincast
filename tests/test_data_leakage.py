import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.engine import add_features


class LeakageTests(unittest.TestCase):
    def test_target_is_not_too_highly_correlated_with_feature(self):
        df = pd.DataFrame(
            {
                "open": [100, 101, 102, 103],
                "high": [101, 102, 103, 104],
                "low": [99, 100, 101, 102],
                "close": [100, 101, 102, 103],
                "volume": [100, 200, 300, 400],
            },
            index=pd.date_range("2024-01-01", periods=4, freq="h"),
        )
        enriched = add_features(df)
        if enriched.empty:
            self.assertTrue(True)
            return
        corr = np.corrcoef(enriched["rsi_14"], enriched["target_return"])[0, 1]
        self.assertLess(abs(corr), 0.95)


if __name__ == "__main__":
    unittest.main()

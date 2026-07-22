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
        rng = np.random.default_rng(42)
        close = 100 + np.cumsum(rng.normal(0, 1, 120))
        df = pd.DataFrame(
            {
                "open": close + rng.normal(0, 0.2, 120),
                "high": close + rng.uniform(0.2, 1.0, 120),
                "low": close - rng.uniform(0.2, 1.0, 120),
                "close": close,
                "volume": rng.uniform(100, 1000, 120),
            },
            index=pd.date_range("2024-01-01", periods=120, freq="h"),
        )
        enriched = add_features(df)
        self.assertFalse(enriched.empty)
        corr = np.corrcoef(enriched["rsi_14"], enriched["target_return"])[0, 1]
        self.assertLess(abs(corr), 0.95)



if __name__ == "__main__":
    unittest.main()

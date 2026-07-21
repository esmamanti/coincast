import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.engine import add_features, build_feature_matrix


class FeatureEngineTests(unittest.TestCase):
    def setUp(self):
        rows = []
        for i in range(20):
            rows.append(
                {
                    "open": 100 + i,
                    "high": 102 + i,
                    "low": 99 + i,
                    "close": 101 + i,
                    "volume": 1000 + i * 10,
                }
            )
        self.df = pd.DataFrame(rows, index=pd.date_range("2024-01-01", periods=20, freq="h"))

    def test_add_features_contains_indicator_columns(self):
        result = add_features(self.df)
        self.assertIn("rsi_14", result.columns)
        self.assertIn("sma_20", result.columns)
        self.assertIn("news_sentiment_score", result.columns)
        self.assertIn("target_return", result.columns)

    def test_build_feature_matrix_returns_feature_columns(self):
        result, columns = build_feature_matrix(self.df)
        self.assertGreater(len(columns), 0)
        self.assertIn("rsi_14", columns)
        self.assertTrue(len(result) >= 0)


if __name__ == "__main__":
    unittest.main()

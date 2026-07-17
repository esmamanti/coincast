import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import add_features
from src.split import chronological_split


class FeaturePipelineTests(unittest.TestCase):
    def setUp(self):
        rows = []
        for i in range(60):
            rows.append(
                {
                    "open": 100 + i,
                    "high": 102 + i,
                    "low": 99 + i,
                    "close": 101 + i,
                    "volume": 1000 + i * 10,
                }
            )
        self.df = pd.DataFrame(rows)

    def test_add_features_adds_indicators_and_sentiment(self):
        result = add_features(self.df, sentiment_scores=[0.1] * len(self.df))

        self.assertIn("rsi_14", result.columns)
        self.assertIn("sma_20", result.columns)
        self.assertIn("ema_12", result.columns)
        self.assertIn("news_sentiment_score", result.columns)
        self.assertIn("target_return", result.columns)
        self.assertFalse(result["news_sentiment_score"].isna().any())

    def test_chronological_split_preserves_time_order(self):
        df = pd.DataFrame(
            {"value": range(20)},
            index=pd.date_range("2024-01-01", periods=20, freq="D"),
        )

        train, test = chronological_split(df, test_ratio=0.2)

        self.assertEqual(len(train), 16)
        self.assertEqual(len(test), 4)
        self.assertTrue(train.index.max() < test.index.min())


if __name__ == "__main__":
    unittest.main()

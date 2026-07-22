"""Compatibility entry point for the canonical feature engine in ``src.features.engine``."""

from src.features.engine import (
    add_features,
    build_feature_matrix,
    infer_sentiment_score,
    process_all_coins,
)

__all__ = ["add_features", "build_feature_matrix", "infer_sentiment_score", "process_all_coins"]


if __name__ == "__main__":
    process_all_coins()

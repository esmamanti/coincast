from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data_processed"
MODEL_DIR = ROOT / "models_saved"
RESULTS_DIR = ROOT / "results"

MODEL_GRID = [
    {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8},
    {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8},
]
ONE_WAY_FEE = 0.001
ONE_WAY_SLIPPAGE = 0.0005
ROUND_TRIP_COST = 2 * (ONE_WAY_FEE + ONE_WAY_SLIPPAGE)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"open", "high", "low", "close", "target", "target_return"}
    return [column for column in frame.columns if column not in excluded]


def strategy_metrics(actual: np.ndarray, predicted: np.ndarray, threshold: float) -> dict[str, float | int]:
    signals = np.zeros(len(predicted), dtype=float)
    signals[predicted > threshold] = 1.0
    signals[predicted < -threshold] = -1.0
    net_returns = signals * actual - np.abs(signals) * ROUND_TRIP_COST
    equity = np.cumprod(1.0 + net_returns)
    if len(equity):
        peak = np.maximum.accumulate(equity)
        max_drawdown = float(np.min(equity / peak - 1.0))
        compounded_return = float(equity[-1] - 1.0)
    else:
        max_drawdown = 0.0
        compounded_return = 0.0
    return {
        "trade_count": int(np.count_nonzero(signals)),
        "net_return": compounded_return,
        "mean_net_return": float(np.mean(net_returns)) if len(net_returns) else 0.0,
        "max_drawdown": max_drawdown,
    }


def select_threshold(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, dict[str, float | int]]:
    candidates = [ROUND_TRIP_COST, 0.004, 0.005, 0.0075, 0.01]
    scored = [(threshold, strategy_metrics(actual, predicted, threshold)) for threshold in candidates]
    return max(scored, key=lambda item: (item[1]["net_return"], item[1]["trade_count"]))


def train_symbol_horizon(symbol: str, horizon: int) -> dict:
    source_path = DATA_DIR / f"{symbol}_features.csv"
    frame = pd.read_csv(source_path, parse_dates=["open_time"]).set_index("open_time")
    columns = feature_columns(frame)
    frame = frame.copy()
    frame["target_return_h"] = frame["close"].shift(-horizon) / frame["close"] - 1.0
    frame = frame.dropna(subset=[*columns, "target_return_h"])

    train_end = int(len(frame) * 0.60)
    validation_end = int(len(frame) * 0.80)
    if train_end < 500 or validation_end <= train_end or validation_end >= len(frame):
        raise ValueError(f"{symbol} h={horizon}: insufficient rows for train/validation/test")

    train = frame.iloc[:train_end]
    validation = frame.iloc[train_end:validation_end]
    test = frame.iloc[validation_end:]
    best: dict | None = None

    for params in MODEL_GRID:
        candidate = xgb.XGBRegressor(**params, random_state=42, verbosity=0, n_jobs=2)
        candidate.fit(train[columns], train["target_return_h"])
        validation_predictions = candidate.predict(validation[columns])
        validation_mae = mean_absolute_error(validation["target_return_h"], validation_predictions)
        if best is None or validation_mae < best["validation_mae"]:
            best = {
                "params": params,
                "validation_mae": float(validation_mae),
                "validation_predictions": validation_predictions,
            }

    assert best is not None
    validation_actual = validation["target_return_h"].to_numpy()
    signal_threshold, validation_strategy = select_threshold(validation_actual, best["validation_predictions"])
    residuals = validation_actual - best["validation_predictions"]

    development = frame.iloc[:validation_end]
    model = xgb.XGBRegressor(**best["params"], random_state=42, verbosity=0, n_jobs=2)
    model.fit(development[columns], development["target_return_h"])
    test_predictions = model.predict(test[columns])
    test_actual = test["target_return_h"].to_numpy()

    test_mae = float(mean_absolute_error(test_actual, test_predictions))
    naive_mae = float(np.mean(np.abs(test_actual)))
    direction_accuracy = float(np.mean(np.sign(test_predictions) == np.sign(test_actual)))
    improvement_ratio = float(naive_mae / test_mae) if test_mae else 0.0
    test_strategy = strategy_metrics(test_actual, test_predictions, signal_threshold)
    verified = bool(
        improvement_ratio >= 1.01
        and direction_accuracy >= 0.52
        and test_strategy["net_return"] > 0
        and test_strategy["trade_count"] >= 30
    )

    trained_at = datetime.now(timezone.utc).isoformat()
    model_id = f"{symbol.lower()}-h{horizon}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    model_path = MODEL_DIR / f"xgb_{symbol.lower()}_h{horizon}_return.pkl"
    metadata_path = MODEL_DIR / f"xgb_{symbol.lower()}_h{horizon}_metadata.json"
    joblib.dump(model, model_path)

    metadata = {
        "model_id": model_id,
        "symbol": symbol,
        "interval": "1h",
        "horizon": horizon,
        "trained_at": trained_at,
        "data_start": frame.index.min().isoformat(),
        "data_end": frame.index.max().isoformat(),
        "feature_columns": columns,
        "best_params": best["params"],
        "signal_threshold": signal_threshold,
        "round_trip_cost_assumption": ROUND_TRIP_COST,
        "residual_q05": float(np.quantile(residuals, 0.05)),
        "residual_q95": float(np.quantile(residuals, 0.95)),
        "validation": {"mae": best["validation_mae"], **validation_strategy},
        "test": {
            "mae": test_mae,
            "naive_mae": naive_mae,
            "price_improvement_ratio": improvement_ratio,
            "direction_accuracy": direction_accuracy,
            **test_strategy,
        },
        "verified": verified,
        "quality_gate": {
            "price_improvement_ratio_min": 1.01,
            "direction_accuracy_min": 0.52,
            "net_return_must_be_positive": True,
            "minimum_trades": 30,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train point-in-time coin and horizon-specific XGBoost models")
    parser.add_argument("--symbols", nargs="*", help="Symbols to train; defaults to every processed feature file")
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 4, 24])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    symbols = args.symbols or sorted(path.name.replace("_features.csv", "") for path in DATA_DIR.glob("*_features.csv"))
    summary = []
    for symbol in symbols:
        for horizon in args.horizons:
            print(f"Training {symbol} horizon={horizon}...")
            metadata = train_symbol_horizon(symbol.upper(), horizon)
            summary.append(metadata)
            metrics = metadata["test"]
            print(
                f"  verified={metadata['verified']} direction={metrics['direction_accuracy']:.3f} "
                f"improvement={metrics['price_improvement_ratio']:.3f} net={metrics['net_return']:.4f}"
            )

    summary_path = RESULTS_DIR / "model_registry_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {len(summary)} model records to {summary_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd


class PredictionTracker:
    """Persist live forecasts and score them once their target candle closes."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path, timeout=15)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS forecasts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    horizon INTEGER NOT NULL,
                    interval TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    data_timestamp TEXT NOT NULL,
                    target_timestamp TEXT NOT NULL,
                    current_price REAL NOT NULL,
                    predicted_price REAL NOT NULL,
                    predicted_return REAL NOT NULL,
                    lower_price REAL NOT NULL,
                    upper_price REAL NOT NULL,
                    action TEXT NOT NULL,
                    actual_price REAL,
                    absolute_error REAL,
                    naive_error REAL,
                    direction_correct INTEGER,
                    interval_hit INTEGER,
                    resolved_at TEXT,
                    UNIQUE(symbol, horizon, model_id, data_timestamp)
                );
                CREATE INDEX IF NOT EXISTS idx_forecasts_lookup
                ON forecasts(symbol, horizon, id DESC);
                CREATE INDEX IF NOT EXISTS idx_forecasts_unresolved
                ON forecasts(symbol, interval, resolved_at, target_timestamp);
                """
            )

    @staticmethod
    def _as_utc(value: str | datetime | pd.Timestamp) -> datetime:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        return timestamp.to_pydatetime()

    @staticmethod
    def _interval_delta(interval: str) -> timedelta:
        if interval == "1h":
            return timedelta(hours=1)
        raise ValueError(f"Unsupported tracking interval: {interval}")

    def record(self, prediction: dict, action: str) -> bool:
        data_timestamp = self._as_utc(prediction["data_timestamp"])
        interval = str(prediction["interval"])
        target_timestamp = data_timestamp + self._interval_delta(interval) * int(prediction["horizon"])
        price_interval = prediction["predicted_price_interval"]
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO forecasts (
                    created_at, symbol, horizon, interval, model_id, data_timestamp,
                    target_timestamp, current_price, predicted_price, predicted_return,
                    lower_price, upper_price, action
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    prediction["symbol"],
                    int(prediction["horizon"]),
                    interval,
                    prediction["model_id"],
                    data_timestamp.isoformat(),
                    target_timestamp.isoformat(),
                    float(prediction["current_price"]),
                    float(prediction["predicted_price"]),
                    float(prediction["predicted_return"]),
                    float(price_interval["lower"]),
                    float(price_interval["upper"]),
                    action,
                ),
            )
            return cursor.rowcount > 0

    def resolve_with_candles(self, symbol: str, interval: str, candles: pd.DataFrame) -> int:
        if candles is None or candles.empty or "open_time" not in candles or "close" not in candles:
            return 0

        delta = self._interval_delta(interval)
        frame = candles[["open_time", "close"]].copy()
        frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
        frame["target_timestamp"] = frame["open_time"] + pd.Timedelta(delta)
        close_by_target = {
            self._as_utc(row.target_timestamp).isoformat(): float(row.close)
            for row in frame.itertuples(index=False)
        }
        latest_target = max(close_by_target)

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM forecasts
                WHERE symbol = ? AND interval = ? AND resolved_at IS NULL
                  AND target_timestamp <= ?
                ORDER BY id
                """,
                (symbol, interval, latest_target),
            ).fetchall()
            resolved = 0
            for row in rows:
                target = self._as_utc(row["target_timestamp"]).isoformat()
                actual_price = close_by_target.get(target)
                if actual_price is None:
                    continue

                current_price = float(row["current_price"])
                predicted_price = float(row["predicted_price"])
                predicted_direction = (predicted_price > current_price) - (predicted_price < current_price)
                actual_direction = (actual_price > current_price) - (actual_price < current_price)
                direction_correct = int(predicted_direction == actual_direction)
                interval_hit = int(float(row["lower_price"]) <= actual_price <= float(row["upper_price"]))
                connection.execute(
                    """
                    UPDATE forecasts
                    SET actual_price = ?, absolute_error = ?, naive_error = ?,
                        direction_correct = ?, interval_hit = ?, resolved_at = ?
                    WHERE id = ? AND resolved_at IS NULL
                    """,
                    (
                        actual_price,
                        abs(actual_price - predicted_price),
                        abs(actual_price - current_price),
                        direction_correct,
                        interval_hit,
                        datetime.now(timezone.utc).isoformat(),
                        int(row["id"]),
                    ),
                )
                resolved += 1
        return resolved

    @staticmethod
    def _recent_row(row: sqlite3.Row) -> dict:
        return {
            "id": int(row["id"]),
            "created_at": row["created_at"],
            "data_timestamp": row["data_timestamp"],
            "target_timestamp": row["target_timestamp"],
            "current_price": float(row["current_price"]),
            "predicted_price": float(row["predicted_price"]),
            "actual_price": None if row["actual_price"] is None else float(row["actual_price"]),
            "absolute_error": None if row["absolute_error"] is None else float(row["absolute_error"]),
            "direction_correct": None if row["direction_correct"] is None else bool(row["direction_correct"]),
            "interval_hit": None if row["interval_hit"] is None else bool(row["interval_hit"]),
            "action": row["action"],
            "status": "resolved" if row["resolved_at"] else "pending",
        }

    def performance(self, symbol: str, horizon: int, limit: int = 20) -> dict:
        safe_limit = max(1, min(int(limit), 100))
        with self._connect() as connection:
            aggregate = connection.execute(
                """
                SELECT
                    COUNT(*) AS total_predictions,
                    COUNT(resolved_at) AS resolved_predictions,
                    AVG(CASE WHEN resolved_at IS NOT NULL THEN direction_correct END) AS direction_accuracy,
                    AVG(absolute_error) AS mae,
                    AVG(naive_error) AS naive_mae,
                    AVG(CASE WHEN resolved_at IS NOT NULL THEN interval_hit END) AS interval_coverage
                FROM forecasts WHERE symbol = ? AND horizon = ?
                """,
                (symbol, int(horizon)),
            ).fetchone()
            recent_rows = connection.execute(
                """
                SELECT * FROM forecasts WHERE symbol = ? AND horizon = ?
                ORDER BY id DESC LIMIT ?
                """,
                (symbol, int(horizon), safe_limit),
            ).fetchall()

        total = int(aggregate["total_predictions"] or 0)
        resolved = int(aggregate["resolved_predictions"] or 0)
        mae = None if aggregate["mae"] is None else float(aggregate["mae"])
        naive_mae = None if aggregate["naive_mae"] is None else float(aggregate["naive_mae"])
        improvement_ratio = None
        if mae is not None and mae > 0 and naive_mae is not None:
            improvement_ratio = naive_mae / mae
        return {
            "symbol": symbol,
            "horizon": int(horizon),
            "total_predictions": total,
            "resolved_predictions": resolved,
            "pending_predictions": total - resolved,
            "direction_accuracy": None if aggregate["direction_accuracy"] is None else float(aggregate["direction_accuracy"]),
            "mae": mae,
            "naive_mae": naive_mae,
            "price_improvement_ratio": improvement_ratio,
            "interval_coverage": None if aggregate["interval_coverage"] is None else float(aggregate["interval_coverage"]),
            "recent": [self._recent_row(row) for row in recent_rows],
        }

    def performance_many(self, symbols: list[str], horizon: int, limit: int = 5) -> list[dict]:
        return [self.performance(symbol, horizon, limit=limit) for symbol in symbols]

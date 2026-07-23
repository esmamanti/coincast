from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


class ShadowPortfolioBroker:
    """Independent long-only paper portfolios for every symbol/horizon strategy."""

    def __init__(
        self,
        db_path: str | Path,
        initial_cash: float = 10_000.0,
        position_fraction: float = 0.10,
        fee_rate: float = 0.001,
        slippage_rate: float = 0.0005,
    ) -> None:
        self.db_path = Path(db_path)
        self.initial_cash = float(initial_cash)
        self.position_fraction = float(position_fraction)
        self.fee_rate = float(fee_rate)
        self.slippage_rate = float(slippage_rate)
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
                CREATE TABLE IF NOT EXISTS shadow_accounts (
                    strategy_key TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    horizon INTEGER NOT NULL,
                    initial_cash REAL NOT NULL,
                    cash REAL NOT NULL,
                    peak_equity REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS shadow_positions (
                    strategy_key TEXT PRIMARY KEY,
                    quantity REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    cost_basis REAL NOT NULL,
                    last_price REAL NOT NULL,
                    opened_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS shadow_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    strategy_key TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    horizon INTEGER NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    market_price REAL NOT NULL,
                    execution_price REAL NOT NULL,
                    notional REAL NOT NULL,
                    fee REAL NOT NULL,
                    realized_pnl REAL,
                    predicted_return REAL NOT NULL,
                    data_timestamp TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS shadow_processed_signals (
                    strategy_key TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    data_timestamp TEXT NOT NULL,
                    action TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    processed_at TEXT NOT NULL,
                    PRIMARY KEY(strategy_key, model_id, data_timestamp)
                );
                CREATE INDEX IF NOT EXISTS idx_shadow_trades_strategy
                ON shadow_trades(strategy_key, id DESC);
                """
            )

    @staticmethod
    def strategy_key(symbol: str, horizon: int) -> str:
        return f"{symbol}:h{int(horizon)}"

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _ensure_account(self, connection: sqlite3.Connection, symbol: str, horizon: int) -> str:
        key = self.strategy_key(symbol, horizon)
        connection.execute(
            """
            INSERT OR IGNORE INTO shadow_accounts (
                strategy_key, symbol, horizon, initial_cash, cash,
                peak_equity, max_drawdown, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, 0, ?)
            """,
            (key, symbol, int(horizon), self.initial_cash, self.initial_cash, self.initial_cash, self._now()),
        )
        return key

    def process_signal(self, prediction: dict, action: str) -> dict:
        symbol = str(prediction["symbol"])
        horizon = int(prediction["horizon"])
        market_price = float(prediction["current_price"])
        data_timestamp = str(prediction["data_timestamp"])
        model_id = str(prediction["model_id"])
        predicted_return = float(prediction["predicted_return"])

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            key = self._ensure_account(connection, symbol, horizon)
            duplicate = connection.execute(
                """
                SELECT outcome FROM shadow_processed_signals
                WHERE strategy_key = ? AND model_id = ? AND data_timestamp = ?
                """,
                (key, model_id, data_timestamp),
            ).fetchone()
            if duplicate:
                snapshot = self._snapshot(connection, key, market_price)
                return {"processed": False, "outcome": duplicate["outcome"], "portfolio": snapshot}

            account = connection.execute(
                "SELECT cash FROM shadow_accounts WHERE strategy_key = ?", (key,)
            ).fetchone()
            position = connection.execute(
                "SELECT * FROM shadow_positions WHERE strategy_key = ?", (key,)
            ).fetchone()
            cash = float(account["cash"])
            timestamp = self._now()
            outcome = "HOLD"

            if action == "BUY" and position is None:
                budget = min(cash, cash * self.position_fraction)
                fee = budget * self.fee_rate
                execution_price = market_price * (1.0 + self.slippage_rate)
                notional = budget - fee
                quantity = notional / execution_price
                cash -= budget
                connection.execute(
                    """
                    INSERT INTO shadow_positions (
                        strategy_key, quantity, entry_price, cost_basis, last_price, opened_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (key, quantity, execution_price, budget, market_price, timestamp),
                )
                connection.execute(
                    """
                    INSERT INTO shadow_trades (
                        timestamp, strategy_key, symbol, horizon, side, quantity,
                        market_price, execution_price, notional, fee, realized_pnl,
                        predicted_return, data_timestamp
                    ) VALUES (?, ?, ?, ?, 'BUY', ?, ?, ?, ?, ?, NULL, ?, ?)
                    """,
                    (
                        timestamp, key, symbol, horizon, quantity, market_price,
                        execution_price, notional, fee, predicted_return, data_timestamp,
                    ),
                )
                outcome = "OPENED"
            elif action == "BUY":
                outcome = "POSITION_EXISTS"
            elif action == "SELL" and position is not None:
                quantity = float(position["quantity"])
                execution_price = market_price * (1.0 - self.slippage_rate)
                notional = quantity * execution_price
                fee = notional * self.fee_rate
                proceeds = notional - fee
                realized_pnl = proceeds - float(position["cost_basis"])
                cash += proceeds
                connection.execute("DELETE FROM shadow_positions WHERE strategy_key = ?", (key,))
                connection.execute(
                    """
                    INSERT INTO shadow_trades (
                        timestamp, strategy_key, symbol, horizon, side, quantity,
                        market_price, execution_price, notional, fee, realized_pnl,
                        predicted_return, data_timestamp
                    ) VALUES (?, ?, ?, ?, 'SELL', ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        timestamp, key, symbol, horizon, quantity, market_price,
                        execution_price, notional, fee, realized_pnl, predicted_return, data_timestamp,
                    ),
                )
                outcome = "CLOSED"
            elif action == "SELL":
                outcome = "NO_POSITION"

            connection.execute(
                "UPDATE shadow_accounts SET cash = ?, updated_at = ? WHERE strategy_key = ?",
                (cash, timestamp, key),
            )
            connection.execute(
                """
                INSERT INTO shadow_processed_signals (
                    strategy_key, model_id, data_timestamp, action, outcome, processed_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (key, model_id, data_timestamp, action, outcome, timestamp),
            )
            snapshot = self._snapshot(connection, key, market_price)
            return {"processed": True, "outcome": outcome, "portfolio": snapshot}

    def _snapshot(self, connection: sqlite3.Connection, key: str, market_price: float | None = None) -> dict:
        account = connection.execute(
            "SELECT * FROM shadow_accounts WHERE strategy_key = ?", (key,)
        ).fetchone()
        position = connection.execute(
            "SELECT * FROM shadow_positions WHERE strategy_key = ?", (key,)
        ).fetchone()
        if position is not None and market_price is not None:
            connection.execute(
                "UPDATE shadow_positions SET last_price = ? WHERE strategy_key = ?",
                (float(market_price), key),
            )
        mark_price = float(market_price) if market_price is not None else (
            float(position["last_price"]) if position is not None else 0.0
        )
        cash = float(account["cash"])
        position_value = float(position["quantity"]) * mark_price if position is not None else 0.0
        equity = cash + position_value
        peak_equity = max(float(account["peak_equity"]), equity)
        drawdown = equity / peak_equity - 1.0 if peak_equity else 0.0
        max_drawdown = min(float(account["max_drawdown"]), drawdown)
        connection.execute(
            """
            UPDATE shadow_accounts SET peak_equity = ?, max_drawdown = ?, updated_at = ?
            WHERE strategy_key = ?
            """,
            (peak_equity, max_drawdown, self._now(), key),
        )
        trade_stats = connection.execute(
            """
            SELECT
                COUNT(*) AS trade_count,
                SUM(CASE WHEN side = 'SELL' THEN 1 ELSE 0 END) AS closed_trades,
                SUM(CASE WHEN side = 'SELL' AND realized_pnl > 0 THEN 1 ELSE 0 END) AS winning_trades,
                COALESCE(SUM(fee), 0) AS fees_paid,
                COALESCE(SUM(realized_pnl), 0) AS realized_pnl
            FROM shadow_trades WHERE strategy_key = ?
            """,
            (key,),
        ).fetchone()
        initial_cash = float(account["initial_cash"])
        closed_trades = int(trade_stats["closed_trades"] or 0)
        winning_trades = int(trade_stats["winning_trades"] or 0)
        return {
            "strategy_key": key,
            "symbol": account["symbol"],
            "horizon": int(account["horizon"]),
            "initial_cash": initial_cash,
            "cash": cash,
            "equity": equity,
            "net_return": equity / initial_cash - 1.0 if initial_cash else 0.0,
            "max_drawdown": max_drawdown,
            "trade_count": int(trade_stats["trade_count"] or 0),
            "closed_trades": closed_trades,
            "winning_trades": winning_trades,
            "win_rate": winning_trades / closed_trades if closed_trades else None,
            "fees_paid": float(trade_stats["fees_paid"] or 0.0),
            "realized_pnl": float(trade_stats["realized_pnl"] or 0.0),
            "open_position": None if position is None else {
                "quantity": float(position["quantity"]),
                "entry_price": float(position["entry_price"]),
                "last_price": mark_price,
                "position_value": position_value,
            },
        }

    def snapshot(self, symbol: str, horizon: int, market_price: float | None = None) -> dict:
        with self._connect() as connection:
            key = self._ensure_account(connection, symbol, horizon)
            return self._snapshot(connection, key, market_price)

    def snapshot_many(self, symbols: list[str], horizon: int) -> list[dict]:
        return [self.snapshot(symbol, horizon) for symbol in symbols]

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


class PaperBroker:
    def __init__(self, db_path: str | Path, initial_cash: float = 10_000.0, fee_rate: float = 0.001) -> None:
        self.db_path = Path(db_path)
        self.initial_cash = float(initial_cash)
        self.fee_rate = float(fee_rate)
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
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS account (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    cash REAL NOT NULL,
                    peak_equity REAL NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    quantity REAL NOT NULL,
                    average_price REAL NOT NULL,
                    last_price REAL NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    notional REAL NOT NULL,
                    fee REAL NOT NULL,
                    predicted_return REAL NOT NULL,
                    reason TEXT NOT NULL,
                    report TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS equity_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    equity REAL NOT NULL,
                    cash REAL NOT NULL
                );
                """
            )
            connection.execute(
                "INSERT OR IGNORE INTO account (id, cash, peak_equity, created_at) VALUES (1, ?, ?, ?)",
                (self.initial_cash, self.initial_cash, datetime.now(timezone.utc).isoformat()),
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def mark_price(self, symbol: str, price: float) -> dict:
        with self._connect() as connection:
            connection.execute("UPDATE positions SET last_price = ?, updated_at = ? WHERE symbol = ?", (price, self._now(), symbol))
        return self.snapshot(record=True)

    def snapshot(self, record: bool = False) -> dict:
        with self._connect() as connection:
            account = connection.execute("SELECT cash, peak_equity FROM account WHERE id = 1").fetchone()
            position_rows = connection.execute(
                "SELECT symbol, quantity, average_price, last_price, updated_at FROM positions WHERE quantity > 0 ORDER BY symbol"
            ).fetchall()
            positions = [dict(row) for row in position_rows]
            cash = float(account["cash"])
            equity = cash + sum(float(row["quantity"]) * float(row["last_price"]) for row in position_rows)
            peak_equity = max(float(account["peak_equity"]), equity)
            connection.execute("UPDATE account SET peak_equity = ? WHERE id = 1", (peak_equity,))
            today = datetime.now(timezone.utc).date().isoformat()
            first_today = connection.execute(
                "SELECT equity FROM equity_snapshots WHERE substr(timestamp, 1, 10) = ? ORDER BY id LIMIT 1",
                (today,),
            ).fetchone()
            day_start_equity = float(first_today["equity"]) if first_today else equity
            if record:
                connection.execute(
                    "INSERT INTO equity_snapshots (timestamp, equity, cash) VALUES (?, ?, ?)",
                    (self._now(), equity, cash),
                )
            return {
                "cash": round(cash, 8),
                "equity": round(equity, 8),
                "peak_equity": round(peak_equity, 8),
                "day_start_equity": round(day_start_equity, 8),
                "positions": positions,
            }

    def execute(
        self,
        symbol: str,
        side: str,
        price: float,
        predicted_return: float,
        reason: str,
        report: str,
        quote_amount: float = 0.0,
    ) -> dict:
        timestamp = self._now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            account = connection.execute("SELECT cash FROM account WHERE id = 1").fetchone()
            position = connection.execute("SELECT quantity, average_price FROM positions WHERE symbol = ?", (symbol,)).fetchone()
            cash = float(account["cash"])

            if side == "BUY":
                spend = min(float(quote_amount), cash)
                if spend <= 0:
                    raise ValueError("Paper BUY amount must be positive")
                fee = spend * self.fee_rate
                notional = spend - fee
                quantity = notional / price
                previous_quantity = float(position["quantity"]) if position else 0.0
                previous_cost = previous_quantity * float(position["average_price"]) if position else 0.0
                total_quantity = previous_quantity + quantity
                average_price = (previous_cost + notional) / total_quantity
                cash -= spend
                connection.execute(
                    """
                    INSERT INTO positions (symbol, quantity, average_price, last_price, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(symbol) DO UPDATE SET quantity=excluded.quantity, average_price=excluded.average_price,
                    last_price=excluded.last_price, updated_at=excluded.updated_at
                    """,
                    (symbol, total_quantity, average_price, price, timestamp),
                )
                trade_quantity = quantity
            elif side == "SELL":
                if not position or float(position["quantity"]) <= 0:
                    raise ValueError(f"No paper position to sell for {symbol}")
                trade_quantity = float(position["quantity"])
                notional = trade_quantity * price
                fee = notional * self.fee_rate
                cash += notional - fee
                connection.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
            else:
                raise ValueError(f"Unsupported paper side: {side}")

            connection.execute("UPDATE account SET cash = ? WHERE id = 1", (cash,))
            cursor = connection.execute(
                """
                INSERT INTO trades (timestamp, symbol, side, quantity, price, notional, fee, predicted_return, reason, report)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (timestamp, symbol, side, trade_quantity, price, notional, fee, predicted_return, reason, report),
            )
            trade_id = int(cursor.lastrowid)

        account_snapshot = self.snapshot(record=True)
        return {
            "id": trade_id,
            "timestamp": timestamp,
            "symbol": symbol,
            "side": side,
            "quantity": round(trade_quantity, 12),
            "price": round(price, 8),
            "notional": round(notional, 8),
            "fee": round(fee, 8),
            "account": account_snapshot,
        }

    def recent_trades(self, limit: int = 50) -> list[dict]:
        safe_limit = max(1, min(int(limit), 500))
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (safe_limit,)
            ).fetchall()
        return [dict(row) for row in rows]

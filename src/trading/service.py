from __future__ import annotations

import os
from pathlib import Path

from ml_backend.inference.service import InferenceService
from src.trading.notifications import TradeNotifier
from src.trading.paper_broker import PaperBroker
from src.trading.prediction_tracker import PredictionTracker
from src.trading.quality_gate import ModelQualityGate
from src.trading.risk import RiskEngine, RiskLimits
from src.trading.shadow_portfolio import ShadowPortfolioBroker


ROOT = Path(__file__).resolve().parents[2]


class TradingService:
    def __init__(
        self,
        inference: InferenceService | None = None,
        broker: PaperBroker | None = None,
        notifier: TradeNotifier | None = None,
        risk_engine: RiskEngine | None = None,
        tracker: PredictionTracker | None = None,
        shadow_broker: ShadowPortfolioBroker | None = None,
        quality_gate: ModelQualityGate | None = None,
    ) -> None:
        db_path = os.getenv("PAPER_DB_PATH", str(ROOT / "data" / "paper_trading.sqlite3"))
        initial_cash = float(os.getenv("PAPER_INITIAL_CASH", "10000"))
        self.inference = inference or InferenceService()
        self.broker = broker or PaperBroker(db_path, initial_cash=initial_cash)
        tracking_db_path = os.getenv("PREDICTION_DB_PATH", str(ROOT / "data" / "prediction_tracking.sqlite3"))
        self.tracker = tracker or PredictionTracker(tracking_db_path)
        shadow_db_path = os.getenv("SHADOW_DB_PATH", str(ROOT / "data" / "shadow_portfolios.sqlite3"))
        self.shadow_broker = shadow_broker or ShadowPortfolioBroker(
            shadow_db_path,
            initial_cash=float(os.getenv("SHADOW_INITIAL_CASH", "10000")),
            position_fraction=float(os.getenv("SHADOW_POSITION_FRACTION", "0.10")),
            fee_rate=float(os.getenv("SHADOW_FEE_RATE", "0.001")),
            slippage_rate=float(os.getenv("SHADOW_SLIPPAGE_RATE", "0.0005")),
        )
        self.quality_gate = quality_gate or ModelQualityGate()
        self.notifier = notifier or TradeNotifier()
        self.risk_engine = risk_engine or RiskEngine(
            RiskLimits(
                max_position_fraction=float(os.getenv("MAX_POSITION_FRACTION", "0.10")),
                max_drawdown_fraction=float(os.getenv("MAX_DRAWDOWN_FRACTION", "0.10")),
                max_daily_loss_fraction=float(os.getenv("MAX_DAILY_LOSS_FRACTION", "0.02")),
                minimum_order_quote=float(os.getenv("MINIMUM_ORDER_QUOTE", "10")),
            )
        )
        self.paper_enabled = os.getenv("PAPER_TRADING_ENABLED", "true").lower() == "true"

    @staticmethod
    def _action(predicted_return: float, threshold: float) -> str:
        if predicted_return > threshold:
            return "BUY"
        if predicted_return < -threshold:
            return "SELL"
        return "HOLD"

    @staticmethod
    def _report(prediction: dict, action: str, status: str, reason: str, account: dict) -> str:
        return (
            f"{prediction['symbol']} | {action} ({status}) | fiyat {prediction['current_price']:.6f} | "
            f"beklenen getiri %{prediction['predicted_return'] * 100:.3f} | "
            f"ufuk {prediction['horizon']}x{prediction['interval']} | model doğrulandı: "
            f"{'evet' if prediction['model_verified'] else 'hayır'} | özsermaye {account['equity']:.2f} | {reason}"
        )

    def signal(self, symbol: str, horizon: int = 1) -> dict:
        tracked = self.track_prediction(symbol, horizon=horizon)
        prediction = tracked["prediction"]
        action = tracked["action"]
        account = self.broker.mark_price(prediction["symbol"], prediction["current_price"])
        decision = self.risk_engine.evaluate(
            action=action,
            symbol=prediction["symbol"],
            account=account,
            model_verified=prediction["model_verified"],
            live_mode=False,
        )
        return {
            **tracked,
            "risk": decision.to_dict(),
            "account": account,
        }

    def track_prediction(self, symbol: str, horizon: int = 1) -> dict:
        prediction = self.inference.predict(symbol, horizon=horizon)
        action = self._action(prediction["predicted_return"], prediction["signal_threshold"])
        self.tracker.record(prediction, action)
        market_frame = self.inference.latest_market_frame(prediction["symbol"], prediction["interval"])
        if market_frame is not None:
            self.tracker.resolve_with_candles(prediction["symbol"], prediction["interval"], market_frame)
        performance = self.tracker.performance(prediction["symbol"], prediction["horizon"])
        shadow_result = self.shadow_broker.process_signal(prediction, action)
        quality = self.quality_gate.evaluate(performance, shadow_result["portfolio"])
        return {
            "prediction": prediction,
            "action": action,
            "performance": performance,
            "shadow": shadow_result["portfolio"],
            "quality": quality,
        }

    def quality_report(self, symbol: str, horizon: int, limit: int = 5) -> dict:
        performance = self.tracker.performance(symbol, horizon, limit=limit)
        shadow = self.shadow_broker.snapshot(symbol, horizon)
        return {
            "symbol": symbol,
            "horizon": int(horizon),
            "performance": performance,
            "shadow": shadow,
            "quality": self.quality_gate.evaluate(performance, shadow),
        }

    def quality_reports(self, symbols: list[str], horizon: int, limit: int = 5) -> list[dict]:
        return [self.quality_report(symbol, horizon, limit=limit) for symbol in symbols]

    def backfill_shadow_portfolios(self, symbols: list[str], horizons: list[int]) -> dict:
        processed = 0
        duplicates = 0
        strategies = 0
        for symbol in symbols:
            for horizon in horizons:
                strategies += 1
                for record in self.tracker.history(symbol, horizon):
                    action = record.pop("action")
                    result = self.shadow_broker.process_signal(record, action)
                    if result["processed"]:
                        processed += 1
                    else:
                        duplicates += 1
        return {
            "strategies": strategies,
            "processed_signals": processed,
            "duplicate_signals": duplicates,
        }

    def run_paper_cycle(self, symbol: str, horizon: int = 1) -> dict:
        signal = self.signal(symbol, horizon=horizon)
        prediction = signal["prediction"]
        decision = signal["risk"]
        action = signal["action"]
        status = "NO_TRADE"
        trade = None

        if not self.paper_enabled:
            reason = "Paper trading yapılandırmada kapalı"
        elif not decision["allowed"]:
            reason = decision["reason"]
        else:
            reason = decision["reason"]
            preliminary_report = self._report(prediction, action, "EXECUTING", reason, signal["account"])
            trade = self.broker.execute(
                symbol=prediction["symbol"],
                side=action,
                price=prediction["current_price"],
                predicted_return=prediction["predicted_return"],
                reason=reason,
                report=preliminary_report,
                quote_amount=float(decision.get("quote_amount", 0.0)),
            )
            status = "EXECUTED"

        account = trade["account"] if trade else signal["account"]
        report = self._report(prediction, action, status, reason, account)
        notifications = []
        if trade:
            notifications = self.notifier.send_trade_report(
                f"CoinCast paper {action}: {prediction['symbol']}", report
            )
        return {
            "mode": "paper",
            "status": status,
            "action": action,
            "reason": reason,
            "report": report,
            "prediction": prediction,
            "risk": decision,
            "trade": trade,
            "account": account,
            "notifications": notifications,
        }

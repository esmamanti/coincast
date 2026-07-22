from __future__ import annotations

import os
from pathlib import Path

from ml_backend.inference.service import InferenceService
from src.trading.notifications import TradeNotifier
from src.trading.paper_broker import PaperBroker
from src.trading.prediction_tracker import PredictionTracker
from src.trading.risk import RiskEngine, RiskLimits


ROOT = Path(__file__).resolve().parents[2]


class TradingService:
    def __init__(
        self,
        inference: InferenceService | None = None,
        broker: PaperBroker | None = None,
        notifier: TradeNotifier | None = None,
        risk_engine: RiskEngine | None = None,
        tracker: PredictionTracker | None = None,
    ) -> None:
        db_path = os.getenv("PAPER_DB_PATH", str(ROOT / "data" / "paper_trading.sqlite3"))
        initial_cash = float(os.getenv("PAPER_INITIAL_CASH", "10000"))
        self.inference = inference or InferenceService()
        self.broker = broker or PaperBroker(db_path, initial_cash=initial_cash)
        tracking_db_path = os.getenv("PREDICTION_DB_PATH", str(ROOT / "data" / "prediction_tracking.sqlite3"))
        self.tracker = tracker or PredictionTracker(tracking_db_path)
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
        return {
            "prediction": prediction,
            "action": action,
            "performance": performance,
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

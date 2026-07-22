from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from src.market_config import COINCAST_HORIZONS, COINCAST_SYMBOLS
from src.trading.notifications import TradeNotifier
from src.trading.paper_broker import PaperBroker
from src.trading.prediction_tracker import PredictionTracker


ISTANBUL = ZoneInfo("Europe/Istanbul")


class DailyReportService:
    def __init__(
        self,
        broker: PaperBroker,
        tracker: PredictionTracker,
        notifier: TradeNotifier | None = None,
    ) -> None:
        self.broker = broker
        self.tracker = tracker
        self.notifier = notifier or TradeNotifier()

    def channel_status(self) -> list[dict]:
        return [
            {"channel": channel.__class__.__name__, "configured": bool(channel.configured)}
            for channel in self.notifier.channels
        ]

    @staticmethod
    def _weighted_metric(coins: list[dict], key: str) -> float | None:
        usable = [coin for coin in coins if coin[key] is not None and coin["resolved_predictions"] > 0]
        total = sum(coin["resolved_predictions"] for coin in usable)
        if total == 0:
            return None
        return sum(float(coin[key]) * coin["resolved_predictions"] for coin in usable) / total

    def build(self, now: datetime | None = None) -> dict:
        generated_at = now or datetime.now(timezone.utc)
        if generated_at.tzinfo is None:
            generated_at = generated_at.replace(tzinfo=timezone.utc)
        local_time = generated_at.astimezone(ISTANBUL)
        account = self.broker.snapshot()
        cutoff = generated_at - timedelta(hours=24)
        trades = [
            trade
            for trade in self.broker.recent_trades(limit=500)
            if datetime.fromisoformat(trade["timestamp"]).astimezone(timezone.utc) >= cutoff
        ]

        horizon_summaries = []
        for horizon in COINCAST_HORIZONS:
            coins = self.tracker.performance_many(COINCAST_SYMBOLS, horizon, limit=1)
            summary = {
                "horizon": horizon,
                "resolved_predictions": sum(coin["resolved_predictions"] for coin in coins),
                "pending_predictions": sum(coin["pending_predictions"] for coin in coins),
                "direction_accuracy": self._weighted_metric(coins, "direction_accuracy"),
                "price_improvement_ratio": self._weighted_metric(coins, "price_improvement_ratio"),
                "interval_coverage": self._weighted_metric(coins, "interval_coverage"),
            }
            horizon_summaries.append(summary)

        positions = account["positions"]
        lines = [
            f"CoinCast günlük paper raporu — {local_time:%d.%m.%Y %H:%M}",
            "",
            f"Özsermaye: {account['equity']:.2f} USDT",
            f"Nakit: {account['cash']:.2f} USDT",
            f"Açık pozisyon: {len(positions)}",
            f"Son 24 saat işlemi: {len(trades)}",
            "",
            "Canlı tahmin başarısı:",
        ]
        for summary in horizon_summaries:
            direction = self._percent(summary["direction_accuracy"])
            coverage = self._percent(summary["interval_coverage"])
            advantage = self._ratio(summary["price_improvement_ratio"])
            lines.append(
                f"- {summary['horizon']} saat: {summary['resolved_predictions']} sonuç, "
                f"{summary['pending_predictions']} bekleyen | yön {direction} | "
                f"avantaj {advantage} | band {coverage}"
            )
        lines.extend(
            [
                "",
                "Not: Sonuç sayısı yeterli olmadan model başarısı kesin kabul edilmez.",
                "Gerçek para işlemi kapalıdır; sistem paper modundadır.",
            ]
        )
        return {
            "subject": f"CoinCast günlük paper raporu — {local_time:%d.%m.%Y}",
            "report": "\n".join(lines),
            "generated_at": generated_at.isoformat(),
            "account": account,
            "trades_last_24h": len(trades),
            "horizons": horizon_summaries,
            "channels": self.channel_status(),
        }

    def send(self, now: datetime | None = None) -> dict:
        payload = self.build(now=now)
        payload["notifications"] = self.notifier.send_trade_report(payload["subject"], payload["report"])
        return payload

    @staticmethod
    def _percent(value: float | None) -> str:
        return "—" if value is None else f"%{value * 100:.1f}"

    @staticmethod
    def _ratio(value: float | None) -> str:
        return "—" if value is None else f"{value:.2f}×"

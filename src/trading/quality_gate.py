from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class QualityThresholds:
    minimum_resolved_predictions: int = 200
    minimum_direction_accuracy: float = 0.55
    minimum_price_improvement_ratio: float = 1.05
    minimum_closed_trades: int = 30
    minimum_net_return: float = 0.0
    maximum_drawdown: float = 0.10


class ModelQualityGate:
    STATUS_LABELS = {
        "INSUFFICIENT_DATA": "Yetersiz veri",
        "INSUFFICIENT_TRADES": "Yetersiz işlem",
        "CANDIDATE": "Aday",
        "REJECTED": "Başarısız",
    }

    def __init__(self, thresholds: QualityThresholds | None = None) -> None:
        self.thresholds = thresholds or QualityThresholds()

    def evaluate(self, performance: dict, shadow: dict) -> dict:
        resolved = int(performance["resolved_predictions"])
        closed_trades = int(shadow["closed_trades"])
        direction = performance["direction_accuracy"]
        improvement = performance["price_improvement_ratio"]
        checks = {
            "resolved_predictions": resolved >= self.thresholds.minimum_resolved_predictions,
            "direction_accuracy": direction is not None and direction >= self.thresholds.minimum_direction_accuracy,
            "price_improvement_ratio": improvement is not None and improvement >= self.thresholds.minimum_price_improvement_ratio,
            "closed_trades": closed_trades >= self.thresholds.minimum_closed_trades,
            "net_return": float(shadow["net_return"]) > self.thresholds.minimum_net_return,
            "max_drawdown": float(shadow["max_drawdown"]) >= -self.thresholds.maximum_drawdown,
        }
        if resolved < self.thresholds.minimum_resolved_predictions:
            status = "INSUFFICIENT_DATA"
        elif closed_trades < self.thresholds.minimum_closed_trades:
            status = "INSUFFICIENT_TRADES"
        elif all(checks.values()):
            status = "CANDIDATE"
        else:
            status = "REJECTED"
        return {
            "status": status,
            "status_label": self.STATUS_LABELS[status],
            "sample_progress": min(1.0, resolved / self.thresholds.minimum_resolved_predictions),
            "checks": checks,
            "thresholds": asdict(self.thresholds),
            "failed_checks": [name for name, passed in checks.items() if not passed],
        }

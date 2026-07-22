from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    reason: str
    quote_amount: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RiskLimits:
    max_position_fraction: float = 0.10
    max_drawdown_fraction: float = 0.10
    max_daily_loss_fraction: float = 0.02
    minimum_order_quote: float = 10.0


class RiskEngine:
    def __init__(self, limits: RiskLimits | None = None) -> None:
        self.limits = limits or RiskLimits()

    def evaluate(self, action: str, symbol: str, account: dict, model_verified: bool, live_mode: bool = False) -> RiskDecision:
        if action == "HOLD":
            return RiskDecision(False, "Sinyal işlem eşiğini aşmadı")
        if live_mode and not model_verified:
            return RiskDecision(False, "Canlı işlem kalite kapısı doğrulanmamış model için kapalı")

        equity = float(account["equity"])
        peak_equity = max(float(account["peak_equity"]), equity)
        day_start_equity = max(float(account["day_start_equity"]), 1e-9)
        drawdown = max(0.0, (peak_equity - equity) / peak_equity) if peak_equity else 0.0
        daily_loss = max(0.0, (day_start_equity - equity) / day_start_equity)
        if drawdown >= self.limits.max_drawdown_fraction:
            return RiskDecision(False, f"Kill switch: drawdown %{drawdown * 100:.2f}")
        if daily_loss >= self.limits.max_daily_loss_fraction:
            return RiskDecision(False, f"Günlük zarar limiti: %{daily_loss * 100:.2f}")

        position = next((item for item in account["positions"] if item["symbol"] == symbol), None)
        if action == "BUY":
            if position and float(position["quantity"]) > 0:
                return RiskDecision(False, "Bu coin için açık spot pozisyon zaten var")
            quote_amount = min(float(account["cash"]), equity * self.limits.max_position_fraction)
            if quote_amount < self.limits.minimum_order_quote:
                return RiskDecision(False, "Kullanılabilir bakiye minimum emir tutarının altında")
            return RiskDecision(True, "Pozisyon ve zarar limitleri uygun", round(quote_amount, 8))

        if action == "SELL":
            if not position or float(position["quantity"]) <= 0:
                return RiskDecision(False, "Satılabilecek açık spot pozisyon yok")
            return RiskDecision(True, "Mevcut spot pozisyon kapatılabilir")

        return RiskDecision(False, f"Bilinmeyen işlem aksiyonu: {action}")


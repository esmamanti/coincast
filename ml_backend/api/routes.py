from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from ml_backend.inference.service import InferenceService, ModelNotFoundError, StaleMarketDataError
from src.data.ingest import normalize_symbol
from src.market_config import COINCAST_SYMBOLS
from src.trading.daily_report import DailyReportService
from src.trading.service import TradingService


router = APIRouter()
CACHE: Dict[str, Dict[str, object]] = {}
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "60"))
inference_service = InferenceService()
trading_service = TradingService(inference=inference_service)
daily_report_service = DailyReportService(
    trading_service.broker, trading_service.tracker, trading_service.notifier
)


class PredictionRequest(BaseModel):
    symbol: str = Field(min_length=5, max_length=20)
    horizon: Literal[1, 4, 24] = 1

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, value: str) -> str:
        return normalize_symbol(value)


class PredictionResponse(BaseModel):
    symbol: str
    horizon: int
    interval: str
    current_price: float
    predicted_price: float
    predicted_price_interval: dict[str, float]
    predicted_return: float
    confidence_interval: dict[str, float]
    mini_chart: list[float]
    data_timestamp: str
    data_age_seconds: float
    model_id: str
    model_verified: bool
    signal_threshold: float
    quality_metrics: dict[str, object]
    generated_at: str
    source: str


class HealthResponse(BaseModel):
    status: str
    mode: str
    timestamp: str


def get_cache_key(symbol: str, horizon: int) -> str:
    return f"{symbol.upper()}:{horizon}"


def get_cached_prediction(symbol: str, horizon: int) -> Optional[Dict[str, object]]:
    key = get_cache_key(symbol, horizon)
    cached_item = CACHE.get(key)
    if not cached_item:
        return None
    expires_at = cached_item.get("expires_at")
    if expires_at is None or datetime.now(timezone.utc) > expires_at:
        CACHE.pop(key, None)
        return None
    return cached_item


def set_cached_prediction(symbol: str, horizon: int, payload: Dict[str, object]) -> None:
    CACHE[get_cache_key(symbol, horizon)] = {
        **payload,
        "expires_at": datetime.now(timezone.utc) + timedelta(seconds=CACHE_TTL_SECONDS),
    }


def raise_api_error(exc: Exception) -> None:
    if isinstance(exc, ModelNotFoundError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, StaleMarketDataError):
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    raise HTTPException(status_code=503, detail=f"Prediction service unavailable: {exc}") from exc


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", mode="paper", timestamp=datetime.now(timezone.utc).isoformat())


@router.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    cached = get_cached_prediction(payload.symbol, payload.horizon)
    if cached is not None:
        return PredictionResponse(**cached)
    try:
        prediction = inference_service.predict(payload.symbol, horizon=payload.horizon)
        response_payload = {**prediction, "generated_at": datetime.now(timezone.utc).isoformat()}
        set_cached_prediction(payload.symbol, payload.horizon, response_payload)
        return PredictionResponse(**response_payload)
    except Exception as exc:
        raise_api_error(exc)


@router.post("/signal")
def signal(payload: PredictionRequest) -> dict:
    try:
        return trading_service.signal(payload.symbol, horizon=payload.horizon)
    except Exception as exc:
        raise_api_error(exc)


@router.post("/paper/run")
def run_paper_cycle(payload: PredictionRequest) -> dict:
    try:
        return trading_service.run_paper_cycle(payload.symbol, horizon=payload.horizon)
    except Exception as exc:
        raise_api_error(exc)


@router.get("/paper/account")
def paper_account() -> dict:
    return trading_service.broker.snapshot()


@router.get("/paper/trades")
def paper_trades(limit: int = Query(default=50, ge=1, le=500)) -> dict:
    return {"trades": trading_service.broker.recent_trades(limit=limit)}


@router.get("/performance")
def prediction_performance(
    symbol: str = Query(min_length=5, max_length=20),
    horizon: int = Query(default=1),
    limit: int = Query(default=20, ge=1, le=100),
) -> dict:
    try:
        if horizon not in (1, 4, 24):
            raise ValueError("Supported horizons are [1, 4, 24]")
        return trading_service.tracker.performance(normalize_symbol(symbol), horizon, limit=limit)
    except Exception as exc:
        raise_api_error(exc)


@router.get("/performance/all")
def all_prediction_performance(
    horizon: int = Query(default=1),
    limit: int = Query(default=5, ge=1, le=20),
) -> dict:
    try:
        if horizon not in (1, 4, 24):
            raise ValueError("Supported horizons are [1, 4, 24]")
        coins = trading_service.tracker.performance_many(COINCAST_SYMBOLS, horizon, limit=limit)
        return {"horizon": horizon, "coins": coins}
    except Exception as exc:
        raise_api_error(exc)


@router.get("/report/daily/preview")
def preview_daily_report() -> dict:
    return daily_report_service.build()


@router.post("/report/daily/send")
def send_daily_report() -> dict:
    return daily_report_service.send()

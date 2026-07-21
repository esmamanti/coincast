from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ml_backend.inference.service import InferenceService

router = APIRouter()

CACHE: Dict[str, Dict[str, object]] = {}
CACHE_TTL_SECONDS = 60
inference_service = InferenceService()


class PredictionRequest(BaseModel):
    symbol: str
    horizon: int = 1


class PredictionResponse(BaseModel):
    symbol: str
    horizon: int
    predicted_return: float
    confidence_interval: dict[str, float]
    generated_at: str
    source: str


class HealthResponse(BaseModel):
    status: str
    timestamp: str


def get_cache_key(symbol: str, horizon: int) -> str:
    return f"{symbol}:{horizon}"


def get_cached_prediction(symbol: str, horizon: int) -> Optional[Dict[str, object]]:
    cached_item = CACHE.get(get_cache_key(symbol, horizon))
    if not cached_item:
        return None
    expires_at = cached_item.get("expires_at")
    if expires_at is None or datetime.now(timezone.utc) > expires_at:
        CACHE.pop(get_cache_key(symbol, horizon), None)
        return None
    return cached_item


def set_cached_prediction(symbol: str, horizon: int, payload: Dict[str, object]) -> None:
    CACHE[get_cache_key(symbol, horizon)] = {
        **payload,
        "expires_at": datetime.now(timezone.utc) + timedelta(seconds=CACHE_TTL_SECONDS),
    }


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", timestamp=datetime.now(timezone.utc).isoformat())


@router.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    cached = get_cached_prediction(payload.symbol, payload.horizon)
    if cached is not None:
        return PredictionResponse(**cached)

    try:
        prediction = inference_service.predict(payload.symbol, horizon=payload.horizon)
        response_payload = {
            **prediction,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        set_cached_prediction(payload.symbol, payload.horizon, response_payload)
        return PredictionResponse(**response_payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

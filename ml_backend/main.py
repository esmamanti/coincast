import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="CoinCast ML Backend", version="1.0.0")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("coincast_backend")

ALLOWED_ORIGINS = [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if origin.strip()]
API_KEYS = {
    "crypto_panic": os.getenv("CRYPTO_PANIC_API_KEY", ""),
    "binance": os.getenv("BINANCE_API_KEY", ""),
}

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    symbol: str
    horizon: int = 1


class PredictionResponse(BaseModel):
    symbol: str
    horizon: int
    predicted_return: float
    generated_at: str
    source: str


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    config: Dict[str, str]


# Simple in-memory cache with TTL
CACHE: Dict[str, Dict[str, object]] = {}
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "60"))


def get_cache_key(symbol: str, horizon: int) -> str:
    return f"{symbol}:{horizon}"


def get_cached_prediction(symbol: str, horizon: int) -> Optional[Dict[str, object]]:
    key = get_cache_key(symbol, horizon)
    cached_item = CACHE.get(key)
    if not cached_item:
        return None

    expires_at = cached_item.get("expires_at")
    if expires_at is None or datetime.utcnow() > expires_at:
        CACHE.pop(key, None)
        return None

    return cached_item


def set_cached_prediction(symbol: str, horizon: int, payload: Dict[str, object]) -> None:
    key = get_cache_key(symbol, horizon)
    CACHE[key] = {
        **payload,
        "expires_at": datetime.utcnow() + timedelta(seconds=CACHE_TTL_SECONDS),
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    logger.info("Health check requested")
    return HealthResponse(
        status="ok",
        timestamp=datetime.utcnow().isoformat(),
        config={
            "allowed_origins": ",".join(ALLOWED_ORIGINS),
            "cache_ttl_seconds": str(CACHE_TTL_SECONDS),
            "crypto_panic_configured": "true" if API_KEYS["crypto_panic"] else "false",
        },
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    logger.info("Prediction request received for %s horizon=%s", payload.symbol, payload.horizon)

    cached = get_cached_prediction(payload.symbol, payload.horizon)
    if cached is not None:
        logger.info("Returning cached prediction for %s", payload.symbol)
        return PredictionResponse(**cached)

    try:
        if not payload.symbol:
            raise ValueError("symbol is required")

        # Lightweight prediction placeholder for now; this can be replaced by the trained model pipeline later.
        predicted_return = 0.0012 if payload.horizon <= 1 else 0.0034

        response_payload = {
            "symbol": payload.symbol.upper(),
            "horizon": payload.horizon,
            "predicted_return": predicted_return,
            "generated_at": datetime.utcnow().isoformat(),
            "source": "mock_model",
        }
        set_cached_prediction(payload.symbol, payload.horizon, response_payload)
        return PredictionResponse(**response_payload)
    except Exception as exc:
        logger.exception("Prediction failed for %s", payload.symbol)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/debug/config")
def debug_config() -> Dict[str, object]:
    return {
        "allowed_origins": ALLOWED_ORIGINS,
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
        "configured_keys": {k: bool(v) for k, v in API_KEYS.items()},
    }

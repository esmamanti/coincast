import os
import logging
import joblib
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models_saved" / "xgb_ethusdt_return.pkl"
DATA_PATH = ROOT / "data_processed" / "ETHUSDT_features.csv"

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
        "expires_at": datetime.now(timezone.utc) + timedelta(seconds=CACHE_TTL_SECONDS),
    }


def load_model_and_features() -> tuple[object, pd.DataFrame]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Feature data not found at {DATA_PATH}")

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH, parse_dates=["open_time"]).set_index("open_time")
    return model, df


MODEL, FEATURE_DF = load_model_and_features()
FEATURE_COLUMNS = [c for c in FEATURE_DF.columns if c not in ["open", "high", "low", "close", "target", "target_return"]]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    logger.info("Health check requested")
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc).isoformat(),
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

        symbol = payload.symbol.upper()

        model = MODEL
        feature_df = FEATURE_DF.copy()
        latest_row = feature_df.iloc[[-1]][FEATURE_COLUMNS]
        predicted_return = float(model.predict(latest_row)[0])

        response_payload = {
            "symbol": symbol,
            "horizon": payload.horizon,
            "predicted_return": predicted_return,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": "xgb_model",
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

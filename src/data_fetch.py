"""
CoinCast - Veri Çekme Scripti
Binance'ten mevcut coinleri, Binance'te olmayanları (HYPE, KAS) MEXC'ten çeker.
Çıktı: data/ klasörüne her coin için bir CSV dosyası.
"""

import os
import time
import pandas as pd
from binance.client import Client
import ccxt

# ---------------------------------------------------------
# Ayarlar
# ---------------------------------------------------------

BINANCE_COINS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
    "LINKUSDT", "INJUSDT", "RENDERUSDT", "SUIUSDT",
    "ONDOUSDT", "DOGEUSDT", "AAVEUSDT",
]

# Binance'te spot olarak bulunmayan coinler -> MEXC'ten çekilecek
MEXC_COINS = {
    "HYPEUSDT": "HYPE/USDT",
    "KASUSDT": "KAS/USDT",
}

INTERVAL = "1h"
LOOKBACK_BINANCE = "730 days ago UTC"   # ~2 yıl
MEXC_START_DATE = "2023-01-01T00:00:00Z"

DATA_DIR = "data"


# ---------------------------------------------------------
# Binance'ten veri çekme
# ---------------------------------------------------------

def fetch_binance(symbol: str, interval: str = INTERVAL, lookback: str = LOOKBACK_BINANCE) -> pd.DataFrame:
    client = Client()  # public veri için API key gerekmiyor
    klines = client.get_historical_klines(symbol, interval, lookback)

    df = pd.DataFrame(klines, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df[["open_time", "open", "high", "low", "close", "volume"]]
    df[["open", "high", "low", "close", "volume"]] = df[
        ["open", "high", "low", "close", "volume"]
    ].astype(float)
    return df


# ---------------------------------------------------------
# MEXC'ten veri çekme (Binance'te olmayan coinler için)
# ---------------------------------------------------------

def fetch_mexc(symbol_ccxt: str, interval: str = INTERVAL, limit: int = 1000) -> pd.DataFrame:
    exchange = ccxt.mexc()
    exchange.load_markets()

    if symbol_ccxt not in exchange.symbols:
        raise ValueError(f"'{symbol_ccxt}' MEXC'te bulunamadı (spot market listesinde yok)")

    all_data = []
    since = exchange.parse8601(MEXC_START_DATE)
    now = exchange.milliseconds()
    ms_per_candle = exchange.parse_timeframe(interval) * 1000

    while since < now:
        ohlcv = exchange.fetch_ohlcv(symbol_ccxt, timeframe=interval, since=since, limit=limit)
        if not ohlcv:
            # Bu tarihte henüz veri yok (coin daha sonra listelenmiş olabilir).
            # Tarihi limit kadar mum ileri atlayıp tekrar dene.
            since += ms_per_candle * limit
            time.sleep(exchange.rateLimit / 1000)
            continue
        all_data += ohlcv
        new_since = ohlcv[-1][0] + 1
        if new_since <= since:
            break
        since = new_since
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(all_data, columns=["open_time", "open", "high", "low", "close", "volume"])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df.drop_duplicates(subset="open_time", inplace=True)
    return df


# ---------------------------------------------------------
# Ana çalıştırma
# ---------------------------------------------------------

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Binance coinleri
    for symbol in BINANCE_COINS:
        try:
            df = fetch_binance(symbol)
            df.to_csv(f"{DATA_DIR}/{symbol}.csv", index=False)
            print(f"OK (Binance): {symbol} -> {len(df)} satır")
        except Exception as e:
            print(f"HATA: {symbol} Binance'te atlandı -> {e}")

    # MEXC fallback coinleri
    for symbol, ccxt_symbol in MEXC_COINS.items():
        try:
            df = fetch_mexc(ccxt_symbol)
            df.to_csv(f"{DATA_DIR}/{symbol}.csv", index=False)
            print(f"OK (MEXC): {symbol} -> {len(df)} satır")
        except Exception as e:
            print(f"HATA: {symbol} MEXC'te de atlandı -> {e}")


if __name__ == "__main__":
    main()
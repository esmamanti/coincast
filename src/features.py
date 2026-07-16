import os
import pandas as pd
import pandas_ta as ta


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Teknik indikatörler
    df["rsi_14"] = ta.rsi(df["close"], length=14)
    df["rsi_7"] = ta.rsi(df["close"], length=7)
    df["rsi_21"] = ta.rsi(df["close"], length=21)

    macd = ta.macd(df["close"])
    df = pd.concat([df, macd], axis=1)

    bbands = ta.bbands(df["close"], length=20)
    df = pd.concat([df, bbands], axis=1)

    df["ema_12"] = ta.ema(df["close"], length=12)
    df["ema_26"] = ta.ema(df["close"], length=26)
    df["ema_diff"] = df["ema_12"] - df["ema_26"]

    df["roc_12"] = ta.roc(df["close"], length=12)
    df["mom_5"] = ta.mom(df["close"], length=5)
    df["willr_14"] = ta.willr(df["high"], df["low"], df["close"], length=14)
    df["adx_14"] = ta.adx(df["high"], df["low"], df["close"], length=14)["ADX_14"]

    stoch = ta.stoch(df["high"], df["low"], df["close"])
    df = pd.concat([df, stoch], axis=1)

    df["obv"] = ta.obv(df["close"], df["volume"])

    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    # Lag features (geçmiş fiyat bilgisi)
    for lag in [1, 2, 3, 4, 5]:
        df[f"close_lag_{lag}"] = df["close"].shift(lag)
    df["close_diff_1"] = df["close"] - df["close"].shift(1)
    df["close_diff_2"] = df["close"] - df["close"].shift(2)

    # Rolling istatistikler
    df["rolling_mean_7"] = df["close"].rolling(7).mean()
    df["rolling_std_7"] = df["close"].rolling(7).std()
    df["rolling_mean_30"] = df["close"].rolling(30).mean()
    df["volume_mean_7"] = df["volume"].rolling(7).mean()
    df["volume_mean_30"] = df["volume"].rolling(30).mean()

    # Hedef değişken: bir sonraki mumun getirisi (% değişim), ham fiyat DEĞİL.
    # Ham fiyat non-stationary bir seridir, getiri tahmini modelin
    # öğrenmesi için çok daha kolay ve anlamlı bir hedeftir.
    df["target_return"] = df["close"].pct_change().shift(-1)
    df["target"] = df["close"].shift(-1)  # geri dönüşüm/karşılaştırma için hâlâ tutuyoruz

    df.dropna(inplace=True)
    return df


def process_all_coins(data_dir="data", output_dir="data_processed"):
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(data_dir):
        if not file.endswith(".csv"):
            continue
        symbol = file.replace(".csv", "")
        df = pd.read_csv(f"{data_dir}/{file}", parse_dates=["open_time"])
        df.set_index("open_time", inplace=True)
        try:
            df_feat = add_features(df)
            df_feat.to_csv(f"{output_dir}/{symbol}_features.csv")
            print(f"OK: {symbol} -> {df_feat.shape[0]} satır, {df_feat.shape[1]} kolon")
        except Exception as e:
            print(f"HATA: {symbol} atlandı -> {e}")


if __name__ == "__main__":
    process_all_coins()
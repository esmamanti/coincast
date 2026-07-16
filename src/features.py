import pandas as pd
import pandas_ta as ta


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Teknik indikatörler
    df["rsi"] = ta.rsi(df["close"], length=14)

    macd = ta.macd(df["close"])
    df = pd.concat([df, macd], axis=1)

    bbands = ta.bbands(df["close"], length=20)
    df = pd.concat([df, bbands], axis=1)

    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    # Lag features (geçmiş fiyat bilgisi)
    for lag in [1, 2, 3]:
        df[f"close_lag_{lag}"] = df["close"].shift(lag)

    # Rolling istatistikler
    df["rolling_mean_7"] = df["close"].rolling(7).mean()
    df["rolling_std_7"] = df["close"].rolling(7).std()
    df["rolling_mean_30"] = df["close"].rolling(30).mean()

    # Hedef değişken: bir sonraki mumun kapanış fiyatı
    df["target"] = df["close"].shift(-1)

    df.dropna(inplace=True)
    return df
import os


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
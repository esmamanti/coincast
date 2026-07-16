"""
Tum coinler icin GRU (getiri bazli) modelini egitir, naive baseline ile karsilastirir.
Sonuclari data_processed/experiment_results.csv dosyasina kaydeder.
 
Calistirma:
    python src\\run_experiment.py
"""
 
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
 
from split import chronological_split, get_X_y, scale_features_and_target
from models.gru_model import create_sequences, train_gru
 
DATA_DIR = "data_processed"
WINDOW = 30
EPOCHS = 50
RESULTS_PATH = f"{DATA_DIR}/experiment_results.csv"
 
 
def run_for_symbol(symbol: str) -> dict:
    path = f"{DATA_DIR}/{symbol}_features.csv"
    df_feat = pd.read_csv(path, parse_dates=["open_time"])
    df_feat.set_index("open_time", inplace=True)
 
    # Cok kucuk veri setlerinde (ornegin HYPE gibi yeni coinler) window + split sonrasi
    # yeterli veri kalmayabilir, guvenlik kontrolu:
    if len(df_feat) < 500:
        raise ValueError(f"{symbol}: yeterli veri yok ({len(df_feat)} satir)")
 
    train, test = chronological_split(df_feat, test_ratio=0.2)
 
    feature_cols = [c for c in df_feat.columns if c not in ["open", "high", "low", "close", "target", "target_return"]]
 
    X_train, y_train = get_X_y(train, feature_cols, target_col="target_return")
    X_test, y_test = get_X_y(test, feature_cols, target_col="target_return")
 
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, x_scaler, y_scaler = scale_features_and_target(
        X_train, X_test, y_train, y_test
    )
 
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, window=WINDOW)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, window=WINDOW)
 
    if len(X_test_seq) < 50:
        raise ValueError(f"{symbol}: test seti window sonrasi cok kucuk kaldi ({len(X_test_seq)} satir)")
 
    model_gru, train_losses, test_losses = train_gru(
        X_train_seq, y_train_seq, X_test_seq, y_test_seq,
        input_size=X_train_seq.shape[2],
        epochs=EPOCHS,
        lr=5e-4,
    )
 
    model_gru.eval()
    with torch.no_grad():
        device = next(model_gru.parameters()).device
        return_preds_scaled = model_gru(
            torch.tensor(X_test_seq, dtype=torch.float32).to(device)
        ).cpu().numpy().flatten()
 
    return_preds = y_scaler.inverse_transform(return_preds_scaled.reshape(-1, 1)).flatten()
 
    close_prices_test = test["close"].values[WINDOW:]
    actual_next_prices = test["target"].values[WINDOW:]
    predicted_next_prices = close_prices_test * (1 + return_preds)
 
    mae_gru = mean_absolute_error(actual_next_prices, predicted_next_prices)
    rmse_gru = np.sqrt(mean_squared_error(actual_next_prices, predicted_next_prices))
 
    # Naive baseline: "bir sonraki fiyat = simdiki fiyat"
    naive_mae = mean_absolute_error(actual_next_prices, close_prices_test)
 
    # Coin'in ortalama fiyati ve volatilitesi (baglam icin)
    avg_price = df_feat["close"].mean()
    price_std = df_feat["close"].std()
    volatility_pct = (df_feat["close"].pct_change().std()) * 100  # ortalama saatlik oynaklik %
 
    return {
        "symbol": symbol,
        "avg_price": round(avg_price, 4),
        "hourly_volatility_pct": round(volatility_pct, 4),
        "gru_mae": round(mae_gru, 4),
        "gru_rmse": round(rmse_gru, 4),
        "naive_mae": round(naive_mae, 4),
        "gru_vs_naive_ratio": round(mae_gru / naive_mae, 3),  # 1.0 = esit, <1 = GRU daha iyi
        "n_test_rows": len(X_test_seq),
    }
 
 
def main():
    files = [f for f in os.listdir(DATA_DIR) if f.endswith("_features.csv")]
    symbols = [f.replace("_features.csv", "") for f in files]
 
    results = []
    for symbol in symbols:
        print(f"\n=== {symbol} ===")
        try:
            res = run_for_symbol(symbol)
            results.append(res)
            print(f"OK: {symbol} -> GRU MAE: {res['gru_mae']}, Naive MAE: {res['naive_mae']}, "
                  f"Oran: {res['gru_vs_naive_ratio']}")
        except Exception as e:
            print(f"HATA: {symbol} atlandi -> {e}")
 
    results_df = pd.DataFrame(results).sort_values("gru_vs_naive_ratio")
    results_df.to_csv(RESULTS_PATH, index=False)
 
    print("\n\n=== SONUC TABLOSU (kucukten buyuge, 1.0 alti GRU'nun naive'i gectigi anlamina gelir) ===")
    print(results_df.to_string(index=False))
    print(f"\nSonuclar kaydedildi: {RESULTS_PATH}")
 
 
if __name__ == "__main__":
    main()
 
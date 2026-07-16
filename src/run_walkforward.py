import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

# Ensure project root is on sys.path so `src` imports work when executed from repo root
sys.path.append(os.path.abspath('.'))
from src.split import get_X_y

DATA_DIR = 'data_processed'
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Walk-forward parameters
TRAIN_START_PCT = 0.6
TEST_PCT = 0.1
MIN_TRAIN_ROWS = 200

summary = []

files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('_features.csv')])
for f in files:
    symbol = f.replace('_features.csv','')
    path = os.path.join(DATA_DIR, f)
    print(f"\nProcessing {symbol}")
    df = pd.read_csv(path, parse_dates=['open_time']).set_index('open_time')
    n = len(df)
    if n < MIN_TRAIN_ROWS:
        print(f"Skipping {symbol}: not enough rows ({n})")
        continue

    train_start = int(n * TRAIN_START_PCT)
    test_size = max(1, int(n * TEST_PCT))
    if train_start + test_size > n:
        print(f"Skipping {symbol}: not enough data for one test fold")
        continue

    feature_cols = [c for c in df.columns if c not in ['open','high','low','close','target','target_return']]

    fold = 0
    maes = []
    while train_start + test_size <= n:
        fold += 1
        train = df.iloc[:train_start]
        test = df.iloc[train_start:train_start+test_size]
        X_train, y_train = get_X_y(train, feature_cols, target_col='target_return')
        X_test, y_test = get_X_y(test, feature_cols, target_col='target_return')
        try:
            model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, verbosity=0)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            maes.append(mae)
            print(f"  Fold {fold}: rows train={len(train)}, test={len(test)}, MAE={mae:.6f}")
        except Exception as e:
            print(f"  Fold {fold} error: {e}")
        train_start += test_size

    if len(maes)==0:
        print(f"No valid folds for {symbol}")
        continue

    result = {
        'symbol': symbol,
        'n_rows': int(n),
        'n_folds': int(len(maes)),
        'mae_mean': float(np.mean(maes)),
        'mae_std': float(np.std(maes)),
        'mae_folds': [float(x) for x in maes]
    }
    summary.append(result)

# Save summary
summary_path = os.path.join(RESULTS_DIR, 'walkforward_summary.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

# Also CSV summary
import csv
csv_path = os.path.join(RESULTS_DIR, 'walkforward_summary.csv')
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['symbol','n_rows','n_folds','mae_mean','mae_std'])
    for r in summary:
        writer.writerow([r['symbol'], r['n_rows'], r['n_folds'], r['mae_mean'], r['mae_std']])

print('\nDone. Summary saved to', summary_path, 'and', csv_path)

import os
import sys
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.split import chronological_split

SYMBOL = 'ETHUSDT'
DATA_PATH = f'data_processed/{SYMBOL}_features.csv'
MODEL_PATH = f'models_saved/xgb_{SYMBOL.lower()}_return.pkl'
REPORT_PATH = f'results/{SYMBOL}_xgb_price_prediction_report.json'

os.makedirs('models_saved', exist_ok=True)
os.makedirs('results', exist_ok=True)

print(f'Loading {SYMBOL} features from {DATA_PATH}')
df = pd.read_csv(DATA_PATH, parse_dates=['open_time']).set_index('open_time')
feature_cols = [c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'target', 'target_return']]

train, test = chronological_split(df, test_ratio=0.2)

X_train = train[feature_cols]
y_train = train['target_return']
X_test = test[feature_cols]
y_test = test['target_return']

print('Train/test shapes:', X_train.shape, X_test.shape)

model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0,
)
model.fit(X_train, y_train)

preds = model.predict(X_test)
return_mae = mean_absolute_error(y_test, preds)
print(f'MAE on target_return (test): {return_mae:.6f}')

close_prices = test['close'].values
predicted_prices = close_prices * (1 + preds)
actual_prices = test['target'].values
price_mae = mean_absolute_error(actual_prices, predicted_prices)
naive_price_mae = mean_absolute_error(actual_prices, close_prices)
print(f'Price MAE: {price_mae:.4f}')
print(f'Naive price MAE (next price = current close): {naive_price_mae:.4f}')
print(f'Improvement vs naive price baseline: {naive_price_mae / price_mae:.3f}')

joblib.dump(model, MODEL_PATH)
print(f'Saved XGBoost return model to {MODEL_PATH}')

# Simple price-based trading signal
signals = np.where(preds > 0, 1, -1)
realized_returns = test['target_return'].values[1:]
signal_returns = signals[:-1] * realized_returns

cum_return = np.nansum(signal_returns)
mean_ret = np.nanmean(signal_returns)
print(f'Simple strategy cumulative return: {cum_return:.6f}')
print(f'Mean return per period: {mean_ret:.6e}')

report = {
    'symbol': SYMBOL,
    'n_test': int(len(X_test)),
    'return_mae': float(return_mae),
    'price_mae': float(price_mae),
    'naive_price_mae': float(naive_price_mae),
    'price_improvement_ratio': float(naive_price_mae / price_mae),
    'strategy_cumulative_return': float(cum_return),
    'strategy_mean_return': float(mean_ret),
}
with open(REPORT_PATH, 'w') as f:
    json.dump(report, f, indent=2)
print(f'Report saved to {REPORT_PATH}')

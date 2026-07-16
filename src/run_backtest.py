import pandas as pd
import numpy as np
import joblib
import os

MODEL_PATH = 'models_saved/xgb_eth_quick.pkl'
DATA_PATH = 'data_processed/ETHUSDT_features.csv'

COMMISSION = 0.0005  # 0.05% per trade
SLIPPAGE = 0.0005    # 0.05% slippage
HOURS_PER_YEAR = 8766

print('Loading model...')
model = joblib.load(MODEL_PATH)
print('Loading data...')
df = pd.read_csv(DATA_PATH, parse_dates=['open_time']).set_index('open_time')
feature_cols = [c for c in df.columns if c not in ['open','high','low','close','target','target_return']]

split = int(len(df)*0.8)
train = df.iloc[:split]
test = df.iloc[split:]

X_test = test[feature_cols]
true_returns = test['target_return'].values

print('Predicting...')
preds = model.predict(X_test)

# Signals: 1 for long, -1 for short, 0 for flat (we'll use hold 1 period)
signals = np.where(preds > 0, 1, -1)

# Simulate execution: assume we enter at close price at time t, realize return at t+1
# Apply slippage and commission at entry and exit (both sides simplified)

prices = test['close'].values

# Align signals -> realized returns
# signal at index i uses return at i+1 (target_return aligned with next period)
signal_aligned = signals[:-1]
realized_returns = true_returns[1:]
entry_prices = prices[:-1]

# Apply slippage (worse fill): for long, effective entry price = entry*(1+slippage); for short, entry*(1-slippage)
entry_adj = np.where(signal_aligned==1, entry_prices*(1+SLIPPAGE), entry_prices*(1-SLIPPAGE))
# exit price assume next close: prices[1:]
exit_prices = prices[1:]
# apply slippage on exit (adverse)
exit_adj = np.where(signal_aligned==1, exit_prices*(1-SLIPPAGE), exit_prices*(1+SLIPPAGE))

# gross return per trade = (exit_adj/entry_adj - 1) * position_sign
returns_per_trade = (exit_adj / entry_adj - 1) * signal_aligned

# Subtract commission (entry + exit)
returns_per_trade -= 2 * COMMISSION * np.abs(signal_aligned)

# Clean NaNs
mask = ~np.isnan(returns_per_trade)
strategy_returns = returns_per_trade[mask]

cum_return = np.nansum(strategy_returns)
mean_ret = np.nanmean(strategy_returns)
vol = np.nanstd(strategy_returns)
sharpe = (mean_ret / vol) * np.sqrt(HOURS_PER_YEAR) if vol>0 else np.nan

# Max drawdown
cum_values = np.cumsum(strategy_returns)
peak = np.maximum.accumulate(cum_values)
drawdowns = (cum_values - peak)
max_dd = drawdowns.min()

print('Results:')
print(f'Periods: {len(strategy_returns)}, Cumulative return (sum of hourly returns): {cum_return:.6f}')
print(f'Mean return per period: {mean_ret:.6e}, Volatility per period: {vol:.6e}')
print(f'Approx Sharpe (annualized): {sharpe:.3f}')
print(f'Max drawdown (sum series): {max_dd:.6f}')

# Save simple report
report = {
    'symbol': 'ETHUSDT',
    'periods': int(len(strategy_returns)),
    'cumulative_return': float(cum_return),
    'mean_ret': float(mean_ret),
    'vol': float(vol),
    'sharpe': float(sharpe) if not np.isnan(sharpe) else None,
    'max_drawdown': float(max_dd),
}

os.makedirs('results', exist_ok=True)
import json
with open('results/backtest_eth_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print('Report saved to results/backtest_eth_report.json')

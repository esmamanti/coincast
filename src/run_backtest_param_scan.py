import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

DATA_DIR = 'data_processed'
RESULTS_DIR = 'results/param_scan'
os.makedirs(RESULTS_DIR, exist_ok=True)

POSITION_FRAC = 0.01
COMMISSION = 0.0005
SLIPPAGE = 0.0005
THRESHOLDS = [0.0, 0.0005, 0.001, 0.002]
HOURS_PER_YEAR = 8766

summary = []
files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('_features.csv')])
for file_name in files:
    symbol = file_name.replace('_features.csv', '')
    path = os.path.join(DATA_DIR, file_name)
    print(f"\n=== {symbol} ===")
    df = pd.read_csv(path, parse_dates=['open_time']).set_index('open_time')
    feature_cols = [c for c in df.columns if c not in ['open','high','low','close','target','target_return']]

    split = int(len(df) * 0.8)
    train = df.iloc[:split]
    test = df.iloc[split:]

    X_train = train[feature_cols]; y_train = train['target_return']
    X_test = test[feature_cols]; y_test = test['target_return']

    model = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, verbosity=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    for threshold in THRESHOLDS:
        signal = np.zeros_like(preds, dtype=int)
        signal[preds > threshold] = 1
        signal[preds < -threshold] = -1

        equity = 10000.0
        equity_curve = []
        trade_count = 0
        pnls = []

        prices = test['close'].values
        returns = test['target_return'].values

        for i in range(len(signal)-1):
            sig = signal[i]
            if sig == 0:
                equity_curve.append(equity)
                continue

            entry_price = prices[i]
            exit_price = prices[i+1]
            position_value = equity * POSITION_FRAC
            qty = position_value / entry_price if entry_price > 0 else 0

            if sig == 1:
                entry_adj = entry_price * (1 + SLIPPAGE)
                exit_adj = exit_price * (1 - SLIPPAGE)
            else:
                entry_adj = entry_price * (1 - SLIPPAGE)
                exit_adj = exit_price * (1 + SLIPPAGE)

            gross = qty * (exit_adj - entry_adj) * sig
            commission_cost = (entry_adj * qty + exit_adj * qty) * COMMISSION
            pnl = gross - commission_cost
            equity += pnl
            equity_curve.append(equity)
            pnls.append(pnl)
            trade_count += 1

        if len(pnls) == 0:
            avg_pnl = 0.0
            vol = 0.0
            sharpe = None
            cum = 0.0
            max_dd = 0.0
        else:
            avg_pnl = float(np.mean(pnls))
            vol = float(np.std(pnls))
            sharpe = float((avg_pnl / vol) * np.sqrt(HOURS_PER_YEAR)) if vol > 0 else None
            cum = float(np.sum(pnls))
            equity_vals = np.array(equity_curve)
            peaks = np.maximum.accumulate(equity_vals)
            drawdowns = equity_vals - peaks
            max_dd = float(np.min(drawdowns))

        result = {
            'symbol': symbol,
            'threshold': threshold,
            'position_frac': POSITION_FRAC,
            'trade_count': int(trade_count),
            'cumulative_pnl': cum,
            'final_equity': float(equity),
            'mean_pnl': avg_pnl,
            'pnl_vol': vol,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'mae': float(mean_absolute_error(y_test, preds)),
        }
        summary.append(result)
        print(f" t={threshold}: trades={trade_count}, final_equity={equity:.2f}, cum_pnl={cum:.2f}, sharpe={sharpe}")

with open(os.path.join(RESULTS_DIR, 'param_scan_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

pd.DataFrame(summary).to_csv(os.path.join(RESULTS_DIR, 'param_scan_summary.csv'), index=False)
print('\nDone. Summary saved to', os.path.join(RESULTS_DIR, 'param_scan_summary.csv'))

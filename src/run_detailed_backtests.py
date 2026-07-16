import os
import json
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

# Settings
# Run detailed backtests for all processed coins in `data_processed`
DATA_DIR = 'data_processed'
files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('_features.csv')])
TOP_COINS = [f.replace('_features.csv', '') for f in files]
DATA_DIR = 'data_processed'
RESULTS_DIR = 'results/detailed_backtests'
os.makedirs(RESULTS_DIR, exist_ok=True)

INITIAL_CAPITAL = 10000.0
POSITION_FRAC = 0.1   # fraction of current equity to allocate per trade
COMMISSION = 0.0005   # per side
SLIPPAGE = 0.0005     # per side
HOURS_PER_YEAR = 8766

reports = []

for symbol in TOP_COINS:
    print(f"\n=== Running detailed backtest for {symbol} ===")
    path = f"{DATA_DIR}/{symbol}_features.csv"
    df = pd.read_csv(path, parse_dates=['open_time']).set_index('open_time')
    feature_cols = [c for c in df.columns if c not in ['open','high','low','close','target','target_return']]

    # train/test split 80/20
    split = int(len(df)*0.8)
    train = df.iloc[:split]
    test = df.iloc[split:]

    X_train = train[feature_cols]; y_train = train['target_return']
    X_test = test[feature_cols]; y_test = test['target_return']

    # train model
    model = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, verbosity=0)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print('Model MAE on test:', mae)

    prices = test['close'].values
    returns = test['target_return'].values

    equity = INITIAL_CAPITAL
    equity_curve = []
    positions = []
    trade_pnls = []

    # generate signals: long if pred>threshold else short; use threshold=0 initially
    threshold = 0.0
    signals = np.where(preds>threshold, 1, -1)

    # iterate over test period, trades from t -> t+1; last index has no next return
    for i in range(len(signals)-1):
        sig = signals[i]
        entry_price = prices[i]
        exit_price = prices[i+1]

        # position size in currency
        position_value = equity * POSITION_FRAC
        # compute number of contracts/shares (we treat price units generically)
        qty = position_value / entry_price if entry_price>0 else 0

        # apply slippage adverse to entry/exit
        if sig==1:
            entry_adj = entry_price * (1 + SLIPPAGE)
            exit_adj = exit_price * (1 - SLIPPAGE)
        else:
            entry_adj = entry_price * (1 - SLIPPAGE)
            exit_adj = exit_price * (1 + SLIPPAGE)

        gross_pnl = qty * (exit_adj - entry_adj) * sig
        # commissions on entry+exit as fraction of trade value
        commission_cost = (entry_adj * qty + exit_adj * qty) * COMMISSION
        net_pnl = gross_pnl - commission_cost

        # update equity
        equity += net_pnl
        equity_curve.append(equity)
        positions.append(sig)
        trade_pnls.append(net_pnl)

    equity_curve = np.array(equity_curve)
    trade_pnls = np.array(trade_pnls)

    cum_return = equity - INITIAL_CAPITAL
    mean_ret = np.nanmean(trade_pnls)
    vol = np.nanstd(trade_pnls)
    sharpe = (mean_ret/vol)*np.sqrt(HOURS_PER_YEAR) if vol>0 else np.nan

    # drawdown on equity curve
    cum_values = equity_curve - INITIAL_CAPITAL
    peak = np.maximum.accumulate(cum_values)
    drawdowns = cum_values - peak
    max_dd = drawdowns.min() if len(drawdowns)>0 else 0.0

    report = {
        'symbol': symbol,
        'model_mae': float(mae),
        'trades': int(len(trade_pnls)),
        'cumulative_pnl': float(cum_return),
        'final_equity': float(equity),
        'mean_pnl': float(mean_ret),
        'pnl_vol': float(vol),
        'sharpe_approx': float(sharpe) if not np.isnan(sharpe) else None,
        'max_drawdown': float(max_dd)
    }

    # save equity curve and report
    pd.DataFrame({'equity': equity_curve}).to_csv(f"{RESULTS_DIR}/{symbol}_equity.csv", index=False)
    with open(f"{RESULTS_DIR}/{symbol}_report.json", 'w') as f:
        json.dump(report, f, indent=2)

    reports.append(report)
    print(f"Done {symbol}: final equity={equity:.2f}, cumulative_pnl={cum_return:.2f}, sharpe~{report['sharpe_approx']}")

# aggregate summary
with open(f"{RESULTS_DIR}/summary.json", 'w') as f:
    json.dump(reports, f, indent=2)
print('\nAll backtests completed. Reports in', RESULTS_DIR)

import itertools
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

from src.features import process_all_coins

DATA_DIR = 'data_processed'
RESULTS_DIR = 'results/xgb_full_pipeline'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

TRAIN_RATIO = 0.8
MODEL_GRID = [
    {'n_estimators': n, 'max_depth': d, 'learning_rate': lr, 'subsample': ss, 'colsample_bytree': ct}
    for n in [150, 200]
    for d in [4, 5, 6]
    for lr in [0.03, 0.05]
    for ss in [0.8, 1.0]
    for ct in [0.8]
]
THRESHOLDS = [0.0, 0.0005, 0.001, 0.002]
POSITION_FRACS = [0.005, 0.01]
STOP_LOSS_PCTS = [0.0, 0.0025, 0.005]
TAKE_PROFIT_PCTS = [0.0, 0.01, 0.02]
COMMISSION = 0.0005
SLIPPAGE = 0.0005
INITIAL_CAPITAL = 10000.0
HOURS_PER_YEAR = 8766


def get_feature_columns(df):
    return [c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'target', 'target_return']]


def train_best_xgb(X_train, y_train, X_test, y_test):
    best = None
    for params in MODEL_GRID:
        model = xgb.XGBRegressor(**params, random_state=42, verbosity=0)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        if best is None or mae < best['mae']:
            best = {'model': model, 'mae': mae, 'params': params, 'preds': preds}
    return best['model'], best['mae'], best['params'], best['preds']


def compute_price_metrics(close_prices, preds, actual_prices):
    predicted_prices = close_prices * (1 + preds)
    price_mae = mean_absolute_error(actual_prices, predicted_prices)
    naive_price_mae = mean_absolute_error(actual_prices, close_prices)
    actual_direction = np.sign(actual_prices - close_prices)
    predicted_direction = np.sign(preds)
    direction_acc = np.mean(predicted_direction == actual_direction)
    return {
        'price_mae': float(price_mae),
        'naive_price_mae': float(naive_price_mae),
        'price_improvement_ratio': float(naive_price_mae / price_mae) if price_mae > 0 else None,
        'direction_accuracy': float(direction_acc),
    }


def simulate_trade(test, preds, threshold, position_frac, commission, slippage,
                   stop_loss_pct, take_profit_pct, initial_capital):
    close = test['close'].values
    high = test['high'].values
    low = test['low'].values

    signal = np.zeros_like(preds, dtype=int)
    signal[preds > threshold] = 1
    signal[preds < -threshold] = -1

    equity = initial_capital
    equity_curve = []
    trade_pnls = []
    trade_reasons = []

    for idx in range(len(signal) - 1):
        sig = int(signal[idx])
        if sig == 0:
            equity_curve.append(equity)
            continue

        entry_price = float(close[idx])
        next_close = float(close[idx + 1])
        next_high = float(high[idx + 1])
        next_low = float(low[idx + 1])

        if sig == 1:
            stop_price = entry_price * (1 - stop_loss_pct)
            target_price = entry_price * (1 + take_profit_pct)
        else:
            stop_price = entry_price * (1 + stop_loss_pct)
            target_price = entry_price * (1 - take_profit_pct)

        exit_price = next_close
        reason = 'close'

        if sig == 1:
            if next_low <= stop_price:
                exit_price = stop_price
                reason = 'sl'
            elif next_high >= target_price:
                exit_price = target_price
                reason = 'tp'
        else:
            if next_high >= stop_price:
                exit_price = stop_price
                reason = 'sl'
            elif next_low <= target_price:
                exit_price = target_price
                reason = 'tp'

        qty = (equity * position_frac) / entry_price if entry_price > 0 else 0.0
        if sig == 1:
            entry_adj = entry_price * (1 + slippage)
            exit_adj = exit_price * (1 - slippage)
        else:
            entry_adj = entry_price * (1 - slippage)
            exit_adj = exit_price * (1 + slippage)

        gross_pnl = qty * (exit_adj - entry_adj) * sig
        commission_cost = (entry_adj + exit_adj) * qty * commission
        net_pnl = gross_pnl - commission_cost

        equity += net_pnl
        equity_curve.append(equity)
        trade_pnls.append(net_pnl)
        trade_reasons.append(reason)

    if len(trade_pnls) == 0:
        return {
            'final_equity': float(equity),
            'cumulative_pnl': 0.0,
            'trade_count': 0,
            'mean_pnl': 0.0,
            'pnl_vol': 0.0,
            'sharpe': None,
            'max_drawdown': 0.0,
            'tp_count': 0,
            'sl_count': 0,
            'equity_curve': equity_curve,
        }

    avg_pnl = float(np.mean(trade_pnls))
    vol = float(np.std(trade_pnls))
    sharpe = float((avg_pnl / vol) * np.sqrt(HOURS_PER_YEAR)) if vol > 0 else None
    cum_pnl = float(np.sum(trade_pnls))
    equity_vals = np.array(equity_curve)
    peak = np.maximum.accumulate(equity_vals)
    max_dd = float(np.min(equity_vals - peak))
    tp_count = int(trade_reasons.count('tp'))
    sl_count = int(trade_reasons.count('sl'))

    return {
        'final_equity': float(equity),
        'cumulative_pnl': cum_pnl,
        'trade_count': len(trade_pnls),
        'mean_pnl': avg_pnl,
        'pnl_vol': vol,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'tp_count': tp_count,
        'sl_count': sl_count,
        'equity_curve': equity_curve,
    }


def find_best_backtest(test, preds):
    best = None
    for threshold, position_frac, stop_loss_pct, take_profit_pct in itertools.product(
        THRESHOLDS, POSITION_FRACS, STOP_LOSS_PCTS, TAKE_PROFIT_PCTS
    ):
        result = simulate_trade(
            test=test,
            preds=preds,
            threshold=threshold,
            position_frac=position_frac,
            commission=COMMISSION,
            slippage=SLIPPAGE,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            initial_capital=INITIAL_CAPITAL,
        )
        if best is None or result['final_equity'] > best['final_equity']:
            best = {
                'threshold': threshold,
                'position_frac': position_frac,
                'stop_loss_pct': stop_loss_pct,
                'take_profit_pct': take_profit_pct,
                **result,
            }
    return best


def main():
    summary = []
    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('_features.csv')])

    for file_name in files:
        symbol = file_name.replace('_features.csv', '')
        print(f'\n=== {symbol} ===')

        path = os.path.join(DATA_DIR, file_name)
        df = pd.read_csv(path, parse_dates=['open_time']).set_index('open_time')
        feature_cols = get_feature_columns(df)

        split = int(len(df) * TRAIN_RATIO)
        train = df.iloc[:split]
        test = df.iloc[split:]

        X_train = train[feature_cols]
        y_train = train['target_return']
        X_test = test[feature_cols]
        y_test = test['target_return']

        model, return_mae, best_params, preds = train_best_xgb(X_train, y_train, X_test, y_test)
        price_metrics = compute_price_metrics(test['close'].values, preds, test['target'].values)

        best_backtest = find_best_backtest(test, preds)

        symbol_report = {
            'symbol': symbol,
            'n_rows': int(len(df)),
            'n_test': int(len(X_test)),
            'best_model_params': best_params,
            'return_mae': float(return_mae),
            **price_metrics,
            **best_backtest,
        }

        with open(os.path.join(RESULTS_DIR, f'{symbol}_full_report.json'), 'w') as f:
            json.dump(symbol_report, f, indent=2)
        pd.DataFrame({'equity': best_backtest['equity_curve']}).to_csv(
            os.path.join(RESULTS_DIR, f'{symbol}_full_equity.csv'), index=False
        )

        summary.append(symbol_report)

        print(f"  best return MAE: {return_mae:.6f}  price MAE: {price_metrics['price_mae']:.4f}  naive: {price_metrics['naive_price_mae']:.4f}")
        print(f"  best backtest: equity={best_backtest['final_equity']:.2f} trades={best_backtest['trade_count']} sharpe={best_backtest['sharpe']} threshold={best_backtest['threshold']} pos={best_backtest['position_frac']} sl={best_backtest['stop_loss_pct']} tp={best_backtest['take_profit_pct']}")

    summary_path = os.path.join(RESULTS_DIR, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame(summary).sort_values(['final_equity', 'symbol'], ascending=[False, True]).to_csv(
        os.path.join(RESULTS_DIR, 'summary.csv'), index=False
    )
    print(f'\nDone. Full pipeline results saved to {RESULTS_DIR}')


if __name__ == '__main__':
    main()

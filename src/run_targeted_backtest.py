import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.models.xgb_model import train_xgb

DATA_DIR = 'data_processed'
RESULTS_DIR = 'results/targeted_backtest'
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_RATIO = 0.8
TARGET_SYMBOLS = ['KASUSDT', 'ONDOUSDT']
MODEL_PARAMS_BY_SYMBOL = {
    'KASUSDT': {
        'n_estimators': 150,
        'max_depth': 4,
        'learning_rate': 0.03,
        'subsample': 1.0,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbosity': 0,
    },
    'ONDOUSDT': {
        'n_estimators': 150,
        'max_depth': 4,
        'learning_rate': 0.03,
        'subsample': 1.0,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbosity': 0,
    },
}
THRESHOLDS = [0.003, 0.004, 0.005, 0.006]
POSITION_FRACS = [0.005, 0.01]
STOP_LOSS_PCTS = [0.0025, 0.005]
TAKE_PROFIT_PCTS = [0.02, 0.03]
COMMISSION = 0.0005
SLIPPAGE = 0.0005
INITIAL_CAPITAL = 10000.0
HOURS_PER_YEAR = 8766


def get_feature_columns(df):
    return [c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'target', 'target_return']]


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


def simulate_trade(test, preds, threshold, position_frac, stop_loss_pct, take_profit_pct):
    close = test['close'].values
    high = test['high'].values
    low = test['low'].values

    signal = np.zeros_like(preds, dtype=int)
    signal[preds > threshold] = 1
    signal[preds < -threshold] = -1

    equity = INITIAL_CAPITAL
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
            entry_adj = entry_price * (1 + SLIPPAGE)
            exit_adj = exit_price * (1 - SLIPPAGE)
        else:
            entry_adj = entry_price * (1 - SLIPPAGE)
            exit_adj = exit_price * (1 + SLIPPAGE)

        gross_pnl = qty * (exit_adj - entry_adj) * sig
        commission_cost = (entry_adj + exit_adj) * qty * COMMISSION
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
    for threshold in THRESHOLDS:
        for position_frac in POSITION_FRACS:
            for stop_loss_pct in STOP_LOSS_PCTS:
                for take_profit_pct in TAKE_PROFIT_PCTS:
                    result = simulate_trade(
                        test=test,
                        preds=preds,
                        threshold=threshold,
                        position_frac=position_frac,
                        stop_loss_pct=stop_loss_pct,
                        take_profit_pct=take_profit_pct,
                    )
                    result.update({
                        'threshold': threshold,
                        'position_frac': position_frac,
                        'stop_loss_pct': stop_loss_pct,
                        'take_profit_pct': take_profit_pct,
                    })
                    if best is None or result['final_equity'] > best['final_equity']:
                        best = result
    return best


def main():
    summary = []

    for symbol in TARGET_SYMBOLS:
        print(f'\n=== {symbol} ===')
        path = os.path.join(DATA_DIR, f'{symbol}_features.csv')
        if not os.path.exists(path):
            print('  missing data file:', path)
            continue

        df = pd.read_csv(path, parse_dates=['open_time']).set_index('open_time')
        feature_cols = get_feature_columns(df)
        split = int(len(df) * TRAIN_RATIO)
        train = df.iloc[:split]
        test = df.iloc[split:]

        X_train = train[feature_cols]
        y_train = train['target_return']
        X_test = test[feature_cols]
        y_test = test['target_return']

        coin_params = MODEL_PARAMS_BY_SYMBOL.get(symbol, {})
        model, preds, metrics = train_xgb(X_train, y_train, X_test, y_test, **coin_params)
        price_metrics = compute_price_metrics(test['close'].values, preds, test['target'].values)
        best_backtest = find_best_backtest(test, preds)

        report = {
            'symbol': symbol,
            'n_rows': int(len(df)),
            'n_test': int(len(X_test)),
            'model_params': coin_params,
            **metrics,
            **price_metrics,
            **best_backtest,
        }

        with open(os.path.join(RESULTS_DIR, f'{symbol}_targeted_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        pd.DataFrame({'equity': best_backtest['equity_curve']}).to_csv(
            os.path.join(RESULTS_DIR, f'{symbol}_targeted_equity.csv'), index=False
        )

        summary.append(report)
        print(f"  price_mae={price_metrics['price_mae']:.6f} naive={price_metrics['naive_price_mae']:.6f} dir_acc={price_metrics['direction_accuracy']:.4f}")
        print(f"  best equity={best_backtest['final_equity']:.2f} pnl={best_backtest['cumulative_pnl']:.2f} trades={best_backtest['trade_count']} sharpe={best_backtest['sharpe']} threshold={best_backtest['threshold']} pos={best_backtest['position_frac']} sl={best_backtest['stop_loss_pct']} tp={best_backtest['take_profit_pct']}")

    pd.DataFrame(summary).sort_values('final_equity', ascending=False).to_csv(
        os.path.join(RESULTS_DIR, 'summary.csv'), index=False
    )
    print(f'\nDone. Targeted results saved to {RESULTS_DIR}')


if __name__ == '__main__':
    main()

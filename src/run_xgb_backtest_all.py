import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

DATA_DIR = 'data_processed'
RESULTS_DIR = 'results/xgb_backtest_all'
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_RATIO = 0.8
POSITION_FRAC = 0.01
COMMISSION = 0.0005
SLIPPAGE = 0.0005
THRESHOLD = 0.002
STOP_LOSS_PCT = 0.005
TAKE_PROFIT_PCT = 0.01
INITIAL_CAPITAL = 10000.0
HOURS_PER_YEAR = 8766


def get_feature_columns(df):
    return [c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'target', 'target_return']]


def evaluate_price_prediction(close_prices, preds, actual_prices):
    predicted_prices = close_prices * (1 + preds)
    price_mae = mean_absolute_error(actual_prices, predicted_prices)
    naive_price_mae = mean_absolute_error(actual_prices, close_prices)
    direction_acc = np.mean(np.sign(preds) == np.sign(actual_prices - close_prices))
    return predicted_prices, price_mae, naive_price_mae, direction_acc


def backtest_with_risk(test, preds, threshold, position_frac, commission, slippage,
                       stop_loss_pct, take_profit_pct, initial_capital):
    close = test['close'].values
    high = test['high'].values
    low = test['low'].values
    returns = test['target_return'].values
    signal = np.zeros_like(preds, dtype=int)
    signal[preds > threshold] = 1
    signal[preds < -threshold] = -1

    equity = initial_capital
    equity_curve = []
    trade_pnls = []
    trade_reasons = []
    trade_count = 0

    for i in range(len(signal) - 1):
        sig = signal[i]
        if sig == 0:
            equity_curve.append(equity)
            continue

        entry_price = close[i]
        exit_price = close[i + 1]
        sl_price = entry_price * (1 - stop_loss_pct if sig == 1 else 1 + stop_loss_pct)
        tp_price = entry_price * (1 + take_profit_pct if sig == 1 else 1 - take_profit_pct)

        next_high = high[i + 1]
        next_low = low[i + 1]
        reason = 'close'

        if sig == 1:
            if next_low <= sl_price and next_high >= tp_price:
                exit_price = sl_price
                reason = 'sl'
            elif next_low <= sl_price:
                exit_price = sl_price
                reason = 'sl'
            elif next_high >= tp_price:
                exit_price = tp_price
                reason = 'tp'
        else:
            if next_high >= sl_price and next_low <= tp_price:
                exit_price = sl_price
                reason = 'sl'
            elif next_high >= sl_price:
                exit_price = sl_price
                reason = 'sl'
            elif next_low <= tp_price:
                exit_price = tp_price
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
        trade_count += 1

    if len(trade_pnls) > 0:
        avg_pnl = float(np.mean(trade_pnls))
        vol = float(np.std(trade_pnls))
        sharpe = float((avg_pnl / vol) * np.sqrt(HOURS_PER_YEAR)) if vol > 0 else None
        cum_pnl = float(np.sum(trade_pnls))
        equity_vals = np.array(equity_curve)
        peaks = np.maximum.accumulate(equity_vals)
        max_dd = float(np.min(equity_vals - peaks))
        tp_count = trade_reasons.count('tp')
        sl_count = trade_reasons.count('sl')
    else:
        avg_pnl = 0.0
        vol = 0.0
        sharpe = None
        cum_pnl = 0.0
        max_dd = 0.0
        tp_count = 0
        sl_count = 0

    return {
        'final_equity': float(equity),
        'cumulative_pnl': cum_pnl,
        'trade_count': int(trade_count),
        'mean_pnl': avg_pnl,
        'pnl_vol': vol,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'tp_count': int(tp_count),
        'sl_count': int(sl_count),
        'equity_curve': equity_curve,
    }


def main():
    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('_features.csv')])
    summary = []

    for file_name in files:
        symbol = file_name.replace('_features.csv', '')
        path = os.path.join(DATA_DIR, file_name)
        print(f'\n=== {symbol} ===')

        df = pd.read_csv(path, parse_dates=['open_time']).set_index('open_time')
        feature_cols = get_feature_columns(df)

        split = int(len(df) * TRAIN_RATIO)
        train = df.iloc[:split]
        test = df.iloc[split:]

        X_train = train[feature_cols]
        y_train = train['target_return']
        X_test = test[feature_cols]
        y_test = test['target_return']

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
        close_prices = test['close'].values
        actual_prices = test['target'].values
        _, price_mae, naive_price_mae, direction_acc = evaluate_price_prediction(close_prices, preds, actual_prices)

        backtest = backtest_with_risk(
            test=test,
            preds=preds,
            threshold=THRESHOLD,
            position_frac=POSITION_FRAC,
            commission=COMMISSION,
            slippage=SLIPPAGE,
            stop_loss_pct=STOP_LOSS_PCT,
            take_profit_pct=TAKE_PROFIT_PCT,
            initial_capital=INITIAL_CAPITAL,
        )

        print(f'  return MAE: {return_mae:.6f}')
        print(f'  price MAE: {price_mae:.4f}  naive price MAE: {naive_price_mae:.4f}  direction acc: {direction_acc:.3f}')
        print(f'  trades: {backtest['trade_count']}  final equity: {backtest['final_equity']:.2f}  pnl: {backtest['cumulative_pnl']:.2f}  sharpe: {backtest['sharpe']}')

        symbol_report = {
            'symbol': symbol,
            'n_rows': int(len(df)),
            'n_test': int(len(X_test)),
            'return_mae': float(return_mae),
            'price_mae': float(price_mae),
            'naive_price_mae': float(naive_price_mae),
            'price_improvement_ratio': float(naive_price_mae / price_mae) if price_mae > 0 else None,
            'direction_accuracy': float(direction_acc),
            'threshold': THRESHOLD,
            'position_frac': POSITION_FRAC,
            'stop_loss_pct': STOP_LOSS_PCT,
            'take_profit_pct': TAKE_PROFIT_PCT,
            'trade_count': backtest['trade_count'],
            'cumulative_pnl': backtest['cumulative_pnl'],
            'final_equity': backtest['final_equity'],
            'mean_pnl': backtest['mean_pnl'],
            'pnl_vol': backtest['pnl_vol'],
            'sharpe': backtest['sharpe'],
            'max_drawdown': backtest['max_drawdown'],
            'tp_count': backtest['tp_count'],
            'sl_count': backtest['sl_count'],
        }

        with open(os.path.join(RESULTS_DIR, f'{symbol}_report.json'), 'w') as f:
            json.dump(symbol_report, f, indent=2)
        pd.DataFrame({'equity': backtest['equity_curve']}).to_csv(os.path.join(RESULTS_DIR, f'{symbol}_equity.csv'), index=False)

        summary.append(symbol_report)

    summary_path = os.path.join(RESULTS_DIR, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame(summary).sort_values(['final_equity', 'symbol'], ascending=[False, True]).to_csv(
        os.path.join(RESULTS_DIR, 'summary.csv'), index=False
    )
    print(f'\nDone. Reports saved to {RESULTS_DIR}')


if __name__ == '__main__':
    main()

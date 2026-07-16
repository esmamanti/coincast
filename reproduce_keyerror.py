import pandas as pd
from src.split import chronological_split, get_X_y

print('Reading CSV')
df = pd.read_csv('data_processed/ETHUSDT_features.csv', parse_dates=['open_time'])
print('Columns:', df.columns.tolist())
df.set_index('open_time', inplace=True)
print('Index set. Length:', len(df))
train, test = chronological_split(df, test_ratio=0.2)
print('Train/Test sizes:', len(train), len(test))
feature_cols = [c for c in df.columns if c not in ['open','high','low','close','target','target_return']]
print('Feature cols count:', len(feature_cols))
print('Sample feature cols:', feature_cols[:10])
print('Has target_return:', 'target_return' in df.columns)
print('Has target:', 'target' in df.columns)

try:
    X_train, y_train = get_X_y(train, feature_cols, target_col='target_return')
    print('X_train shape', X_train.shape)
    print('y_train shape', y_train.shape)
except Exception as e:
    print('ERROR during get_X_y:', repr(e))
    raise

import pandas as pd
from sklearn.preprocessing import StandardScaler


def chronological_split(df: pd.DataFrame, test_ratio: float = 0.2, min_train_rows: int = 1):
    """
    Time-series split: the most recent test_ratio portion is reserved for testing.
    This avoids data leakage that would happen with random splitting.
    """
    if len(df) < 2:
        raise ValueError("Dataframe must contain at least two rows")

    split_idx = int(len(df) * (1 - test_ratio))
    split_idx = max(min_train_rows, split_idx)
    if split_idx >= len(df):
        raise ValueError("Not enough rows for a valid train/test split")

    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    return train, test


def get_X_y(df: pd.DataFrame, feature_cols: list, target_col: str = "target"):
    X = df[feature_cols]
    y = df[target_col]
    return X, y


def scale_features_and_target(X_train, X_test, y_train, y_test):
    """
    Feature'lar ve target AYRI scaler'larla scale edilir.
    Scaler'lar sadece train verisiyle fit edilir (veri sızıntısını önlemek için).
    y_scaler'ı sakla -> tahminleri gerçek dolar değerine geri çevirmek için gerekecek.
    """
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, x_scaler, y_scaler
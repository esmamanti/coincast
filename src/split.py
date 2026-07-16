import pandas as pd


def chronological_split(df: pd.DataFrame, test_ratio: float = 0.2):
    """
    Zaman serisinde rastgele split YAPILMAZ - veri sızıntısına yol açar.
    Son %test_ratio kısmı test seti olarak ayrılır.
    """
    split_idx = int(len(df) * (1 - test_ratio))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    return train, test


def get_X_y(df: pd.DataFrame, feature_cols: list, target_col: str = "target"):
    X = df[feature_cols]
    y = df[target_col]
    return X, y
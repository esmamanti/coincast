import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


def train_xgb(X_train, y_train, X_test, y_test, **kwargs):
    params = dict(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    params.update(kwargs)

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

    metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape}
    return model, preds, metrics
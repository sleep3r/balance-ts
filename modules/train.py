import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from utils.rates import get_rates

warnings.filterwarnings("ignore")


def train_model(
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        visualize: bool = False, **model_params
) -> LGBMRegressor:
    def calc_pnl(y_true: np.ndarray, pred: np.ndarray):
        y_true, pred = pd.Series(y_true, index=val_dates), pd.Series(pred, index=val_dates)
        ts = (y_true - pred).rename("ts")
        rates = get_rates()

        result = pd.merge(ts, rates, left_index=True, right_index=True, how='left')
        result['key_rate'].ffill(axis=0, inplace=True)

        result['pnl'] = 0
        result.loc[pred > 0, 'pnl'] += pred[pred > 0] * (result.loc[pred > 0, 'key_rate'] + 0.005)
        result.loc[ts > 0, 'pnl'] += ts[ts > 0] * (result.loc[ts > 0, 'key_rate'] - 0.009)
        result.loc[ts < 0, 'pnl'] += ts[ts < 0] * (result.loc[ts < 0, 'key_rate'] + 0.01)
        return "PnL", result['pnl'].sum(), True

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        shuffle=False
    )
    val_dates = X_val.index

    model = LGBMRegressor(**model_params)

    model.fit(
        X_train.values,
        y_train,
        eval_set=[(X_val.values, y_val)],
        eval_metric=calc_pnl,
        verbose=False
    )
    print("Best MAE:", mean_absolute_error(y_val, model.predict(X_val)))
    print("Best PnL:", calc_pnl(y_val, model.predict(X_val))[1])

    if visualize:
        plt.figure(figsize=(15, 8))
        plt.plot(y.values, label="Actual")
        plt.plot(model.predict(X), label="Predicted", alpha=0.9)
        plt.axvspan(int(len(X) * (1 - test_size)), len(X), alpha=0.5, color='lightgrey')
        plt.legend()
    return model

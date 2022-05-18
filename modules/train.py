import warnings

import pandas as pd
from lightgbm import LGBMRegressor
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def train_model(
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        visualize: bool = False, **model_params
) -> LGBMRegressor:
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        shuffle=False
    )

    model = LGBMRegressor(**model_params)

    model.fit(
        X_train.values,
        y_train,
        eval_set=[(X_val.values, y_val)],
        eval_metric="mae",
        verbose=False
    )

    print("Best MAE:", mean_absolute_error(y_val, model.predict(X_val)))

    if visualize:
        plt.figure(figsize=(15, 8))
        plt.plot(y.values, label="Actual")
        plt.plot(model.predict(X), label="Predicted")
        plt.axvspan(int(len(X) * (1 - test_size)), len(X), alpha=0.5, color='lightgrey')
        plt.legend()
    return model

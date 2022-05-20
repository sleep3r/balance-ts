import warnings

import pandas as pd
from lightgbm import LGBMRegressor
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

def calc_pnl( y_true : pd.Series, pred : pd.Series) -> pd.Series:
    
    ts = y_true - pred
    min_date = ts.index.min().strftime("%d.%m.%Y")
    max_date = ts.index.max().strftime("%d.%m.%Y")

    url =f'https://www.cbr.ru/hd_base/KeyRate/?UniDbQuery.Posted=True&UniDbQuery.From={min_date}&UniDbQuery.To={max_date}'
    rates = pd.read_html(io=url)[0]

    rates.columns = ['Date','key_rate']
    rates.Date = rates.Date.apply(pd.Timestamp)
    rates.set_index('Date',inplace = True)
    rates['key_rate']/=1e4
    result = pd.merge(ts, rates, left_index=True, right_index = True,how='left')
    result['key_rate'].ffill(axis = 0,inplace = True)
    result['pnl']=0
    result.loc[ts>0,'pnl'] = ts[ts>0]*(result.loc[ts>0,'key_rate']-0.009)
    result.loc[ts<0,'pnl'] = ts[ts<0]*(result.loc[ts<0,'key_rate']+0.01)
    return result['pnl'].cumsum()

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

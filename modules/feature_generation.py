import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import roll_time_series
from tsfresh.utilities.dataframe_functions import impute

warnings.filterwarnings("ignore")


def smoothing(series: pd.DataFrame, length):
    result = pd.Series(series.values).copy()
    ind = np.where(series != 0)[0]
    for i in range(1, length):
        tmp_idx_mn = ind - i
        tmp_idx_mn = tmp_idx_mn[np.isin(tmp_idx_mn, result.index.values)]
        calc1 = np.repeat(1 / np.exp(i), len(tmp_idx_mn))
        tmp_idx_pl = ind + i
        tmp_idx_pl = tmp_idx_pl[np.isin(tmp_idx_pl, result.index.values)]
        calc2 = np.repeat(1 / np.exp(2 * i), len(tmp_idx_pl))
        result[tmp_idx_mn] = calc1
        result[tmp_idx_pl] = calc2
    return result.values


def add_taxes(X: pd.DataFrame) -> pd.DataFrame:
    nalogi = pd.read_csv('./nalogs.csv', index_col=0)
    nalogi['last_date'] = pd.to_datetime(nalogi['last_date'])
    NDS = nalogi[nalogi['type'] == 'НДС'].last_date.unique()
    NDFL = nalogi[nalogi['type'] == 'НДФЛ'].last_date.unique()
    USN = nalogi[nalogi['type'] == 'УСН'].last_date.unique()
    month_profit = nalogi[nalogi['type'] == 'Прибыль_мес'].last_date.unique()
    quart_profit = nalogi[nalogi['type'] == 'Прибыль_кварт'].last_date.unique()

    cols = dict(zip(['NDS', 'NDFL', 'USN', 'month_profit', 'quart_profit'],
                    [NDS, NDFL, USN, month_profit, quart_profit]))

    for col, df in cols.items():
        X[col] = 0
        X.loc[np.isin(X.index, df), col] = 1
        X[col] = smoothing(X[col], 4)
    return X


def add_macro(X: pd.DataFrame) -> pd.DataFrame:
    brent = yf.download("BZ=F", start="2017-01-09", end="2021-03-31", progress=False)
    moex = yf.download("IMOEX.ME", start="2017-01-09", end="2021-03-31", progress=False)
    usd_rub = yf.download("RUB=X", start="2017-01-09", end="2021-03-31", progress=False)
    brent = brent[['Adj Close']].rename(columns={'Adj Close': 'BR'})
    moex = moex[['Adj Close']].rename(columns={'Adj Close': 'MOEX'})
    usd_rub = usd_rub[['Adj Close']].rename(columns={'Adj Close': 'USD'})

    X = X.merge(usd_rub, how='left', left_index=True, right_index=True) \
        .merge(brent, how='left', left_index=True, right_index=True) \
        .merge(moex, how='left', left_index=True, right_index=True)

    for col in X.columns[1:]:
        X[col] = X[col].ffill()
    return X


def add_dating(X: pd.DataFrame) -> pd.DataFrame:
    X["weekday"] = X.index.weekday
    X["month"] = X.index.month
    X["year"] = X.index.year
    X["quarter"] = X.index.quarter
    X["day"] = X.index.day
    return X


def extract(timeseries: pd.DataFrame, max_timeshift: int, min_timeshift: int) -> (pd.DataFrame, pd.Series):
    """Extracts features from timeseries using tsfresh."""
    timeseries = timeseries.reset_index()
    timeseries.columns = ["date", "value"]
    timeseries["index"] = "Balance"

    df_rolled = roll_time_series(
        timeseries,
        column_id="index", column_sort="date",
        max_timeshift=max_timeshift, min_timeshift=min_timeshift
    )

    X = extract_features(
        df_rolled.drop("index", axis=1),
        column_id="id", column_sort="date", column_value="value",
        impute_function=impute, show_warnings=False
    )

    X = X.set_index(X.index.map(lambda x: x[1]), drop=True)
    X.index.name = "last_date"
    y = timeseries.set_index("date").sort_index().value.shift(-1)
    y = y[y.index.isin(X.index)].dropna()
    X = X[X.index.isin(y.index)]
    return X, y


def generate_features(
        timeseries: pd.DataFrame,
        max_timeshift: int,
        min_timeshift: int
) -> (pd.DataFrame, pd.Series):
    """Generates features from timeseries."""
    X, y = extract(timeseries, max_timeshift, min_timeshift)
    X = select_features(X, y)
    X = add_taxes(X)
    X = add_macro(X)
    X = add_dating(X)
    return X, y

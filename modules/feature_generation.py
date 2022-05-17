import pandas as pd
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import roll_time_series
from tsfresh.utilities.dataframe_functions import impute


def extract(timeseries: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    """Extracts features from timeseries using tsfresh."""
    timeseries = timeseries.reset_index()
    timeseries.columns = ["date", "value"]
    timeseries["index"] = "Balance"

    df_rolled = roll_time_series(
        timeseries,
        column_id="index", column_sort="date",
        max_timeshift=20, min_timeshift=10
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


def generate_features(timeseries: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    """Generates features from timeseries."""
    X, y = extract(timeseries)
    X = select_features(X, y)
    return X, y

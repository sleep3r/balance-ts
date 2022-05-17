import pandas as pd
from tsfresh import extract_relevant_features


def extract_features(timeseries: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    """Extracts features from timeseries using tsfresh."""
    extracted_features = extract_relevant_features(
        timeseries, target, column_id="Date", column_sort="Date"
    )
    return extracted_features


def generate_features(timeseries: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    """Generates features from timeseries."""
    extracted_features = extract_features(timeseries, target)
    return extracted_features

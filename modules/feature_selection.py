from typing import Dict

import numpy as np
import pandas as pd
from mufs import MUFS
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel, mutual_info_classif as MIC, RFECV
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC


class CorrelationSelector:
    """Correlation-based Feature Selection with a forward best first heuristic search"""

    def __init__(self):
        self.estimator = MUFS()

    def fit(self, X: np.ndarray, y: np.ndarray):
        selection = self.estimator.cfs(X, y)
        features = np.arange(X.shape[1])
        selected_features = selection.get_results()
        self.selected_features = features[selected_features]
        return self

    def transform(self, X: np.ndarray):
        return X[:, self.selected_features]

    def get_support(self):
        return self.selected_features


class MutualInformationSelector:
    """Mutual Information-based Feature Selection."""

    def __init__(self):
        self.estimator = MIC

    def fit(self, X: np.ndarray, y: np.ndarray):
        mi_scores = self.estimator(X, y > 0)
        self.selected_features = mi_scores >= np.quantile(mi_scores, 0.95)

    def transform(self, X: np.ndarray):
        return X[:, self.selected_features]

    def get_support(self):
        return self.selected_features


def select_features(extracted_features: pd.DataFrame, target: pd.Series) -> Dict[str, list]:
    """Selects the most relevant features comparing different methods."""
    wrapper_methods = {"OLS recursive": RFECV(LinearRegression(), step=1, cv=5)}
    filter_methods = {
        "correlation": CorrelationSelector(),
        "mutual_info": MutualInformationSelector()
    }
    embedded_methods = {
        "lasso l1": SelectFromModel(estimator=LinearSVC(C=0.01, penalty="l1", dual=False)),
        "ramdom_forest": SelectFromModel(estimator=RandomForestRegressor(n_estimators=100))
    }

    res = {}
    for method, estimator in {*wrapper_methods, *filter_methods, *embedded_methods}.items():  # noqa
        estimator.fit(extracted_features.values, target.values)
        selected_features = estimator.get_support()
        res[method] = selected_features
    return res

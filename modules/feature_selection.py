from typing import Dict

import numpy as np
import pandas as pd
from mufs import MUFS
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel, mutual_info_classif as MIC, RFECV
from sklearn.linear_model import LinearRegression, Lasso
from tqdm import tqdm

from utils.stabitily import nogueira


class CorrelationSelector:
    """Correlation-based Feature Selection with a forward best first heuristic search"""

    def __init__(self):
        self.estimator = MUFS()

    def fit(self, X: np.ndarray, y: np.ndarray):
        selection = self.estimator.cfs(X, y)
        features = np.zeros(X.shape[1])
        features[selection.get_results()] = 1
        self.selected_features = features.astype(bool)
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
        return self

    def transform(self, X: np.ndarray):
        return X[:, self.selected_features]

    def get_support(self):
        return self.selected_features


def select_features(extracted_features: pd.DataFrame, target: pd.Series) -> Dict[str, list]:
    """Selects the most relevant features comparing different methods."""
    wrapper_methods = {
        "OLS recursive": RFECV(LinearRegression(), step=1, cv=5)
    }
    filter_methods = {
        "correlation": CorrelationSelector(),
        "mutual_info": MutualInformationSelector()
    }
    embedded_methods = {
        "random_forest": SelectFromModel(estimator=RandomForestRegressor(n_estimators=100)),
        "lasso l1": SelectFromModel(estimator=Lasso(alpha=1.5))
    }

    res = {}
    for method, estimator in {**filter_methods, **wrapper_methods, **embedded_methods}.items():
        estimator.fit(extracted_features.values, target.values)
        selected_features = estimator.get_support()
        res[method] = selected_features

    res["ensemble"] = np.array([*res.values()]).sum(axis=0) > 1
    return res


def test_stability(extracted_features: pd.DataFrame, target: pd.Series, n_iterations: int = 20) -> Dict[str, float]:
    """Tests the stability of the selected features."""
    stability_res = {
        'OLS recursive': [],
        'correlation': [],
        'mutual_info': [],
        'random_forest': [],
        'lasso l1': [],
        'ensemble': []
    }

    for _ in tqdm(range(n_iterations)):
        res = select_features(extracted_features.sample(1000), target.sample(1000))
        for method in stability_res.keys():
            stability_res[method].append(res[method])

    for method in stability_res:
        stability_res[method] = nogueira(stability_res[method])
    return stability_res

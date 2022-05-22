import numpy as np
import pandas as pd
from pylab import *


class Stat(object):
    def __init__(self, threshold, direction="unknown", init_stat=0.0):
        self._direction = str(direction)
        self._threshold = float(threshold)
        self._stat = float(init_stat)
        self._alarm = self._stat / self._threshold

    @property
    def direction(self):
        return self._direction

    @property
    def stat(self):
        return self._stat

    @property
    def alarm(self):
        return self._alarm

    @property
    def threshold(self):
        return self._threshold

    def update(self, **kwargs):
        # Statistics may use any of the following kwargs:
        #   ts - timestamp for the value
        #   value - original value
        #   mean - current estimated mean
        #   std - current estimated std
        #   adjusted_value - usually (value - mean) / std
        # Statistics call this after updating '_stat'
        self._alarm = self._stat / self._threshold


class AdjustedShiryaevRoberts(Stat):
    def __init__(self, mean_diff, threshold, max_stat=float("+inf"), init_stat=0.0):
        super(AdjustedShiryaevRoberts, self).__init__(threshold,
                                                      direction="up",
                                                      init_stat=init_stat)
        self._mean_diff = mean_diff
        self._max_stat = max_stat

    def update(self, adjusted_value, **kwargs):
        likelihood = np.exp(self._mean_diff * (adjusted_value - self._mean_diff / 2.))
        self._stat = min(self._max_stat, (1. + self._stat) * likelihood)
        Stat.update(self)


class MeanExpNoDataException(Exception):
    pass


class MeanExp(object):
    def __init__(self, new_value_weight, load_function=median):
        self._load_function = load_function
        self._new_value_weight = new_value_weight
        self.load([])

    @property
    def value(self):
        if self._weights_sum <= 1:
            raise MeanExpNoDataException('self._weights_sum <= 1')
        return self._values_sum / self._weights_sum

    def update(self, new_value, **kwargs):
        self._values_sum = (1 - self._new_value_weight) * self._values_sum + new_value
        self._weights_sum = (1 - self._new_value_weight) * self._weights_sum + 1.0

    def load(self, old_values):
        if old_values:
            old_values = [value for ts, value in old_values]
            mean = float(self._load_function(old_values))
            self._weights_sum = min(float(len(old_values)), 1.0 / self._new_value_weight)
            self._values_sum = mean * self._weights_sum
        else:
            self._values_sum = 0.0
            self._weights_sum = 0.0


def detection_change_point(ts: pd.Series, visualize: bool = False) -> pd.Series:
    alpha = 0.01
    beta = 0.05
    sigma_diff = 2.0

    stat_trajectory, mean_values, var_values, diff_values = [], [], [], []
    timestamps, values, changepoint = [], [], []

    mean_exp = MeanExp(new_value_weight=alpha)
    var_exp = MeanExp(new_value_weight=beta)
    sr = AdjustedShiryaevRoberts(sigma_diff, 1000., max_stat=1e9)

    X = ts
    for x_k in X:
        values.append(x_k)
        try:
            mean_estimate = mean_exp.value
        except MeanExpNoDataException:
            mean_estimate = 0.

        try:
            var_estimate = var_exp.value
        except MeanExpNoDataException:
            var_estimate = 1.

        predicted_diff_value = (x_k - mean_estimate) ** 2
        predicted_diff_mean = var_estimate
        sr.update(predicted_diff_value - predicted_diff_mean)
        diff_values.append(predicted_diff_value - predicted_diff_mean)

        mean_exp.update(x_k)
        diff_value = (x_k - mean_estimate) ** 2
        var_exp.update(diff_value)

        stat_trajectory.append(sr._stat)
        mean_values.append(mean_estimate)
        var_values.append(np.sqrt(var_estimate))

    if visualize:
        figure(figsize=(12, 6))
        plot(values)
        plot(np.array(mean_values), 'k')
        plot(np.array(mean_values) + np.sqrt(var_values), 'k')
        plot(np.array(mean_values) - np.sqrt(var_values), 'k')

        figure(figsize=(12, 6))
        fig = semilogy(stat_trajectory)
        grid('on')
    return stat_trajectory


def add_training_change_point(stat_trajectory):
    count = 0
    treshhold = 5
    count_back = 3
    for i in stat_trajectory[-count_back:]:
        if i >= treshhold:
            count = count + 1
    if count >= count_back:
        return 'signal'
    if count < count_back:
        return 'anomaly'

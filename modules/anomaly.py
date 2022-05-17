import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def detect_anomalies(ts: pd.Series, visualize: bool = False) -> np.ndarray:
    """
    Detects anomalies in time series.
    """
    # Calculate the mean and standard deviation of the time series.
    mean = ts.mean()
    std = ts.std()

    # Calculate the anomaly score for each point in the time series.
    anomaly_score = (ts - mean) / std

    # Find the anomaly indices.
    anomaly_indices = anomaly_score[anomaly_score > anomaly_score.quantile(0.99)].index
    anomaly_indices = np.append(
        anomaly_indices, anomaly_score[anomaly_score < anomaly_score.quantile(0.01)].index
    )

    # Visualize the anomaly indices.
    if visualize:
        plt.figure(figsize=(7, 3))
        plt.plot(ts)
        plt.plot(anomaly_indices, ts.loc[anomaly_indices], "o", color="red")
        plt.show()
    return anomaly_indices

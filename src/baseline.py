# src/baseline.py

import numpy as np


def last_value_baseline(X):
    """
    Last-observation baseline.

    Given:

        X = [y(t-23), ..., y(t-1), y(t)]

    predict:

        y_hat(t+1) = y(t)

    Input:
        X: (N, 24)

    Output:
        predictions: (N,)
    """

    X = np.asarray(X)

    return X[:, -1]


def moving_average_baseline(X):
    """
    24-hour moving-average baseline.

        y_hat(t+1)
        =
        (1/24) * sum_{i=0}^{23} y(t-i)

    Input:
        X: (N, 24)

    Output:
        predictions: (N,)
    """

    X = np.asarray(X)

    return np.mean(
        X,
        axis=1
    )


def evaluate_baseline(
    X,
    y,
    baseline_function,
    metrics_function
):
    """
    Evaluate a baseline using the same test set
    as the attention model.
    """

    predictions = baseline_function(X)

    return metrics_function(
        y,
        predictions
    )
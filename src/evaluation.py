# src/evaluation.py

import numpy as np
import pandas as pd


# ============================================================
# Time-Aware Dataset Split
# ============================================================

def time_based_split(
    values,
    train_ratio=0.70,
    validation_ratio=0.15,
    test_ratio=0.15
):
    """
    Chronological split for time-dependent data.

    No shuffling is performed.

    Example:

        Earlier data
        |----------------------|---------------|---------------|
               Train              Validation        Test

    This prevents future observations from entering the
    training set.
    """

    if not np.isclose(
        train_ratio
        + validation_ratio
        + test_ratio,
        1.0
    ):
        raise ValueError(
            "train_ratio + validation_ratio + "
            "test_ratio must equal 1."
        )

    values = np.asarray(
        values
    )

    n = len(values)

    train_end = int(
        n * train_ratio
    )

    validation_end = (
        train_end
        +
        int(n * validation_ratio)
    )

    train = values[
        :train_end
    ]

    validation = values[
        train_end:validation_end
    ]

    test = values[
        validation_end:
    ]

    return {
        "train": train,
        "validation": validation,
        "test": test
    }


# ============================================================
# Sequence Construction
# ============================================================

def create_forecasting_sequences(
    values,
    sequence_length=24
):
    """
    Converts a chronological demand series into:

        X[t] = previous 24 observations
        y[t] = next-hour demand
    """

    values = np.asarray(
        values,
        dtype=np.float32
    )

    X = []
    y = []

    for i in range(
        sequence_length,
        len(values)
    ):

        X.append(
            values[
                i - sequence_length:i
            ]
        )

        y.append(
            values[i]
        )

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.float32
    )

    return X, y


# ============================================================
# MAE
# ============================================================

def mae(
    y_true,
    y_pred
):
    """
    Mean Absolute Error.

        MAE =
        (1/N) * sum |y - y_hat|
    """

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    return float(
        np.mean(
            np.abs(
                y_true - y_pred
            )
        )
    )


# ============================================================
# MSE
# ============================================================

def mse(
    y_true,
    y_pred
):
    """
    Mean Squared Error.

        MSE =
        (1/N) * sum (y - y_hat)^2
    """

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    return float(
        np.mean(
            (
                y_true - y_pred
            ) ** 2
        )
    )


# ============================================================
# RMSE
# ============================================================

def rmse(
    y_true,
    y_pred
):
    """
    Root Mean Squared Error.

        RMSE = sqrt(MSE)
    """

    return float(
        np.sqrt(
            mse(
                y_true,
                y_pred
            )
        )
    )


# ============================================================
# MAPE
# ============================================================

def mape(
    y_true,
    y_pred,
    epsilon=1e-8
):
    """
    Mean Absolute Percentage Error.

    MAPE is included as an optional additional forecasting
    metric.

    Small or zero demand values can make percentage error
    unstable, so epsilon is used.
    """

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    denominator = np.maximum(
        np.abs(y_true),
        epsilon
    )

    return float(
        np.mean(
            np.abs(
                (y_true - y_pred)
                /
                denominator
            )
        )
        * 100.0
    )


# ============================================================
# All Metrics
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):
    """
    Compute all predefined forecasting metrics.
    """

    return {
        "MAE": mae(
            y_true,
            y_pred
        ),

        "MSE": mse(
            y_true,
            y_pred
        ),

        "RMSE": rmse(
            y_true,
            y_pred
        ),

        "MAPE_percent": mape(
            y_true,
            y_pred
        )
    }


# ============================================================
# Split Summary
# ============================================================

def split_summary(
    train,
    validation,
    test
):
    """
    Create a summary of the chronological split.
    """

    return pd.DataFrame([
        {
            "Split": "Training",
            "Samples": len(train),
            "First_Index": 0,
            "Last_Index": len(train) - 1
        },

        {
            "Split": "Validation",
            "Samples": len(validation),
            "First_Index": len(train),
            "Last_Index":
                len(train)
                + len(validation)
                - 1
        },

        {
            "Split": "Test",
            "Samples": len(test),
            "First_Index":
                len(train)
                + len(validation),
            "Last_Index":
                len(train)
                + len(validation)
                + len(test)
                - 1
        }
    ])

def inverse_transform(values, mean, std):
    """Undo the train-set normalization used by ``prepare_data``."""
    return np.asarray(values) * float(std) + float(mean)


# Backwards-compatible short name used by experiment scripts.
metrics = calculate_metrics

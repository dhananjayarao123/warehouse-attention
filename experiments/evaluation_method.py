# experiments/evaluation_methodology.py

import sys
from pathlib import Path

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


import json

import numpy as np
import pandas as pd

from src.data_generator import (
    generate_warehouse_data
)

from src.evaluation import (
    time_based_split,
    calculate_metrics,
    split_summary
)


# ============================================================
# Configuration
# ============================================================

SEED = 42

TRAIN_RATIO = 0.70

VALIDATION_RATIO = 0.15

TEST_RATIO = 0.15

SEQUENCE_LENGTH = 24


# ============================================================
# Generate Complete Time Series
# ============================================================

df = generate_warehouse_data(
    n_hours=24 * 365,
    seed=SEED
)

demand = (
    df["demand"]
    .to_numpy(
        dtype=np.float32
    )
)


# ============================================================
# Chronological Split
# ============================================================

splits = time_based_split(
    demand,

    train_ratio=TRAIN_RATIO,

    validation_ratio=
        VALIDATION_RATIO,

    test_ratio=
        TEST_RATIO
)


train = splits["train"]

validation = splits["validation"]

test = splits["test"]


# ============================================================
# Split Summary
# ============================================================

summary = split_summary(
    train,
    validation,
    test
)


# ============================================================
# Print Split Information
# ============================================================

print("=" * 70)
print("SECTION 18 - EVALUATION METHODOLOGY")
print("=" * 70)

print("\nChronological split:")
print(
    summary.to_string(
        index=False
    )
)

print("\nSplit percentages:")

print(
    f"Training   : "
    f"{TRAIN_RATIO * 100:.1f}%"
)

print(
    f"Validation : "
    f"{VALIDATION_RATIO * 100:.1f}%"
)

print(
    f"Test       : "
    f"{TEST_RATIO * 100:.1f}%"
)


# ============================================================
# Create Forecasting Sequences Separately
# ============================================================
#
# Important:
#
# Each split is converted into sequences independently.
# Therefore a sequence from validation/test does not use
# demand values from a later split.
#
# ============================================================

def create_sequences(
    values,
    sequence_length
):

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

    return (
        np.asarray(
            X,
            dtype=np.float32
        ),
        np.asarray(
            y,
            dtype=np.float32
        )
    )


X_train, y_train = create_sequences(
    train,
    SEQUENCE_LENGTH
)

X_validation, y_validation = create_sequences(
    validation,
    SEQUENCE_LENGTH
)

X_test, y_test = create_sequences(
    test,
    SEQUENCE_LENGTH
)


# ============================================================
# Print Sequence Shapes
# ============================================================

print("\nForecasting sequence shapes:")

print(
    "Training X:",
    X_train.shape
)

print(
    "Training y:",
    y_train.shape
)

print(
    "Validation X:",
    X_validation.shape
)

print(
    "Validation y:",
    y_validation.shape
)

print(
    "Test X:",
    X_test.shape
)

print(
    "Test y:",
    y_test.shape
)


# ============================================================
# Metric Definitions
# ============================================================

metric_definitions = {

    "MAE": {
        "formula":
            "MAE = (1/N) * sum(|y - y_hat|)",

        "reason":
            "MAE is easy to interpret because it "
            "expresses the average absolute forecasting "
            "error in the same units as warehouse demand.",

        "lower_is_better":
            True
    },

    "MSE": {
        "formula":
            "MSE = (1/N) * sum((y - y_hat)^2)",

        "reason":
            "MSE penalizes larger forecasting errors more "
            "strongly, making it useful when large demand "
            "prediction errors are particularly important.",

        "lower_is_better":
            True
    },

    "RMSE": {
        "formula":
            "RMSE = sqrt(MSE)",

        "reason":
            "RMSE retains the original demand units while "
            "still giving greater influence to large errors.",

        "lower_is_better":
            True
    },

    "MAPE": {
        "formula":
            "MAPE = mean(|(y-y_hat)/y|) * 100",

        "reason":
            "MAPE provides a percentage-based view of "
            "forecasting error. It is treated as an "
            "additional metric rather than the primary "
            "metric because percentage error can become "
            "unstable when actual demand is close to zero.",

        "lower_is_better":
            True
    }
}


# ============================================================
# Metric Demonstration
# ============================================================
#
# The actual model evaluation will call calculate_metrics()
# using predictions generated by the trained attention model.
#
# Here we verify the metric functions using the test target
# and a simple last-observation prediction.
# ============================================================

baseline_predictions = X_test[:, -1]

baseline_metrics = calculate_metrics(
    y_test,
    baseline_predictions
)


print("\nPredefined metrics:")

for name, definition in metric_definitions.items():

    print(
        f"\n{name}"
    )

    print(
        f"Formula: "
        f"{definition['formula']}"
    )

    print(
        f"Reason: "
        f"{definition['reason']}"
    )

    print(
        f"Lower is better: "
        f"{definition['lower_is_better']}"
    )


print("\nExample last-observation baseline metrics:")

for metric, value in baseline_metrics.items():

    print(
        f"{metric}: "
        f"{value:.6f}"
    )


# ============================================================
# Leakage Checks
# ============================================================

leakage_checks = {

    "random_shuffle":
        False,

    "chronological_order_preserved":
        True,

    "future_data_used_for_training":
        False,

    "validation_after_training_period":
        True,

    "test_after_validation_period":
        True,

    "same_test_set_for_final_model_comparison":
        True
}


print("\nLeakage checks:")

for name, value in leakage_checks.items():

    print(
        f"{name}: {value}"
    )


# ============================================================
# Save Results
# ============================================================

Path(
    "results/metrics"
).mkdir(
    parents=True,
    exist_ok=True
)


summary.to_csv(
    "results/metrics/"
    "section18_split_summary.csv",
    index=False
)


pd.DataFrame({
    "metric": list(
        baseline_metrics.keys()
    ),

    "value": list(
        baseline_metrics.values()
    )
}).to_csv(
    "results/metrics/"
    "section18_metric_example.csv",
    index=False
)


# ============================================================
# Complete Methodology
# ============================================================

methodology = {

    "section": 18,

    "objective":
        "Evaluate the warehouse next-hour demand "
        "forecast using a leakage-safe time-dependent "
        "evaluation methodology.",

    "split_strategy": {

        "type":
            "chronological",

        "train_ratio":
            TRAIN_RATIO,

        "validation_ratio":
            VALIDATION_RATIO,

        "test_ratio":
            TEST_RATIO,

        "random_split":
            False,

        "explanation":
            "The earliest observations are used for training, "
            "the following observations are used for validation, "
            "and the latest observations are reserved for the "
            "final test. This prevents future demand observations "
            "from leaking into model training."
    },

    "forecasting_input":
        "Previous 24 hourly observations",

    "forecasting_target":
        "Next-hour demand",

    "metrics_defined_before_experiment": [
        "MAE",
        "MSE",
        "RMSE",
        "MAPE"
    ],

    "primary_metrics": [
        "MAE",
        "MSE",
        "RMSE"
    ],

    "additional_metric":
        "MAPE",

    "metric_definitions":
        metric_definitions,

    "leakage_checks":
        leakage_checks,

    "split_sizes": {

        "training":
            len(train),

        "validation":
            len(validation),

        "test":
            len(test)
    },

    "sequence_sizes": {

        "training_sequences":
            len(X_train),

        "validation_sequences":
            len(X_validation),

        "test_sequences":
            len(X_test)
    }
}


with open(
    "results/metrics/"
    "section18_evaluation_methodology.json",
    "w"
) as f:

    json.dump(
        methodology,
        f,
        indent=4
    )


print("\n")
print("=" * 70)
print("SECTION 18 COMPLETE")
print("=" * 70)

print(
    "\nTraining set:"
)

print(
    f"  {len(train)} hourly observations"
)

print(
    "\nValidation set:"
)

print(
    f"  {len(validation)} hourly observations"
)

print(
    "\nTest set:"
)

print(
    f"  {len(test)} hourly observations"
)

print(
    "\nPrimary metrics:"
)

print(
    "  MAE"
)

print(
    "  MSE"
)

print(
    "  RMSE"
)

print(
    "\nAdditional metric:"
)

print(
    "  MAPE"
)

print(
    "\nNo random splitting was used."
)

print(
    "\nResults saved to:"
)

print(
    "results/metrics/section18_split_summary.csv"
)

print(
    "results/metrics/section18_evaluation_methodology.json"
)
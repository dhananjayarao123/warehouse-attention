import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import pandas as pd

from src.data_generator import generate_warehouse_data
from src.dataset import create_sequences, chronological_split
from src.warehouse_model import WarehouseAttentionModel
from src.train import train_model, predict_model
from src.evaluation import inverse_transform, metrics

SEED = 42
LOOKBACK = 24


def prepare_original_data(demand):
    """Chronologically split original data and normalize using train stats."""
    X, y = create_sequences(demand, sequence_length=LOOKBACK)
    X_train, y_train, X_val, y_val, X_test, y_test = chronological_split(X, y)

    # Fit preprocessing ONLY on the original training data.
    mean = float(X_train.mean())
    std = max(float(X_train.std()), 1e-8)

    return {
        "X_train": ((X_train - mean) / std).astype(np.float32),
        "y_train": ((y_train - mean) / std).astype(np.float32),
        "X_val": ((X_val - mean) / std).astype(np.float32),
        "y_val": ((y_val - mean) / std).astype(np.float32),
        "X_test": ((X_test - mean) / std).astype(np.float32),
        "y_test": ((y_test - mean) / std).astype(np.float32),
        "mean": mean,
        "std": std,
        "raw_X_test": X_test,
        "raw_y_test": y_test,
    }


def prepare_shifted_data(demand, mean, std):
    """Create shifted sequences using ORIGINAL training normalization."""
    X, y = create_sequences(demand, sequence_length=LOOKBACK)

    # The shifted distribution is evaluation-only. It is not used to fit
    # normalization statistics.
    X_n = ((X - mean) / std).astype(np.float32)
    y_n = ((y - mean) / std).astype(np.float32)

    return X_n, y_n, X, y


def entropy(row):
    row = np.asarray(row, dtype=np.float64)
    return float(-np.sum(row * np.log(row + 1e-12)))


def describe_attention(weights):
    """Summarize the attention matrix for one sample."""
    A = np.asarray(weights)

    # Expected [batch, query_position, key_position].
    if A.ndim != 3:
        return {
            "available": False,
            "reason": f"Unexpected attention shape {A.shape}",
        }

    matrix = A[0]

    # The prediction uses mean aggregation over attended positions. We report
    # both the full matrix and its average key-position importance.
    key_importance = matrix.mean(axis=0)
    strongest_key = int(np.argmax(key_importance))

    return {
        "available": True,
        "shape": list(matrix.shape),
        "mean_entropy": float(np.mean([entropy(r) for r in matrix])),
        "mean_max_weight": float(np.mean(np.max(matrix, axis=1))),
        "strongest_key_position_zero_based": strongest_key,
        "strongest_key_position_hours_before_prediction": LOOKBACK - strongest_key,
        "key_position_mean_weights": key_importance.tolist(),
        "matrix": matrix.tolist(),
    }


def main():
    result_dir = PROJECT_ROOT / "results" / "failure_analysis"
    result_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # 1. Original training distribution
    # ------------------------------------------------------------
    original_df = generate_warehouse_data(
        seed=SEED,
        noise_std=10.0,
        spike_probability=0.01,
        spike_low=50.0,
        spike_high=120.0,
    )

    original = prepare_original_data(
        original_df["demand"].to_numpy(dtype=np.float32)
    )

    # ------------------------------------------------------------
    # 2. Train the same model used by warehouse experiments
    # ------------------------------------------------------------
    model = WarehouseAttentionModel(
        d_model=32,
        hidden_dim=16,
        scale=True,
    )

    train_model(
        model,
        original["X_train"],
        original["y_train"],
        original["X_val"],
        original["y_val"],
        epochs=30,
        batch_size=128,
        learning_rate=1e-3,
        seed=SEED,
    )

    # ------------------------------------------------------------
    # 3. Build the shifted distribution
    # ------------------------------------------------------------
    shifted_df = generate_warehouse_data(
        seed=SEED + 1000,
        noise_std=20.0,
        spike_probability=0.03,
        spike_low=100.0,
        spike_high=200.0,
    )

    X_shift_n, y_shift_n, X_shift_raw, y_shift_raw = prepare_shifted_data(
        shifted_df["demand"].to_numpy(dtype=np.float32),
        original["mean"],
        original["std"],
    )

    # ------------------------------------------------------------
    # 4. Evaluate without retraining
    # ------------------------------------------------------------
    pred_n, weights = predict_model(model, X_shift_n)

    y_true = inverse_transform(
        y_shift_n,
        original["mean"],
        original["std"],
    )
    y_pred = inverse_transform(
        pred_n,
        original["mean"],
        original["std"],
    )

    abs_error = np.abs(y_true - y_pred)
    signed_error = y_pred - y_true
    squared_error = signed_error ** 2

    # ------------------------------------------------------------
    # 5. Select genuine failure: largest shifted absolute error
    # ------------------------------------------------------------
    failure_index = int(np.argmax(abs_error))
    failure_window = X_shift_raw[failure_index]
    actual = float(y_true[failure_index])
    predicted = float(y_pred[failure_index])
    error = float(abs_error[failure_index])

    # Last-observation baseline on exactly the same failure window.
    baseline_prediction = float(failure_window[-1])
    baseline_error = float(abs(actual - baseline_prediction))

    # ------------------------------------------------------------
    # 6. Determine whether the failure is associated with a spike
    # ------------------------------------------------------------
    window_mean = float(np.mean(failure_window))
    window_std = float(np.std(failure_window))
    window_max = float(np.max(failure_window))
    window_min = float(np.min(failure_window))

    # A simple, transparent event diagnostic. It is not claimed as a causal
    # explanation; it only identifies unusual magnitude in the selected input.
    spike_like = bool(
        actual > window_mean + 2.0 * max(window_std, 1e-8)
        or actual > np.quantile(shifted_df["demand"], 0.95)
    )

    attention_info = describe_attention(weights[failure_index:failure_index + 1])

    # ------------------------------------------------------------
    # 7. Overall shifted metrics provide evidence that this is not an
    #    isolated invented example.
    # ------------------------------------------------------------
    shifted_metrics = metrics(y_true, y_pred)

    # ------------------------------------------------------------
    # 8. Save the 24-hour failure input
    # ------------------------------------------------------------
    history = pd.DataFrame({
        "hour_before_prediction": np.arange(-LOOKBACK, 0),
        "demand": failure_window,
    })
    history.to_csv(
        result_dir / "failure_input_window.csv",
        index=False,
    )

    # Save shifted predictions so the selected index is auditable.
    all_predictions = pd.DataFrame({
        "test_index": np.arange(len(y_true)),
        "actual": y_true,
        "predicted": y_pred,
        "signed_error": signed_error,
        "absolute_error": abs_error,
        "squared_error": squared_error,
    })
    all_predictions.to_csv(
        result_dir / "shifted_failure_search.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # 9. Human-readable summary
    # ------------------------------------------------------------
    summary = {
        "section": "20_failure_investigation",
        "selection_rule": "largest absolute prediction error on the shifted evaluation distribution",
        "scenario": (
            "A model trained on the original warehouse distribution is evaluated "
            "without retraining on a shifted distribution with higher noise and "
            "larger/more frequent demand spikes. The selected example is the worst "
            "absolute-error prediction in that shifted evaluation set."
        ),
        "input": {
            "lookback_hours": LOOKBACK,
            "historical_demand": failure_window.tolist(),
            "window_mean": window_mean,
            "window_std": window_std,
            "window_min": window_min,
            "window_max": window_max,
        },
        "expected_behavior": (
            "Given the previous 24 hours, the model should predict the next-hour "
            "demand close to the actual value and should retain useful daily/weekly "
            "temporal structure despite the distribution shift."
        ),
        "actual_behavior": {
            "test_index": failure_index,
            "actual_next_hour_demand": actual,
            "predicted_next_hour_demand": predicted,
            "absolute_error": error,
            "relative_error_percent": float(
                error / max(abs(actual), 1e-8) * 100.0
            ),
            "last_observation_baseline_prediction": baseline_prediction,
            "last_observation_baseline_absolute_error": baseline_error,
            "model_better_than_last_value_on_this_case": bool(error < baseline_error),
            "spike_like_failure": spike_like,
        },
        "explanation": (
            "The failure is consistent with the model being evaluated outside the "
            "stochastic regime used during training. The shifted distribution has "
            "higher noise and substantially larger/more frequent spikes. The model "
            "can still use historical temporal structure, but it has no training "
            "evidence covering every extreme configuration. A large next-hour event "
            "therefore can produce a residual that is much larger than typical test "
            "errors. This is an interpretation supported by the distribution design "
            "and error statistics, not a causal claim about individual attention weights."
        ),
        "evidence": {
            "shifted_test_metrics": shifted_metrics,
            "failure_absolute_error": error,
            "failure_squared_error": float(squared_error[failure_index]),
            "shifted_error_mean": float(np.mean(abs_error)),
            "shifted_error_median": float(np.median(abs_error)),
            "shifted_error_95th_percentile": float(np.quantile(abs_error, 0.95)),
            "error_vs_95th_percentile_ratio": float(
                error / max(float(np.quantile(abs_error, 0.95)), 1e-8)
            ),
            "attention": attention_info,
            "distribution_shift": {
                "training_noise_std": 10.0,
                "shifted_noise_std": 20.0,
                "training_spike_probability": 0.01,
                "shifted_spike_probability": 0.03,
                "training_spike_range": [50.0, 120.0],
                "shifted_spike_range": [100.0, 200.0],
            },
        },
        "mathematical_reasoning": {
            "predictor": "y_hat_t = f_theta(X_(t-24):t)",
            "training_risk": "E_{P_train}[(f_theta(X)-Y)^2]",
            "shifted_risk": "E_{P_shift}[(f_theta(X)-Y)^2]",
            "mse_tail_sensitivity": "MSE squares the residual, so extreme shifted errors contribute disproportionately.",
            "interpretation": "The selected failure is evidence of increased prediction risk under P_shift, not evidence that the attention mechanism is mathematically incorrect.",
        },
        "potential_improvement": [
            "Train with a mixture containing representative high-noise/high-spike regimes.",
            "Add event/spike indicators or other exogenous operational features if they are available at prediction time.",
            "Use a robust loss or explicitly model the conditional variance if extreme-demand errors are especially costly.",
            "Perform a dedicated stress-test set during model selection instead of selecting hyperparameters only on ordinary-distribution validation data.",
        ],
        "limitation": (
            "The potential improvements are proposals, not validated fixes. A follow-up "
            "experiment would be required to establish whether any of them actually reduces "
            "the observed failure."
        ),
    }

    with open(
        result_dir / "section20_failure_report.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame([{
        "test_index": failure_index,
        "actual": actual,
        "predicted": predicted,
        "absolute_error": error,
        "relative_error_percent": error / max(abs(actual), 1e-8) * 100.0,
        "last_value_baseline": baseline_prediction,
        "last_value_error": baseline_error,
        "window_mean": window_mean,
        "window_std": window_std,
        "window_min": window_min,
        "window_max": window_max,
        "spike_like_failure": spike_like,
    }]).to_csv(
        result_dir / "section20_failure_summary.csv",
        index=False,
    )

    print("\n" + "=" * 72)
    print("SECTION 20 — FAILURE INVESTIGATION")
    print("=" * 72)
    print(f"Selected shifted-test index : {failure_index}")
    print(f"Actual next-hour demand     : {actual:.4f}")
    print(f"Predicted next-hour demand  : {predicted:.4f}")
    print(f"Absolute error              : {error:.4f}")
    print(f"Last-value baseline error   : {baseline_error:.4f}")
    print(f"Spike-like failure          : {spike_like}")
    print(f"Shifted MAE                 : {shifted_metrics['MAE']:.4f}")
    print(f"Shifted RMSE                : {shifted_metrics['RMSE']:.4f}")
    print(f"\nArtifacts: {result_dir}")


if __name__ == "__main__":
    main()
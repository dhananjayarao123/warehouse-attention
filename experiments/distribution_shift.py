"""Section 19: evaluate the trained model under a changed distribution."""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_generator import generate_warehouse_data
from src.dataset import prepare_data, create_sequences
from src.warehouse_model import WarehouseAttentionModel
from src.train import train_model, predict_model
from src.evaluation import inverse_transform, metrics

SEED = 42
LOOKBACK = 24


def normalize_with_training_stats(values, mean, std):
    X, y = create_sequences(values, LOOKBACK)
    return ((X - mean) / std).astype(np.float32), ((y - mean) / std).astype(np.float32), X, y


def main():
    out = ROOT / "results" / "generalization"
    out.mkdir(parents=True, exist_ok=True)

    train_df = generate_warehouse_data(seed=SEED, noise_std=10.0, spike_probability=0.01,
                                       spike_low=50.0, spike_high=120.0)
    data = prepare_data(train_df["demand"].to_numpy(), LOOKBACK)

    model = WarehouseAttentionModel(d_model=32, hidden_dim=16, scale=True)
    history = train_model(model, data.X_train, data.y_train, data.X_val, data.y_val,
                          epochs=15, batch_size=128, learning_rate=1e-3, seed=SEED)

    normal_pred, _ = predict_model(model, data.X_test)
    normal_true = inverse_transform(data.y_test, data.mean, data.std)
    normal_pred = inverse_transform(normal_pred, data.mean, data.std)

    shifted_df = generate_warehouse_data(seed=SEED + 1, noise_std=20.0, spike_probability=0.03,
                                         spike_low=100.0, spike_high=200.0)
    Xs, ys, Xs_raw, ys_raw = normalize_with_training_stats(
        shifted_df["demand"].to_numpy(dtype=np.float32), data.mean, data.std
    )
    shifted_pred, _ = predict_model(model, Xs)
    shifted_pred = inverse_transform(shifted_pred, data.mean, data.std)

    normal_metrics = metrics(normal_true, normal_pred)
    shifted_metrics = metrics(ys_raw, shifted_pred)
    table = pd.DataFrame([
        {"distribution": "Original", **normal_metrics},
        {"distribution": "Shifted", **shifted_metrics},
    ])
    table["MAE_degradation_vs_original"] = table["MAE"] - normal_metrics["MAE"]
    table["RMSE_degradation_vs_original"] = table["RMSE"] - normal_metrics["RMSE"]
    table.to_csv(out / "distribution_shift_metrics.csv", index=False)
    pd.DataFrame({"epoch": range(1, len(history.train_loss)+1), "train_loss": history.train_loss,
                  "validation_loss": history.val_loss}).to_csv(out / "training_history.csv", index=False)

    (out / "configuration.json").write_text(json.dumps({
        "original": {"noise_std": 10.0, "spike_probability": 0.01, "spike_low": 50.0, "spike_high": 120.0},
        "shifted": {"noise_std": 20.0, "spike_probability": 0.03, "spike_low": 100.0, "spike_high": 200.0},
        "critical_protocol": "Shifted observations are normalized using the original training mean/std; shifted data is never used to fit preprocessing or model parameters."
    }, indent=2))
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

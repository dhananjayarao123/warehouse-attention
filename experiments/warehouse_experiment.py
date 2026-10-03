"""Section 15/16: train and evaluate the canonical warehouse attention model."""

from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_generator import generate_warehouse_data
from src.dataset import prepare_data
from src.warehouse_model import WarehouseAttentionModel
from src.train import train_model, predict_model
from src.evaluation import inverse_transform, metrics

SEED = 42
LOOKBACK = 24


def main():
    out = ROOT / "results" / "warehouse"
    fig = ROOT / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)

    df = generate_warehouse_data(seed=SEED, n_hours=24 * 365)
    df.to_csv(out / "generated_warehouse.csv", index=False)
    data = prepare_data(df["demand"].to_numpy(), LOOKBACK)

    model = WarehouseAttentionModel(d_model=32, hidden_dim=16, scale=True)
    history = train_model(model, data.X_train, data.y_train, data.X_val, data.y_val,
                          epochs=15, batch_size=128, learning_rate=1e-3, seed=SEED)
    pred_n, weights = predict_model(model, data.X_test)

    y_true = inverse_transform(data.y_test, data.mean, data.std)
    y_pred = inverse_transform(pred_n, data.mean, data.std)
    X_raw = inverse_transform(data.X_test, data.mean, data.std)

    result = metrics(y_true, y_pred)
    pd.DataFrame([result]).to_csv(out / "attention_metrics.csv", index=False)
    pd.DataFrame({"actual_demand": y_true, "attention_prediction": y_pred,
                  "last_observation_prediction": X_raw[:, -1],
                  "moving_average_prediction": X_raw.mean(axis=1)}).to_csv(out / "predictions.csv", index=False)
    history_df = pd.DataFrame({"epoch": range(1, len(history.train_loss)+1),
                               "train_loss": history.train_loss, "validation_loss": history.val_loss,
                               "gradient_norm": history.gradient_norm,
                               "attention_entropy": history.attention_entropy,
                               "attention_max": history.attention_max})
    history_df.to_csv(out / "training_history.csv", index=False)

    # Attention diagnostics for a representative test sample.
    mean_weights = weights.mean(axis=0)
    pd.DataFrame({"historical_position": [f"t-{LOOKBACK-i}" for i in range(LOOKBACK)],
                  "mean_attention_weight": mean_weights.mean(axis=0)}).to_csv(out / "attention_positions.csv", index=False)

    plt.figure(figsize=(9, 4))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="train")
    plt.plot(history_df["epoch"], history_df["validation_loss"], label="validation")
    plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.title("Warehouse training dynamics"); plt.legend(); plt.tight_layout()
    plt.savefig(fig / "warehouse_training.png", dpi=150); plt.close()

    (out / "configuration.json").write_text(json.dumps({
        "seed": SEED, "lookback_hours": LOOKBACK, "d_model": 32, "hidden_dim": 16,
        "epochs": 15, "batch_size": 128, "learning_rate": 1e-3,
        "split": "70/15/15 chronological", "normalization": "training mean/std only",
        "attention": "Q=XW_Q, K=XW_K, V=XW_V, QK^T/sqrt(d_k), stable softmax, AV"
    }, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

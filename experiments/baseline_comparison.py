"""Section 17: compare attention forecasting with simple baselines."""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import tensorflow as tf

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
    out.mkdir(parents=True, exist_ok=True)

    df = generate_warehouse_data(seed=SEED)
    data = prepare_data(df["demand"].to_numpy(), LOOKBACK)

    model = WarehouseAttentionModel(d_model=32, hidden_dim=16, scale=True)
    history = train_model(model, data.X_train, data.y_train, data.X_val, data.y_val,
                          epochs=15, batch_size=128, learning_rate=1e-3, seed=SEED)
    pred_n, weights = predict_model(model, data.X_test)

    y_true = inverse_transform(data.y_test, data.mean, data.std)
    attention_pred = inverse_transform(pred_n, data.mean, data.std)
    X_raw = inverse_transform(data.X_test, data.mean, data.std)
    last_pred = X_raw[:, -1]
    moving_pred = X_raw.mean(axis=1)

    table = pd.DataFrame([
        {"model": "Last Observation", **metrics(y_true, last_pred)},
        {"model": "Moving Average", **metrics(y_true, moving_pred)},
        {"model": "Attention", **metrics(y_true, attention_pred)},
    ])
    table.to_csv(out / "baseline_comparison.csv", index=False)

    pd.DataFrame({
        "actual_demand": y_true,
        "last_observation_prediction": last_pred,
        "moving_average_prediction": moving_pred,
        "attention_prediction": attention_pred,
    }).to_csv(out / "baseline_predictions.csv", index=False)

    pd.DataFrame({"epoch": range(1, len(history.train_loss)+1),
                  "train_loss": history.train_loss,
                  "validation_loss": history.val_loss}).to_csv(out / "training_history.csv", index=False)

    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

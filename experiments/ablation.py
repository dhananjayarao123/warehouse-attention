"""Section 14: controlled scaled-vs-unscaled attention ablation."""

from pathlib import Path
import json
import random
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_generator import generate_warehouse_data
from src.dataset import prepare_data, make_dataset
from src.warehouse_model import WarehouseAttentionModel
from src.evaluation import inverse_transform, metrics

SEED = 42
EPOCHS = 10
BATCH_SIZE = 128
LR = 1e-3
D_MODEL = 32
HIDDEN = 16


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train_variant(data, scale):
    set_seed(SEED)
    model = WarehouseAttentionModel(d_model=D_MODEL, hidden_dim=HIDDEN, scale=scale)
    _ = model(tf.convert_to_tensor(data.X_train[:1]), training=False)
    opt = tf.keras.optimizers.Adam(LR)
    loss_fn = tf.keras.losses.MeanSquaredError()
    train_ds = make_dataset(data.X_train, data.y_train, BATCH_SIZE, shuffle=True, seed=SEED)
    history = []
    diagnostics = []

    for epoch in range(EPOCHS):
        losses, grads = [], []
        for xb, yb in train_ds:
            with tf.GradientTape() as tape:
                pred, weights = model(xb, training=True)
                loss = loss_fn(yb, pred)
            g = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(g, model.trainable_variables))
            losses.append(float(loss))
            valid = [x for x in g if x is not None]
            grads.append(float(tf.linalg.global_norm(valid)) if valid else 0.0)

        pred, weights = model(tf.convert_to_tensor(data.X_val), training=False)
        val_loss = float(loss_fn(data.y_val, pred))
        w = weights.numpy()
        entropy = float(np.mean(-np.sum(w * np.log(w + 1e-12), axis=-1)))
        max_weight = float(np.mean(np.max(w, axis=-1)))
        saturated = float(np.mean(w > 0.95))
        scores = model.attention.last_scores if hasattr(model.attention, "last_scores") else None
        history.append({"epoch": epoch + 1, "train_loss": np.mean(losses), "validation_loss": val_loss,
                        "gradient_norm": np.mean(grads), "attention_entropy": entropy,
                        "mean_max_attention": max_weight, "saturated_fraction": saturated})

    pred, weights = model(tf.convert_to_tensor(data.X_test), training=False)
    pred_raw = inverse_transform(pred.numpy(), data.mean, data.std)
    y_raw = inverse_transform(data.y_test, data.mean, data.std)
    result = metrics(y_raw, pred_raw)
    return model, pd.DataFrame(history), result


def main():
    out = ROOT / "results" / "ablation"
    out.mkdir(parents=True, exist_ok=True)

    df = generate_warehouse_data(seed=SEED)
    data = prepare_data(df["demand"].to_numpy(), sequence_length=24)

    rows = []
    for name, scale in [("Scaled Attention", True), ("Unscaled Attention", False)]:
        model, history, result = train_variant(data, scale)
        history.to_csv(out / ("scaled_training_dynamics.csv" if scale else "unscaled_training_dynamics.csv"), index=False)
        rows.append({"experiment": name, "scale": scale, **result,
                     "final_train_loss": float(history.iloc[-1].train_loss),
                     "final_validation_loss": float(history.iloc[-1].validation_loss),
                     "final_gradient_norm": float(history.iloc[-1].gradient_norm),
                     "final_attention_entropy": float(history.iloc[-1].attention_entropy),
                     "final_saturated_fraction": float(history.iloc[-1].saturated_fraction)})

    table = pd.DataFrame(rows)
    table.to_csv(out / "scaled_vs_unscaled.csv", index=False)
    (out / "hypothesis.json").write_text(json.dumps({
        "hypothesis": "Scaling should reduce logit magnitude and generally improve numerical/training stability.",
        "controlled_variables": ["dataset", "seed", "architecture", "optimizer", "learning_rate", "epochs", "batch_size"],
        "only_changed": "division of QK^T by sqrt(d_k)"
    }, indent=2))
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

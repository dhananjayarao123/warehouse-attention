import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.toy_model import (
    ToySequenceAttentionModel
)


# ============================================================
# Reproducibility
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# Configuration
# ============================================================

CONFIG = {
    "seed": SEED,

    "sequence_length": 12,

    "d_model": 16,

    "hidden_units": 16,

    "train_samples": 4000,

    "validation_samples": 1000,

    "test_samples": 1000,

    "batch_size": 64,

    "epochs": 10,

    "learning_rate": 0.001,

    "noise_std": 0.05,

    "important_positions": [
        11,
        9,
        6
    ]
}


# ============================================================
# Output directory
# ============================================================

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# Synthetic Dataset
# ============================================================

def generate_dataset(
    n_samples,
    sequence_length,
    noise_std,
    seed
):

    rng = np.random.default_rng(seed)

    X = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(
            n_samples,
            sequence_length
        )
    ).astype(np.float32)

    # --------------------------------------------------------
    # Target depends on selected historical positions
    #
    # y =
    # 0.5*x[-1]
    # + 0.3*x[-3]
    # + 0.2*x[-6]
    # + noise
    # --------------------------------------------------------

    y = (
        0.5 * X[:, -1]
        + 0.3 * X[:, -3]
        + 0.2 * X[:, -6]
    )

    noise = rng.normal(
        loc=0.0,
        scale=noise_std,
        size=n_samples
    )

    y = y + noise

    y = y.astype(np.float32)

    return (
        X[..., np.newaxis],
        y[..., np.newaxis]
    )


# ============================================================
# Dataset Preparation
# ============================================================

X_train, y_train = generate_dataset(
    CONFIG["train_samples"],
    CONFIG["sequence_length"],
    CONFIG["noise_std"],
    SEED
)

X_val, y_val = generate_dataset(
    CONFIG["validation_samples"],
    CONFIG["sequence_length"],
    CONFIG["noise_std"],
    SEED + 1
)

X_test, y_test = generate_dataset(
    CONFIG["test_samples"],
    CONFIG["sequence_length"],
    CONFIG["noise_std"],
    SEED + 2
)


train_dataset = tf.data.Dataset.from_tensor_slices(
    (X_train, y_train)
)

train_dataset = train_dataset.shuffle(
    buffer_size=len(X_train),
    seed=SEED
)

train_dataset = train_dataset.batch(
    CONFIG["batch_size"]
)

train_dataset = train_dataset.prefetch(
    tf.data.AUTOTUNE
)


val_dataset = tf.data.Dataset.from_tensor_slices(
    (X_val, y_val)
)

val_dataset = val_dataset.batch(
    CONFIG["batch_size"]
)

val_dataset = val_dataset.prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# Model
# ============================================================

model = ToySequenceAttentionModel(
    d_model=CONFIG["d_model"],
    hidden_units=CONFIG["hidden_units"],
    sequence_length=CONFIG["sequence_length"]
)

model(
    tf.zeros((1, CONFIG["sequence_length"], 1)),
    training=False
)

attention_variable_ids = {
    id(variable)
    for layer in (
        model.attention.W_Q,
        model.attention.W_K,
        model.attention.W_V
    )
    for variable in layer.trainable_variables
}


optimizer = tf.keras.optimizers.Adam(
    learning_rate=CONFIG["learning_rate"]
)


loss_function = tf.keras.losses.MeanSquaredError()


# ============================================================
# Training
# ============================================================

train_losses = []
val_losses = []

train_maes = []
val_maes = []
attention_gradient_norms = []

best_val_loss = float("inf")
best_val_epoch = 0


for epoch in range(
    CONFIG["epochs"]
):

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    epoch_train_loss = []
    epoch_train_mae = []
    epoch_attention_gradient_norms = []

    for X_batch, y_batch in train_dataset:

        with tf.GradientTape() as tape:

            predictions = model(
                X_batch,
                training=True
            )

            loss = loss_function(
                y_batch,
                predictions
            )

        gradients = tape.gradient(
            loss,
            model.trainable_variables
        )

        attention_gradients = [
            gradient
            for gradient, variable in zip(
                gradients,
                model.trainable_variables
            )
            if id(variable) in attention_variable_ids
            and gradient is not None
        ]

        epoch_attention_gradient_norms.append(
            float(
                tf.linalg.global_norm(
                    attention_gradients
                ).numpy()
            )
        )

        optimizer.apply_gradients(
            zip(
                gradients,
                model.trainable_variables
            )
        )

        mae = tf.reduce_mean(
            tf.abs(
                y_batch - predictions
            )
        )

        epoch_train_loss.append(
            float(loss.numpy())
        )

        epoch_train_mae.append(
            float(mae.numpy())
        )

    train_loss = np.mean(
        epoch_train_loss
    )

    train_mae = np.mean(
        epoch_train_mae
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    epoch_val_loss = []
    epoch_val_mae = []

    for X_batch, y_batch in val_dataset:

        predictions = model(
            X_batch,
            training=False
        )

        loss = loss_function(
            y_batch,
            predictions
        )

        mae = tf.reduce_mean(
            tf.abs(
                y_batch - predictions
            )
        )

        epoch_val_loss.append(
            float(loss.numpy())
        )

        epoch_val_mae.append(
            float(mae.numpy())
        )

    val_loss = np.mean(
        epoch_val_loss
    )

    val_mae = np.mean(
        epoch_val_mae
    )

    train_losses.append(
        train_loss
    )

    val_losses.append(
        val_loss
    )

    train_maes.append(
        train_mae
    )

    val_maes.append(
        val_mae
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_epoch = epoch + 1

    attention_gradient_norm = np.mean(
        epoch_attention_gradient_norms
    )
    attention_gradient_norms.append(
        attention_gradient_norm
    )

    print(
        f"Epoch {epoch + 1:02d}/{CONFIG['epochs']} "
        f"| Train Loss: {train_loss:.6f} "
        f"| Val Loss: {val_loss:.6f} "
        f"| Train MAE: {train_mae:.6f} "
        f"| Val MAE: {val_mae:.6f} "
        f"| Attention Grad: {attention_gradient_norm:.6f}"
    )


# ============================================================
# Training History
# ============================================================

history = pd.DataFrame({
    "epoch": np.arange(
        1,
        CONFIG["epochs"] + 1
    ),

    "train_loss": train_losses,

    "validation_loss": val_losses,

    "train_mae": train_maes,

    "validation_mae": val_maes,

    "attention_gradient_norm": attention_gradient_norms
})


history.to_csv(
    RESULTS_DIR / "toy_training_history.csv",
    index=False
)


# ============================================================
# Test Evaluation
# ============================================================

test_predictions, test_attention = model(
    X_test,
    training=False,
    return_attention=True
)

test_predictions = (
    test_predictions
    .numpy()
    .reshape(-1)
)

y_test_flat = (
    y_test
    .reshape(-1)
)


test_mse = np.mean(
    (
        y_test_flat -
        test_predictions
    ) ** 2
)

test_mae = np.mean(
    np.abs(
        y_test_flat -
        test_predictions
    )
)

test_rmse = np.sqrt(
    test_mse
)
baseline_test_mse = np.mean(
    y_test_flat ** 2
)
test_mse_reduction = 1.0 - test_mse / baseline_test_mse


# ============================================================
# Attention Statistics
# ============================================================

attention_np = (
    test_attention
    .numpy()
)

mean_attention = np.mean(
    attention_np,
    axis=0
)
mean_attention_by_position = np.mean(
    attention_np,
    axis=(0, 1)
)

attention_entropy = -np.sum(
    attention_np *
    np.log(
        attention_np + 1e-12
    ),
    axis=-1
)

mean_attention_entropy = float(
    np.mean(
        attention_entropy
    )
)


# ============================================================
# Model Configuration
# ============================================================

model_config = {
    "task": "synthetic_sequence_prediction",

    "sequence_length":
        CONFIG["sequence_length"],

    "d_model":
        CONFIG["d_model"],

    "hidden_units":
        CONFIG["hidden_units"],

    "train_samples":
        CONFIG["train_samples"],

    "validation_samples":
        CONFIG["validation_samples"],

    "test_samples":
        CONFIG["test_samples"],

    "batch_size":
        CONFIG["batch_size"],

    "epochs":
        CONFIG["epochs"],

    "learning_rate":
        CONFIG["learning_rate"],

    "noise_std":
        CONFIG["noise_std"],

    "optimizer": "Adam",

    "loss": "MeanSquaredError",

    "positional_encoding": "trainable_absolute_embedding",

    "attention_type":
        "scaled_dot_product_self_attention"
}


# ============================================================
# Observations
# ============================================================

observations = {
    "initial_train_loss": float(train_losses[0]),
    "final_train_loss": float(train_losses[-1]),
    "initial_validation_loss": float(val_losses[0]),
    "final_validation_loss": float(val_losses[-1]),
    "best_validation_loss": float(best_val_loss),
    "best_validation_epoch": best_val_epoch,
    "validation_loss_decreased": bool(val_losses[-1] < val_losses[0]),
    "zero_prediction_baseline_mse": float(baseline_test_mse),
    "test_mse_reduction_vs_baseline": float(test_mse_reduction),
    "final_attention_gradient_norm": float(attention_gradient_norms[-1]),
    "attention_received_nonzero_gradients": bool(
        any(norm > 0.0 for norm in attention_gradient_norms)
    ),
    "mean_attention_by_position": {
        str(position): float(mean_attention_by_position[position])
        for position in range(CONFIG["sequence_length"])
    },
    "important_positions_zero_based": CONFIG["important_positions"]
}


# ============================================================
# Save Results
# ============================================================

results = {
    "model_configuration": model_config,

    "test_metrics": {
        "MSE": float(test_mse),
        "MAE": float(test_mae),
        "RMSE": float(test_rmse)
    },

    "best_validation_loss":
        float(best_val_loss),

    "zero_prediction_baseline_mse":
        float(baseline_test_mse),

    "test_mse_reduction_vs_baseline":
        float(test_mse_reduction),

    "mean_attention_entropy":
        mean_attention_entropy,

    "observations":
        observations
}


with open(
    RESULTS_DIR / "toy_results.json",
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


# ============================================================
# Save Attention
# ============================================================

np.save(
    RESULTS_DIR / "toy_attention.npy",
    attention_np
)


np.save(
    RESULTS_DIR / "toy_predictions.npy",
    test_predictions
)


# ============================================================
# Training Loss Plot
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history["epoch"],
    history["train_loss"],
    label="Training Loss"
)

plt.plot(
    history["epoch"],
    history["validation_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("MSE Loss")

plt.title(
    "Toy Sequence Prediction - Training Behavior"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "toy_training_loss.png",
    dpi=150
)

plt.close()


# ============================================================
# MAE Plot
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history["epoch"],
    history["train_mae"],
    label="Training MAE"
)

plt.plot(
    history["epoch"],
    history["validation_mae"],
    label="Validation MAE"
)

plt.xlabel("Epoch")

plt.ylabel("MAE")

plt.title(
    "Toy Sequence Prediction - MAE"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "toy_training_mae.png",
    dpi=150
)

plt.close()


# ============================================================
# Attention Heatmap
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.imshow(
    mean_attention,
    aspect="auto"
)

plt.colorbar(
    label="Mean Attention Weight"
)

plt.xlabel(
    "Key Position"
)

plt.ylabel(
    "Query Position"
)

plt.title(
    "Mean Attention Matrix"
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "toy_attention_heatmap.png",
    dpi=150
)

plt.close()


# ============================================================
# Final Report
# ============================================================

print("\n")
print("=" * 70)
print("SECTION 12 - PROBLEM 1: TOY LEARNING TASK")
print("=" * 70)

print("\nTask:")
print(
    "Synthetic sequence prediction"
)

print("\nLearning Objective:")
print(
    "Predict a target generated from multiple "
    "positions in the input sequence."
)

print("\nTarget equation:")
print(
    "y = 0.5*x[-1] + 0.3*x[-3] + 0.2*x[-6] + noise"
)

print("\nModel Configuration:")
for key, value in model_config.items():
    print(
        f"  {key}: {value}"
    )

print("\nFinal Training Loss:")
print(
    f"  {train_losses[-1]:.6f}"
)

print("\nFinal Validation Loss:")
print(
    f"  {val_losses[-1]:.6f}"
)

print("\nTest Metrics:")

print(
    f"  MSE  : {test_mse:.6f}"
)

print(
    f"  MAE  : {test_mae:.6f}"
)

print(
    f"  RMSE : {test_rmse:.6f}"
)

print("\nMean Attention Entropy:")
print(
    f"  {mean_attention_entropy:.6f}"
)

print("\nImportant sequence positions:")
print(
    "  Last position      : -1"
)

print(
    "  Third-last position: -3"
)

print(
    "  Sixth-last position: -6"
)

print("\nResults saved to:")
print(
    "  results/toy_training_history.csv"
)

print(
    "  results/toy_training_loss.png"
)

print(
    "  results/toy_training_mae.png"
)

print(
    "  results/toy_attention_heatmap.png"
)

print(
    "  results/toy_results.json"
)

print(
    "  results/toy_attention.npy"
)

print(
    "  results/toy_predictions.npy"
)

print("\n" + "=" * 70)
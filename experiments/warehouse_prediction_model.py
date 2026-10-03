# experiments/warehouse_prediction_model.py

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from src.data_generator import generate_warehouse_data
from src.dataset import prepare_data
from src.warehouse_model import WarehouseAttentionModel


# ============================================================
# Configuration
# ============================================================

SEED = 42

SEQUENCE_LENGTH = 24
D_MODEL = 32
HIDDEN_DIM = 16

EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.001


np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# Model Learning Explanation
# ============================================================

MODEL_EXPLANATION = {
    "input": (
        "Previous 24 hourly warehouse demand observations "
        "from t-23 through t."
    ),

    "input_representation": (
        "Each scalar hourly demand observation is projected "
        "into a d_model-dimensional representation."
    ),

    "self_attention": (
        "Self-attention computes relationships between every "
        "historical position in the 24-hour sequence."
    ),

    "attention_equation": (
        "A = softmax(QK^T / sqrt(d_k))"
    ),

    "context": (
        "The attention-weighted value representations are "
        "aggregated into a single context vector."
    ),

    "prediction": (
        "The context vector is passed through a prediction "
        "head to estimate the next-hour demand."
    ),

    "interpretation_warning": (
        "Attention weights are treated as a visualization "
        "of the model's internal weighting behavior. They "
        "are not automatically interpreted as causal "
        "explanations."
    )
}


# ============================================================
# Generate Dataset
# ============================================================

df = generate_warehouse_data(
    n_hours=24 * 365,
    seed=SEED
)


# ============================================================
# Prepare 24-hour sequences
# ============================================================

data = prepare_data(
    df["demand"].values,
    sequence_length=SEQUENCE_LENGTH
)


# ============================================================
# Build Model
# ============================================================

model = WarehouseAttentionModel(
    d_model=D_MODEL,
    hidden_dim=HIDDEN_DIM,
    scale=True
)


# Build model
dummy_input = tf.convert_to_tensor(
    data.X_train[:1],
    dtype=tf.float32
)

model(
    dummy_input,
    training=False
)


# ============================================================
# Optimizer
# ============================================================

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)

loss_function = tf.keras.losses.MeanSquaredError()


# ============================================================
# Training Dataset
# ============================================================

train_dataset = (
    tf.data.Dataset
    .from_tensor_slices(
        (
            data.X_train,
            data.y_train
        )
    )
    .shuffle(
        len(data.X_train),
        seed=SEED
    )
    .batch(
        BATCH_SIZE
    )
)


# ============================================================
# Validation Dataset
# ============================================================

validation_dataset = (
    tf.data.Dataset
    .from_tensor_slices(
        (
            data.X_val,
            data.y_val
        )
    )
    .batch(
        BATCH_SIZE
    )
)


# ============================================================
# Training
# ============================================================

training_history = []

print("=" * 70)
print("SECTION 16 - WAREHOUSE PREDICTION MODEL")
print("=" * 70)

print("\nTraining model...\n")


for epoch in range(EPOCHS):

    train_losses = []

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for X_batch, y_batch in train_dataset:

        with tf.GradientTape() as tape:

            predictions, _ = model(
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

        optimizer.apply_gradients(
            zip(
                gradients,
                model.trainable_variables
            )
        )

        train_losses.append(
            float(loss.numpy())
        )

    train_loss = np.mean(
        train_losses
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validation_losses = []

    for X_batch, y_batch in validation_dataset:

        predictions, _ = model(
            X_batch,
            training=False
        )

        loss = loss_function(
            y_batch,
            predictions
        )

        validation_losses.append(
            float(loss.numpy())
        )

    validation_loss = np.mean(
        validation_losses
    )

    training_history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "validation_loss": validation_loss
    })

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.6f} | "
        f"Validation Loss: {validation_loss:.6f}"
    )


# ============================================================
# Test Prediction
# ============================================================

predictions, attention_weights = model(
    data.X_test,
    training=False
)

predictions = (
    predictions
    .numpy()
    .reshape(-1)
)

y_test = (
    np.asarray(data.y_test)
    .reshape(-1)
)


# ============================================================
# Metrics
# ============================================================

mse = np.mean(
    (y_test - predictions) ** 2
)

mae = np.mean(
    np.abs(
        y_test - predictions
    )
)

rmse = np.sqrt(
    mse
)


# ============================================================
# Attention Statistics
# ============================================================

attention_weights = (
    attention_weights
    .numpy()
)

mean_attention = np.mean(
    attention_weights,
    axis=0
)

attention_entropy = -np.sum(
    attention_weights
    *
    np.log(
        attention_weights + 1e-12
    ),
    axis=-1
)

mean_entropy = np.mean(
    attention_entropy
)


# ============================================================
# Historical Position Mapping
# ============================================================

positions = [
    f"t-{23-i}"
    for i in range(24)
]

position_statistics = []

for i, position in enumerate(positions):

    position_statistics.append({
        "position": position,
        "mean_attention": float(
            mean_attention[:, i].mean()
        )
    })


position_df = pd.DataFrame(
    position_statistics
)

position_df = position_df.sort_values(
    "mean_attention",
    ascending=False
)


# ============================================================
# Save Results
# ============================================================

Path("results/metrics").mkdir(
    parents=True,
    exist_ok=True
)

Path("results/figures").mkdir(
    parents=True,
    exist_ok=True
)


position_df.to_csv(
    "results/metrics/"
    "section16_attention_positions.csv",
    index=False
)


history_df = pd.DataFrame(
    training_history
)

history_df.to_csv(
    "results/metrics/"
    "section16_training_history.csv",
    index=False
)


# ============================================================
# Training Curve
# ============================================================

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    history_df["epoch"],
    history_df["train_loss"],
    label="Training Loss"
)

plt.plot(
    history_df["epoch"],
    history_df["validation_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")

plt.title(
    "Warehouse Demand Prediction Training"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "results/figures/"
    "section16_training_curve.png",
    dpi=150
)

plt.close()


# ============================================================
# Attention Heatmap
# ============================================================

plt.figure(
    figsize=(10, 8)
)

plt.imshow(
    attention_weights[0],
    aspect="auto"
)

plt.colorbar(
    label="Attention Weight"
)

plt.xlabel(
    "Historical Position"
)

plt.ylabel(
    "Query Position"
)

plt.title(
    "Attention Weights for One 24-Hour Window"
)

plt.xticks(
    range(24),
    positions,
    rotation=90
)

plt.tight_layout()

plt.savefig(
    "results/figures/"
    "section16_attention_heatmap.png",
    dpi=150
)

plt.close()


# ============================================================
# Mean Attention by Historical Position
# ============================================================

plt.figure(
    figsize=(11, 5)
)

plt.bar(
    position_df["position"],
    position_df["mean_attention"]
)

plt.xlabel(
    "Historical Position"
)

plt.ylabel(
    "Mean Attention Weight"
)

plt.title(
    "Historical Positions Receiving Attention"
)

plt.xticks(
    rotation=90
)

plt.tight_layout()

plt.savefig(
    "results/figures/"
    "section16_position_attention.png",
    dpi=150
)

plt.close()


# ============================================================
# Example Prediction
# ============================================================

example_index = 0

example_history = (
    data.X_test[
        example_index
    ]
)

example_prediction = (
    predictions[
        example_index
    ]
)

example_actual = (
    y_test[
        example_index
    ]
)


example_df = pd.DataFrame({
    "historical_position": positions,
    "demand": example_history,
    "attention_weight": (
        attention_weights[
            example_index
        ].mean(axis=0)
    )
})

example_df.to_csv(
    "results/metrics/"
    "section16_example_window.csv",
    index=False
)


# ============================================================
# Explanation
# ============================================================

explanation = {

    "section": 16,

    "objective":
        "Predict next-hour warehouse demand from "
        "the previous 24 hourly observations.",

    "pipeline": [
        "t-23 ... t-2 t-1",
        "Input representation",
        "Self-attention",
        "Aggregated context",
        "Prediction",
        "Next-hour demand"
    ],

    "what_model_learns":
        "The model learns parameterized relationships "
        "between historical hourly demand representations "
        "through Q, K and V projections. Attention determines "
        "how strongly each historical representation "
        "contributes to each query representation.",

    "test_mse": float(mse),

    "test_mae": float(mae),

    "test_rmse": float(rmse),

    "mean_attention_entropy":
        float(mean_entropy),

    "highest_attention_positions":
        position_df.head(5).to_dict(
            orient="records"
        ),

    "interpretation_warning":
        "Attention weights describe model weighting behavior "
        "and should not automatically be interpreted as "
        "causal explanations."
}


with open(
    "results/metrics/"
    "section16_model_explanation.json",
    "w"
) as f:

    json.dump(
        explanation,
        f,
        indent=4
    )


# ============================================================
# Print
# ============================================================

print("\n")
print("=" * 70)
print("SECTION 16 RESULTS")
print("=" * 70)

print("\nPipeline:")

print(
    "t-23 ... t-2 t-1"
)

print("      ↓")
print("Input representation")
print("      ↓")
print("Self-attention")
print("      ↓")
print("Aggregated context")
print("      ↓")
print("Prediction")
print("      ↓")
print("ŷ(t+1)")


print("\nTest MSE:")
print(
    f"{mse:.6f}"
)

print("\nTest MAE:")
print(
    f"{mae:.6f}"
)

print("\nTest RMSE:")
print(
    f"{rmse:.6f}"
)

print("\nMean attention entropy:")
print(
    f"{mean_entropy:.6f}"
)

print("\nHistorical positions with highest mean attention:")

print(
    position_df.head(5).to_string(
        index=False
    )
)

print("\nImportant interpretation note:")
print(
    "Attention weights are visualization of model "
    "weighting behavior, not automatic causal explanations."
)

print("\nResults saved under:")
print("results/metrics/")
print("results/figures/")
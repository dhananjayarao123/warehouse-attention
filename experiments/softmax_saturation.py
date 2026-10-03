import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from src.data_generator import (
    generate_warehouse_data
)

from src.dataset import (
    prepare_data
)

from src.warehouse_model import (
    WarehouseAttentionModel
)


def entropy(weights):

    eps = 1e-8

    return -tf.reduce_sum(
        weights *
        tf.math.log(weights + eps),
        axis=-1,
    )


def gradient_norm(gradients):

    gradients = [
        g for g in gradients
        if g is not None
    ]

    if len(gradients) == 0:
        return 0.0

    return float(
        tf.linalg.global_norm(
            gradients
        ).numpy()
    )


def run_experiment(
    X_train,
    y_train,
    scale,
    epochs=30,
    learning_rate=1e-3,
):

    model = WarehouseAttentionModel(
        d_model=64,
        hidden_dim=32,
        scale=scale,
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    loss_fn = (
        tf.keras.losses.MeanSquaredError()
    )

    # Build model
    model(
        tf.convert_to_tensor(
            X_train[:1],
            dtype=tf.float32,
        )
    )

    records = []

    for epoch in range(epochs):

        epoch_losses = []
        epoch_gradients = []
        epoch_entropy = []
        epoch_max_attention = []
        epoch_saturated_fraction = []

        # Mini-batches
        indices = np.arange(
            len(X_train)
        )

        np.random.shuffle(indices)

        for start in range(
            0,
            len(indices),
            128,
        ):

            batch_idx = indices[
                start:start + 128
            ]

            X_batch = tf.convert_to_tensor(
                X_train[batch_idx],
                dtype=tf.float32,
            )

            y_batch = tf.convert_to_tensor(
                y_train[batch_idx],
                dtype=tf.float32,
            )

            with tf.GradientTape() as tape:

                prediction, weights, scores = (
                    model(
                        X_batch,
                        training=True,
                    )
                )

                loss = loss_fn(
                    y_batch,
                    prediction,
                )

            gradients = tape.gradient(
                loss,
                model.trainable_variables,
            )

            optimizer.apply_gradients(
                zip(
                    gradients,
                    model.trainable_variables,
                )
            )

            # --------------------------------
            # Loss
            # --------------------------------

            epoch_losses.append(
                float(loss.numpy())
            )

            # --------------------------------
            # Gradient norm
            # --------------------------------

            epoch_gradients.append(
                gradient_norm(
                    gradients
                )
            )

            # --------------------------------
            # Attention entropy
            # --------------------------------

            batch_entropy = tf.reduce_mean(
                entropy(weights)
            )

            epoch_entropy.append(
                float(
                    batch_entropy.numpy()
                )
            )

            # --------------------------------
            # Maximum attention
            # --------------------------------

            max_attention = tf.reduce_mean(
                tf.reduce_max(
                    weights,
                    axis=-1,
                )
            )

            epoch_max_attention.append(
                float(
                    max_attention.numpy()
                )
            )

            # --------------------------------
            # Saturation
            # --------------------------------

            saturated = tf.reduce_mean(
                tf.cast(
                    weights > 0.95,
                    tf.float32,
                )
            )

            epoch_saturated_fraction.append(
                float(
                    saturated.numpy()
                )
            )

        records.append({
            "epoch": epoch + 1,

            "loss":
                np.mean(epoch_losses),

            "gradient_norm":
                np.mean(epoch_gradients),

            "attention_entropy":
                np.mean(epoch_entropy),

            "max_attention":
                np.mean(
                    epoch_max_attention
                ),

            "saturated_fraction":
                np.mean(
                    epoch_saturated_fraction
                ),
        })

    return pd.DataFrame(records)


def main():

    Path(
        "results/metrics"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(
        "results/figures"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    # ==================================================
    # Generate data
    # ==================================================

    df = generate_warehouse_data(
        seed=42
    )

    data = prepare_data(
        df["demand"].values,
        sequence_length=24,
    )

    # ==================================================
    # Hypothesis
    # ==================================================

    print(
        """
        H3 — Softmax Saturation Hypothesis

        Unscaled attention should produce larger
        QK^T logits.

        Larger logits should make softmax more
        concentrated.

        Therefore we expect:

        Unscaled:
            lower entropy
            higher max attention
            higher saturation

        Scaled:
            higher entropy
            less concentration
            better-controlled gradients
        """
    )

    # ==================================================
    # Scaled
    # ==================================================

    print(
        "\nRunning SCALED attention..."
    )

    scaled = run_experiment(
        data.X_train,
        data.y_train,
        scale=True,
    )

    scaled["attention_type"] = "Scaled"

    # ==================================================
    # Unscaled
    # ==================================================

    print(
        "\nRunning UNSCALED attention..."
    )

    unscaled = run_experiment(
        data.X_train,
        data.y_train,
        scale=False,
    )

    unscaled["attention_type"] = "Unscaled"

    # ==================================================
    # Combine
    # ==================================================

    results = pd.concat(
        [
            scaled,
            unscaled,
        ],
        ignore_index=True,
    )

    results.to_csv(
        "results/metrics/"
        "softmax_saturation.csv",
        index=False,
    )

    # ==================================================
    # Print final comparison
    # ==================================================

    print(
        "\nFinal epoch comparison:"
    )

    print(
        results.groupby(
            "attention_type"
        ).tail(1)
    )

    # ==================================================
    # Plot 1: Loss
    # ==================================================

    plt.figure(
        figsize=(9, 5)
    )

    for name, group in results.groupby(
        "attention_type"
    ):

        plt.plot(
            group["epoch"],
            group["loss"],
            label=name,
        )

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")

    plt.title(
        "Training Dynamics: "
        "Scaled vs Unscaled Attention"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "results/figures/"
        "scaled_vs_unscaled_loss.png",
        dpi=150,
    )

    plt.close()

    # ==================================================
    # Plot 2: Gradient norm
    # ==================================================

    plt.figure(
        figsize=(9, 5)
    )

    for name, group in results.groupby(
        "attention_type"
    ):

        plt.plot(
            group["epoch"],
            group["gradient_norm"],
            label=name,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Gradient L2 Norm")

    plt.title(
        "Gradient Behavior"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "results/figures/"
        "scaled_vs_unscaled_gradients.png",
        dpi=150,
    )

    plt.close()

    # ==================================================
    # Plot 3: Attention entropy
    # ==================================================

    plt.figure(
        figsize=(9, 5)
    )

    for name, group in results.groupby(
        "attention_type"
    ):

        plt.plot(
            group["epoch"],
            group["attention_entropy"],
            label=name,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Attention Entropy")

    plt.title(
        "Softmax Saturation: Attention Entropy"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "results/figures/"
        "attention_entropy_scaled_vs_unscaled.png",
        dpi=150,
    )

    plt.close()

    # ==================================================
    # Plot 4: Maximum attention
    # ==================================================

    plt.figure(
        figsize=(9, 5)
    )

    for name, group in results.groupby(
        "attention_type"
    ):

        plt.plot(
            group["epoch"],
            group["max_attention"],
            label=name,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Maximum Attention Weight")

    plt.title(
        "Attention Concentration"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "results/figures/"
        "attention_concentration_scaled_vs_unscaled.png",
        dpi=150,
    )

    plt.close()

    # ==================================================
    # Plot 5: Saturation fraction
    # ==================================================

    plt.figure(
        figsize=(9, 5)
    )

    for name, group in results.groupby(
        "attention_type"
    ):

        plt.plot(
            group["epoch"],
            group["saturated_fraction"],
            label=name,
        )

    plt.xlabel("Epoch")

    plt.ylabel(
        "Fraction of weights > 0.95"
    )

    plt.title(
        "Softmax Saturation"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "results/figures/"
        "softmax_saturation_fraction.png",
        dpi=150,
    )

    plt.close()

    print(
        "\nResults saved to:"
    )

    print(
        "results/metrics/"
        "softmax_saturation.csv"
    )


if __name__ == "__main__":
    main()
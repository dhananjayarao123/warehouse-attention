from dataclasses import dataclass
import random
import numpy as np
import tensorflow as tf

from .dataset import make_dataset


@dataclass
class TrainingHistory:
    train_loss: list
    val_loss: list

    # Section 13
    gradient_norm: list
    attention_entropy: list
    attention_max: list
    attention_min: list


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def attention_entropy(weights):
    """
    H(A) = -sum_i A_i log(A_i)

    Lower entropy:
        attention is concentrated.

    Higher entropy:
        attention is more distributed.
    """
    eps = tf.constant(1e-8, dtype=weights.dtype)

    entropy = -tf.reduce_sum(
        weights * tf.math.log(weights + eps),
        axis=-1
    )

    return tf.reduce_mean(entropy)


def compute_gradient_norm(gradients):
    """
    Global L2 norm of all trainable-variable gradients.
    """
    valid_gradients = [
        g for g in gradients
        if g is not None
    ]

    if not valid_gradients:
        return 0.0

    return float(
        tf.linalg.global_norm(valid_gradients).numpy()
    )


def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=30,
    batch_size=128,
    learning_rate=1e-3,
    seed=42,
    collect_dynamics=True,
):
    set_seed(seed)

    train_ds = make_dataset(
        X_train,
        y_train,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    val_ds = make_dataset(
        X_val,
        y_val,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    loss_fn = tf.keras.losses.MeanSquaredError()

    history = TrainingHistory(
        train_loss=[],
        val_loss=[],
        gradient_norm=[],
        attention_entropy=[],
        attention_max=[],
        attention_min=[],
    )

    # Build model variables.
    _ = model(
        tf.convert_to_tensor(
            X_train[:1],
            dtype=tf.float32
        ),
        training=False,
    )

    for epoch in range(epochs):

        train_metric = tf.keras.metrics.Mean()

        epoch_gradient_norms = []
        epoch_entropies = []
        epoch_attention_max = []
        epoch_attention_min = []

        for X_batch, y_batch in train_ds:

            with tf.GradientTape() as tape:

                prediction, attention_weights = model(
                    X_batch,
                    training=True,
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

            train_metric.update_state(loss)

            if collect_dynamics:

                # Gradient magnitude
                grad_norm = compute_gradient_norm(
                    gradients
                )

                # Attention distribution
                entropy = float(
                    attention_entropy(
                        attention_weights
                    ).numpy()
                )

                max_attention = float(
                    tf.reduce_max(
                        attention_weights
                    ).numpy()
                )

                min_attention = float(
                    tf.reduce_min(
                        attention_weights
                    ).numpy()
                )

                epoch_gradient_norms.append(
                    grad_norm
                )

                epoch_entropies.append(
                    entropy
                )

                epoch_attention_max.append(
                    max_attention
                )

                epoch_attention_min.append(
                    min_attention
                )

        # -------------------------
        # Validation
        # -------------------------

        val_metric = tf.keras.metrics.Mean()

        for X_batch, y_batch in val_ds:

            prediction, _ = model(
                X_batch,
                training=False,
            )

            loss = loss_fn(
                y_batch,
                prediction,
            )

            val_metric.update_state(loss)

        train_loss = float(
            train_metric.result().numpy()
        )

        val_loss = float(
            val_metric.result().numpy()
        )

        history.train_loss.append(
            train_loss
        )

        history.val_loss.append(
            val_loss
        )

        if collect_dynamics:

            history.gradient_norm.append(
                float(
                    np.mean(epoch_gradient_norms)
                )
            )

            history.attention_entropy.append(
                float(
                    np.mean(epoch_entropies)
                )
            )

            history.attention_max.append(
                float(
                    np.mean(epoch_attention_max)
                )
            )

            history.attention_min.append(
                float(
                    np.mean(epoch_attention_min)
                )
            )

        print(
            f"Epoch {epoch + 1:03d}/{epochs} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"grad_norm="
            f"{history.gradient_norm[-1]:.6f} "
            f"entropy="
            f"{history.attention_entropy[-1]:.6f} "
            f"max_attention="
            f"{history.attention_max[-1]:.6f}"
        )

    return history


def predict_model(model, X):

    tensor_x = tf.convert_to_tensor(
        np.asarray(X),
        dtype=tf.float32,
    )

    predictions, weights = model(
        tensor_x,
        training=False,
    )

    return (
        predictions.numpy(),
        weights.numpy(),
    )
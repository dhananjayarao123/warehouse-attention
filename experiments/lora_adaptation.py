"""
Parameter-Efficient Adaptation Experiment — LoRA

This experiment demonstrates minimal low-rank adaptation on the
warehouse attention model.

The experiment:

1. Builds a small warehouse forecasting model.
2. Initializes its base parameters.
3. Freezes the base model.
4. Trains only LoRA parameters.
5. Reports trainable vs total parameters.
6. Evaluates forecasting performance.
7. Verifies that the frozen base parameters did not change.
"""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.data_generator import generate_warehouse_data
from src.dataset import create_sequences
from src.warehouse_model import (
    LoRAWarehouseAttentionModel,
)


SEED = 42

D_MODEL = 32
HIDDEN_DIM = 32

LOOKBACK = 24

RANK = 2
ALPHA = 4.0

EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-3


def set_seed():

    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def prepare_data():

    df = generate_warehouse_data(
        n_hours=24 * 365,
        seed=SEED,
    )

    demand = df["demand"].to_numpy(
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Chronological split
    # --------------------------------------------------------

    n = len(demand)

    train_end = int(0.70 * n)
    val_end = int(0.85 * n)

    train = demand[:train_end]

    validation = demand[
        train_end:val_end
    ]

    test = demand[
        val_end:
    ]

    # --------------------------------------------------------
    # Normalize using training statistics only
    # --------------------------------------------------------

    mean = train.mean()
    std = train.std() + 1e-8

    train = (
        train - mean
    ) / std

    validation = (
        validation - mean
    ) / std

    test = (
        test - mean
    ) / std

    # --------------------------------------------------------
    # Convert into supervised sequences
    # --------------------------------------------------------

    X_train, y_train = create_sequences(
        train,
        lookback=LOOKBACK,
    )

    X_val, y_val = create_sequences(
        validation,
        lookback=LOOKBACK,
    )

    X_test, y_test = create_sequences(
        test,
        lookback=LOOKBACK,
    )

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )


def build_model():

    model = LoRAWarehouseAttentionModel(
        d_model=D_MODEL,
        hidden_dim=HIDDEN_DIM,
        rank=RANK,
        alpha=ALPHA,
        scaled=True,
    )

    # Build the model.
    dummy_input = tf.zeros(
        shape=(1, LOOKBACK, 1),
        dtype=tf.float32,
    )

    model(dummy_input)

    return model


def count_parameters(model):

    total = np.sum([
        np.prod(variable.shape)
        for variable in model.variables
    ])

    trainable = np.sum([
        np.prod(variable.shape)
        for variable in model.trainable_variables
    ])

    non_trainable = np.sum([
        np.prod(variable.shape)
        for variable in model.non_trainable_variables
    ])

    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "non_trainable_parameters": int(
            non_trainable
        ),
    }


def snapshot_base_parameters(model):

    snapshots = {}

    for variable in model.non_trainable_variables:

        snapshots[
            variable.name
        ] = variable.numpy().copy()

    return snapshots


def verify_frozen_parameters(
    model,
    snapshots,
):

    maximum_difference = 0.0

    for variable in model.non_trainable_variables:

        old_value = snapshots[
            variable.name
        ]

        new_value = variable.numpy()

        difference = np.max(
            np.abs(
                old_value - new_value
            )
        )

        maximum_difference = max(
            maximum_difference,
            float(difference),
        )

    return maximum_difference


def main():

    set_seed()

    print("=" * 70)
    print("PARAMETER-EFFICIENT ADAPTATION — LoRA")
    print("=" * 70)

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = prepare_data()

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_model()

    # --------------------------------------------------------
    # Freeze base model
    #
    # Only LoRA A/B matrices remain trainable.
    # --------------------------------------------------------

    for layer in model.layers:

        if hasattr(
            layer,
            "q_projection"
        ):

            layer.q_projection.base_kernel._trainable = False
            layer.k_projection.base_kernel._trainable = False
            layer.v_projection.base_kernel._trainable = False

    # --------------------------------------------------------
    # Freeze input projection and prediction head.
    # --------------------------------------------------------

    model.input_projection.trainable = False
    model.hidden.trainable = False
    model.output_layer.trainable = False

    # --------------------------------------------------------
    # Explicitly identify LoRA parameters.
    # --------------------------------------------------------

    model.attention.q_projection.lora_A._trainable = True
    model.attention.q_projection.lora_B._trainable = True

    model.attention.k_projection.lora_A._trainable = True
    model.attention.k_projection.lora_B._trainable = True

    model.attention.v_projection.lora_A._trainable = True
    model.attention.v_projection.lora_B._trainable = True

    # --------------------------------------------------------
    # Compile AFTER trainable configuration.
    # --------------------------------------------------------

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(
                name="mae"
            )
        ],
    )

    parameter_counts = count_parameters(model)

    print("\nParameter counts:")
    print(
        f"Total parameters      : "
        f"{parameter_counts['total_parameters']}"
    )

    print(
        f"Trainable parameters  : "
        f"{parameter_counts['trainable_parameters']}"
    )

    print(
        f"Frozen parameters     : "
        f"{parameter_counts['non_trainable_parameters']}"
    )

    print(
        f"LoRA parameters       : "
        f"{model.lora_parameter_count()}"
    )

    # --------------------------------------------------------
    # Snapshot frozen parameters
    # --------------------------------------------------------

    frozen_before = snapshot_base_parameters(
        model
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    history = model.fit(
        X_train[..., None],
        y_train,
        validation_data=(
            X_val[..., None],
            y_val,
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    test_results = model.evaluate(
        X_test[..., None],
        y_test,
        verbose=0,
        return_dict=True,
    )

    # --------------------------------------------------------
    # Verify frozen weights
    # --------------------------------------------------------

    frozen_difference = verify_frozen_parameters(
        model,
        frozen_before,
    )

    print("\nTest results:")

    for name, value in test_results.items():

        print(
            f"{name}: {value:.6f}"
        )

    print(
        "\nMaximum change in frozen "
        f"parameters: {frozen_difference:.6e}"
    )

    assert frozen_difference < 1e-12, (
        "Frozen base parameters changed!"
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    output_dir = Path(
        "results/lora"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = {
        "seed": SEED,
        "lookback": LOOKBACK,
        "d_model": D_MODEL,
        "hidden_dim": HIDDEN_DIM,
        "lora_rank": RANK,
        "lora_alpha": ALPHA,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "parameter_counts": parameter_counts,
        "lora_parameters": int(
            model.lora_parameter_count()
        ),
        "test_results": {
            key: float(value)
            for key, value in test_results.items()
        },
        "maximum_frozen_parameter_change": float(
            frozen_difference
        ),
    }

    with open(
        output_dir / "lora_results.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
        )

    print(
        "\nSaved:"
        "\nresults/lora/lora_results.json"
    )


if __name__ == "__main__":
    main()
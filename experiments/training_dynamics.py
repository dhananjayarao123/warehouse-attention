import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

import matplotlib.pyplot as plt
import pandas as pd

from src.data_generator import generate_warehouse_data
from src.dataset import prepare_data
from src.warehouse_model import WarehouseAttentionModel
from src.train import train_model


def main():

    # ------------------------------------------------
    # Directories
    # ------------------------------------------------

    Path("results/figures").mkdir(
        parents=True,
        exist_ok=True,
    )

    Path("results/metrics").mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------
    # Generate synthetic warehouse data
    # ------------------------------------------------

    df = generate_warehouse_data(
        seed=42
    )

    data = prepare_data(
        df["demand"].values,
        sequence_length=24,
    )

    # ------------------------------------------------
    # Attention model
    # ------------------------------------------------

    model = WarehouseAttentionModel(
        d_model=32,
        hidden_dim=16,
        scale=True,
    )

    # ------------------------------------------------
    # Train
    # ------------------------------------------------

    history = train_model(
        model=model,

        X_train=data.X_train,
        y_train=data.y_train,

        X_val=data.X_val,
        y_val=data.y_val,

        epochs=15,
        batch_size=128,

        learning_rate=1e-3,

        seed=42,

        collect_dynamics=True,
    )

    # ------------------------------------------------
    # Save results
    # ------------------------------------------------

    results = pd.DataFrame({
        "epoch": range(
            1,
            len(history.train_loss) + 1
        ),

        "train_loss":
            history.train_loss,

        "val_loss":
            history.val_loss,

        "gradient_norm":
            history.gradient_norm,

        "attention_entropy":
            history.attention_entropy,

        "attention_max":
            history.attention_max,

        "attention_min":
            history.attention_min,
    })

    results.to_csv(
        "results/metrics/"
        "training_dynamics.csv",
        index=False,
    )

    print("\nTraining dynamics:")
    print(results)

    # ------------------------------------------------
    # Figure 1: Loss
    # ------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.plot(
        results["epoch"],
        results["train_loss"],
        label="Training Loss",
    )

    plt.plot(
        results["epoch"],
        results["val_loss"],
        label="Validation Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(
        "Training Dynamics: Loss"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "results/figures/"
        "training_dynamics_loss.png",
        dpi=150,
    )

    plt.close()

    # ------------------------------------------------
    # Figure 2: Gradient norm
    # ------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.plot(
        results["epoch"],
        results["gradient_norm"],
        label="Gradient Norm",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Global Gradient L2 Norm")
    plt.title(
        "Training Dynamics: Gradient Norm"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "results/figures/"
        "training_gradient_norm.png",
        dpi=150,
    )

    plt.close()

    # ------------------------------------------------
    # Figure 3: Attention entropy
    # ------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.plot(
        results["epoch"],
        results["attention_entropy"],
        label="Attention Entropy",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Entropy")
    plt.title(
        "Training Dynamics: Attention Entropy"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "results/figures/"
        "attention_entropy.png",
        dpi=150,
    )

    plt.close()

    # ------------------------------------------------
    # Figure 4: Attention concentration
    # ------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.plot(
        results["epoch"],
        results["attention_max"],
        label="Maximum Attention",
    )

    plt.plot(
        results["epoch"],
        results["attention_min"],
        label="Minimum Attention",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Attention Weight")
    plt.title(
        "Training Dynamics: Attention Concentration"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "results/figures/"
        "attention_concentration.png",
        dpi=150,
    )

    plt.close()

    print(
        "\nSaved training dynamics to:"
    )

    print(
        "results/metrics/"
        "training_dynamics.csv"
    )


if __name__ == "__main__":
    main()
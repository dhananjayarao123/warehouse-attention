# experiments/gradient_comparison.py

import numpy as np
import tensorflow as tf
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.manual_attention import (
    attention_forward,
    attention_backward,
)


def main():

    np.random.seed(42)
    tf.random.set_seed(42)

    B = 2
    T = 4
    D = 3

    X_np = np.random.randn(B, T, D).astype(np.float64)

    W_Q_np = np.random.randn(D, D).astype(np.float64)
    W_K_np = np.random.randn(D, D).astype(np.float64)
    W_V_np = np.random.randn(D, D).astype(np.float64)

    dY_np = np.random.randn(B, T, D).astype(np.float64)

    # =========================================================
    # Manual NumPy implementation
    # =========================================================

    Y_np, cache = attention_forward(
        X_np,
        W_Q_np,
        W_K_np,
        W_V_np,
    )

    manual_grads = attention_backward(
        dY_np,
        cache,
        W_Q_np,
        W_K_np,
        W_V_np,
    )

    # =========================================================
    # TensorFlow independent implementation
    # =========================================================

    X = tf.Variable(X_np)
    W_Q = tf.Variable(W_Q_np)
    W_K = tf.Variable(W_K_np)
    W_V = tf.Variable(W_V_np)

    dY = tf.constant(dY_np)

    with tf.GradientTape() as tape:

        Q = tf.matmul(X, W_Q)

        K = tf.matmul(X, W_K)

        V = tf.matmul(X, W_V)

        scores = tf.matmul(
            Q,
            K,
            transpose_b=True,
        )

        scores = scores / tf.sqrt(
            tf.cast(D, tf.float64)
        )

        attention = tf.nn.softmax(
            scores,
            axis=-1,
        )

        Y = tf.matmul(
            attention,
            V,
        )

        loss = tf.reduce_sum(
            Y * dY
        )

    tf_grads = tape.gradient(
        loss,
        [
            X,
            W_Q,
            W_K,
            W_V,
        ],
    )

    names = [
        "dX",
        "dW_Q",
        "dW_K",
        "dW_V",
    ]

    print("=" * 60)
    print("MANUAL BACKWARD vs TENSORFLOW GRADIENT")
    print("=" * 60)

    for name, manual, tf_grad in zip(
        names,
        [
            manual_grads["dX"],
            manual_grads["dW_Q"],
            manual_grads["dW_K"],
            manual_grads["dW_V"],
        ],
        tf_grads,
    ):

        tf_grad_np = tf_grad.numpy()

        absolute_error = np.max(
            np.abs(
                manual - tf_grad_np
            )
        )

        relative_error = (
            absolute_error
            /
            (
                np.max(
                    np.abs(tf_grad_np)
                )
                + 1e-12
            )
        )

        print(f"\n{name}")
        print(
            f"Max absolute error : "
            f"{absolute_error:.12e}"
        )

        print(
            f"Max relative error : "
            f"{relative_error:.12e}"
        )

        if relative_error < 1e-5:
            print("Status              : PASS")
        else:
            print("Status              : CHECK")


if __name__ == "__main__":
    main()
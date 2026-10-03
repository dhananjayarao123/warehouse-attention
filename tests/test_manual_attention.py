# tests/test_manual_attention.py

import numpy as np

from src.manual_attention import (
    attention_forward,
    attention_backward,
    numerical_gradient,
)


def test_manual_attention_shapes():

    np.random.seed(42)

    B = 2
    T = 4
    D = 3

    X = np.random.randn(B, T, D)

    W_Q = np.random.randn(D, D)
    W_K = np.random.randn(D, D)
    W_V = np.random.randn(D, D)

    Y, cache = attention_forward(
        X,
        W_Q,
        W_K,
        W_V,
    )

    assert X.shape == (B, T, D)

    assert cache["Q"].shape == (B, T, D)
    assert cache["K"].shape == (B, T, D)
    assert cache["V"].shape == (B, T, D)

    assert cache["S"].shape == (B, T, T)
    assert cache["A"].shape == (B, T, T)

    assert Y.shape == (B, T, D)


def test_manual_backward():

    np.random.seed(42)

    B = 2
    T = 3
    D = 4

    X = np.random.randn(B, T, D)

    W_Q = np.random.randn(D, D)
    W_K = np.random.randn(D, D)
    W_V = np.random.randn(D, D)

    Y, cache = attention_forward(
        X,
        W_Q,
        W_K,
        W_V,
    )

    dY = np.random.randn(*Y.shape)

    grads = attention_backward(
        dY,
        cache,
        W_Q,
        W_K,
        W_V,
    )

    assert grads["dX"].shape == X.shape
    assert grads["dW_Q"].shape == W_Q.shape
    assert grads["dW_K"].shape == W_K.shape
    assert grads["dW_V"].shape == W_V.shape


def test_gradient_against_finite_difference():

    np.random.seed(123)

    B = 1
    T = 3
    D = 2

    X = np.random.randn(B, T, D)

    W_Q = np.random.randn(D, D)
    W_K = np.random.randn(D, D)
    W_V = np.random.randn(D, D)

    Y, cache = attention_forward(
        X,
        W_Q,
        W_K,
        W_V,
    )

    dY = np.random.randn(*Y.shape)

    manual = attention_backward(
        dY,
        cache,
        W_Q,
        W_K,
        W_V,
    )

    def loss_X(x):
        y, _ = attention_forward(
            x,
            W_Q,
            W_K,
            W_V,
        )

        return np.sum(y * dY)

    numerical_X = numerical_gradient(
        loss_X,
        X.copy(),
    )

    error_X = np.max(
        np.abs(
            manual["dX"] - numerical_X
        )
    )

    assert error_X < 1e-5
# src/manual_attention.py

import numpy as np


def softmax(x, axis=-1):
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def attention_forward(X, W_Q, W_K, W_V, scale=True):
    """
    Scaled dot-product self-attention.

    Shapes:
        X   : (B, T, D)
        W_Q : (D, D)
        W_K : (D, D)
        W_V : (D, D)

        Q   : (B, T, D)
        K   : (B, T, D)
        V   : (B, T, D)

        S   : (B, T, T)
        A   : (B, T, T)
        Y   : (B, T, D)
    """

    B, T, D = X.shape

    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    S = Q @ np.swapaxes(K, -1, -2)

    if scale:
        scale_factor = np.sqrt(Q.shape[-1])
        S = S / scale_factor
    else:
        scale_factor = 1.0

    A = softmax(S, axis=-1)

    Y = A @ V

    cache = {
        "X": X,
        "Q": Q,
        "K": K,
        "V": V,
        "S": S,
        "A": A,
        "scale_factor": scale_factor,
    }

    return Y, cache


def softmax_backward(dA, A):
    """
    Softmax derivative.

    For each row:

        dS_i = A_i * (dA_i - sum_j(dA_j * A_j))
    """

    dot = np.sum(dA * A, axis=-1, keepdims=True)

    dS = A * (dA - dot)

    return dS


def attention_backward(dY, cache, W_Q, W_K, W_V):
    """
    Manual reverse-mode backward pass.

    Forward:

        Q = XW_Q
        K = XW_K
        V = XW_V

        S = QK^T / sqrt(D)

        A = softmax(S)

        Y = AV

    Backward:

        dV = A^T dY
        dA = dY V^T

        dS = softmax_backward(dA, A)

        dQ = dS K / sqrt(D)
        dK = dS^T Q / sqrt(D)

        dX_Q = dQ W_Q^T
        dX_K = dK W_K^T
        dX_V = dV W_V^T

        dX = dX_Q + dX_K + dX_V

        dW_Q = X^T dQ
        dW_K = X^T dK
        dW_V = X^T dV
    """

    X = cache["X"]
    Q = cache["Q"]
    K = cache["K"]
    V = cache["V"]
    A = cache["A"]
    scale_factor = cache["scale_factor"]

    # ---------------------------------------------------------
    # Y = A V
    # ---------------------------------------------------------

    dA = dY @ np.swapaxes(V, -1, -2)

    dV = np.swapaxes(A, -1, -2) @ dY

    # ---------------------------------------------------------
    # A = softmax(S)
    # ---------------------------------------------------------

    dS = softmax_backward(dA, A)

    # ---------------------------------------------------------
    # S = Q K^T / sqrt(D)
    # ---------------------------------------------------------

    dQ = (dS @ K) / scale_factor

    dK = (np.swapaxes(dS, -1, -2) @ Q) / scale_factor

    # ---------------------------------------------------------
    # Q = X W_Q
    # K = X W_K
    # V = X W_V
    # ---------------------------------------------------------

    dX_Q = dQ @ W_Q.T
    dX_K = dK @ W_K.T
    dX_V = dV @ W_V.T

    dX = dX_Q + dX_K + dX_V

    # ---------------------------------------------------------
    # Weight gradients
    # ---------------------------------------------------------

    dW_Q = np.einsum("btd,bte->de", X, dQ)

    dW_K = np.einsum("btd,bte->de", X, dK)

    dW_V = np.einsum("btd,bte->de", X, dV)

    return {
        "dX": dX,
        "dW_Q": dW_Q,
        "dW_K": dW_K,
        "dW_V": dW_V,
        "dQ": dQ,
        "dK": dK,
        "dV": dV,
        "dS": dS,
        "dA": dA,
    }


def numerical_gradient(func, x, eps=1e-5):
    """
    Independent finite-difference gradient.

        df/dx ≈ [f(x+eps) - f(x-eps)] / (2eps)
    """

    grad = np.zeros_like(x)

    iterator = np.nditer(
        x,
        flags=["multi_index"],
        op_flags=["readwrite"],
    )

    while not iterator.finished:
        idx = iterator.multi_index

        original = x[idx]

        x[idx] = original + eps
        f_plus = func(x)

        x[idx] = original - eps
        f_minus = func(x)

        x[idx] = original

        grad[idx] = (f_plus - f_minus) / (2.0 * eps)

        iterator.iternext()

    return grad


def compare_gradients(manual, reference):
    """
    Compare two gradients.
    """

    absolute_error = np.max(np.abs(manual - reference))

    relative_error = (
        np.max(np.abs(manual - reference))
        / (np.max(np.abs(reference)) + 1e-12)
    )

    return {
        "max_absolute_error": float(absolute_error),
        "max_relative_error": float(relative_error),
    }
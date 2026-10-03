import numpy as np

from src.manual_attention import attention_forward, attention_backward, softmax
from src.data_generator import generate_warehouse_data


def test_attention_tensor_shapes():
    rng = np.random.default_rng(23)
    batch, time, input_dim, d_k = 2, 7, 3, 5
    X = rng.normal(size=(batch, time, input_dim))
    W_Q = rng.normal(size=(input_dim, d_k))
    W_K = rng.normal(size=(input_dim, d_k))
    W_V = rng.normal(size=(input_dim, d_k))
    Y, cache = attention_forward(X, W_Q, W_K, W_V)
    assert cache["Q"].shape == (batch, time, d_k)
    assert cache["K"].shape == (batch, time, d_k)
    assert cache["V"].shape == (batch, time, d_k)
    assert cache["S"].shape == (batch, time, time)
    assert cache["A"].shape == (batch, time, time)
    assert Y.shape == (batch, time, d_k)


def test_softmax_normalization():
    rng = np.random.default_rng(24)
    logits = rng.normal(size=(4, 6, 6))
    A = softmax(logits, axis=-1)
    assert np.isfinite(A).all()
    assert np.all(A >= 0.0)
    np.testing.assert_allclose(np.sum(A, axis=-1), 1.0, atol=1e-10)


def test_attention_output_dimensions():
    rng = np.random.default_rng(25)
    X = rng.normal(size=(3, 10, 4))
    W_Q = rng.normal(size=(4, 6))
    W_K = rng.normal(size=(4, 6))
    W_V = rng.normal(size=(4, 8))
    Y, cache = attention_forward(X, W_Q, W_K, W_V)
    assert cache["A"].shape == (3, 10, 10)
    assert cache["V"].shape == (3, 10, 8)
    assert Y.shape == (3, 10, 8)


def test_scaling_uses_projected_dimension_dk():
    rng = np.random.default_rng(26)
    X = rng.normal(size=(1, 4, 3))
    W_Q = rng.normal(size=(3, 5))
    W_K = rng.normal(size=(3, 5))
    W_V = rng.normal(size=(3, 5))
    _, cache = attention_forward(X, W_Q, W_K, W_V, scale=True)
    np.testing.assert_allclose(cache["S"], cache["Q"] @ np.swapaxes(cache["K"], -1, -2) / np.sqrt(5.0))


def test_manual_gradient_matches_finite_difference():
    rng = np.random.default_rng(29)
    X = rng.normal(size=(1, 3, 2))
    W_Q = rng.normal(size=(2, 3))
    W_K = rng.normal(size=(2, 3))
    W_V = rng.normal(size=(2, 3))
    dY = rng.normal(size=(1, 3, 3))
    _, cache = attention_forward(X, W_Q, W_K, W_V, scale=True)
    analytic = attention_backward(dY, cache, W_Q, W_K, W_V)

    def scalar_loss(x, q, k, v):
        y, _ = attention_forward(x, q, k, v, scale=True)
        return float(np.sum(y * dY))

    eps = 1e-5
    for name, arr, analytic_grad, index in [
        ("X", X, analytic["dX"], (0, 1, 0)),
        ("W_Q", W_Q, analytic["dW_Q"], (1, 2)),
        ("W_K", W_K, analytic["dW_K"], (0, 1)),
        ("W_V", W_V, analytic["dW_V"], (1, 2)),
    ]:
        plus = arr.copy(); minus = arr.copy()
        plus[index] += eps; minus[index] -= eps
        if name == "X": lp = scalar_loss(plus, W_Q, W_K, W_V); lm = scalar_loss(minus, W_Q, W_K, W_V)
        elif name == "W_Q": lp = scalar_loss(X, plus, W_K, W_V); lm = scalar_loss(X, minus, W_K, W_V)
        elif name == "W_K": lp = scalar_loss(X, W_Q, plus, W_V); lm = scalar_loss(X, W_Q, minus, W_V)
        else: lp = scalar_loss(X, W_Q, W_K, plus); lm = scalar_loss(X, W_Q, W_K, minus)
        numerical = (lp - lm) / (2 * eps)
        assert abs(float(analytic_grad[index]) - numerical) < 1e-5, name


def test_data_generation_is_deterministic():
    a = generate_warehouse_data(n_hours=100, seed=42)
    b = generate_warehouse_data(n_hours=100, seed=42)
    np.testing.assert_allclose(a["demand"].values, b["demand"].values)

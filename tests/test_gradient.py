import numpy as np
import pytest

from src.manual_attention import attention_forward, attention_backward, numerical_gradient


def test_manual_attention_gradient_matches_finite_difference():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(1, 3, 2))
    W_Q = rng.normal(size=(2, 3))
    W_K = rng.normal(size=(2, 3))
    W_V = rng.normal(size=(2, 3))
    dY = rng.normal(size=(1, 3, 3))

    _, cache = attention_forward(X, W_Q, W_K, W_V)
    analytic = attention_backward(dY, cache, W_Q, W_K, W_V)

    def loss(x):
        y, _ = attention_forward(x, W_Q, W_K, W_V)
        return float(np.sum(y * dY))

    numerical = numerical_gradient(loss, X.copy(), eps=1e-5)
    np.testing.assert_allclose(analytic["dX"], numerical, rtol=1e-5, atol=1e-6)


def test_tensorflow_gradient_check_when_tensorflow_is_installed():
    tf = pytest.importorskip("tensorflow")
    from src.warehouse_model import WarehouseAttentionModel

    tf.random.set_seed(42)
    model = WarehouseAttentionModel(d_model=8, hidden_dim=4, scale=True)
    x = tf.Variable(tf.random.normal((2, 6)), dtype=tf.float32)

    with tf.GradientTape() as tape:
        prediction, _ = model(x, training=False)
        loss = tf.reduce_mean(tf.square(prediction))

    grad = tape.gradient(loss, x)
    assert grad is not None
    assert np.isfinite(grad.numpy()).all()

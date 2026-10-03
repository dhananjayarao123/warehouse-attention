import numpy as np

from src.manual_attention import attention_forward, softmax


def test_attention_shapes():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(4, 24, 3))
    wq = rng.normal(size=(3, 5))
    wk = rng.normal(size=(3, 5))
    wv = rng.normal(size=(3, 7))

    y, cache = attention_forward(x, wq, wk, wv)

    assert cache["Q"].shape == (4, 24, 5)
    assert cache["K"].shape == (4, 24, 5)
    assert cache["V"].shape == (4, 24, 7)
    assert cache["S"].shape == (4, 24, 24)
    assert cache["A"].shape == (4, 24, 24)
    assert y.shape == (4, 24, 7)


def test_softmax_rows_sum_to_one():
    rng = np.random.default_rng(43)
    x = rng.normal(size=(3, 8, 8))
    result = softmax(x, axis=-1)
    sums = np.sum(result, axis=-1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-12)


def test_attention_output_is_finite():
    rng = np.random.default_rng(44)
    x = rng.normal(size=(2, 24, 3))
    wq = rng.normal(size=(3, 6))
    wk = rng.normal(size=(3, 6))
    wv = rng.normal(size=(3, 6))
    y, cache = attention_forward(x, wq, wk, wv)
    assert np.isfinite(y).all()
    assert np.isfinite(cache["A"]).all()

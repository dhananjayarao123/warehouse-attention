import numpy as np

from src.data_generator import generate_warehouse_data
from src.dataset import create_sequences


def test_data_generation_is_deterministic():
    a = generate_warehouse_data(
        n_hours=100,
        seed=42,
    )
    b = generate_warehouse_data(
        n_hours=100,
        seed=42,
    )

    assert np.allclose(
        a["demand"].values,
        b["demand"].values,
    )


def test_sequence_shapes():
    demand = np.arange(100, dtype=np.float32)

    X, y = create_sequences(
        demand,
        sequence_length=24,
    )

    assert X.shape == (76, 24)
    assert y.shape == (76,)


def test_sequence_target_is_next_hour():
    demand = np.arange(30, dtype=np.float32)

    X, y = create_sequences(
        demand,
        sequence_length=24,
    )

    assert np.allclose(X[0], np.arange(24))
    assert y[0] == 24

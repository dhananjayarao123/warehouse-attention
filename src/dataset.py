from dataclasses import dataclass
import numpy as np


@dataclass
class PreparedData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    mean: float
    std: float


def create_sequences(demand: np.ndarray, sequence_length: int = 24):
    demand = np.asarray(demand, dtype=np.float32)
    X, y = [], []

    for i in range(len(demand) - sequence_length):
        X.append(demand[i:i + sequence_length])
        y.append(demand[i + sequence_length])

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def chronological_split(X, y, train_ratio=0.70, val_ratio=0.15):
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return (
        X[:train_end], y[:train_end],
        X[train_end:val_end], y[train_end:val_end],
        X[val_end:], y[val_end:],
    )


def prepare_data(demand, sequence_length=24):
    X, y = create_sequences(demand, sequence_length)
    X_train, y_train, X_val, y_val, X_test, y_test = chronological_split(X, y)

    mean = float(X_train.mean())
    std = max(float(X_train.std()), 1e-8)

    return PreparedData(
        ((X_train - mean) / std).astype(np.float32),
        ((y_train - mean) / std).astype(np.float32),
        ((X_val - mean) / std).astype(np.float32),
        ((y_val - mean) / std).astype(np.float32),
        ((X_test - mean) / std).astype(np.float32),
        ((y_test - mean) / std).astype(np.float32),
        mean,
        std,
    )


def make_dataset(X, y, batch_size=128, shuffle=False, seed=42):
    import tensorflow as tf
    ds = tf.data.Dataset.from_tensor_slices(
        (
            tf.convert_to_tensor(X, dtype=tf.float32),
            tf.convert_to_tensor(y, dtype=tf.float32),
        )
    )
    if shuffle:
        ds = ds.shuffle(
            buffer_size=len(X),
            seed=seed,
            reshuffle_each_iteration=True,
        )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

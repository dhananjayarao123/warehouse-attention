import numpy as np
import pandas as pd
from .config import DataConfig


def generate_warehouse_data(
    n_hours: int = 24 * 365,
    seed: int = 42,
    noise_std: float = 10.0,
    spike_probability: float = 0.01,
    spike_low: float = 50.0,
    spike_high: float = 120.0,
) -> pd.DataFrame:
    """Generate a reproducible synthetic warehouse demand time series."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_hours)

    base = 100.0
    trend = 0.01 * t

    # Daily and weekly recurring structure.
    daily = 30.0 * np.sin(2.0 * np.pi * t / 24.0)
    weekly = 15.0 * np.sin(2.0 * np.pi * t / (24.0 * 7.0))

    noise = rng.normal(0.0, noise_std, n_hours)

    spikes = np.zeros(n_hours)
    mask = rng.random(n_hours) < spike_probability
    spikes[mask] = rng.uniform(spike_low, spike_high, mask.sum())

    demand = np.maximum(base + trend + daily + weekly + spikes + noise, 0.0)

    timestamps = pd.date_range(
        start="2025-01-01",
        periods=n_hours,
        freq="h",
    )

    return pd.DataFrame({"timestamp": timestamps, "demand": demand})


def generate_from_config(config: DataConfig) -> pd.DataFrame:
    return generate_warehouse_data(
        n_hours=config.n_hours,
        seed=config.seed,
        noise_std=config.noise_std,
        spike_probability=config.spike_probability,
        spike_low=config.spike_low,
        spike_high=config.spike_high,
    )


if __name__ == "__main__":
    df = generate_warehouse_data()
    print(df.head())
    print(df["demand"].describe())

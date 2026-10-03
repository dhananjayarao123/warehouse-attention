from dataclasses import dataclass


@dataclass
class DataConfig:
    n_hours: int = 24 * 365
    seed: int = 42
    noise_std: float = 10.0
    spike_probability: float = 0.01
    spike_low: float = 50.0
    spike_high: float = 120.0


@dataclass
class ModelConfig:
    d_model: int = 32
    hidden_dim: int = 16
    sequence_length: int = 24


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 1e-3
    seed: int = 42

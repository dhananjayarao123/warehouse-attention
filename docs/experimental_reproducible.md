# Section 22 — Experimental Reproducibility

## Environment

Recommended environment:

- Python 3.12.x
- TensorFlow 2.20.0
- NumPy 1.26+ / 2.x compatible with the installed TensorFlow build
- Pandas 2.2+
- Matplotlib 3.8+
- Pytest 8+

Install with:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Record the exact environment before submission:

```bash
python --version
python -c "import tensorflow as tf; print(tf.__version__)"
python -c "import numpy as np; print(np.__version__)"
python -c "import pandas as pd; print(pd.__version__)"
pip freeze > requirements-lock.txt
```

## Seeds

Primary seed: `42`.

- NumPy: `np.random.seed(42)` where legacy RNG is used.
- Generator: `np.random.default_rng(42)`.
- TensorFlow: `tf.random.set_seed(42)`.
- Derived seeds are explicitly documented for independent toy validation/test data and shifted evaluation data.

## Dataset

`src/data_generator.py` is the single source for the warehouse environment. It generates:

- base demand = 100;
- linear trend = `0.01 * t`;
- daily sinusoid with period 24;
- weekly sinusoid with period 168;
- Gaussian additive noise;
- rare positive spikes;
- non-negative clipping.

Default configuration:

| Parameter | Value |
|---|---:|
| Hours | 8760 |
| Seed | 42 |
| Noise std | 10 |
| Spike probability | 0.01 |
| Spike low | 50 |
| Spike high | 120 |

## Forecasting protocol

The model receives the previous 24 observations and predicts the next hour.

```text
X[t] = demand[t:t+24]
y[t] = demand[t+24]
```

The complete sequence dataset is split chronologically:

```text
70% train | 15% validation | 15% test
```

Normalization statistics are fitted only on the training input window. The same mean/std are applied to validation and test data.

For distribution shift, the shifted data is also normalized using the **original training mean/std**, never its own statistics.

## Model

- Input: 24 scalar observations.
- Linear input embedding: 32 dimensions.
- Single-head self-attention.
- \(Q=XW_Q\), \(K=XW_K\), \(V=XW_V\).
- Score scaling by \(\sqrt{d_k}\).
- Numerically stable softmax.
- Mean aggregation over the 24 output positions.
- ReLU hidden layer: 16 units.
- Scalar prediction head.

## Training

Canonical warehouse experiments use:

- Adam;
- learning rate `1e-3`;
- batch size `128`;
- 15 epochs;
- MSE training loss.

The toy experiment uses its own documented configuration in `experiments/toy_experiment.py`.

## Primary commands

```bash
PYTHONPATH=. pytest -q
python experiments/toy_experiment.py
python experiments/gradient_comparison.py
python experiments/warehouse_experiment.py
python experiments/baseline_comparison.py
python experiments/ablation.py
python experiments/training_dynamics.py
python experiments/distribution_shift.py
python experiments/failure_analysis.py
```

Or:

```bash
python run_all.py
```

## Result locations

Canonical regenerated results are written below `results/`:

```text
results/warehouse/
results/ablation/
results/generalization/
results/figures/
```

The supplied archive also retains the result artifacts that were present in the uploaded project snapshot for audit/history. Regenerate the canonical results after installing dependencies before treating numerical values as final submission evidence.

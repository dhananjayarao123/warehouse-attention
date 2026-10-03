# From First Principles: Warehouse Demand Attention Model

A small, reproducible mathematical ML investigation implementing scaled dot-product self-attention from first principles and applying the mechanism to next-hour synthetic warehouse-demand forecasting.

## What is implemented

1. **First-principles attention:** NumPy implementation of
   \(Q=XW_Q\), \(K=XW_K\), \(V=XW_V\), \(QK^T/\sqrt{d_k}\), stable softmax, and \(AV\).
2. **Manual reverse-mode backpropagation** plus finite-difference verification.
3. **Toy sequence-learning task** with a known dependency structure.
4. **Warehouse environment** with hourly data, daily/weekly seasonality, trend, Gaussian noise, and rare spikes.
5. **Warehouse attention model** using the same attention equations.
6. **Last-observation and moving-average baselines.**
7. **Training-dynamics analysis** using loss, gradient norm, entropy, and attention concentration.
8. **Controlled scaled vs unscaled ablation.**
9. **Distribution-shift evaluation** without retraining and without refitting normalization.
10. **Genuine failure selection** using the largest absolute prediction error.
11. **Tests** targeting mathematical invariants rather than only code coverage.
12. **Optional LoRA extension** kept separate from the required core.

## Core constraint

The core attention implementation does not use `tf.keras.layers.MultiHeadAttention`, Transformer encoder layers, or any high-level attention library. The authoritative mathematical implementation is `src/manual_attention.py`.

## Repository layout

```text
warehouse_attention/
├── README.md
├── requirements.txt
├── pytest.ini
├── run_all.py
├── ai_assistance_log.md
├── docs/
│   ├── phase0.md
│   ├── mathematics.md
│   ├── numerical_stability.md
│   ├── training_dynamics.md
│   ├── ablation.md
│   ├── problem2_warehouse.md
│   ├── evaluation_method.md
│   ├── failure_analysis.md
│   ├── failure_debugging.md
│   ├── crossproblem_integration.md
│   ├── reproducible.md
│   ├── testing.md
│   ├── demo_instructions.md
│   ├── requirements_traceability.md
│   └── reflect_intellect_honesty.md
├── src/
│   ├── manual_attention.py
│   ├── data_generator.py
│   ├── dataset.py
│   ├── warehouse_model.py
│   ├── toy_model.py
│   ├── baseline.py
│   ├── evaluation.py
│   ├── train.py
│   └── lora.py
├── experiments/
│   ├── toy_experiment.py
│   ├── gradient_comparison.py
│   ├── warehouse_experiment.py
│   ├── baseline_comparison.py
│   ├── ablation.py
│   ├── training_dynamics.py
│   ├── distribution_shift.py
│   ├── failure_analysis.py
│   └── debugging_gradient.py
├── tests/
└── results/
```

## Setup

Python 3.12 is recommended for the declared TensorFlow environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Check the environment:

```bash
python --version
python -c "import tensorflow as tf; print(tf.__version__)"
```

## Tests

```bash
PYTHONPATH=. pytest -q
```

The mathematical NumPy tests run without TensorFlow. TensorFlow-specific tests use `pytest.importorskip`, so an environment without TensorFlow reports those checks as skipped instead of failing test collection.

## Experiments

```bash
python experiments/toy_experiment.py
python experiments/gradient_comparison.py
python experiments/warehouse_experiment.py
python experiments/baseline_comparison.py
python experiments/ablation.py
python experiments/training_dynamics.py
python experiments/distribution_shift.py
python experiments/failure_analysis.py
```

Or run the primary sequence:

```bash
python run_all.py
```

## Reproducibility decisions

- Seed: `42`.
- Lookback: 24 hours.
- Chronological split: 70% / 15% / 15%.
- Normalization: training-set statistics only.
- Warehouse attention: single-head scaled dot-product attention.
- Optimizer: Adam, learning rate `1e-3`.
- Metrics: MAE, MSE, RMSE; MAPE as an additional diagnostic.

See `docs/reproducible.md` for the complete protocol.

## Interpreting attention

Attention weights are treated as model diagnostics/visualizations. They are not automatically interpreted as causal explanations.

## Important submission note

The archive includes experiment artifacts from the supplied project snapshot. The canonical scripts above regenerate the results under `results/`. After installing dependencies, run `python run_all.py` before a final submission if you want every numerical artifact regenerated from the corrected code.

## Requirement traceability

`docs/requirements_traceability.md` maps the assessment requirements to the exact source, experiment, test, and documentation files.

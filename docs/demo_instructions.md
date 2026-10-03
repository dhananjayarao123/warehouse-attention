# Reproducible Live Demo

Run from the repository root after installing `requirements.txt`.

## 0. Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1. First-principles attention

```bash
python -c "import numpy as np; from src.manual_attention import attention_forward; X=np.random.default_rng(42).normal(size=(1,4,3)); W=np.random.default_rng(43); Y,c=attention_forward(X,W.normal(size=(3,5)),W.normal(size=(3,5)),W.normal(size=(3,5))); print('Q',c['Q'].shape,'scores',c['S'].shape,'A',c['A'].shape,'Y',Y.shape); print('row sums',c['A'].sum(axis=-1))"
```

Show `src/manual_attention.py` and trace `Q → K/V → QKᵀ → scaling → stable softmax → AV`.

## 2. Gradient verification

```bash
python experiments/gradient_comparison.py
```

Also run:

```bash
PYTHONPATH=. pytest -q
```

The NumPy finite-difference tests do not require TensorFlow. TensorFlow-specific tests are skipped if TensorFlow is not installed.

## 3. Toy learning

```bash
python experiments/toy_experiment.py
```

Show training loss/MAE and the generated attention diagnostics.

## 4. Warehouse prediction and baseline

```bash
python experiments/warehouse_experiment.py
python experiments/baseline_comparison.py
```

Show `results/warehouse/baseline_comparison.csv` and explain why a simple baseline is a required reference.

## 5. Ablation

```bash
python experiments/ablation.py
```

Show `results/ablation/scaled_vs_unscaled.csv` and the two training-dynamics CSVs. Explain the hypothesis before discussing the observed result.

## 6. Distribution shift

```bash
python experiments/distribution_shift.py
```

Show `results/generalization/distribution_shift_metrics.csv`. Emphasize that shifted data uses the **original training mean/std** for normalization.

## 7. Failure

```bash
python experiments/failure_analysis.py
```

Show the largest-error case and `docs/failure_analysis.md`. Explain the evidence and avoid treating attention weights as causal proof.

## 8. Full rerun

```bash
python run_all.py
```

This runs the test suite and the primary experiments in a fixed order.

## Demo narrative

The intended story is:

**formulate → implement → verify → learn → ablate → forecast → compare → shift → investigate failure → reflect.**

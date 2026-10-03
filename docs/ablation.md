# docs/ablation.md

# Section 14 - Required Ablation

## 1. Hypothesis

The ablation compares scaled and unscaled dot-product attention.

### Hypothesis

Scaled attention:

\[
S=\frac{QK^T}{\sqrt{d_k}}
\]

is expected to produce more stable attention behavior than:

\[
S=QK^T
\]

Removing the scaling factor is expected to increase the magnitude of
attention logits.

Larger logits can produce a more concentrated softmax distribution,
lower attention entropy, greater saturation, and different gradient
behavior.

---

# 2. Experimental Setup

Two conditions are evaluated.

## Condition A: Scaled Attention

\[
S=\frac{QK^T}{\sqrt{d_k}}
\]

## Condition B: Unscaled Attention

\[
S=QK^T
\]

Everything else is held constant.

The two experiments use:

- identical synthetic warehouse data
- identical train/validation/test split
- identical sequence length
- identical model dimension
- identical optimizer
- identical learning rate
- identical batch size
- identical number of epochs
- identical random seed

This makes the attention scaling factor the controlled experimental
variable.

---

# 3. Observed Behavior

The experiment measures:

### Training behavior

- training loss
- validation loss
- gradient norm

### Attention behavior

- attention entropy
- maximum attention weight
- fraction of weights greater than `0.95`

### Numerical behavior

- mean absolute attention logit
- maximum absolute attention logit

### Prediction performance

- MSE
- MAE
- RMSE

The results are generated automatically after running:

```bash
PYTHONPATH=. python experiments/ablation.py
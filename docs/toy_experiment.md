# Section 12 - Problem 1: Toy Learning Task

## 1. Task

A small synthetic sequence prediction task is used to demonstrate that the
self-attention mechanism participates in an actual learning process.

The model receives a sequence of 12 scalar observations and predicts one
continuous target.

The task is intentionally small and does not require a large model.

---

## 2. Synthetic Learning Objective

For each sequence:

\[
X = [x_1,x_2,\ldots,x_{12}]
\]

the target is generated as:

\[
y =
0.5x_{12}
+
0.3x_{10}
+
0.2x_7
+
\epsilon
\]

where:

\[
\epsilon \sim N(0,\sigma^2)
\]

with:

\[
\sigma=0.05
\]

Therefore, the target depends on multiple positions in the input sequence.

The relevant positions are:

- last position: `-1`
- third-last position: `-3`
- sixth-last position: `-6`

The remaining positions provide distractor information.

---

## 3. Why Attention Is Relevant

The model must process information from multiple positions in the sequence.

The attention mechanism computes:

\[
Q=XW_Q
\]

\[
K=XW_K
\]

\[
V=XW_V
\]

and:

\[
A =
softmax
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)
\]

followed by:

\[
C=AV
\]

The resulting context representation is used to produce the prediction.

The matrices:

\[
W_Q,\;W_K,\;W_V
\]

are trainable parameters.

They receive gradients from the prediction loss.

Therefore the attention mechanism participates directly in the optimization process.

---

## 4. Loss

Mean Squared Error is used during training:

\[
MSE =
\frac{1}{N}
\sum_{i=1}^{N}
(y_i-\hat y_i)^2
\]

---

## 5. Evaluation Metrics

The following metrics are reported on the held-out test set.

### MSE

\[
MSE =
\frac{1}{N}
\sum_i(y_i-\hat y_i)^2
\]

### MAE

\[
MAE =
\frac{1}{N}
\sum_i|y_i-\hat y_i|
\]

### RMSE

\[
RMSE=\sqrt{MSE}
\]

---

## 6. Model Configuration

| Parameter | Value |
|---|---:|
| Sequence length | 12 |
| Input dimension | 1 |
| Attention dimension | 16 |
| Hidden units | 16 |
| Training samples | 4000 |
| Validation samples | 1000 |
| Test samples | 1000 |
| Batch size | 64 |
| Epochs | 30 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Loss | MSE |
| Attention | Scaled dot-product self-attention |
| Random seed | 42 |

---

## 7. Training Behavior

The experiment records:

- training loss
- validation loss
- training MAE
- validation MAE

These values are stored in:

```text
results/toy_training_history.csv
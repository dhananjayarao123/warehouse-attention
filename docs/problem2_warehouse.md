# Section 15 - Problem 2: Warehouse Demand Prediction

## 1. Objective

The objective is to predict the next hour of warehouse demand using the
synthetic warehouse dataset created in Phase 0.

The model uses:

Previous 24 hourly observations

↓

Attention mechanism

↓

Context representation

↓

Prediction head

↓

Next-hour demand

---

## 2. Input

For each prediction:

\[
X_t =
[d_{t-23},d_{t-22},...,d_{t}]
\]

The sequence contains exactly 24 hourly observations.

Therefore:

\[
X_t \in \mathbb{R}^{24}
\]

Each scalar hourly demand value is projected into a learned representation:

\[
X' \in \mathbb{R}^{24\times d_{model}}
\]

---

## 3. Attention

The attention implementation follows the first-principles equations used in
Problem 1.

\[
Q=X'W_Q
\]

\[
K=X'W_K
\]

\[
V=X'W_V
\]

The attention scores are:

\[
S=
\frac{QK^T}{\sqrt{d_k}}
\]

The attention matrix is:

\[
A=softmax(S)
\]

The attended sequence is:

\[
C=AV
\]

No high-level Transformer or MultiHeadAttention layer is used.

---

## 4. Context Representation

The attended representations from the 24 positions are aggregated:

\[
c=
\frac{1}{24}
\sum_{i=1}^{24}C_i
\]

This produces a single context vector for the prediction head.

---

## 5. Prediction Head

The context vector is passed through:

1. Dense hidden layer
2. ReLU activation
3. Linear output layer

The final scalar is the predicted next-hour demand:

\[
\hat d_{t+1}=f(c)
\]

---

## 6. Training

The model is trained using:

- Adam optimizer
- Mean Squared Error loss
- chronological train/validation/test split
- fixed random seed

The attention parameters receive gradients through the prediction loss.

---

## 7. Evaluation

The following metrics are reported:

### MAE

\[
MAE =
\frac{1}{N}
\sum_i
|y_i-\hat y_i|
\]

### MSE

\[
MSE =
\frac{1}{N}
\sum_i
(y_i-\hat y_i)^2
\]

### RMSE

\[
RMSE=\sqrt{MSE}
\]

---

## 8. Baselines

Two simple baselines are evaluated on the same test set.

### Last-value baseline

\[
\hat d_{t+1}=d_t
\]

### Moving-average baseline

\[
\hat d_{t+1}
=
\frac{1}{24}
\sum_{i=0}^{23}d_{t-i}
\]

The attention model is compared against both baselines.

---

## 9. Outputs

The experiment produces:

```text
results/
├── metrics/
│   ├── generated_warehouse.csv
│   ├── warehouse_model_comparison.csv
│   ├── warehouse_predictions.csv
│   ├── warehouse_training_history.csv
│   ├── warehouse_attention_statistics.json
│   └── warehouse_experiment_config.json
│
└── figures/
    ├── warehouse_training.png
    ├── warehouse_predictions.png
    └── warehouse_attention_heatmap.png
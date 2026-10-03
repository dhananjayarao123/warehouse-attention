# Engineering Decisions

## 1. Programming Language

The project uses Python.

Python was selected because it provides strong numerical and machine
learning tooling while remaining suitable for implementing the attention
mathematics directly.

NumPy is used for the first-principles mathematical implementation.

TensorFlow/Keras is used for model training and independent gradient
verification.

---

## 2. Framework Choice

The project uses:

- NumPy
- TensorFlow/Keras
- Pandas
- Matplotlib
- Pytest

The core attention mechanism is not implemented using a high-level
Transformer or attention API.

The first-principles implementation explicitly computes:

Q = XW_Q

K = XW_K

V = XW_V

S = QK^T / sqrt(d_k)

A = softmax(S)

Y = AV

---

## 3. Dataset Design

A synthetic warehouse-demand environment was chosen instead of an
external dataset.

This allows the experiment to control the statistical properties of
the environment.

The generated demand contains temporal structure such as:

- hourly observations;
- daily seasonality;
- weekly seasonality;
- random noise;
- unusual demand behavior;
- distribution-shift conditions.

The generator uses a fixed random seed when reproducibility is required.

---

## 4. Forecasting Problem

The primary task is:

> Predict the next hour of warehouse demand from the previous 24 hours.

Therefore the input window is:

[t-23, ..., t]

and the target is:

y(t+1)

A 24-hour lookback was selected because it provides one complete daily
cycle while keeping the model small.

---

## 5. Model Architecture

The model intentionally uses a small architecture.

Conceptually:

Historical demand
        ↓
Input representation
        ↓
Q/K/V projections
        ↓
Scaled dot-product attention
        ↓
Context aggregation
        ↓
Prediction head
        ↓
Next-hour demand

A large Transformer was deliberately not used.

This makes the mathematical mechanism easier to inspect and defend.

---

## 6. Attention Design

The project implements single-head scaled dot-product self-attention.

The attention equations are:

Q = XW_Q

K = XW_K

V = XW_V

S = QK^T / sqrt(d_k)

A = softmax(S)

Y = AV

The scaling factor is explicitly implemented because its mathematical
effect is investigated in the ablation experiment.

---

## 7. Baseline Decisions

Two simple forecasting baselines are used.

### Last observation

y_hat(t+1) = y(t)

### Moving average

y_hat(t+1) =
mean(y(t-23), ..., y(t))

These baselines provide simple reference points.

The attention model is not assumed to outperform them.

---

## 8. Data Splitting

The dataset is divided chronologically.

Conceptually:

Training → Validation → Test

Randomly shuffling the time series before splitting was avoided because
future observations could otherwise influence training.

Normalization statistics are calculated using training data only.

---

## 9. Evaluation Metrics

The primary metrics include:

- MAE
- MSE
- RMSE
- sMAPE where appropriate

MAE provides an interpretable average prediction error.

MSE and RMSE emphasize larger errors and are useful when investigating
demand spikes and failure cases.

---

## 10. Ablation Decision

The primary ablation compares:

Scaled attention:

S = QK^T / sqrt(d_k)

against:

Unscaled attention:

S = QK^T

The purpose is to investigate the mathematical effect of attention
scaling rather than simply changing the architecture.

---

## 11. Generalization Decision

The generalization experiment trains on one synthetic distribution and
evaluates on a shifted distribution without retraining.

The shift changes characteristics such as:

- noise standard deviation;
- spike frequency;
- spike magnitude.

The current canonical generator does not include explicit drop or operational-event terms; those are optional guidance in the assessment and are intentionally not part of this controlled environment.

This tests whether the learned model remains effective when the
data-generating process changes.

---

## 12. Failure Analysis Decision

The failure investigation identifies a concrete prediction with a large
absolute error.

For each prediction:

e_i = y_i - y_hat_i

The investigated case is selected using:

i* = argmax |e_i|

The corresponding historical input window is then examined.

This avoids manually selecting a convenient example.

---

## 13. Testing Decision

Tests focus on mathematical invariants rather than only code coverage.

Examples include:

- tensor shape correctness;
- softmax normalization;
- attention output dimensions;
- finite outputs;
- numerical gradient verification;
- deterministic data generation;
- correct use of d_k in attention scaling.

---

## 14. UI and Deployment

No production UI or deployment system was implemented.

This was an intentional scope decision.

The assessment focuses on:

Mathematical formulation
→ Implementation
→ Verification
→ Experimentation
→ Analysis

rather than production deployment.

---

## 15. Optional LoRA Extension

LoRA was implemented as an optional parameter-efficient adaptation
experiment.

It is not required for the core attention investigation.

The LoRA experiment adapts Q/K/V projections using low-rank updates while
keeping the base weights frozen.

It is kept separate from the core pipeline so that the main experiment
remains easy to understand.

---

## 16. Overall Engineering Principle

The project prioritizes:

1. mathematical correctness;
2. reproducibility;
3. controlled experimentation;
4. failure investigation;
5. explainability.

Additional complexity was avoided when it did not directly contribute
to these goals.
# Parameter-Efficient Adaptation — LoRA

## Objective

Implement a minimal parameter-efficient adaptation method on the
small warehouse attention model.

The project uses Low-Rank Adaptation (LoRA).

---

## Mathematical Formulation

For a frozen weight matrix W, LoRA learns a low-rank update:

\[
W' = W + \Delta W
\]

where:

\[
\Delta W =
\frac{\alpha}{r}AB
\]

and:

- \(W\) is frozen;
- \(A\) is trainable;
- \(B\) is trainable;
- \(r\) is the LoRA rank;
- \(\alpha\) controls the scaling.

For an input X:

\[
Y =
XW +
X\frac{\alpha}{r}AB
\]

---

## Application to Attention

LoRA is applied to the Q, K and V projections.

Instead of:

\[
Q=XW_Q
\]

the adapted projection is:

\[
Q=X(W_Q+\Delta W_Q)
\]

where:

\[
\Delta W_Q =
\frac{\alpha}{r}A_QB_Q
\]

Similarly:

\[
K=X(W_K+\Delta W_K)
\]

and:

\[
V=X(W_V+\Delta W_V)
\]

The attention calculation itself remains unchanged:

\[
S =
\frac{QK^T}{\sqrt{d_k}}
\]

\[
A=\operatorname{softmax}(S)
\]

\[
Y=AV
\]

Therefore LoRA changes the parameterization of the projections,
not the fundamental attention mathematics.

---

## Configuration

The experiment uses:

```text
d_model = 32
LoRA rank = 2
alpha = 4
# Section 11 — Numerical Stability

The attention implementation uses a stable softmax transformation.

Instead of directly evaluating

\[
\frac{e^{s_i}}{\sum_j e^{s_j}},
\]

it computes

\[
m=\max_j s_j,
\qquad
\frac{e^{s_i-m}}{\sum_j e^{s_j-m}}.
\]

Subtracting the row maximum does not change the mathematical probability distribution, because the same constant is removed from every logit. It prevents large positive logits from overflowing the exponential.

The project also investigates softmax saturation through the attention entropy and fraction of weights above 0.95 in the scaled/unscaled ablation.

The scaling factor

\[
1/\sqrt{d_k}
\]

addresses a related issue earlier in the computation: dot products tend to grow in magnitude as the query/key dimension increases. Large logits can make softmax nearly one-hot, which can change gradient behavior.

The implementation is intentionally educational rather than a claim of production-level numerical optimization.

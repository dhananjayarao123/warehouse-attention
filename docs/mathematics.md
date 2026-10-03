# Mathematical Derivation

## Input

Let X be a sequence representation:

X in R^(B x T x d)

where:
- B = batch size
- T = sequence length
- d = embedding dimension

For this project, T = 24.

## Query, Key, Value

Q = X W_Q

K = X W_K

V = X W_V

Each projection produces a representation in R^(B x T x d).

## Similarity

S = Q K^T

For each sequence item, this produces a similarity score against every sequence item.

S has shape:

(B, T, T)

## Scaling

S' = S / sqrt(d_k)

The scale reduces the magnitude of dot products when the representation dimension grows.

## Softmax

A_ij = exp(S'_ij) / sum_j exp(S'_ij)

Each row of A sums to approximately 1.

The implementation uses the numerically stable equivalent:

exp(x - max(x))

before normalization.

## Weighted aggregation

Y = A V

Y has shape:

(B, T, d)

Each output position is a weighted combination of value vectors.

## Forecasting

The attention output is mean-pooled over the 24 positions:

C = mean(Y, dimension=T)

A small prediction head maps C to the next-hour demand.

## Important interpretation

Attention weights are treated as a visualization of learned weighting, not as proof of causal influence.


## TensorFlow implementation

The implementation uses `tf.matmul`, `tf.exp`, `tf.reduce_sum`, and `tf.GradientTape`; it does not use a high-level attention layer.

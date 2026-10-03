## Objective

The test suite focuses on mathematical correctness and reproducibility rather
than code coverage alone. The tests are designed to catch subtle errors in
attention tensor dimensions, softmax normalization, scaling, gradient
propagation, and synthetic-data reproducibility.

## Required Tests

### 1. Attention tensor shapes

For an input

\[
X \in R^{B \times T \times D}
\]

and projected dimension \(d_k\), the expected tensors are:

```text
Q       : B × T × d_k
K       : B × T × d_k
V       : B × T × d_v
QK^T    : B × T × T
A       : B × T × T
Y       : B × T × d_v
```

The test suite verifies these dimensions directly.

### 2. Softmax normalization

For every attention row:

\[
\sum_j A_{ij}=1
\]

and:

\[
A_{ij}\geq0
\]

The tests also check that the result contains no NaN or infinity.

### 3. Attention output dimensions

Because:

\[
Y=AV
\]

where:

\[
A\in R^{T\times T},\qquad V\in R^{T\times d_v},
\]

we require:

\[
Y\in R^{T\times d_v}.
\]

A dedicated test verifies this independently of the query/key dimension.

### 4. Correct scaling dimension

The scaling term must be:

\[
\frac{QK^T}{\sqrt{d_k}}
\]

where \(d_k\) is the projected query/key dimension.

A subtle regression test deliberately uses:

```text
input dimension D = 3
action/query-key dimension d_k = 5
```

and verifies that the implementation divides the raw scores by
\(\sqrt{5}\), not \(\sqrt{3}\).

This test is important because both values can be equal in simple examples,
allowing an incorrect implementation to appear correct.

### 5. Manual gradient verification

The manually implemented backward pass is checked against finite differences.
For a scalar objective \(L\):

\[
\frac{\partial L}{\partial\theta}
\approx
\frac{L(\theta+\epsilon)-L(\theta-\epsilon)}{2\epsilon}
\]

Representative gradients are checked for:

```text
X
W_Q
W_K
W_V
```

The manual input gradient is also compared with an independent TensorFlow
GradientTape calculation using the same mathematical equations.

### 6. Deterministic dataset generation

The same generator configuration and seed must produce identical timestamps
and demand values.

The test also verifies that changing the seed changes the stochastic demand
sequence, ensuring that the seed is actually controlling the random process.

## Test Files

```text
tests/
├── test_section23.py
└── test_data.py
```

## Run the Test Suite

From the project root:

```bash
PYTHONPATH=. pytest -q
```

For more detail:

```bash
PYTHONPATH=. pytest -vv
```

Run only Section 23:

```bash
PYTHONPATH=. pytest -q tests/test_section23.py tests/test_data.py
```

## What These Tests Protect Against

| Mathematical/engineering issue | Test |
|---|---|
| Wrong Q/K/V dimensions | `test_attention_tensor_shapes` |
| Wrong score matrix dimensions | `test_attention_tensor_shapes` |
| Softmax rows not summing to 1 | `test_softmax_normalization` |
| Negative/NaN attention weights | `test_softmax_normalization` |
| Incorrect `AV` output dimension | `test_attention_output_dimensions` |
| Scaling by input dimension instead of \(d_k\) | `test_scaling_uses_projected_dimension_dk` |
| Non-finite attention output | `test_attention_output_is_finite` |
| Incorrect manual backward equations | `test_manual_gradient_matches_finite_difference` |
| Incorrect TensorFlow/manual gradient agreement | `test_manual_gradient_matches_tensorflow` |
| Non-reproducible synthetic data | `test_data_generation_is_deterministic` |
| Seed ignored by generator | `test_data_generation_changes_with_seed` |
| Incorrect sequence/target alignment | `test_sequence_target_is_next_hour` |

## Testing Philosophy

The goal is not to maximize the number of tests. The goal is to test invariants
that would reveal mathematical implementation errors.

The most important invariants are:

\[
Q=XW_Q
\]

\[
K=XW_K
\]

\[
V=XW_V
\]

\[
S=\frac{QK^T}{\sqrt{d_k}}
\]

\[
A=softmax(S)
\]

\[
\sum_jA_{ij}=1
\]

and:

\[
Y=AV.
\]

Gradient verification then tests whether the implemented computational graph
has the expected derivatives.

This provides stronger evidence of mathematical correctness than tests that
only check whether functions execute without exceptions.
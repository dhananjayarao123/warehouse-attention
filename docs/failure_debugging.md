# Section 24 — Failure Handling and Debugging

## 24.1 Purpose

This section documents one complete debugging process rather than only
showing the final corrected implementation.

The debugging process follows:

Initial implementation
↓
Unexpected mathematical / gradient result
↓
Hypothesis about bug
↓
Diagnostic experiment
↓
Correction
↓
Verification

The investigated issue is the scaling factor in scaled dot-product
self-attention.

---

## 24.2 Initial Implementation

The intended attention equations are:

Q = XW_Q

K = XW_K

V = XW_V

S = QK^T / sqrt(d_k)

A = softmax(S)

Y = AV

where d_k is the dimensionality of the projected query/key vectors.

The initial implementation incorrectly used the input dimension:

S_bug = QK^T / sqrt(d_input)

instead of:

S_correct = QK^T / sqrt(d_k)

---

## 24.3 Unexpected Result

The issue can remain hidden when:

input_dim == d_k

because:

sqrt(input_dim) == sqrt(d_k)

Therefore, a test using equal dimensions is insufficient to detect
this mathematical bug.

The debugging experiment deliberately uses:

input_dim = 3
d_k = 5

so that:

sqrt(3) != sqrt(5)

---

## 24.4 Hypothesis

The hypothesis was:

> The attention scaling factor is using the wrong dimension.

The correct dimension is the projected query/key dimension d_k.

---

## 24.5 Diagnostic Experiment

The experiment compares:

Buggy:

raw_scores / sqrt(input_dim)

Correct:

raw_scores / sqrt(d_k)

The diagnostic also verifies the corrected implementation using
central finite differences:

dL/dθ ≈ [L(θ + ε) - L(θ - ε)] / (2ε)

The following gradients are checked:

- X
- W_Q
- W_K
- W_V

---

## 24.6 Correction

The correction is:

```python
scale_factor = np.sqrt(W_Q.shape[-1])
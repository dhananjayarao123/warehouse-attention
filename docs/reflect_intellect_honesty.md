# Section 26 — Reflection and Intellectual Honesty

## 26.1 Purpose

This reflection records what was expected before the experiments, what
was learned during implementation, which assumptions were challenged,
and which questions remain unresolved.

The goal is not to present the project as more successful or more
certain than the evidence supports.

---

# 1. What did I initially expect?

I initially expected scaled dot-product self-attention to learn useful
relationships between different positions in a historical demand
window.

For the toy sequence problem, I expected the attention mechanism to
learn that some historical positions are more relevant to the target
than others.

For the warehouse forecasting problem, I expected the previous
24 hours to contain useful information about the next-hour demand.

I also expected scaled attention to behave more stably than unscaled
attention because the factor

\[
\frac{1}{\sqrt{d_k}}
\]

controls the magnitude of the attention logits before softmax.

Finally, I expected the attention model to provide useful information
beyond simple forecasting baselines, although I did not assume in
advance that it would outperform every baseline.

---

# 2. Which hypotheses were correct?

## Hypothesis 1 — Attention can learn relationships between sequence positions

The toy problem was designed specifically to test whether the
attention mechanism could learn position-dependent relationships.

The target was constructed from several historical positions:

\[
y =
0.5x_{-1}
+
0.3x_{-3}
+
0.2x_{-6}
+
\epsilon
\]

This gives the experiment a known dependency structure.

The experiment therefore provides a controlled way to investigate
whether the model can learn relationships among sequence positions.

The final numerical result should be reported from the generated
experiment output rather than claimed without running the experiment.

---

## Hypothesis 2 — Scaling should affect attention stability

The ablation hypothesis was:

> Scaled dot-product attention should be more stable than unscaled
> dot-product attention.

The mathematical reasoning for this hypothesis is clear.

Scaled attention uses:

\[
S =
\frac{QK^T}{\sqrt{d_k}}
\]

while the ablation removes the denominator:

\[
S =
QK^T
\]

Larger logits can make softmax more concentrated, which can reduce
attention entropy and affect gradient behavior.

The hypothesis should therefore be evaluated using the actual ablation
results rather than assuming the hypothesis is correct merely because
the mathematical argument is plausible.

---

# 3. Which hypotheses were wrong?

A hypothesis should only be labelled correct or wrong after examining
the corresponding experiment.

The project deliberately avoids claiming that the attention model
must outperform the simple baselines.

If the last-value or moving-average baseline performs better on the
warehouse test set, that is an important result rather than something
to hide.

Similarly, if scaled attention does not outperform unscaled attention
on a particular metric, the observed result should be reported
directly and then interpreted using the actual experimental evidence.

Therefore, any hypothesis whose numerical result has not yet been
verified remains **unresolved**, rather than being declared correct
or wrong.

This is intentional intellectual honesty.

---

# 4. What surprised me?

One of the most important surprises was how easily a mathematically
subtle implementation error can remain hidden.

The attention scaling bug used:

```python
sqrt(input_dim)
```

instead of:

```python
sqrt(d_k)
```

The bug can remain invisible when:

```text
input_dim == d_k
```

because both expressions then have the same value.

The debugging experiment deliberately used:

```text
input_dim = 3
d_k = 5
```

which exposed the difference.

This showed that a test can appear successful while still failing to
test the underlying mathematical requirement.

Another important observation was that attention weights should not
automatically be interpreted as causal explanations.

They show how the implemented attention mechanism weights internal
representations, but they do not by themselves establish that a
particular historical event caused the prediction.

---

# 5. What was the hardest implementation problem?

The hardest implementation problem was maintaining mathematical
consistency across the complete attention computation.

It was not sufficient to implement only:

```text
Q
K
V
softmax
output
```

The same mathematical definitions had to remain consistent across:

- forward propagation;
- manual backward propagation;
- finite-difference verification;
- TensorFlow gradient verification;
- scaled/unscaled ablation;
- unit tests.

The scaling factor was particularly difficult because an apparently
reasonable implementation could still be mathematically wrong.

The manual backward pass was another difficult part because an error
in one derivative can produce incorrect gradients even when the
forward prediction looks reasonable.

---

# 6. What failure did I investigate?

The main implementation failure investigated was the incorrect
attention scaling dimension.

The initial implementation used:

```python
raw_scores / sqrt(input_dim)
```

instead of:

```python
raw_scores / sqrt(d_k)
```

The debugging process was:

```text
Initial implementation
        ↓
Unexpected mathematical / gradient behavior
        ↓
Hypothesis: wrong scaling dimension
        ↓
Diagnostic experiment
        ↓
input_dim != d_k
        ↓
Correction
        ↓
Finite-difference verification
```

The corrected implementation uses:

```python
scale_factor = np.sqrt(W_Q.shape[-1])
```

which corresponds to \(d_k\).

The correction was then checked against numerical finite differences.

---

# 7. What do I now understand better?

I now understand several aspects of attention more concretely.

### 7.1 Attention is a mathematical computation

The mechanism is not a black box:

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
A=\operatorname{softmax}(S)
\]

\[
Y=AV
\]

Each operation affects the next operation.

---

### 7.2 Shape correctness is necessary but not sufficient

A program can have correct tensor shapes and still implement the wrong
mathematics.

The scaling-dimension bug is an example.

Therefore, testing must include mathematical invariants and numerical
verification, not only shape checks.

---

### 7.3 Gradient verification is valuable

The manual backward pass can be checked independently using numerical
finite differences.

This provides evidence that the implementation of the derivatives
matches the forward computation.

---

### 7.4 Synthetic data can be useful for controlled experiments

Synthetic warehouse data makes it possible to control:

- seasonality;
- noise;
- spikes;
- unusual drops;
- operational events;
- distribution shifts.

This makes it possible to investigate specific model behaviors without
depending on an uncontrolled external dataset.

---

### 7.5 Baselines are important

A more complicated model is not automatically a better forecasting
model.

Comparing against:

```text
last observation
moving average
```

provides a reference point for interpreting the attention model.

---

# 8. What remains uncertain?

Several questions remain unresolved unless the corresponding final
experiments are actually executed and inspected.

### 8.1 Whether attention consistently beats simple baselines

The model may or may not outperform the last-value and moving-average
baselines.

The final numerical test-set metrics must determine this.

---

### 8.2 Whether scaling produces a large practical improvement

The mathematical reason for scaling is well understood, but the size
of its practical benefit depends on the chosen model dimensions,
dataset, initialization, and training setup.

The ablation results should therefore be reported rather than assumed.

---

### 8.3 How robust the model is under distribution shift

The distribution-shift experiment changes demand disturbances.

The exact degradation must be measured from the experiment outputs.

A good result on the original distribution does not establish
robustness to the shifted distribution.

---

### 8.4 Whether attention weights provide useful forecasting explanations

Attention weights can be inspected as model diagnostics.

However, they should not automatically be interpreted as causal
explanations.

A stronger interpretability investigation would be required to make
strong causal claims.

---

# 9. Where did AI assistance help?

AI assistance was useful for:

- proposing project structure;
- explaining mathematical formulations;
- generating initial code scaffolding;
- suggesting controlled experiments;
- suggesting baseline models;
- designing test cases;
- identifying possible debugging strategies;
- organizing documentation.

AI assistance was particularly useful for turning the requirements
document into concrete implementation tasks.

It also helped identify useful mathematical tests such as finite-
difference gradient verification.

---

# 10. Where did I need to correct or reject AI-generated suggestions?

AI-generated suggestions were not accepted automatically.

One important correction involved the attention scaling dimension.

The implementation needed to use:

```python
sqrt(d_k)
```

rather than:

```python
sqrt(input_dim)
```

The project therefore added a regression test where:

```text
input_dim != d_k
```

so that this mistake could not pass accidentally.

Another important design decision was to keep
`src/manual_attention.py` as the authoritative first-principles
attention implementation instead of introducing an unrelated duplicate
attention implementation.

This kept the mathematical implementation centralized and easier to
verify.

The candidate also retained responsibility for checking whether
suggested experiments actually matched the assessment requirements.

---

# 11. If given one additional day, what would I investigate?

With one additional day, I would focus on three areas.

## 11.1 Complete the controlled ablation analysis

I would run the scaled-versus-unscaled experiment multiple times with
different random seeds and compare:

- test MAE;
- test RMSE;
- attention entropy;
- maximum attention weight;
- gradient norms;
- score magnitude.

This would distinguish a consistent effect from a single-run effect.

---

## 11.2 Investigate distribution-shift failures in more detail

I would examine several of the largest errors instead of only one.

I would categorize failures according to:

- spikes;
- demand drops;
- operational events;
- unusually noisy periods;
- combinations of these conditions.

This could reveal whether the failure is systematic or isolated.

---

## 11.3 Investigate alternative aggregation

The current warehouse model aggregates the attended sequence
representation before prediction.

I would investigate whether using the representation of the most
recent position, rather than a simple mean aggregation, changes the
forecasting behavior.

This would be a controlled architectural experiment rather than an
unrestricted model change.

---

# 12. Intellectual Honesty Statement

The project does not assume that a sophisticated attention model must
produce the best forecasting result.

The main objective was to understand and verify the mathematics,
construct a controlled experiment, compare against simple baselines,
investigate distribution shift, and document failures.

Where an experimental result has not been executed or verified, it is
treated as unresolved rather than reported as a fact.

Where a debugging issue was identified, the correction is accompanied
by an independent verification procedure.

This distinction between:

```text
expected behavior
observed behavior
and unresolved behavior
```

is an important part of the final interpretation of the project.
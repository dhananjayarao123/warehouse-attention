# AI Assistance Log

AI assistance was used as a development aid. The candidate remained responsible for understanding, modifying, testing, and defending the final implementation.

## Entry 1 — Repository design

**Tool used:** ChatGPT.

**Task requested:** Turn the assessment requirements into a small modular repository.

**Generated output:** Suggested `src/`, `experiments/`, `tests/`, `docs/`, and `results/` separation.

**Candidate modifications:** Removed unnecessary production infrastructure and kept the project focused on mathematical correctness, experiments, and evidence.

**Verification performed:** Compared the structure against the submission and demo requirements.

**Resulting understanding:** Repository organization should make the Phase 0 → implementation → experiment → analysis chain obvious.

## Entry 2 — Attention mathematics

**Task requested:** Explain and implement Q/K/V, scaling, softmax, and AV.

**Generated output:** Mathematical equations and shape reasoning.

**Candidate modifications:** Implemented the authoritative NumPy version in `src/manual_attention.py` and explicitly exposed every operation.

**Verification performed:** Shape tests and softmax normalization tests.

**Resulting understanding:** The first-principles requirement is about exposing the computation rather than calling a Transformer abstraction.

## Entry 3 — Manual backpropagation

**Task requested:** Derive the reverse pass through `Y=AV`, softmax, and `QK^T/sqrt(d_k)`.

**Generated output:** Reverse-mode equations for `dX`, `dW_Q`, `dW_K`, and `dW_V`.

**Candidate modifications:** Implemented the backward pass in NumPy and retained it as optional bonus work rather than making it the only verification path.

**Verification performed:** Central finite differences on X and representative projection entries.

**Resulting understanding:** A plausible forward result does not establish derivative correctness.

## Entry 4 — Toy learning task

**Task requested:** Design a small sequence problem where attention has a known dependency structure.

**Generated output:** Target based on `0.5*x[-1] + 0.3*x[-3] + 0.2*x[-6] + noise`.

**Candidate modifications:** Implemented the task in `experiments/toy_experiment.py` with a small TensorFlow model.

**Verification performed:** Training loss, validation loss, MAE, and attention diagnostics are recorded.

**Resulting understanding:** A toy problem can test whether the mechanism participates in learning without requiring a large model.

## Entry 5 — Warehouse data generation

**Task requested:** Design a reproducible synthetic warehouse process.

**Generated output:** Daily/weekly seasonality, trend, noise, and rare events.

**Candidate modifications:** The final generator deliberately uses daily and weekly sinusoidal structure, a small trend, Gaussian noise, and positive spikes. Drops/events were not added because they were guidance rather than a mandatory schema.

**Verification performed:** Fixed-seed determinism tests and descriptive inspection of generated data.

**Resulting understanding:** Synthetic data is useful only when its statistical assumptions are explicit.

## Entry 6 — Baselines

**Task requested:** Compare attention with simple forecasting references.

**Generated output:** Last-observation and moving-average baselines.

**Candidate modifications:** Canonical baseline evaluation now converts predictions back to original demand units before computing metrics.

**Verification performed:** Confirmed all methods use the same chronological test windows.

**Resulting understanding:** A complex model is not useful merely because it is more sophisticated.

## Entry 7 — Scaled/unscaled ablation

**Task requested:** Design a controlled attention-scaling ablation.

**Generated output:** Compare scaled `QK^T/sqrt(d_k)` against unscaled `QK^T` while controlling all other variables.

**Candidate modifications:** Rewrote the experiment to reuse `WarehouseAttentionModel`; the scaling flag is the only model-level change.

**Verification performed:** Recorded loss, gradient norm, entropy, maximum attention, and saturation.

**Resulting understanding:** The hypothesis concerns numerical/training behavior; it does not guarantee lower forecasting error.

## Entry 8 — Distribution shift

**Task requested:** Create a generalization experiment with changed noise/event statistics.

**Generated output:** Higher noise and larger/more frequent spikes.

**Candidate modifications:** Corrected preprocessing so shifted data is normalized using the original training mean/std rather than refitting statistics on the shifted distribution.

**Verification performed:** Documented the original and shifted configurations and wrote degradation metrics.

**Resulting understanding:** Distribution-shift experiments can be invalid if preprocessing is accidentally adapted to the evaluation distribution.

## Entry 9 — Failure investigation

**Task requested:** Select a genuine failure rather than inventing one.

**Generated output:** Select the largest absolute error from the shifted evaluation set and inspect its 24-hour input window.

**Candidate modifications:** Added baseline comparison, input statistics, attention diagnostics, and evidence fields.

**Verification performed:** Failure selection is deterministic from the generated predictions.

**Resulting understanding:** Failure analysis should explain evidence and uncertainty rather than assign causal meaning to attention weights automatically.

## Entry 10 — Debugging

**Task requested:** Construct a documented mathematical debugging example.

**Generated output:** A scaling-dimension bug scenario using input dimension different from projected `d_k`.

**Candidate modifications:** Added `experiments/debugging_gradient.py` and regression tests using `D=3`, `d_k=5`.

**Verification performed:** Checked the corrected scaling against the expected `sqrt(5)` factor and gradient finite differences.

**Resulting understanding:** A targeted regression test can protect against a subtle mathematical error that shape-only tests miss.

## Entry 11 — Test/review pass

**Task requested:** Audit the repository against the assessment.

**Generated output:** Identified missing demo detail, duplicate APIs, inconsistent imports, and stale experiment interfaces.

**Candidate modifications:** Consolidated the authoritative NumPy attention implementation, fixed imports, corrected the distribution-shift protocol, rewrote the ablation/baseline runners, and added a requirement traceability document.

**Verification performed:** Python compilation, AST import-reference audit, and `pytest`.

**Resulting understanding:** AI-generated code must be treated as a proposal. The final repository is responsible for its own internal consistency and mathematical evidence.

## Candidate responsibility

The candidate must rerun the TensorFlow experiments in the final environment and inspect the generated numerical results before submitting them as final evidence. The repository intentionally avoids claiming experimental outcomes that were not regenerated after the final code corrections.

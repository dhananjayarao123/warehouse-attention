# Requirements Traceability

This table maps the assessment requirements to concrete repository artifacts.

| Requirement | Evidence in repository |
|---|---|
| Mathematical formulation | `docs/phase0.md`, `docs/mathematics.md` |
| Synthetic warehouse environment | `src/data_generator.py` |
| Statistical assumptions | `docs/phase0.md` |
| Two hypotheses before experiments | `docs/phase0.md`, `docs/ablation.md`, `docs/toy_experiment.md` |
| Three system-specific failure modes | `docs/phase0.md` |
| First-principles attention | `src/manual_attention.py` |
| Q/K/V, scaling, softmax, AV derivation | `docs/mathematics.md` |
| Gradient verification | `src/manual_attention.py`, `tests/test_gradient.py`, `experiments/gradient_comparison.py` |
| Manual backprop bonus | `src/manual_attention.py` |
| Numerical stability | `docs/numerical_stability.md`, `experiments/softmax_saturation.py` |
| Toy learning | `src/toy_model.py`, `experiments/toy_experiment.py` |
| Training dynamics | `src/train.py`, `experiments/training_dynamics.py`, `docs/training_dynamics.md` |
| Controlled scaled/unscaled ablation | `experiments/ablation.py`, `docs/ablation.md` |
| Warehouse forecasting | `src/warehouse_model.py`, `experiments/warehouse_experiment.py` |
| Simple baselines | `src/baseline.py`, `experiments/baseline_comparison.py` |
| Quantitative evaluation | `src/evaluation.py`, `results/` |
| Leakage-safe chronological split | `src/dataset.py`, `docs/reproducible.md` |
| Distribution shift | `experiments/distribution_shift.py`, `results/generalization/` |
| Genuine failure investigation | `experiments/failure_analysis.py`, `docs/failure_analysis.md` |
| Cross-problem integration | `docs/crossproblem_integration.md`, `experiments/crossproblem_integration.py` |
| Reproducibility | `docs/reproducible.md`, `run_all.py` |
| Basic tests | `tests/` |
| Debugging process | `experiments/debugging_gradient.py`, `docs/failure_debugging.md` |
| AI assistance log | `ai_assistance_log.md` |
| Reflection | `docs/reflect_intellect_honesty.md` |
| Demo instructions | `docs/demo_instructions.md` |
| Optional LoRA | `src/lora.py`, `experiments/lora_adaptation.py`, `docs/lora_adaptation.md` |

## Scope decisions

The project deliberately does not implement a full Transformer, multi-head attention, production deployment, GPU optimization, or distributed training. These are outside the mandatory scope. LoRA is retained only as an optional extension.

## Important implementation principle

The NumPy implementation in `src/manual_attention.py` is the authoritative first-principles mathematical reference. The TensorFlow warehouse/toy layers reproduce the same equations using low-level tensor operations so that the mechanism can participate in gradient-based learning. No high-level Transformer or attention API is used.

# Historical experiment artifacts

This directory contains the experiment-result files that were present in the uploaded project archive before the consistency corrections.

They are retained for audit/history only. They are **not** the canonical final numerical results because the experiment scripts were corrected after the snapshot was created.

For final evidence, install the declared dependencies and regenerate the canonical outputs with:

```bash
python run_all.py
```

Canonical output directories are:

```text
results/warehouse/
results/ablation/
results/generalization/
results/figures/
```

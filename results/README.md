# Experiment results

## Verified artifacts in this archive

- `verification/gradient_check.json` — NumPy manual-backprop finite-difference check generated from the corrected source.
- `data/warehouse_sample_30_days.csv` and `data/generator_summary.json` — deterministic generator audit artifacts.
- `toy_*` — toy experiment artifacts retained from the supplied project snapshot.

## Canonical final experiment outputs

After installing the dependencies, regenerate the primary numerical evidence:

```bash
python run_all.py
```

This creates:

```text
results/warehouse/
results/ablation/
results/generalization/
results/figures/
```

The historical snapshot of the uploaded archive's prior experiment outputs is kept separately under `results/archive_snapshot/` so those numbers are not accidentally presented as results from the corrected scripts.

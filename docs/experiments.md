# Experiments

## Experiment 1 — Toy learning

Purpose:
Verify that the first-principles attention mechanism participates in an actual learning problem.

Record:
- training loss
- validation loss
- convergence behavior

## Experiment 2 — Gradient verification

Compare TensorFlow GradientTape gradients with finite-difference gradients for representative parameters.

Report:
- analytical gradient
- numerical gradient
- absolute difference
- relative difference

## Experiment 3 — Scaled vs unscaled attention

Compare:

softmax(QK^T / sqrt(d_k))

against:

softmax(QK^T)

Keep the dataset, seed, optimizer, learning rate, and training duration controlled.

## Experiment 4 — Warehouse forecasting

Compare:
- last-value baseline
- moving-average baseline
- attention model

Metrics:
- MAE
- MSE
- RMSE

## Experiment 5 — Distribution shift

Train under the original distribution and test under higher noise and larger/more frequent spikes.

Compare performance degradation.

## Experiment 6 — Failure analysis

Locate the test example with the largest absolute error and investigate:
- historical input
- expected output
- actual output
- attention behavior
- plausible explanation
- potential improvement

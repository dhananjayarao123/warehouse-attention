# Phase 0 — Design Before Code

## 1. Mathematical problem formulation

For each time index t, the input is the previous 24 hourly observations:

\[
X_t=[D_{t-24},D_{t-23},...,D_{t-1}]
\]

and the target is the next hour:

\[
y_t=D_t.
\]

The forecasting model learns \(\hat y_t=f(X_t;\theta)\). Evaluation is chronological so that observations from the future are never used to train preprocessing or model parameters.

## 2. Synthetic warehouse environment

The generator in `src/data_generator.py` creates one deterministic hourly demand process:

\[
D_t=100+0.01t+30\sin(2\pi t/24)+15\sin(2\pi t/168)+\epsilon_t+S_t
\]

where:

- \(\epsilon_t\sim N(0,\sigma^2)\), with configurable `noise_std`;
- \(S_t\) is a rare positive spike drawn uniformly between `spike_low` and `spike_high`;
- demand is clipped at zero;
- timestamps are hourly and reproducible from the seed.

The assessment explicitly allows the exact synthetic schema to be candidate-designed. This implementation uses daily and weekly seasonality, trend, Gaussian noise, and rare spikes. Explicit drop events and operational events are not part of the current generator; the distribution-shift experiment instead changes noise and spike statistics.

## 3. Statistical assumptions

### Temporal structure

- Daily periodicity: 24 hours.
- Weekly periodicity: 168 hours.
- A small deterministic trend is present.
- Nearby observations are correlated because seasonality is smooth and the target is adjacent to the input window.

### Noise

- Additive Gaussian noise.
- Constant variance within a configured regime.
- `noise_std` controls its magnitude.

### Unusual events

Rare positive spikes are generated independently at each hour with configurable probability. Their magnitude is sampled from a bounded uniform interval. The model is therefore expected to perform less reliably on rare extreme values than on ordinary seasonal observations.

### Informative history

The most recent 24 hours contain daily-cycle information. The weekly component also makes positions separated by roughly 168 hours informative across windows, although a 24-hour-only input cannot observe a full weekly cycle directly.

## 4. Pre-experiment hypotheses

### H1 — Attention can exploit unequal temporal relevance

Because different historical positions can contain different information about the next hour, learned attention weights should not be forced to be uniform. The toy task tests this directly with a known sparse dependency structure.

### H2 — Scaling improves numerical/training stability

Scaled attention

\[
A=\operatorname{softmax}(QK^T/\sqrt{d_k})
\]

should generally have less extreme logits than unscaled attention. Therefore the scaled variant is expected to show less softmax saturation and more stable gradient behavior, although a lower loss is not guaranteed.

## 5. System-specific failure modes

1. **Extreme spike underprediction:** the next-hour value may exceed the range represented by the previous 24 hours.
2. **Distribution-shift degradation:** higher noise and more/larger spikes can increase forecast error without retraining.
3. **Unhelpful attention concentration:** the model may place weight on a historical position that is correlated with the target in the training distribution but is not useful for a particular case.

## 6. Evaluation methodology

- Lookback: 24 hours.
- Chronological split: 70% train / 15% validation / 15% test.
- Normalization: training-set mean and standard deviation only.
- Primary metrics: MAE, MSE, RMSE; MAPE is retained as an additional diagnostic.
- Baselines use exactly the same raw-unit test targets and test windows as the attention model.
- No future observations are used to fit preprocessing.

## 7. Expected baseline behavior

The last-observation baseline

\[
\hat D_t=D_{t-1}
\]

should be strong when adjacent demand values are smooth. A 24-hour moving average should suppress noise but can lag sudden changes. The attention model is useful only if its learned representation adds predictive information beyond these simple references.

## 8. Planned generalization experiment

Train on the original distribution:

- `noise_std=10`
- spike probability `0.01`
- spike range `50–120`

Evaluate without retraining on:

- `noise_std=20`
- spike probability `0.03`
- spike range `100–200`

Crucially, shifted inputs are normalized using the original training mean/std. The shifted distribution is not allowed to change preprocessing statistics.

## 9. Verification plan

Before interpreting model results:

1. Verify tensor shapes.
2. Verify softmax rows sum to one.
3. Verify scaling uses the projected \(d_k\), not input dimension.
4. Verify manual gradients against finite differences.
5. Compare manual gradients against TensorFlow independently in `experiments/gradient_comparison.py`.
6. Run controlled scaled/unscaled ablation.
7. Select a genuine failure using the largest absolute test error rather than inventing a scenario.

## 10. Reproducibility

Primary seed: `42`. Dataset generation is deterministic for a fixed seed. Commands and exact experiment configuration are documented in `docs/reproducible.md` and `docs/demo_instructions.md`.

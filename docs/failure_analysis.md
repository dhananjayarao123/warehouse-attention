# Section 20 — Failure Investigation

## Objective

The project must contain at least one **genuine failure case**. Rather than
manually choosing a visually interesting example, the experiment searches the
entire shifted evaluation set and selects the example with the largest absolute
prediction error.

This connects Section 19 (distribution shift) to Section 20 (failure analysis).

## Scenario

The attention model is trained on the original synthetic warehouse
 distribution:

- noise standard deviation = 10;
- spike probability = 0.01;
- spike magnitude = 50–120 additive demand units.

The trained model is then evaluated **without retraining** on a shifted
distribution:

- noise standard deviation = 20;
- spike probability = 0.03;
- spike magnitude = 100–200 additive demand units.

The failure case is the shifted-test example with the largest absolute
prediction error.

Therefore, the failure is data-driven and reproducible rather than a manually
invented example.

## Input — What produced the failure?

The selected example contains the previous 24 hourly demand observations:

```text
results/failure_analysis/failure_input_window.csv
```

The exact numerical values are generated when the experiment is run and are
also stored in:

```text
results/failure_analysis/section20_failure_report.json
```

The experiment also records the window mean, standard deviation, minimum and
maximum so that the unusual nature of the input can be inspected quantitatively.

## Expected behavior

The expected behavior is that the model should use the historical sequence to
predict the next-hour demand reasonably closely, retaining useful daily and
weekly temporal structure even though the test distribution has changed.

The last-observation baseline is also calculated for the exact same failure
case. This provides a simple reference without changing the failure definition.

## Actual behavior

The experiment records:

- actual next-hour demand;
- predicted next-hour demand;
- absolute error;
- relative error;
- last-observation baseline prediction;
- last-observation baseline error;
- whether the selected target is unusually large relative to its history.

The generated summary is:

```text
results/failure_analysis/section20_failure_summary.csv
```

Do not insert numerical values into the final written report before running the
experiment. The experiment generates the actual observed values.

## Explanation

The primary explanation tested by this experiment is **distribution shift**.

The model minimizes prediction error on examples drawn from the original
training distribution. Section 19 deliberately changes the noise and event
statistics. Consequently, the shifted test distribution can contain more
extreme observations than the model encountered during training.

The model can therefore retain useful seasonal behavior while still making a
large error on an unusually extreme next-hour demand value.

This should be stated as an evidence-supported interpretation rather than as a
causal claim about a particular attention weight.

## Evidence

The experiment records several independent pieces of evidence:

1. **Failure magnitude** — absolute and squared prediction error.
2. **Error distribution** — mean, median and 95th-percentile absolute error.
3. **Tail ratio** — failure error divided by the 95th-percentile error.
4. **Input statistics** — mean, standard deviation, minimum and maximum of the
   24-hour window.
5. **Distribution-shift parameters** — original and shifted noise/spike
   settings.
6. **Baseline comparison** — last-observation prediction on exactly the same
   case.
7. **Attention diagnostics** — attention entropy and key-position weights when
   available.

The attention diagnostics are used as model diagnostics only. Attention weights
should not automatically be interpreted as causal explanations.

## Potential improvement

Several technically plausible improvements can be investigated:

### 1. Train on multiple regimes

Include representative high-noise and high-spike regimes in the training data.
This should expose the model to a wider range of conditional patterns.

### 2. Add operational features

If operational event information is available before prediction time, include
an event indicator or event severity as an input feature.

### 3. Robust objective

A robust loss can reduce the dominance of extreme residuals during training.
Alternatively, conditional-variance modeling can explicitly represent changing
uncertainty.

### 4. Stress-test validation

Include a separate stress-test validation set when selecting model
hyperparameters, rather than validating only on ordinary-distribution data.

These are **potential** improvements. They are not claimed to fix the failure
until a follow-up experiment demonstrates that they do.


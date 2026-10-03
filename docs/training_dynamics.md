# Section 13 — Training Dynamics Analysis

The project records more than final forecasting error. During training it records:

- training and validation MSE;
- global gradient norm;
- mean attention entropy;
- maximum attention weight.

The mathematical chain investigated is:

\[
|QK^T|\uparrow
\rightarrow
\text{softmax concentration}\uparrow
\rightarrow
H(A)\downarrow
\rightarrow
\text{some local gradients may become small}.
\]

The controlled ablation in `experiments/ablation.py` compares scaled and unscaled attention while keeping the dataset, seed, architecture, optimizer, learning rate, batch size, and number of epochs fixed.

A result is not interpreted as evidence of instability merely because a curve looks irregular. The analysis should compare the recorded logit/entropy/saturation/gradient statistics and connect them to the softmax equations.

Generated artifacts:

```text
results/ablation/scaled_training_dynamics.csv
results/ablation/unscaled_training_dynamics.csv
results/ablation/scaled_vs_unscaled.csv
```

If the scaled and unscaled variants have similar final loss, that is also a valid observation: scaling can improve numerical behavior without guaranteeing a large forecasting improvement on this small synthetic task.

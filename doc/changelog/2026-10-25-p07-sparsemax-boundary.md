# P07 sparsemax support-boundary derivative

The bounded operator model now uses ReLU for the final positive-part operation in
sparsemax. It returns the same finite forward projection as `clamp_min(0)), while
selecting zero derivative for coordinates exactly at the threshold, consistently
with the strictly positive support used to compute that threshold.

For scores `[1, 0, -2, -3, -4, -5]`, the previous composition could backpropagate
`[-1, 1, 0, 0, 0, 0]` through the constant sum of the probabilities. The regression
checks this simplex identity and the positive-support Jacobian at singleton and
two-active-coordinate boundaries in float32 and float64. It runs in the existing
CPU Torch 2.6 job alongside the bounded model tests.

The forward model, operators, CE/residual objectives, optimizer settings, sample
selection, thresholds and budgets are unchanged. This software correction does
not establish the cause of P07's archived six-window fitting failure: its saved
singleton supports may lie strictly inside a constant region. No industrial
training, HPO or target evaluation is performed by this change.

References: [sparsemax §2.5, Eq. 12 and footnote 2](https://proceedings.mlr.press/v48/martins16.pdf);
[PyTorch 2.6 derivative definitions](https://github.com/pytorch/pytorch/blob/v2.6.0/tools/autograd/derivatives.yaml).

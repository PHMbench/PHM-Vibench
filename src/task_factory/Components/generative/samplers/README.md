# Generative Samplers

Samplers convert trained generative models into synthetic PHM signal windows.
They must stay stateless across probability-flow time steps unless a later goal
explicitly proves a stateful method is valid.

## V0 Sampling Contract

- input/output tensors use `[N, C, L]`
- condition keys are `fault_label` and `domain_id`
- sampler time is generative probability-flow time, not physical sequence time
- each ODE/SDE update checks finite values, shape, dtype, and device
- failures are explicit; no hidden CPU fallback for paper GPU runs

The Euler ODE sampler is compatible with stateless velocity-field models such
as CFM and Rectified Flow. DDPM and Score SDE require their own scheduler and
reverse-process evidence before benchmark-valid promotion.

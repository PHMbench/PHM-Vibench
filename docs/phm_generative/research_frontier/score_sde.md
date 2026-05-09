# Score SDE

Status: research-only.

Score SDE enters PHM-Vibench as a shape-checked skeleton, not a benchmark
baseline. The expected score prediction and denoising target shapes are
`[N, C, L]`; condition keys remain `fault_label` and `domain_id`.

Future sampler requirements include a reviewed predictor-corrector or ODE
sampler, finite drift/diffusion guards, manifest fields for continuous-time
settings, and leakage checks before any benchmark-valid output.

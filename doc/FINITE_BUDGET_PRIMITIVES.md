# Common-intervention numerical primitives

`phmfactory.finite_budget` supplies small functions consumed by the P02 industrial
representation study and P20 general finite-budget study. It does not import either
paper, load their data, choose a checkpoint, run a training workflow or claim a
real-data baseline.

- `orthogonal_basis`: identity, real orthonormal DCT, complete Haar packet levels.
- `truncate`: deterministic global scalar-coordinate budget.
- `exact_sparse_risk`: exhaustive second-moment oracle, dimension at most 16.
- `sparse_response_fit`: greedy least squares using calibration queries only.
- `coefficients`: signed gradient, path-average gradient and central secant.

The dense basis implementation is bounded to windows of at most 4096 samples.
It is a reference primitive, not an optimized FFT/wavelet backend. The matrix acts
on time within each channel. Channels are never silently flattened or reordered.

The input model is already loaded and in evaluation mode. Targets and path
baselines are explicit. Outputs are sensitivities, not interchangeable with IG
contributions, occlusion effects or Grad-CAM heatmaps. `path_gradient` returns the
mean IG integrand; it does not claim to be vanilla IG. No division by a baseline
coordinate is used. Evaluation interventions are supplied by the consumer and
must not be the calibration queries used to fit an explanation.

```bash
python -m pytest test/test_finite_budget.py -q
```

The tests establish numerical semantics on analytic functions. They do not
establish an industrial diagnosis, explanation or forecast baseline. The module
has no new registry, CLI, optional backend or fallback path.

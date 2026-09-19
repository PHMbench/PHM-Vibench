# Physical frontend component check

`src/data_factory/tii_physical.py::project_window` consumes one explicitly selected
`[L,C]` raw window. It converts the configured channel by the supplied physical
unit scale, resamples the fixed physical interval to the frozen grid, and projects
onto disjoint `[low, high)` Hz sets. Outputs `common` and `incremental` have shape
`[K,P]` and retain physical units; `increment_available` is a scalar boolean tensor.
No RMS, mean, per-window normalization, band selection, or recording/window
selection is fitted in this function.

The projector uses the periodic Fourier basis of the entire physical window.
Grid conversion truncates frequencies that the output grid cannot represent and
handles the even-length Nyquist coefficient. Acquisition and effective rates are
both checked as necessary bounds on usable bands. Those bounds do not qualify a
device's passband. `support_basis` and the cited `support_evidence` must come from
qualification; the function does not authenticate documentary evidence.
`designed_filter` and `tensor_fixture` do not establish natural hardware support.

Executed from the child repository on 2026-09-19:

```bash
conda run -n LQ_signal --no-capture-output python -m unittest test.test_tii_physical -v
```

PASS: 4 explicit analytic tensor-fixture tests in 0.532 seconds. They check unit
conversion, channel selection, physical duration and grid geometry, orthogonal
common/increment projections and union reconstruction, projection idempotence,
different physical sampling rates, unavailable-increment value/gradient
isolation, and rejection of missing common support, partial/unknown increment,
Nyquist-only qualification, overlapping/empty bands, and unknown units.

NOT RUN in this component check: real-corpus qualification, native CLI, real
two-source shared micro-run, or scientific outcome estimation. No dataset is
qualified by these tensor results.

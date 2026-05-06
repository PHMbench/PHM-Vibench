# Generative Tasks

This package contains PHM generative benchmark tasks.

V0 supports Conditional Flow Matching for 1D vibration windows with explicit
conditions:

- `fault_label`
- `domain_id`

The training contract is intentionally separate from fault classification
tasks. Generative training optimizes velocity matching only; FFT and
distributional metrics are evaluation signals, not training losses.


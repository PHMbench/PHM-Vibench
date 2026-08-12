# Generative experiments

This directory contains **unpromoted** Pipeline 06 configurations. Files here
may be useful for config inspection and focused development, but they are not
part of the maintained demo or release-supported surface.

The public entrypoint remains:

```bash
python main.py --config <yaml> [--override key=value ...]
```

A generative configuration may move to `configs/demo/10_generative/` and receive
`status=sanity_ok` only after the exact method/model/data combination completes:

```text
CPU seed 0: train -> checkpoint -> sample -> synthetic manifest -> eval -> evidence manifest
GPU seed 0: train -> checkpoint -> sample -> synthetic manifest -> eval -> evidence manifest
```

Additional requirements include strict checkpoint loading, finite-value and
shape/device contracts, provenance hashes, leakage/duplicate checks, focused
and maintained tests, and green post-merge CI.

`sanity_ok` is functional smoke evidence. It is not benchmark-performance or
scientific-validity evidence.

Available CFM contracts:

- `dummy_generative_cfm.yaml` keeps the baseline velocity-MSE objective;
- `dummy_generative_cfm_population.yaml` adds a same-time population-correlation
  MMD regularizer.

The population variant is a PHM CFM adaptation, not a full PaD-TS or DDPM
implementation. Its population dependency metric is emitted only when the
regularizer is enabled and remains exploratory evidence.

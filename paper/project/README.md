# P02 Vibench execution overlay

This directory preserves the runnable source surface of the legacy P02
explainability toolkit inside its isolated PHM-Vibench paper branch. It is an
execution overlay, not the canonical manuscript or evidence ledger.

The authoritative paper state remains in
`AI4Engineering-L/P02-XFD-Benchmark-Toolkit`. Source provenance and filtering
are recorded in [SOURCE_MAP.yaml](SOURCE_MAP.yaml). The full legacy project
documentation is retained as [LEGACY_README.md](LEGACY_README.md).

## What is here

- `toolkit_integration/`: legacy interfaces, adapters, explainers, metrics,
  and report-generation code.
- `scripts/`, `demos/`, `examples/`, and `configs/`: runnable project
  entrypoints and fixtures.
- `schema/`: the legacy run and metric schema examples.
- `doc/`, `manuscript/`, and `submission_prep/`: source documentation
  retained for traceability.
- `configs/experiments/p02_xfd_benchmark_toolkit/` at repository root: the
  current engine-facing P02 configuration set.

Historical outputs, benchmark results, generated visualizations, checkpoints,
caches, archives, agent prompts, and session material are intentionally absent.

## Engine-facing checks

Run from the PHM-Vibench repository root:

```bash
python -m scripts.validate_configs
python -m scripts.config_inspect \
  --config configs/experiments/p02_xfd_benchmark_toolkit/p02_resnet1d_cwru.yaml
python -m scripts.validate_docs
```

The legacy toolkit is not automatically registered as a PHM-Vibench factory
component. These checks establish config and documentation validity only; they
do not establish benchmark comparability or reproduce legacy metrics.

The three `p02_resnet1d_*.yaml` files are engine configs. The
`p02_toolkit_benchmark.yaml` and `p02_toolkit_ablation.yaml` files are
planned paper-protocol records, not valid 5-block `main.py` configs.

## Evidence boundary

No legacy benchmark result is promoted by this copy. The source snapshot was
dirty primarily in generated result and figure paths, which were excluded.
Use the PaperTrace claim registry and run ledger before citing any number.

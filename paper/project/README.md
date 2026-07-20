# P01 Vibench execution overlay

This directory preserves the runnable source surface of the legacy P01 project
inside its isolated PHM-Vibench paper branch. It is an execution overlay, not
the canonical manuscript or evidence ledger.

The authoritative paper state remains in
`AI4Engineering-L/P01-UXFD-Multimodal-Alignment`. Source provenance and the
copy filter are recorded in [SOURCE_MAP.yaml](SOURCE_MAP.yaml). The legacy
project documentation is retained as [LEGACY_README.md](LEGACY_README.md).

## What is here

- `code/`, `model/`, and `explainers/`: legacy fusion and explanation code.
- `configs/`, `experiments/`, and `scripts/`: paper-local experiment
  configurations and runners.
- `doc/`, `manuscript/`, `paper_draft/`, and `submission_prep/`:
  source documentation retained for traceability.
- `configs/experiments/p01/` at the PHM-Vibench repository root: the current
  engine-facing P01 configuration set.

Historical outputs, result folders, checkpoints, weights, caches, generated
figures, PDFs, archives, agent prompts, and session material are intentionally
absent.

## Engine-facing checks

Run from the PHM-Vibench repository root:

```bash
python -m scripts.validate_configs
python -m scripts.config_inspect \
  --config configs/experiments/p01/p01_baseline_cwru_dlinear.yaml
python -m scripts.validate_docs
```

These checks validate configuration and documentation contracts. They do not
establish paper accuracy, explanation faithfulness, or submission readiness.
Real-data runs still require locally available datasets and an explicitly
audited split protocol.

## Evidence boundary

No legacy metric is promoted by this copy. The source snapshot was dirty and
included unverified results; generated evidence was excluded from this branch.
Use the PaperTrace claim registry and run ledger before citing any number.

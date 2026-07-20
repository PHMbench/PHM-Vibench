# P03 Vibench execution overlay

This directory preserves the runnable source surface of the legacy P03
evidence-grounded LLM toolkit inside its isolated PHM-Vibench paper branch. It
is an execution overlay, not the canonical manuscript or evidence ledger.

The authoritative paper state remains in
`AI4Engineering-L/P03-Evidence-Grounded-LLM-XFD`. Source provenance and
filtering are recorded in [SOURCE_MAP.yaml](SOURCE_MAP.yaml). The full legacy
project documentation is retained as [LEGACY_README.md](LEGACY_README.md).

## What is here

- `code/llm_explainable_toolkit/`: legacy evidence IR, adapters, generation,
  evaluation, and dialogue code.
- `code/tests/`, `experiments/scripts/`, and `scripts/`: legacy test and
  execution entrypoints.
- `config/`, `configs/`, and `experiments/configs/`: project-local
  configuration sources.
- `doc/`, `manuscript/`, and `submission_prep/`: source documentation
  retained for traceability.
- `configs/experiments/p03/` at repository root: the current engine-facing
  P03 configuration set.

Generated test results, conversation sessions, caches, figures, PDFs, agent
material, and archives are intentionally absent.

## Engine-facing checks

Run from the PHM-Vibench repository root:

```bash
python -m scripts.validate_configs
python -m scripts.config_inspect \
  --config configs/experiments/p03/e1_tspn_uxfd_cwru_dg.yaml
python -m scripts.validate_docs
```

The legacy LLM toolkit is not automatically registered as a PHM-Vibench
factory component. Provider-dependent or human-study execution is not implied
by the overlay, and no external LLM call is required for the checks above.

## Evidence boundary

No legacy metric, user study, industrial deployment, or generated explanation
is promoted by this copy. The source repository was clean, but its own
readiness record states that only synthetic/dummy smoke evidence exists.
Use the PaperTrace claim registry and run ledger before citing any number.

# GOAL-FFU-P0-001: PHM-GenBench Constitution

## Objective

Finalize `.specify/memory/constitution.md` as the binding PHM generative
benchmark governance document.

## Required Behavior

- Define configuration-first and 5-block config rules.
- Define factory-first runtime boundaries and forbid `src/phm_factory/`.
- Define benchmark-valid, exploratory, and docs-only validity rules.
- Define forbidden synthetic source splits.
- Define eval-only FFT/spectral/leakage/utility evidence rules.
- Define frontier promotion policy for core-fast experimental methods.

## Acceptance Criteria

- No template placeholders remain.
- Version is `1.0.0`.
- `python -m scripts.validate_docs` passes.

## Validation Commands

```bash
python -m scripts.validate_docs
rg -n "\[PROJECT_NAME\]|\[PRINCIPLE_|CONSTITUTION_VERSION" .specify/memory/constitution.md
```

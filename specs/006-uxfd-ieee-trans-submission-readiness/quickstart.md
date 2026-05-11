# Quickstart: UXFD IEEE Transactions Submission Readiness

Use these commands from the repository root.

## Confirm Active Feature

```bash
cat .specify/feature.json
```

Expected:

```json
{
  "feature_directory": "specs/006-uxfd-ieee-trans-submission-readiness"
}
```

## Inspect Goal Package

```bash
find paper/UXFD_paper/goal -maxdepth 1 -type f | sort
```

Expected: 11 files: index, overall goal, seven paper files, recent-work citation README, and readiness matrix.

## Inspect Spec Kit Artifacts

```bash
find specs/006-uxfd-ieee-trans-submission-readiness -maxdepth 3 -type f | sort
```

Expected: `spec.md`, `plan.md`, `research.md`, `data-model.md`, `quickstart.md`, `tasks.md`, one contract, and two checklists.

## Inspect Claude Team Spec

```bash
sed -n '1,220p' .codex/claude-team-runs/20260511-uxfd-ieee-trans-review/TASK_SPEC.md
```

The task spec is prepared for read-only review. Launching Claude teammates is a later explicit action.
See `.codex/claude-team-runs/20260511-uxfd-ieee-trans-review/LAUNCH_LOG.md` for the blocked launch attempt.

## Inspect Handoff

```bash
sed -n '1,220p' .claude/handoffs/2026-05-11-uxfd-ieee-trans-submission-readiness.md
```

## Validate Docs

```bash
python -m scripts.validate_docs
```

If this fails, record the failing paths. Do not claim documentation validation passed.

## Validate Literature And Goal Gates

```bash
python -m scripts.baseline_mapping
python -m pytest -q test/test_uxfd_paper_alignment_contract.py test/test_collect_uxfd_runs.py
```

The broad `docs/literature/` inventory is no longer sufficient for UXFD paper
submission positioning. Use `paper/UXFD_paper/goal/08_recent_work_citation_readme.md`
as the TOP-source gate for core related work and baselines.

## Validate Compute Budget Gate

All paper evidence plans must fit local RTX 4090 GPUs `0,1`.

```bash
rg -n "CUDA_VISIBLE_DEVICES=0,1|RTX 4090|Compute Budget|resource-blocked" paper/UXFD_paper/goal specs/006-uxfd-ieee-trans-submission-readiness
python -m pytest -q test/test_uxfd_paper_alignment_contract.py
```

Recorded on 2026-05-11:

- `python -m scripts.validate_docs`: `[OK] Documentation checks passed (128 files scanned).`
- `python -m pytest -q test/test_uxfd_paper_alignment_contract.py test/test_collect_uxfd_runs.py`: `9 passed in 0.32s`.
- `.specify/scripts/bash/check-prerequisites.sh --json --require-tasks --include-tasks`: passed and resolved `FEATURE_DIR` to `specs/006-uxfd-ieee-trans-submission-readiness`.
- Cross-artifact consistency pass: no critical/high issues found; remaining unchecked tasks are intentional future Claude launch and paper-production backlog.
- Claude Code Team launch: prepared, attempted, and blocked by external-service policy; see `LAUNCH_LOG.md`.
- `python -m scripts.phm_literature_matrix --min-count 50`: passed with 58 entries from 2025-2026.
- `python -m scripts.baseline_mapping`: passed and rendered the current baseline mapping.
- `python -m pytest -q test/test_uxfd_paper_alignment_contract.py test/test_collect_uxfd_runs.py test/test_phm_literature_matrix.py test/test_baseline_mapping_contract.py`: `23 passed in 2.51s`.
- Base-env representative model smoke command `python -m pytest -q test/test_model_registry_contract.py test/test_x_model_smoke.py`: failed because `pytorch_lightning` is missing from the base Python environment.
- `eval "$(conda shell.bash hook)" && conda activate LQ_signal && python -m pytest -q test/test_model_registry_contract.py test/test_x_model_smoke.py`: `23 passed, 1 skipped`.

Recorded after 2x4090 compute-gate update on 2026-05-11:

- `.specify/scripts/bash/check-prerequisites.sh --json --require-tasks --include-tasks`: passed and resolved `FEATURE_DIR` to `specs/006-uxfd-ieee-trans-submission-readiness`.
- `python -m scripts.validate_docs`: `[OK] Documentation checks passed (128 files scanned).`
- `python -m scripts.baseline_mapping`: passed and rendered the current baseline mapping.
- `python -m pytest -q test/test_uxfd_paper_alignment_contract.py`: `16 passed in 0.27s`.
- `python -m pytest -q test/test_uxfd_paper_alignment_contract.py test/test_collect_uxfd_runs.py test/test_baseline_mapping_contract.py`: `21 passed in 2.22s`.
- `git diff --check -- paper/UXFD_paper/goal specs/006-uxfd-ieee-trans-submission-readiness test/test_uxfd_paper_alignment_contract.py .claude/handoffs/2026-05-11-uxfd-ieee-trans-submission-readiness.md docs/literature/README.md .specify/feature.json`: passed.

Recorded after TOP-source gate update on 2026-05-11:

- `.specify/scripts/bash/check-prerequisites.sh --json --require-tasks --include-tasks`: passed and resolved `FEATURE_DIR` to `specs/006-uxfd-ieee-trans-submission-readiness`.
- `python -m scripts.validate_docs`: `[OK] Documentation checks passed (128 files scanned).`
- `python -m scripts.baseline_mapping`: passed and rendered the current baseline mapping.
- `python -m pytest -q test/test_uxfd_paper_alignment_contract.py test/test_collect_uxfd_runs.py test/test_baseline_mapping_contract.py`: `19 passed in 2.10s`.
- Base-env representative model smoke command `python -m pytest -q test/test_model_registry_contract.py test/test_x_model_smoke.py`: failed because `pytorch_lightning` is missing from the base Python environment.
- `eval "$(conda shell.bash hook)" && conda activate LQ_signal && python -m pytest -q test/test_model_registry_contract.py test/test_x_model_smoke.py`: `23 passed, 1 skipped`.

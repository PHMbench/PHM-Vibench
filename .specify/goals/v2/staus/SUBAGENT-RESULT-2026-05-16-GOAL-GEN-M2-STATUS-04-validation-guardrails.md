# Subagent Result 04: Validation And Guardrails

**Date**: 2026-05-16
**Mode**: read-only advisory analysis
**Scope**: validation commands, guardrails, stale evidence risk
**Mutation**: none

## Validation Gates

| Gate | Status |
| --- | --- |
| `python -m scripts.validate_docs` | current pass reported by subagent, 120 files scanned |
| `python -m scripts.validate_configs` | current pass reported by subagent, 22/22 configs passed |
| Forbidden path guard | active in `scripts.validate_docs` |
| Module-local README checks | active in `scripts.validate_docs` |
| M2 preflight mirror/source consistency | active in `scripts.validate_docs` |
| Run-status ledger and paper sidecar checks | active in `scripts.validate_docs` |

Forbidden path guard covers:

```text
docs/phm_generative
docs/generative
src/phm_factory
projects
projects/phm_generative
packs
templates
schemas
```

## Known Passing Gates

These are latest-known recorded results unless explicitly rerun by Codex after
this status package is written.

| Command | Latest-known result |
| --- | --- |
| `python -m scripts.validate_docs` | pass |
| `python -m scripts.validate_configs` | 22/22 configs passed |
| `python -m pytest test/smoke/test_validate_docs.py -q` | 91 passed |
| `python -m pytest test/smoke -q` | 98 passed |
| `python -m pytest test/generative/test_benchmark_effect.py -q` | 17 passed |
| `python -m pytest test/generative/test_benchmark_effect.py test/generative/test_six_dataset_submission.py -q` | 35 passed |
| `python -m pytest test/generative -q` | 103 passed, 1 warning |
| `conda activate LQ_signal && python -m pytest test/ -q` | 220 passed, 1 warning |

## Gaps

- M2-003 real evidence remains blocked.
- `results/paper/phm_generative/six_dataset_submission_v1/runs` is absent.
- GPU evidence records failure for GPU 6 and GPU 7.
- Blocked ledgers have 36 `BLOCKED_GPU_PREFLIGHT` rows.
- Paper artifacts remain `NOT_SUBMISSION_READY`.
- Full pytest status is latest-known unless rerun.
- `results/` freshness is not comprehensively guaranteed by doc scanning.

## Stale Evidence Risk

- Several generated evidence artifacts are dated 2026-05-12, while the current
  status snapshot is 2026-05-16.
- Pytest counts should be reported as latest-known unless rerun in the current
  pass.
- The dirty worktree is substantial, so status should distinguish local current
  state from a committed baseline.

## Recommended Status Decomposition

- Current verified gates: commands rerun in the current pass with timestamps.
- Recorded historical gates: pytest and GPU results copied from evidence
  snapshots.
- Structural guardrails: forbidden paths, module README contracts, and local
  doc references.
- Runtime blocker: GPU 6/7 CUDA failure and missing real `runs/`.
- Submission readiness: paper remains scaffolded but not benchmark-valid.
- Next validation bundle after GPU fix: CUDA probes, M2 preflight, staged
  train/sample/eval/paperpack, aggregation, paperpack, draft generation, and
  full `LQ_signal` pytest.

# GitHub Issues Draft: Core Runtime And Config Contract

**Repository:** `PHMbench/PHM-Vibench`
**Source tasks:** `specs/001-core-runtime-config-contract/tasks.md`
**Status:** Draft only. Do not treat as completed `speckit-taskstoissues`.

Issue creation is blocked until GitHub authentication and duplicate detection are safe.

## Issue: Slice 1 Phase 1 - Setup And Existing Coverage

Labels: `speckit`, `slice-1`, `phase-setup`

Tasks:

- T001 Verify `.specify/feature.json` points to `specs/001-core-runtime-config-contract`
- T002 Run targeted existing tests and record current failures in `quickstart.md`
- T003 Inspect `main.py` and `src/configs/preflight.py` against the runtime contract
- T004 Inspect manifest and artifact helpers against the runtime contract

Acceptance:

- Active feature resolves to Slice 1.
- Current coverage gaps are recorded before implementation edits.

## Issue: Slice 1 Phase 2 - Foundational Contract Tests

Labels: `speckit`, `slice-1`, `tests`

Tasks:

- T005 Create or extend strict CLI/config contract tests in `test/test_main_strictness.py`
- T006 Create or extend run artifact contract tests in `test/test_run_artifacts_contract.py`
- T007 Create or extend config inspect/validate contract tests in `test/test_config_tools_contract.py`
- T008 Run these tests and record pre-implementation failures in `quickstart.md`

Acceptance:

- Each uncovered contract has a targeted test before code changes.
- Any new failing tests identify the exact implementation gap.

## Issue: Slice 1 US1 - Valid Config Run Emits Required Artifacts

Labels: `speckit`, `slice-1`, `US1`, `artifacts`

Tasks:

- T009 Add a required-manifest-fields test in `test/test_run_artifacts_contract.py`
- T010 Add a smoke-run artifact expectation test in `test/test_run_contract_helper.py`
- T011 Patch artifact writing only if the tests fail
- T012 Patch run context or metrics writing only if the tests fail
- T013 Run artifact contract tests

Acceptance:

- Required manifest fields and required run artifact files are covered.
- No explainability artifact is made mandatory unless the run enables explainability.

## Issue: Slice 1 US2 - Invalid Runtime Inputs Fail Early

Labels: `speckit`, `slice-1`, `US2`, `fail-fast`

Tasks:

- T014 Add or verify missing `--config` and missing file tests
- T015 Add or verify missing top-level `pipeline` and unknown pipeline tests
- T016 Add or verify invalid override failure tests
- T017 Patch strict input handling only if tests fail
- T018 Patch preflight failure surfacing only if tests expose a gap
- T019 Run strict main tests

Acceptance:

- Invalid inputs fail before trainer setup.
- No implicit demo, default pipeline, or legacy fallback is used for invalid runtime contracts.

## Issue: Slice 1 US3 - Inspect And Validate Configs Before Running

Labels: `speckit`, `slice-1`, `US3`, `config-tools`

Tasks:

- T020 Add config inspect contract tests in `test/test_config_tools_contract.py`
- T021 Add config validation registry coverage tests in `test/test_config_tools_contract.py`
- T022 Patch inspect output only if tests fail
- T023 Patch validation path collection only if tests fail
- T024 Run config tool contract tests

Acceptance:

- Inspect output exposes resolved config, sources, and targets.
- Validation covers maintained demos, Hydra experiments, and active registry rows.

## Issue: Slice 1 Phase 6 - Cross-Cutting Validation And Handoff

Labels: `speckit`, `slice-1`, `validation`

Tasks:

- T025 Run config inspect and record status in `quickstart.md`
- T026 Run config validation and record status in `quickstart.md`
- T027 Regenerate config atlas and inspect `docs/CONFIG_ATLAS.md` diff
- T028 Run targeted Slice 1 tests and record status in `quickstart.md`
- T029 Run smoke demo matrix and record status in `quickstart.md`
- T030 Update `quickstart.md` with actual command results and skipped gates
- T031 Write final Slice 1 implementation handoff

Acceptance:

- Actual command results are recorded.
- Any skipped validation gate has an explicit reason.


# Tasks: PHM-GenBench Frontier

**Input**: Design documents from `specs/002-phm-genbench-frontier/`
**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/`, `quickstart.md`

**Tests**: Required for runtime changes and evidence contracts.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Make governance and tracking artifacts authoritative.

- [x] T001 Finalize `.gitignore` exceptions for `.specify/feature.json`, `.specify/goals/*.md`, `.specify/memory/constitution.md`, and new `test/**/*.py`
- [x] T002 Replace `.specify/memory/constitution.md` with PHM-GenBench v1.0.0 principles
- [x] T003 [P] Replace `.specify/goals/` with the P0-P3 roadmap from this feature
- [x] T004 [P] Add Speckit feature artifacts under `specs/002-phm-genbench-frontier/`
- [x] T005 [P] Create Claude Teams review task spec under `specs/002-phm-genbench-frontier/reviews/claude-team/`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Evidence gates that block all model-family work.

- [x] T006 Add tests for strict config preflight in `test/smoke/test_preflight.py`
- [x] T007 Implement `--preflight-only` and strict YAML errors in `main.py`
- [x] T008 Add schema fields for generative method family, experimental flag, and condition policy in `src/config_schema/models.py`
- [x] T009 Add condition sampling tests in `test/generative/test_condition_sampling.py`
- [x] T010 Implement condition policies in `src/Pipeline_06_generative.py`
- [x] T011 Add normalization manifest tests in `test/generative/test_normalization_manifest.py`
- [x] T012 Implement normalization artifact writing and manifest attachment in generative train/sample flow
- [x] T013 Add missing metric reason tests in `test/generative/test_generative_metrics.py`
- [x] T014 Implement metric value/status/reason outputs in generative metric modules

**Checkpoint**: P0 evidence loop can preflight, sample condition grids, record normalization, and explain missing metrics.

---

## Phase 3: User Story 1 - Govern Benchmark Validity (Priority: P1) MVP

**Goal**: Prevent benchmark-valid claims without complete evidence.

**Independent Test**: A manifest fixture with missing evidence is downgraded or rejected.

- [x] T015 [P] [US1] Add manifest validity tests in `test/generative/test_manifest_validity.py`
- [x] T016 [US1] Tighten synthetic manifest validity checks in `src/task_factory/Components/generative/manifests/synthetic_data_manifest.py`
- [x] T017 [US1] Update generative module READMEs to explain validity statuses and required evidence

---

## Phase 4: User Story 2 - Preflight And Evidence Loop (Priority: P2)

**Goal**: Make train/sample/eval/paperpack reproducible.

**Independent Test**: Dummy generative config completes preflight and the smoke evidence loop.

- [x] T018 [P] [US2] Add Euler sampler guard tests in `test/generative/test_euler_ode_sampler.py`
- [x] T019 [US2] Add post-update finite/shape/dtype/device checks in `src/task_factory/Components/generative/samplers/euler_ode.py`
- [x] T020 [P] [US2] Add paperpack aggregation tests in `test/generative/test_paperpack_generative.py`
- [x] T021 [US2] Upgrade `scripts/paperpack_generative.py` with run index, mean/std tables, manifest completeness, and missing metric appendix
- [x] T022 [US2] Add quickstart validation commands to the relevant module READMEs

---

## Phase 5: User Story 3 - Integrate Frontier Model Families (Priority: P3)

**Goal**: Add factory-integrated frontier coverage with exploratory defaults.

**Independent Test**: Each family has a CPU smoke config and cannot be benchmark-valid without evidence.

- [x] T023 [P] [US3] Add Rectified Flow / FlowTS task and config tests under `test/generative/`
- [x] T024 [US3] Promote Rectified Flow / FlowTS through existing generative factories and configs
- [x] T025 [P] [US3] Add DDPM / Diffusion-TS task and config tests under `test/generative/`
- [x] T026 [US3] Promote DDPM / Diffusion-TS through existing generative factories and configs
- [x] T027 [P] [US3] Add TimeFlow / Score-SDE task and config tests under `test/generative/`
- [x] T028 [US3] Promote TimeFlow / Score-SDE through existing generative factories and configs
- [x] T029 [P] [US3] Add UNet1D, DiT1D, and Mamba/SSM backbone tests under `test/generative/`
- [x] T030 [US3] Add UNet1D, DiT1D, and stateless Mamba/SSM backbones under `src/model_factory/generative_model/`
- [x] T031 [P] [US3] Add one-step experimental method tests for MeanFlow/iMF, Drifting, TFM, and OT-NFM
- [x] T032 [US3] Add one-step experimental methods with `experimental=true` and exploratory defaults

---

## Phase 6: User Story 4 - Produce Paper-Grade Review Artifacts (Priority: P4)

**Goal**: Produce evidence suitable for PHM application paper review.

**Independent Test**: Multi-seed runs produce reproducibility tables and appendices with traceable rows.

- [x] T033 [P] [US4] Add paper config matrix under `configs/paper/phm_generative/`
- [x] T034 [US4] Extend `scripts/generative_sweep.py` for multi-family and multi-seed matrices
- [x] T035 [P] [US4] Add utility protocol tests for TSTR/TRTS and real+synthetic augmentation
- [x] T036 [US4] Implement paper utility protocol runners or documented smoke wrappers
- [x] T037 [US4] Add final paper table/figure documentation to `scripts/README.md`, `configs/paper/phm_generative/README.md`, and metric component README

---

## Final Phase: Polish & Cross-Cutting Concerns

- [x] T038 Run `python -m scripts.validate_docs`
- [x] T039 Run `python -m scripts.validate_configs`
- [x] T040 Run `python -m pytest test/`
- [x] T041 Write phase handoff under `specs/002-phm-genbench-frontier/handoffs/`
- [x] T042 [P] Add M2 queue index under `specs/002-phm-genbench-frontier/m2/`
- [x] T043 [P] Add M2 cross-artifact analysis under `specs/002-phm-genbench-frontier/analysis/`
- [x] T044 [P] Add Claude teammate task spec and blocked-review evidence under `specs/002-phm-genbench-frontier/reviews/claude-team/`
- [x] T045 Add Codex verification log under `specs/002-phm-genbench-frontier/reviews/codex/`
- [x] T046 Add M2 paper readiness notes under `specs/002-phm-genbench-frontier/paper/`
- [ ] T047 [M2-003] Execute real six-dataset GPU train/sample/eval/paperpack
  after GPU 6/7 preflight passes and attach benchmark-valid run evidence
- [ ] T048 [M2-002] Aggregate real six-dataset run directories into the
  benchmark effect summary and manifest after T047 completes
- [ ] T049 [M2-004] Generate final paper tables and figure sources from the
  real benchmark effect artifacts after T048 completes
- [ ] T050 [M2-005] Regenerate the submission draft and readiness sidecars from
  real benchmark-valid evidence after T049 completes
- [ ] T051 [M2-006] Run final Codex verification and advisory review against
  the completed evidence package after T050 completes

**Current verification note (2026-05-16 15:09:57 CST)**: T038 and T039 pass.
`python -m scripts.validate_docs` passes with 120 files scanned,
`python main.py --config configs/demo/00_smoke/dummy_dg.yaml --preflight-only`
and `python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml
--preflight-only` both pass,
`GOAL-GEN-M2-003` now includes preflight, staged execute, and aggregation
commands directly in its validation block and activates `LQ_signal` for those
benchmark commands; `configs/paper/phm_generative/README.md` and
`scripts/README.md` mirror the same M2 execution-environment guidance, and
`scripts.validate_docs` now enforces that guidance plus the M2-003 staged
validation command contract,
and the M2 GPU gate now requires individual `CUDA_VISIBLE_DEVICES=6` and
`CUDA_VISIBLE_DEVICES=7` torch probes. Default sandboxed probes cannot see
`/dev/nvidia*`, but elevated `nvidia-modprobe -u -c=0` plus `LQ_signal` probes
report `cuda_available True` and `device_count 2` for GPU 6/7 combined,
`python -m pytest test/smoke/test_validate_docs.py -q` passes with 91 tests,
and `python -m pytest test/smoke -q` passes with 98 tests. In the project
`LQ_signal` environment, `python -m pytest test/ -q` passes with `220 passed, 1
warning`. The base Python environment still lacks `torchmetrics`, so use
`LQ_signal` for the full repository test gate. M2 real GPU execution is now
partially progressed in the elevated context: GPU 6/7 preflight passes after
`nvidia-modprobe -u -c=0`, and
`results/paper/phm_generative/six_dataset_submission_v1/runs` contains partial
train evidence.
After the canonical source-report and execute-preflight-failure safe-stop gates, `python -m pytest
test/generative/test_benchmark_effect.py test/generative/test_six_dataset_submission.py
-q` passes with 35 tests.
After making `source_report` mandatory, `python -m pytest test/generative -q`
passes with 103 tests and 1 warning.
`python -m scripts.generative_submission_draft ... --require-submission-ready`
still exits 2 and keeps the draft sidecars at `NOT_SUBMISSION_READY` because
the real effect summary/manifest are absent.
The latest canonical M2-003 GPU dry-run preflight under `LQ_signal` exits 2 as
expected, refreshes `gpu_preflight_report.json` with `created_at`
`2026-05-16T07:10:59.524502+00:00`, and writes a 37-line blocked ledger.
The six-dataset matrix now overrides `model.num_fault_classes=32` and
`model.num_domains=16`, which fixed the first real train failure
(`fault_label exceeds configured embedding size`). The elevated M2-003 train
stage produced partial evidence, then was intentionally interrupted at
`2026-05-16 16:26:39 CST` to avoid leaving an unattended long GPU run: 7
checkpoints exist, including all six CWRU method/seed train jobs and XJTU CFM
seed 0. A bounded train resume with `--skip-existing --max-runs 1` later
skipped the six completed CWRU rows and produced XJTU CFM seed 0
`train_result_0.csv`. A second bounded train resume completed XJTU CFM seed 1
with `train_wall_clock_sec=1268.225`. Current partial evidence is 8
`train_result_0.csv`, 9 checkpoints, and 6 manifest files; sample/eval/
paperpack artifacts are still absent. The latest M2-002 aggregation recheck at
`2026-05-16 15:16:54 CST`
exits 2
because complete eval metric runs are absent. The latest M2-005 draft recheck
at the same timestamp exits 2 and keeps `PAPER_DRAFT.md`,
`evidence_gaps.md`, and `submission_readiness.md` at `NOT_SUBMISSION_READY`.
The staged execute command `--execute --preflight-gpu --stages train` also
exits 2 during GPU preflight and creates no `runs/` directory.
Focused benchmark-effect tests now cover that safe-stop path; `python -m pytest
test/generative/test_benchmark_effect.py test/generative/test_six_dataset_submission.py
-q` passes with 35 tests.
T006 and T007 are covered by `test/smoke/test_preflight.py` and the existing
`main.py --preflight-only` implementation, including malformed YAML, invalid
pipeline, missing required section, generative sample-without-checkpoint, and
no pipeline import during preflight.
`test/smoke/test_validate_docs.py` covers the documentation placement gate that
blocks deprecated central PHM generative docs directories and the v2 GOAL-GEN
structure, filename/ID matching, and required module README gates used by
handoff/review automation.

## Dependencies & Execution Order

- Phase 1 blocks all later work.
- Phase 2 blocks model-family promotion.
- US1 and US2 can proceed after Phase 2; US3 depends on US1/US2 evidence contracts.
- US4 depends on at least two model families and paperpack aggregation.

## Parallel Opportunities

- T003, T004, T005 can run in parallel.
- Test-writing tasks marked `[P]` can run before implementation tasks in the same story.
- Model-family tests can be split by family once Phase 2 is complete.
- M2 process-artifact tasks T042-T044 can run in parallel because they own
  disjoint directories under the active feature.

## Implementation Strategy

Start with P0 evidence infrastructure. Do not add new model families until
preflight, condition sampling, normalization evidence, manifest validity, and
metric missing reasons are in place.

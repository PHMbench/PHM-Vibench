# Benchmark Readiness Checklist: PHM-GenBench Frontier

**Purpose**: Unit tests for requirement quality around benchmark evidence, model promotion, and paper readiness
**Created**: 2026-05-10
**Feature**: `specs/002-phm-genbench-frontier/spec.md`

## Requirement Completeness

- [x] CHK001 Are benchmark-valid evidence requirements complete for config, protocol, normalization, conditions, leakage, and metrics? [Completeness, Spec §FR-005]
- [x] CHK002 Are model-promotion requirements defined for both stable baselines and one-step frontier methods? [Completeness, Spec §FR-008]
- [x] CHK003 Are paperpack requirements complete for tables, figure sources, appendices, and reproducibility statements? [Completeness, Spec §FR-007]

## Requirement Clarity

- [x] CHK004 Is `benchmark-valid` objectively distinguishable from `exploratory` and `docs-only`? [Clarity, Spec §FR-005]
- [x] CHK005 Is "Core Fast" constrained enough to avoid hidden benchmark-valid claims for immature frontier methods? [Clarity, Spec §FR-009]
- [x] CHK006 Are condition sampling policies named and scoped unambiguously? [Clarity, Spec §FR-003]

## Requirement Consistency

- [x] CHK007 Are factory-first integration requirements consistent with the no-parallel-runtime constitution rule? [Consistency, Spec §FR-008]
- [x] CHK008 Are paper-grade claims consistent with exploratory defaults for frontier methods? [Consistency, Spec §FR-009]

## Acceptance Criteria Quality

- [x] CHK009 Can every success criterion be validated without subjective judgment? [Measurability, Spec §SC-001-SC-007]
- [x] CHK010 Do acceptance scenarios cover both successful evidence loops and invalid evidence downgrades? [Coverage, Spec §User Scenarios]

## Edge Case Coverage

- [x] CHK011 Are missing metric, missing normalization, and forbidden split cases covered in requirements? [Edge Case, Spec §Edge Cases]
- [x] CHK012 Are optional dependency and CUDA-specific risks addressed in requirements? [Dependency, Spec §Assumptions]

## Dependencies & Assumptions

- [x] CHK013 Are external research anchors separated from implementation obligations? [Assumption, Spec §Assumptions]
- [x] CHK014 Is Claude Teams usage scoped as review/planning rather than final authority? [Dependency, Spec §FR-010]

## Evidence Notes

- Requirement quality is covered by `spec.md`, `data-model.md`, and
  `contracts/generative-benchmark-contract.md`.
- This checklist validates requirement readiness only. It does not mean M2-003
  real six-dataset GPU execution is complete.
- M2-003 remains blocked until GPU 6/7 pass CUDA preflight and real run
  evidence exists.

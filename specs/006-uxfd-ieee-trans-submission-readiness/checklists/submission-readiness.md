# Submission-Readiness Requirements Checklist: UXFD IEEE Transactions Submission Readiness

**Purpose**: Unit tests for the English requirements governing seven-paper submission readiness
**Created**: 2026-05-11
**Feature**: `specs/006-uxfd-ieee-trans-submission-readiness/spec.md`

## Requirement Completeness

- [x] CHK001 Are all seven UXFD papers explicitly represented in the requirements? [Completeness, Spec FR-002]
- [x] CHK002 Are per-paper goal-file contents specified with enough fields to guide implementation? [Completeness, Spec FR-003]
- [x] CHK003 Are Claude Code Team requirements documented as a first-class workflow component? [Completeness, Spec FR-007]
- [x] CHK004 Are handoff requirements documented for multi-session continuity? [Completeness, Spec FR-009]
- [x] CHK016 Are the six-baseline, ablation, and SOTA-gate requirements specified for all seven papers? [Completeness, Spec FR-013/FR-014/FR-015]
- [x] CHK017 Is a 2024-2026 TOP-source related-work citation/reproduction README required? [Completeness, Spec FR-016]
- [x] CHK021 Are low-tier sources excluded from core related work, baseline, novelty, and SOTA evidence? [Completeness, Spec FR-019]
- [x] CHK022 Does every paper require at least three TOP recent methods and one runnable TOP baseline? [Completeness, Spec FR-020]
- [x] CHK026 Is the local 2x4090 compute budget specified for all papers? [Completeness, Spec FR-021/FR-022]
- [x] CHK027 Is resource-blocked exact reproduction distinguished from representative-runnable baselines? [Completeness, Spec FR-023]

## Requirement Clarity

- [x] CHK005 Is "submission-ready" distinguished from minimal root-gate success? [Clarity, Spec Edge Cases]
- [x] CHK006 Is the Spec Kit workflow order stated exactly and consistently? [Clarity, Spec FR-006]
- [x] CHK007 Are unsupported claims and missing evidence handled as blockers rather than vague risks? [Clarity, Spec FR-011]
- [x] CHK018 Is exact reproduction distinguished from representative runnable baselines? [Clarity, Spec FR-017]
- [x] CHK023 Is TOP venue quality distinguished from broad PHM field proximity? [Clarity, Spec Assumptions]
- [x] CHK028 Are GPU IDs, device metadata, and no-cloud/no-extra-GPU assumptions explicit? [Clarity, Spec FR-024]

## Requirement Consistency

- [x] CHK008 Are submodule commit rules consistent between paper goal files and Spec Kit requirements? [Consistency, Spec FR-010]
- [x] CHK009 Are Claude Team permissions consistent with dirty-worktree safety constraints? [Consistency, Spec FR-008]

## Acceptance Criteria Quality

- [x] CHK010 Can the goal package file count be objectively verified? [Measurability, Spec SC-001]
- [x] CHK011 Can the Claude team task spec requirements be objectively verified? [Measurability, Spec SC-004]
- [x] CHK012 Can the handoff content requirement be objectively verified? [Measurability, Spec SC-005]
- [x] CHK019 Can the six-baseline minimum be objectively checked per paper goal file? [Measurability, Spec SC-008]
- [x] CHK024 Can the accepted TOP method pool and per-paper TOP quota be objectively checked? [Measurability, Spec SC-009]
- [x] CHK029 Can compute-budget presence be objectively checked in goal files and readiness matrix? [Measurability, Spec SC-010/SC-011]

## Edge Case Coverage

- [x] CHK013 Does the spec address existing `specs/005-*` collision risk? [Coverage, Spec Edge Cases]
- [x] CHK014 Does the spec address dirty submodule state before commits? [Coverage, Spec Edge Cases]
- [x] CHK015 Does the spec address missing or placeholder manuscript entrypoints? [Coverage, Spec Edge Cases]
- [x] CHK020 Does the spec prevent literature-only recent work from being counted as reproduced SOTA evidence? [Coverage, Spec Edge Cases]
- [x] CHK025 Does the spec prevent low-tier broad-literature inventory entries from entering UXFD accepted method pools? [Coverage, Spec Edge Cases]
- [x] CHK030 Does the spec prevent large TOP methods from being counted as exact SOTA evidence when they exceed 2x4090? [Coverage, Spec Edge Cases]

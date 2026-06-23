# Checklist: UXFD Paper Alignment Requirements

**Purpose**: Unit-test the Slice 4 requirements before task generation
**Created**: 2026-05-10
**Feature**: ../spec.md
**Plan**: ../plan.md

## Requirement Completeness

- [x] CHK001 Are all seven UXFD submodule contract dimensions specified? [Completeness, Spec FR-001]
- [x] CHK002 Are root CLI and paper-local command statuses clearly distinguished? [Completeness, Spec FR-002]
- [x] CHK003 Are evidence statuses defined for smoke-passed, blocked, skipped, paper-local-only, and unverified cases? [Completeness, Spec FR-003]
- [x] CHK004 Are artifact, command, and blocker recording requirements specified for minimal gates? [Completeness, Spec FR-004]
- [x] CHK005 Are LaTeX entrypoint discovery and compile-tooling requirements specified before compile tasks? [Completeness, Spec FR-007]
- [x] CHK006 Are claim-to-evidence requirements specified for figures, tables, metrics, baselines, and text claims? [Completeness, Spec FR-008]

## Requirement Clarity

- [x] CHK007 Is the submodule ownership rule clear enough to prevent accidental parent gitlink changes? [Clarity, Spec FR-006, Contract Submodule Safety]
- [x] CHK008 Is missing TeX toolchain behavior specified as skipped or blocked rather than pass/fail ambiguity? [Clarity, Spec FR-011]
- [x] CHK009 Is the parent-doc boundary clear enough to avoid duplicating submodule roadmaps? [Clarity, Spec FR-012]
- [x] CHK010 Are unsupported or placeholder claims required to be removed, marked unresolved, or blocked? [Clarity, Spec FR-009]

## Requirement Consistency

- [x] CHK011 Do the spec, plan, research, and contract agree that `VIBENCH.md` and min configs are the parent-facing reproduction contract? [Consistency, Spec US1, Research, Contract Sources]
- [x] CHK012 Do paper evidence requirements align with Slice 1 runtime artifacts and Slice 2/3 task/model support status? [Consistency, Spec Assumptions]
- [x] CHK013 Do compile requirements avoid assuming one uniform LaTeX entrypoint across submodules? [Consistency, Clarifications, Research]
- [x] CHK014 Does AGENTS point to the active Slice 4 plan without conflicting Speckit guidance? [Consistency, AGENTS Speckit block]

## Acceptance Criteria Quality

- [x] CHK015 Are success criteria measurable through submodule status, commands, artifacts, logs, or blockers? [Measurability, Spec SC-001..SC-006]
- [x] CHK016 Is claim traceability objectively verifiable through artifact/source/blocker mapping? [Measurability, Spec SC-003]
- [x] CHK017 Is compile status measurable through command, PDF, log, and first error fields? [Measurability, Spec SC-004, Contract Compile Gate]

## Scenario Coverage

- [x] CHK018 Are primary flows covered for contract audit, minimal evidence, claim alignment, and compile gates? [Coverage, Spec User Stories 1..4]
- [x] CHK019 Are historical README paths and old CLI flags represented as edge cases? [Edge Case, Spec Edge Cases]
- [x] CHK020 Are missing artifacts and unsupported baselines represented as paper-claim blockers? [Coverage, Spec Edge Cases]
- [x] CHK021 Are dirty submodules and unintended gitlink changes represented as edge cases? [Edge Case, Spec Edge Cases]

## Dependencies And Assumptions

- [x] CHK022 Are dependencies on Slice 1 artifacts and Slice 2/3 support status documented? [Assumption, Spec Assumptions]
- [x] CHK023 Is the no-recursive-paper-read constraint explicit enough for implementation tasks? [Assumption, Spec Assumptions, Plan Technical Context]
- [x] CHK024 Is skipped validation required to record reason and impact? [Traceability, Spec SC-006, Quickstart]

## Notes

All checklist items are complete. Proceed to `/speckit-tasks`.

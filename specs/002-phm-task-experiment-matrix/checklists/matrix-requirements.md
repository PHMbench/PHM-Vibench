# Checklist: PHM Task Experiment Matrix Requirements

**Purpose**: Unit-test the Slice 2 requirements before task generation
**Created**: 2026-05-10
**Feature**: ../spec.md
**Plan**: ../plan.md

## Requirement Completeness

- [x] CHK001 Are task-family sources of truth explicitly limited to registry and atlas inputs? [Completeness, Spec FR-001, Plan Phase 0]
- [x] CHK002 Are all required support statuses defined with mutually exclusive meanings? [Completeness, Spec FR-002, Data Model Support Status]
- [x] CHK003 Does the spec define how absent task families are recorded without inventing support? [Clarity, Spec FR-003, Clarifications]
- [x] CHK004 Are offline smoke requirements separate from real-data full-matrix requirements? [Completeness, Spec FR-004, Spec FR-005]
- [x] CHK005 Are task/data compatibility requirements defined beyond generic "invalid config" language? [Completeness, Spec FR-008, Contract Task/Data Compatibility]
- [x] CHK006 Are atlas and registry synchronization requirements included when maintained entries change? [Completeness, Spec FR-012]

## Requirement Clarity

- [x] CHK007 Is `PHM_VIBENCH_DATA` or an equivalent explicit real-data root required for full mode? [Clarity, Spec FR-010, Quickstart]
- [x] CHK008 Are unsupported and unverified statuses required to fail or skip with recorded reasons? [Clarity, Spec FR-011]
- [x] CHK009 Is "offline" defined enough to prevent private raw-data dependency in smoke mode? [Clarity, Spec User Story 2, Assumptions]
- [x] CHK010 Are matrix evidence fields specific enough to support later paper traceability? [Clarity, Data Model Matrix Evidence]

## Requirement Consistency

- [x] CHK011 Do the spec, plan, and contract agree that task support is derived from source-of-truth registries rather than prose duplication? [Consistency, Spec FR-001, Plan Summary, Contract Sources]
- [x] CHK012 Do the spec and plan agree that Slice 3 owns model/loss/baseline coverage? [Consistency, Spec Assumptions, Plan Summary]
- [x] CHK013 Do smoke/full requirements align with the constitution's offline-vs-real-data constraint? [Consistency, Constitution Benchmark Constraints, Spec User Story 2, Spec User Story 3]
- [x] CHK014 Does AGENTS point to the active Slice 2 plan without conflicting Speckit guidance? [Consistency, AGENTS Speckit block]

## Acceptance Criteria Quality

- [x] CHK015 Are success criteria measurable through commands, statuses, registry checks, or recorded evidence? [Measurability, Spec SC-001..SC-006]
- [x] CHK016 Can the full-matrix missing-data-root behavior be objectively verified in one command? [Measurability, Spec SC-003, Quickstart]
- [x] CHK017 Is representative DG, CDDG, FS, GFS, and pretrain coverage stated as a measurable matrix outcome? [Coverage, Spec SC-004]

## Scenario Coverage

- [x] CHK018 Are primary user flows covered for seeing task support, running smoke, running full, and detecting compatibility errors? [Coverage, Spec User Stories 1..4]
- [x] CHK019 Are registry/config mismatches represented as requirement-level scenarios? [Coverage, Spec Edge Cases, Contract Registry Consistency]
- [x] CHK020 Are few-shot class/sample feasibility and domain/system metadata gaps represented as edge cases? [Edge Case, Spec Edge Cases]
- [x] CHK021 Are pretraining objective batch-field differences captured as a requirement concern? [Coverage, Spec Edge Cases, Data Model Compatibility Contract]

## Dependencies And Assumptions

- [x] CHK022 Are dependencies on Slice 1 artifacts and Slice 3 model/loss coverage documented? [Assumption, Spec Assumptions]
- [x] CHK023 Is the no-new-dependency and no-unrelated-algorithm stance explicit enough for task generation? [Assumption, Plan Technical Context, Research]
- [x] CHK024 Is skipped real-data validation required to include a reason? [Traceability, Quickstart, Data Model Matrix Evidence]

## Notes

All checklist items are complete. Proceed to `/speckit-tasks`.

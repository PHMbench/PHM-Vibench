# Checklist: Model, Loss, And Baseline Registry Requirements

**Purpose**: Unit-test the Slice 3 requirements before task generation
**Created**: 2026-05-10
**Feature**: ../spec.md
**Plan**: ../plan.md

## Requirement Completeness

- [x] CHK001 Are model support sources explicitly limited to the model registry and smoke evidence? [Completeness, Spec FR-001]
- [x] CHK002 Are ISFM component support sources explicitly defined? [Completeness, Spec FR-002]
- [x] CHK003 Are all support statuses defined, including dependency-blocked and failed? [Completeness, Spec FR-003, Data Model Support Status]
- [x] CHK004 Are model smoke failure classes enumerated by import, constructor, dependency, and output-shape failures? [Completeness, Spec FR-004]
- [x] CHK005 Are loss, metric, contrastive strategy, and regularization discovery requirements specified? [Completeness, Spec FR-006]
- [x] CHK006 Are baseline mapping requirements tied to model, task/data, config, command, and evidence? [Completeness, Spec FR-008]

## Requirement Clarity

- [x] CHK007 Is the distinction between smoke-tested support and full benchmark evidence explicit? [Clarity, Contract Model Smoke]
- [x] CHK008 Is "dependency-blocked" defined as non-passing support? [Clarity, Clarifications, Contract Support Status]
- [x] CHK009 Are impossible contrastive/loss pairings required to fail explicitly rather than return zero loss? [Clarity, Spec FR-007]
- [x] CHK010 Are archived or legacy wrappers prevented from appearing supported unless registered? [Clarity, Spec FR-012]

## Requirement Consistency

- [x] CHK011 Do the spec, plan, research, and contract agree that no duplicate model inventory is maintained in prose? [Consistency, Spec FR-001, Plan Summary, Research]
- [x] CHK012 Do Slice 3 assumptions align with Slice 1 runtime artifacts and Slice 2 task compatibility dependencies? [Consistency, Spec Assumptions]
- [x] CHK013 Do optional dependency rules align with the constitution's no-silent-fallback principle? [Consistency, Constitution III, Research]
- [x] CHK014 Does AGENTS point to the active Slice 3 plan without conflicting Speckit guidance? [Consistency, AGENTS Speckit block]

## Acceptance Criteria Quality

- [x] CHK015 Are success criteria measurable through registry status, focused tests, or recorded blocker evidence? [Measurability, Spec SC-001..SC-006]
- [x] CHK016 Is X-model and ISFM smoke coverage defined without claiming every architecture is fully benchmarked? [Scope, Spec SC-002]
- [x] CHK017 Is baseline mapping measurable without freezing a baseline list in the spec? [Measurability, Spec SC-004, Clarifications]

## Scenario Coverage

- [x] CHK018 Are primary flows covered for support inspection, model smoke, loss validation, and baseline mapping? [Coverage, Spec User Stories 1..4]
- [x] CHK019 Are optional dependency and missing module paths represented as edge cases? [Edge Case, Spec Edge Cases]
- [x] CHK020 Are loss and contrastive impossible-pairing scenarios represented as edge cases? [Edge Case, Spec Edge Cases]
- [x] CHK021 Are prose-only baselines and incompatible task-family baselines addressed as gaps? [Coverage, Spec Edge Cases]

## Dependencies And Assumptions

- [x] CHK022 Are dependencies on Slice 1 artifacts, Slice 2 task matrix, and Slice 4 paper alignment documented? [Assumption, Spec Assumptions]
- [x] CHK023 Is the no-new-architecture/no-new-dependency default explicit enough for task generation? [Assumption, Plan Technical Context]
- [x] CHK024 Is skipped validation required to record command, prerequisite, and impact? [Traceability, Spec SC-006, Quickstart]

## Notes

All checklist items are complete. Proceed to `/speckit-tasks`.

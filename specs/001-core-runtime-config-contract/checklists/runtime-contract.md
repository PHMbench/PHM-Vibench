# Checklist: Core Runtime And Config Contract Requirements

**Purpose**: Unit-test the Slice 1 requirements before task generation
**Created**: 2026-05-10
**Feature**: ../spec.md
**Plan**: ../plan.md

## Requirement Completeness

- [x] CHK001 Are the canonical CLI requirements explicit and bounded to the maintained command? [Completeness, Spec FR-001, Contract CLI]
- [x] CHK002 Are required resolved-config blocks and `pipeline` specified? [Completeness, Spec FR-002, Contract Config Resolution]
- [x] CHK003 Is config precedence specified in a testable order? [Clarity, Spec FR-003]
- [x] CHK004 Are fail-fast input classes enumerated without relying on generic "invalid input" wording? [Completeness, Spec FR-004, Spec Edge Cases]
- [x] CHK005 Is silent fallback explicitly forbidden for demo, pipeline, task, model, and legacy path cases? [Consistency, Spec FR-005, Constitution III]
- [x] CHK006 Are inspect and validate tool expectations specified separately from runtime execution? [Completeness, Spec FR-006, Spec FR-007]
- [x] CHK007 Are required artifact files and manifest fields specified by name? [Clarity, Spec FR-008, Spec FR-008a, Contract Artifact]

## Requirement Consistency

- [x] CHK008 Do the spec, plan, and contract agree that this slice does not add new algorithms, models, losses, or paper text? [Consistency, Spec Assumptions, Plan Summary]
- [x] CHK009 Do validation requirements align with the constitution's evidence-backed reproducibility principle? [Consistency, Constitution IV, Plan Constitution Check]
- [x] CHK010 Does AGENTS point to the active Slice 1 plan without introducing conflicting workflow guidance? [Consistency, AGENTS Speckit block]

## Acceptance Criteria Quality

- [x] CHK011 Are success criteria measurable through concrete commands, files, or tests? [Measurability, Spec SC-001..SC-005, Quickstart]
- [x] CHK012 Is the offline smoke acceptance criterion separated from real-data validation? [Scope, Spec Assumptions, Constitution Benchmark Constraints]
- [x] CHK013 Is atlas synchronization tied to registry changes rather than required as an unrelated rewrite? [Scope, Spec FR-009, Quickstart]

## Scenario Coverage

- [x] CHK014 Is the valid-run scenario independently testable? [Coverage, Spec User Story 1]
- [x] CHK015 Is the invalid-input scenario independently testable before trainer setup? [Coverage, Spec User Story 2]
- [x] CHK016 Is the inspect/validate scenario independently testable before expensive experiments? [Coverage, Spec User Story 3]
- [x] CHK017 Are artifact-write failures represented as an edge case? [Edge Case, Spec Edge Cases]

## Dependencies And Assumptions

- [x] CHK018 Are existing tools and source files named as sources of truth for planning? [Traceability, Plan Phase 0, Research]
- [x] CHK019 Are no-new-dependency and no-new-abstraction assumptions explicit? [Assumption, Plan Technical Context, Constitution V]
- [x] CHK020 Is user worktree safety captured before implementation tasks are generated? [Assumption, Spec Assumptions]

## Notes

All checklist items are complete. Proceed to `/speckit-tasks`.


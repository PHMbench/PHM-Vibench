# B−1 Acceptance Status

Date: 2026-09-18

This page records the completion state of B−1 only. It does not authorize B00, new model implementation, or changes to PR #266.

## Acceptance checklist

- [x] Benchmark coverage matrix frozen.
- [x] Scientific Inclusion Gate frozen.
- [x] Candidate identity schema frozen as name + variant + task + source reference, with provisional source-row IDs retained where the source sheet lacks those fields.
- [x] lifecycle / disposition / reason_code separated.
- [x] E0 / E1 / E2 / E3 evidence ladder frozen.
- [x] Native / pretrained / non-gradient / external-service acceptance profiles frozen.
- [x] Resource-budget method frozen and grounded in existing CI plus a small local baseline measurement.
- [x] Catalogue and current runtime/discovery-registry responsibilities separated without modifying production resolution.
- [x] No new model implementation added by B−1.
- [x] Original 187-row source denominator and 18-row Task planning sheet preserved.
- [x] Final-head repository checks relevant to this documentation branch are green.
- [ ] Independent fresh-context review completed.

## Resource qualification

Measured evidence now includes:
- existing PR #266 JUnit timing and public Dummy timing;
- three independent local source-slice import/inference measurements for the unchanged GlobalAverageLinear baseline.

Still NOT RUN:
- full PHMFactory cold package import;
- normal wheel dependency-size delta;
- full training peak memory;
- GPU / real-PHM resource budget.

These unmeasured dimensions do not invalidate B−1's scientific-selection rules, but they prevent any future candidate from automatically passing G7 without model-specific evidence.

## Merge gate

B−1 implementation/documentation work is complete enough for independent review. Merge only after:
1. current-head CI remains green;
2. no unresolved review thread remains;
3. an independent reviewer finds no blocking P0/P1, or explicitly records the blocking issue for correction.

Until then:
- keep PR #267 unmerged;
- keep PR #266 unchanged at its existing head;
- do not execute B00;
- do not create a model implementation PR;
- do not change dev/main, tags, package publication, THU, or external checkpoints.

Author self-review and green CI are not treated as independent approval.

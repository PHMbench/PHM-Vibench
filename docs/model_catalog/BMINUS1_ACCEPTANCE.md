# B−1 scope-freeze outcome

Date: 2026-09-18

B−1 established durable selection and evidence rules for model-catalogue work.
It did not add or approve model implementations.

## Frozen outcomes

- Benchmark coverage is expressed as scientific roles, not a model-count target.
- Candidates must pass the G1–G7 inclusion gate before implementation is queued.
- Candidate identity separates canonical name, variant, task and source revision.
- lifecycle, disposition and reason_code are orthogonal fields.
- E0 / E1 / E2 / E3 distinguish source/legal, framework, algorithm-fidelity and PHM evidence.
- Native, pretrained, non-gradient and external-service candidates use different acceptance profiles.
- The catalogue is research/source material and is not a second runtime registry.
- No candidate is promoted from DISCOVERED solely by catalogue presence or CI status.
- Resource evidence is recorded with its exact environment and scope.

## Resource qualification at the freeze

Measured evidence includes:
- historical PR #266 JUnit and public Dummy timings at its recorded source revision;
- three local source-slice import/inference measurements for the unchanged
  GlobalAverageLinear baseline.

Not established by B−1:
- full PHMFactory cold-package import cost;
- normal-wheel dependency-size delta;
- full training peak memory;
- GPU or real-PHM resource budgets.

Those missing measurements are candidate-specific evidence requirements where relevant,
not reasons to invent a universal model threshold.

## Review outcome rule

An implementation or adoption decision still requires its own source, fidelity, resource
and scientific evidence. B−1 itself is a scope/policy freeze; it is not E2/E3 evidence
for any model and does not make a later stage automatic.

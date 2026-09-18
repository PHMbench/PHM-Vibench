# Model catalogue — B−1 scientific scope freeze

This directory stores durable model-catalogue source material and the B−1 scientific
selection/evidence decisions. It is not another AI instruction body or an automatic
execution queue.

## Read in this order

1. [Scientific coverage](BENCHMARK_COVERAGE.md): questions, roles and the Inclusion Gate.
2. [Integration policy](MODEL_INTEGRATION_POLICY.md): identity, orthogonal state,
   E0–E3 and acceptance profiles.
3. [B−1 outcome](BMINUS1_ACCEPTANCE.md): durable conclusions and evidence limits.
4. [Resource evidence](RESOURCE_BUDGET.md): measured baselines, engineering warning
   thresholds and explicitly unmeasured dimensions.
5. [Working catalogue](MODEL_CATALOGUE.csv): the 187 original source rows, all left
   DISCOVERED until candidate-specific audit.
6. [Source material](sources/README.md): repository-visible export of the original
   workbook plus the original Task table.

## Important boundaries

The current `model_registry.csv` is a discovery catalogue used alongside dynamic
`model.type/name` resolution; B−1 does not replace it or introduce another runtime
resolver. The working catalogue is never consumed by production execution.

The source workbook did not establish exact variants, revisions, code/weight licenses or
algorithm fidelity for every row. Provisional `source:Mnnn` identities preserve the
input denominator without inventing those facts.

Historical model-integration prompts remain in the conversation delivery archive requested
by the maintainer, not in versioned repository documentation. Durable decisions from those
prompts are captured in the policy/coverage documents here so AGENTS.md remains the single
maintained AI instruction body.

B−1 does not claim all 187 candidates are audited, approved, implemented, benchmark-ready
or within a universal resource budget.

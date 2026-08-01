# P02 measurement-contract configurations

## Maintained G030 entry point

measurement_contract_v1.yaml is the non-evidence conformance configuration for
the maintained measurement object and metric registry.

Run it from the src/vibench repository root:

~~~bash
conda run -n LQ_signal python -m src.explain_factory.contract_cli \
  --config configs/experiments/p02_xfd_benchmark_toolkit/measurement_contract_v1.yaml
~~~

This configuration:

- uses its own strict schema and loader;
- is not a main.py training configuration;
- is not compatible with the standard training-config registry;
- exercises all nine registered metric contracts on deterministic fixtures;
- always reports evidence_eligible=false.

The conformance result supports no paper claim. Real evidence remains blocked
until G040 is human-approved and fixed P07/P08 outputs have versioned,
source-specific adapters.

## Historical traceability configurations

p02_toolkit_benchmark.yaml and p02_toolkit_ablation.yaml describe legacy
paper-side runners. They are retained for audit only. They are not loadable by
the maintained training schema and are ineligible because the historical path
contains synthetic inputs, RNG or hard-coded metric behavior, silent
fallbacks, and arbitrary overall-score semantics.

## Classification-only smoke configurations

p02_resnet1d_cwru.yaml, p02_resnet1d_xjtu.yaml, and
p02_resnet1d_thu018.yaml are historical classification smoke configurations.
Passing those configurations does not execute the P02 measurement contract and
does not support C1, C2, or C3. Their data-boundary claims must be re-audited in
G040 before any reuse.

No file in this directory authorizes a real experiment, GPU run, claim
promotion, or protocol approval.

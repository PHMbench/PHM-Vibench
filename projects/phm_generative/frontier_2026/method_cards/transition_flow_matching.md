# Transition Flow Matching

- Method ID: `transition_flow_matching`
- Family: `transition_flow`
- Year: 2026
- Reference: https://arxiv.org/abs/2603.15689
- Integration level: `runtime_pilot`
- Default claim status: `exploratory`

## Core idea

Learn a global transition flow rather than only a local velocity field.

## PHM integration scope

runtime pilot.

## Shape contract

```text
signal/noise/output: [N,C,L]
condition.fault_label: [N]
condition.domain_id: [N]
```

## Promotion requirements

- method-specific loss
- method-specific sampler/map
- unit test and finite gradients
- dummy E2E evidence
- sample/eval/stage manifests
- reviewer PASS

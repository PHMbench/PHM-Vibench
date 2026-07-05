# Euler Mean Flows

- Method ID: `euler_mean_flow`
- Family: `mean_flow`
- Year: 2026
- Reference: https://arxiv.org/abs/2602.02571
- Integration level: `runtime_pilot`
- Default claim status: `exploratory`

## Core idea

JVP-free long-range trajectory-consistency surrogate for one/few-step generation.

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

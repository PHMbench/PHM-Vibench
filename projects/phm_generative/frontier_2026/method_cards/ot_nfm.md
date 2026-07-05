# OT-NFM

- Method ID: `ot_nfm`
- Family: `one_step_flow_map`
- Year: 2026
- Reference: https://arxiv.org/abs/2604.06413
- Integration level: `runtime_pilot`
- Default claim status: `exploratory`

## Core idea

Learn the transport map directly with consistent optimal-transport pairings.

## PHM integration scope

first runtime pilot.

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

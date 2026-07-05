# DiM-TS

- Method ID: `dim_ts`
- Family: `diffusion_ssm`
- Year: 2025
- Reference: https://arxiv.org/abs/2511.18312
- Integration level: `backbone_pilot`
- Default claim status: `exploratory`

## Core idea

Diffusion with selective state-space modeling for long temporal and cross-channel structure.

## PHM integration scope

backbone pilot; real SSM required.

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

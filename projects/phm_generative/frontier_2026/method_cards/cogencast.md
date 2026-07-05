# CoGenCast

- Method ID: `cogencast`
- Family: `autoregressive_flow_forecasting`
- Year: 2026
- Reference: https://arxiv.org/abs/2602.03564
- Integration level: `project_card_only`
- Default claim status: `exploratory`

## Core idea

Couple an autoregressive context model with flow matching for stochastic forecasting.

## PHM integration scope

project-card only.

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

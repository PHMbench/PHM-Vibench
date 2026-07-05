# Path-dependent Physical Field One-Step FM

- Method ID: `physical_field_one_step_fm`
- Family: `latent_physical_field_flow`
- Year: 2026
- Reference: https://arxiv.org/abs/2606.22752
- Integration level: `project_card_only`
- Default claim status: `exploratory`

## Core idea

Latent transformer flow matching for geometry/loading-conditioned path-dependent fields.

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

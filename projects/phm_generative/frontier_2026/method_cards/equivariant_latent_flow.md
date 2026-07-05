# Equivariant Latent Flow

- Method ID: `equivariant_latent_flow`
- Family: `latent_flow_matching`
- Year: 2026
- Reference: https://arxiv.org/abs/2601.22848
- Integration level: `runtime_pilot_later`
- Default claim status: `exploratory`

## Core idea

Regularize an autoencoder latent space for translation/amplitude equivariance before latent flow matching.

## PHM integration scope

later runtime pilot.

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

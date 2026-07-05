# Wasserstein Gradient Flow

- Method ID: `w_flow`
- Family: `wasserstein_gradient_flow`
- Year: 2026
- Reference: https://arxiv.org/abs/2605.11755
- Integration level: `research_pilot`
- Default claim status: `exploratory`

## Core idea

Compress a Sinkhorn-divergence Wasserstein gradient-flow evolution into a one-step generator.

## PHM integration scope

toy/research pilot; requires Sinkhorn and distribution-level evidence.

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

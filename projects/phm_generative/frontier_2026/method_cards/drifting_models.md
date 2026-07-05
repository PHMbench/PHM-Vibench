# Drifting Models

- Method ID: `drifting_models`
- Family: `distribution_drifting`
- Year: 2026
- Reference: https://arxiv.org/abs/2602.04770
- Integration level: `toy_only`
- Default claim status: `exploratory`

## Core idea

Evolve the pushforward distribution during training with a drifting field and use one-step inference.

## PHM integration scope

toy-only until a method-specific drifting field and stop-gradient target are implemented.

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

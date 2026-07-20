# Physics-Constrained MoE Innovation Contract

## Paper Identity

- Title: `Physics-Constrained MoE Explainable Fault Diagnosis`
- Thesis: make MoE diagnosis auditable by aligning experts with physical signal roles and forcing routing behavior to remain explainable across multiple datasets.

## Core Innovations

### 1. Physics-Aligned Expert Roles

- Innovation: each expert should map to interpretable signal-processing behavior rather than being an anonymous latent component.
- Why nontrivial: ordinary MoE accuracy does not prove expert specialization is meaningful.
- Required evidence:
  - accepted expert ablation
  - accepted routing analysis
  - expert role descriptions consistent with bound artifacts

### 2. Auditable Routing Metrics

- Innovation: route entropy, path signature, and expert activation distribution are treated as first-class comparison outputs.
- Why nontrivial: route visualization without quantitative metrics is too weak for review.
- Required evidence:
  - accepted routing metrics pack
  - accepted manuscript truth sync and evidence binding

### 3. High-Accuracy Multi-Dataset Diagnosis

- Innovation: the model must reach near-100% in-domain diagnosis on three maintained datasets while preserving routing interpretability.
- Why nontrivial: explainable routing is not enough if diagnosis quality collapses on real datasets.
- Required evidence:
  - accepted innovation bind ticket
  - accepted in-domain passes for `CWRU`, `XJTU`, `THU_006`
  - real-data accepted summaries with threshold pass fields and persisted logs

## Required Datasets

- `CWRU`
- `XJTU`
- `THU_006`

## Required Comparison Items

- `route_entropy`
- `path_signature`
- `expert_activation_distribution`

## In-Domain Accuracy Gate

- Target accuracy: `>= 0.98`
- Gate semantics: each required dataset must appear in `in_domain_98_pass`
- Supporting evidence that does not replace the gate:
  - seed stability
  - expert ablation
  - manuscript truth sync

## Current Status

- Status: `partial`
- Truth-first manuscript and routing evidence exist under the older contract.
- Remaining gaps:
  - link this contract into repo authorities
  - extend the real-data probe to `THU_006`
  - produce accepted `>=0.98` passes on all three datasets
  - rerun review/manuscript binding after the new gate

## Blocking Risks

- Current accepted real-data bridge evidence is below the strengthened accuracy gate.
- Probe-scale evidence must remain labeled as bounded support unless upgraded to full accepted in-domain runs.

# 1D-2D Fusion Explainable Innovation Contract

## Paper Identity

- Title: `1D-2D Fusion Explainable Fault Diagnosis`
- Thesis: combine time-domain and time-frequency diagnosis with a physically interpretable three-layer alignment mechanism that can reach near-ceiling in-domain accuracy without giving up quantitative explainability.

## Core Innovations

### 1. Physical-Semantic-Geometric Three-Layer Alignment

- Innovation: align 1D signal dynamics and 2D spectral structure with explicit physical, semantic, and geometric consistency constraints.
- Why nontrivial: most multimodal diagnosis stacks only concatenate features and do not expose why modalities agree.
- Required evidence:
  - accepted truth audit
  - accepted quantitative explainability pack
  - manuscript claims bound to accepted artifacts only

### 2. Interpretable 1D-2D Fusion Network

- Innovation: the fusion module explicitly tracks cross-modal contribution instead of treating the 2D branch as an opaque accuracy booster.
- Why nontrivial: strong performance alone does not prove the fusion mechanism is doing meaningful work.
- Required evidence:
  - accepted comparison suite against `MoE`, `TSPN`, `OperatorAttention`
  - accepted stability statistics and cross-dataset package

### 3. High-Accuracy Multi-Dataset Diagnosis

- Innovation: the method must maintain near-100% in-domain diagnosis on three maintained datasets while preserving explainability outputs.
- Why nontrivial: this paper is only convincing if the fusion method is both interpretable and operationally strong.
- Required evidence:
  - accepted innovation bind ticket
  - accepted in-domain passes for `CWRU`, `XJTU`, `THU_006`
  - per-dataset accepted runs recorded with `run_meta.yaml`, `metrics.json`, summary JSON, and logs

## Required Datasets

- `CWRU`
- `XJTU`
- `THU_006`

## Required Comparison Items

- `MoE`
- `TSPN`
- `OperatorAttention`

## In-Domain Accuracy Gate

- Target accuracy: `>= 0.98`
- Gate semantics: each required dataset must appear in `in_domain_98_pass`
- Supporting sections that do not replace the gate:
  - cross-dataset generalization
  - quantitative explainability
  - multi-seed stability

## Current Status

- Status: `partial`
- Truth-first manuscript sync is already in place for the old contract.
- Remaining gaps:
  - link this contract into repo authorities
  - produce accepted `>=0.98` runs for `CWRU`, `XJTU`, `THU_006`
  - rerun manuscript/evidence binding after the strengthened gate

## Blocking Risks

- Current accepted real-data summaries are far below the new target and cannot be restated as high-accuracy evidence.
- Any legacy manuscript number not mapped to an accepted artifact must stay removed.

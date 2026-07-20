# Explainable FD Toolkit Innovation Contract

## Paper Identity

- Title: `Explainable FD Toolkit`
- Thesis: build a reusable explainability operating layer for fault diagnosis so models, explainers, metrics, and reports can be compared under one reproducible contract.

## Core Innovations

### 1. Explainability OS For Fault Diagnosis

- Innovation: standardize `SignalData / ModelPlugin / ExplainabilityMethod` into one reusable interface boundary for PHM diagnosis.
- Why nontrivial: prior code paths are paper-local and incomparable; this paper turns explainability from ad hoc scripts into infrastructure.
- Required evidence:
  - accepted benchmark bootstrap
  - accepted 5-model unified matrix
  - schema-valid benchmark outputs and reproducible report artifacts

### 2. Reproducible Comparison Matrix For Explainability Methods

- Innovation: make `Captum`, `SHAP`, and `LIME` directly comparable under one artifact format instead of isolated one-off analyses.
- Why nontrivial: comparison methods usually differ in metrics, report shape, and runtime assumptions, which breaks fair comparison.
- Required evidence:
  - accepted `Captum` comparison pack
  - accepted `SHAP/LIME` comparison pack
  - bound comparison artifacts that can be cited from the manuscript

### 3. Multi-Dataset Explainability Execution Contract

- Innovation: the toolkit must expose one maintained explainability execution path that covers multiple datasets and remains schema-valid.
- Why nontrivial: infrastructure papers fail when they only work on a single synthetic example.
- Required evidence:
  - accepted dataset coverage on `CWRU`, `XJTU`, `THU_018_basic`
  - accepted innovation bind ticket linking the contract into `README.md`, `CORE.md`, and `paper_blueprint.md`

## Required Datasets

- `CWRU`
- `XJTU`
- `THU_018_basic`

## Required Comparison Items

- `Captum`
- `SHAP`
- `LIME`

## Accuracy Gate

- Target mode: `benchmark_proxy_high_accuracy`
- Required high-accuracy models: `TSPN`, `Fusion1D2D`
- Proxy threshold: benchmark-backed accuracy `>= 0.98`
- Note: this infrastructure paper currently uses maintained benchmark evidence plus multi-dataset execution coverage, not a standalone classifier-only gate.

## Current Status

- Status: `partial`
- Bound evidence already exists for model matrix, comparison methods, and demos.
- Remaining gaps:
  - link this contract into repo authorities
  - add maintained third-dataset evidence for `THU_018_basic`
  - reopen review under the strengthened 3-dataset contract

## Blocking Risks

- Current `run_benchmark_standalone.py` is synthetic and cannot by itself satisfy per-dataset real-data claims.
- Any manuscript claim about diagnosis accuracy must cite accepted artifact paths, not historical README numbers.

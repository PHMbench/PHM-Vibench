# Review Report 06: Normalization and Domain Evidence (Axis H)
## Reviewer: reviewer-norm-domain
## Date: 2026-06-10
## Branch: Feature_factory-update

## Axis H: Normalization/Domain Evidence
### Score: 4/5 — Benchmark-usable

## Normalization
### Method: standardization or robust_scaler only
Enforced by `resolve_normalization_method` in `src/data_factory/data_utils.py:24-43`.

### Scope: per_channel
Stats computed independently per channel at `Pipeline_06_generative.py:249-254`.

### Source split enforcement: Train only, hardcoded
String `"train"` hardcoded at `Pipeline_06_generative.py:237` (`data_factory.get_dataloader("train")`). Never parameterized from config — this is correct behavior preventing accidental val/test normalization.

### Artifact writing
`normalization_params.json` + `normalization_params.sha256` written to run directory (`data_utils.py:46-56`).

### Hash writing
SHA256 over raw JSON bytes, written as sidecar file.

## Domain Map
### Domain map hash: Required and enforced
`build_synthetic_data_manifest` raises `ValueError` if `domain_map_hash` is empty (`synthetic_data_manifest.py:73-74`).

### load/rpm handling: Correct — through domain_id only, NOT direct model conditions
Model conditions are strictly `fault_label` and `domain_id` only (`conditional_flow_matching.py:129-139`). load/rpm resolved through domain map for audit/reporting only.

## Manifest Evidence
### normalization_params in manifest: YES
`normalization_params_path` and `normalization_params_hash` read from `args_data` and included in manifest (`conditional_flow_matching.py:254-258`).

### condition_counts by fault/domain: YES
Keyed by `fault={label},domain={domain}` at `Pipeline_06_generative.py:325-334`.

## Benchmark-validity Gates
Nine evidence gates at `synthetic_data_manifest.py:103-113` including:
- normalization_params
- leakage_checks
- condition_counts
- config_hash
- protocol_hash
- domain_map_hash

Forbidden source splits rejected with hard `ValueError` before manifest creation.

## Non-blocking Issues
1. **Scope fallback mismatch**: `conditional_flow_matching.py:252` default is `"window"` but actual implementation is always `"per_channel"`. Overwritten by `_attach_normalization_artifacts` in practice, but the fallback string is inconsistent.

## Blocking Issues
**None.**

## Summary
- Normalization and domain evidence is benchmark-usable (4/5)
- Paper-ready gap: needs explicit test for the scope fallback, and potentially confidence interval / statistical tests on normalization stability
- Recommended goal: GOAL-FFU-PAPER-NORM-001 — fix scope fallback string consistency in conditional_flow_matching.py

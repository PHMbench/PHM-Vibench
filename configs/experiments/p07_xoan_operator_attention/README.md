# P07 — XOAN Operator Attention (experiment configs)

Bridge configs mapping P07 experimental arms to PHM-Vibench_fix runs.

All configs here:
- Compose the 5-block model (`environment/data/model/task/trainer`) from `configs/base/*`.
- Use registry-style component IDs and registered model rows from
  `src/model_factory/model_registry.csv` (`X_model/TSPN_UXFD`, `CNN/ResNet1D`, `X_model/MWA_CNN`).
- Default to smoke (1 epoch, CPU) — flip `trainer.num_epochs` and `trainer.device=cuda` for accepted runs.
- Are ADDITIVE: no `configs/base`, `configs/demo`, or `configs/reference` file is mutated.
- Do NOT mutate any source file under `paper/` (read-only).

## Files

| Config | Arm | Status | Derives from |
|---|---|---|---|
| `g030_executable_operator_path_smoke.yaml` | G030 typed executable operator-path wiring | ready (CPU software smoke; not C6-C9 evidence) | `configs/base/model/xoan_operator_path.yaml` + dummy DG bases |
| `p0_synthetic_operator_attention_smoke.yaml` | P0 synthetic operator-selection validation | ready (smoke; cannot verify CLAIM-SYN-* — simplified OperatorAttention1D, not manuscript DSOA) | legacy `configs/vibench/min.yaml` + `configs/demo/00_smoke/dummy_dg.yaml` |
| `case1_cwru_xoan_dg.yaml` | Case 1 industrial proxy — XOAN method arm | ready (simplified OA model) / needs_new_component (full DSOA) | `configs/demo/01_cross_domain/cwru_dg.yaml` |
| `case1_cwru_baselines_dg.yaml` | Case 1 baselines M1 (ResNet1D) + M4 (MWA-CNN) | ready for M1, M4; M2 (SincNet) + M3 (WKN) = needs_new_component | `configs/demo/01_cross_domain/cwru_dg.yaml` |
| `case2_dirg_xoan_dg.yaml` | Case 2 DIRG DG + SNR-sweep hook | ready (clean DG); needs_new_component (test-time SNR sampler) | `configs/demo/01_cross_domain/cwru_dg.yaml` |
| `ablation_cwru_xoan_dg.yaml` | Ablation matrix (L1 gamma, operator/FE subsets) | ready for rows a,b,d,e,f; blocked for (c) w/o residual | sibling of `case1_cwru_xoan_dg.yaml` |

Authoritative machine-readable map: `paper/experiments/config_bridge.yaml` (destination repo).

## Reused IDs / registered model rows (no new component required)

- Active P07 method: standalone `X_model/XOANOperatorPath` with
  `model.operator_path.*`. It uses a typed K-stage DAG chain, Avg+Var gates,
  continuous sparsemax relaxation, deterministic per-sample export, an
  effective-dictionary-bound path artifact, and an independent executor. Its
  post-training counterfactual API supports preregistered dormant-slot
  unmasking, removal, same-signature replacement, and stateless seeded output
  corruption. G030 verifies software wiring only; C6-C9 still require approved
  protocols and accepted runs.
- Method (simplified): `X_model/TSPN_UXFD` with `uxfd.operator_attention.enable=true`
  (uses `OperatorAttention1D`: mean-pool + Linear/ReLU/Linear gate + softmax; ops `{I, HT, FFT}`).
- Baseline M1: `CNN/ResNet1D` (1D ResNet-18).
- Baseline M4: `X_model/MWA_CNN` (multi-window attention CNN — closest registered proxy to the
  manuscript's "Discrete Wavelet Attention CNN" Wang2023; not byte-identical).
- Embedding/backbone/head IDs (when an ISFM/DLinear baseline is needed): `E_01_HSE`, `B_04_Dlinear`,
  `H_01_Linear_cla`.

## Required experiment components (not supplied by the G030 software smoke)

- **E7-E11 protocol runners and metric parsers**: recovery/equivalence,
  matched interventions, risk-coverage, industrial noninferiority, and
  reproducibility audit. The typed path component is implemented, but no
  claim is evidence-eligible until these protocols and thresholds are frozen.
- **Scientific-contract comparator runners** — dense decoder, discrete search,
  feature attention, a parameter-matched black box, and a random-dictionary
  control. Legacy SincNet/WKN placeholders are not C6-C9 contract comparators.
- **Test-time additive-Gaussian SNR-sweep sampler** — for the Case 2 noise-robustness curve
  (CLAIM-CASE2-NOISE-ROBUST, CLAIM-CASE2-NOISE-AS).

## Required data (NOT in metadata — blockers)

- **Zhang 2022 self-powered piezoelectric bearing dataset** (manuscript "Case 1"): NOT present
  in `data/metadata.xlsx` (only 19 datasets, none piezoelectric). All legacy Case 1 numbers
  (100% / 98.75% / 384.64k / 87.00k etc.) are quarantined narrative-only and cannot be verified
  until this dataset is ingested with a metadata row + reader.

## Leakage controls (recorded for every accepted run)

- `data.split.strategy: grouped_metadata` by `Domain_id` (workload/load); held-out target domain.
- CWRU/DIRG: `metadata.xlsx` has no `Bearing_id` column → Domain_id is the fallback group;
  bearing-level grouping is a recorded P2 blocker.
- Pretraining pool (none in P07 single-stage DG arms) never overlaps the held-out target.
- Normalization scope is recorded per-window in the split manifest (treat as evidence).

## Seeds

Accepted runs use `[20, 42, 100, 7, 31]` (>=5, replaces the legacy "n=5 but no artifact" policy).
Smoke runs use a single seed (`environment.seed: 0` or `42`).

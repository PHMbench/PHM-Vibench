# P06 — Verifiable Neural-Symbolic XFD (experiment configs)

Bridge configs mapping P06 experimental arms to PHM-Vibench_fix runs.

All configs here:
- Compose the 5-block model (`environment/data/model/task/trainer`) from `configs/base/*`.
- Use registry-style component IDs (`E_01_HSE` / `B_04_Dlinear` / `H_01_Linear_cla`).
- Default to smoke (1 epoch, CPU) — flip `trainer.num_epochs` and `trainer.device=cuda` for accepted runs.
- Are ADDITIVE: no `configs/base`, `configs/demo`, or `configs/reference` file is mutated.

## Files

| Config | Arm | Status | Derives from |
|---|---|---|---|
| `p1_reliability_cwru_dg.yaml` | P1 standard baseline (CWRU DG) | ready (needs Bearing_id metadata for full leakage control) | `configs/demo/01_cross_domain/cwru_dg.yaml` |
| `p1_symbolic_reliability_cwru_dg.yaml` | P1 symbolic-constrained variant | needs_new_component (symbolic head + constraint loss) | sibling of `p1_reliability_cwru_dg.yaml` |
| `p2_robustness_cwru_dg.yaml` | P2 standard baseline + noise sweep hook | ready (needs registered noise sampler for the sweep) | `configs/demo/01_cross_domain/cwru_dg.yaml` |
| `p2_robustness_physics_cwru_dg.yaml` | P2 physics-informed variant | needs_new_component (physics head + energy/freq loss + noise sampler) | sibling of `p2_robustness_cwru_dg.yaml` |
| `p2_transfer_xjtu_to_cwru_cddg.yaml` | P2 cross-dataset transfer | ready (CDDG demo path) | `configs/demo/02_cross_system/multi_system_cddg.yaml` |
| `p3_pareto_cwru_dg.yaml` | P3 Pareto Dlinear row (ISFM, low interp) | ready | `configs/demo/01_cross_domain/cwru_dg.yaml` |
| `p3_pareto_cwru_tspn.yaml` | P3 Pareto TSPN row (X_model, high interp) | ready | sibling of `p3_pareto_cwru_dg.yaml` |
| `p3_pareto_cwru_expcnn.yaml` | P3 Pareto BASE_ExplainableCNN row (X_model, mid interp) | ready | sibling of `p3_pareto_cwru_dg.yaml` |

All DG/CDDG configs use `data.split.strategy=grouped_metadata` + `test_policy=task_defined`
(required by `src/data_factory/splitting.py`: `partition` is only valid for `Default_task` /
`pretrain`; DG/CDDG must use `task_defined` with `fractions={train,val}`). Test set = the
task's `target_domain_id` (Arm A/B/D) or the transfer target system (Arm C).

## Reused IDs (no new components required)

- ISFM family: model `M_01_ISFM` (type `ISFM`); embedding `E_01_HSE`; backbone `B_04_Dlinear`; head `H_01_Linear_cla`
- X_model family (P3 Pareto rows): `X_model`/`TSPN`, `X_model`/`BASE_ExplainableCNN`
- Tasks: `DG`/`classification`, `CDDG`/`classification`
- Datasets (metadata.xlsx, `Dataset_id` stored as string): `'1'`=CWRU, `'2'`=XJTU, `'6'`=THU

## Required new components (NOT registered — blockers)

- Symbolic constraint head (`H_0X_Symbolic_cla`) + logic-rule constraint loss
- Physics-informed head (`H_0X_PhysicsInformed_cla`) + energy / frequency-smoothness / homomorphism auxiliary losses
- Test-time Gaussian noise sampler (for the P2 sigma sweep)
- `Bearing_id` column in `metadata.xlsx` (for leakage-safe grouped split beyond Domain_id)

## Verified dataset facts (read from `data/metadata.xlsx`, 2026-07-18)

- CWRU (`Dataset_id='1'`): 155 visible rows, `Domain_id in {'0','1','2','3'}` + 5 None; `Label in {0,1,2,3}`.
- XJTU (`Dataset_id='2'`): 9215 visible rows, `Domain_id in {'0','1','2'}`; mixed labels incl. -1.
- THU (`Dataset_id='6'`): 12 visible rows, `Domain_id in {'0','1','2'}`, `Label in {0,1,2,3}`.
- No `Bearing_id` column exists; only `Domain_id` is available for physical grouping.

Authoritative machine-readable map: `paper/experiments/config_bridge.yaml` (destination repo).

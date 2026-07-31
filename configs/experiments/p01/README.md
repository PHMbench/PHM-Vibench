# P01 - UXFD 1D-2D Multimodal Alignment - additive experiment configs

These configs are ADDITIVE: they live under `configs/experiments/p01/` and do NOT
modify any existing config under `configs/base/`, `configs/demo/`, or
`configs/reference/`.

## Purpose

Map every experimental arm of paper P01 (1D-2D physical-semantic-geometric
tri-level alignment XFD) to a PHM-Vibench_fix `main.py --config <yaml>` entrypoint.

The paper's full method (tri-level-aligned 1D+2D fusion) is the central novelty
claim. That model and its ablation variants are NOT yet registered in the
maintained `src/model_factory/model_registry.csv` - they exist only in the
READ-ONLY legacy snapshot under
`paper/UXFD_paper/1D-2D_fusion_explainable/.../code/models/`. Therefore:

- **Method arms (full + ablation)** are recorded as `needs_new_component` and
  bound to the placeholder traceability config `p01_method_placeholder.yaml`.
  They become runnable only after the Fusion1D2D family is ported into the
  maintained factory and given registry IDs `E_*/B_*/H_*`.
- **Baseline arms** (1D-only DLinear, 2D-only ResNet1D, simple-fusion concat,
  TSPN, OperatorAttention, MoE) reuse already-registered model components and
  have runnable configs below.
- **Supporting arms** (DG, CDDG, few-shot, pretrain) reuse the matching demo
  pipelines (`Pipeline_01_default`, `Pipeline_02_pretrain_fewshot`) with paper
  dataset/seed overrides.

## Files

- `p01_baseline_cwru_dlinear.yaml` - 1D single-modal baseline (ISFM/DLinear), CWRU.
- `p01_baseline_cwru_resnet1d.yaml` - 1D single-modal baseline (ResNet1D), CWRU.
- `p01_baseline_xjtu_dlinear.yaml` - 1D single-modal baseline (ISFM/DLinear), XJTU.
- `p01_baseline_xjtu_resnet1d.yaml` - 1D single-modal baseline (ResNet1D), XJTU.
- `p01_cross_system_cddg_cwru_xjtu.yaml` - cross-system generalization source=CWRU target=XJTU.
- `p01_fewshot_cwru_prototypical.yaml` - few-shot K-sweep on CWRU (FS task).
- `p01_cross_system_fewshot_tspn.yaml` - cross-system few-shot baseline (TSPN-style, GFS task).
- `p01_pretrain_hse_cddg.yaml` - HSE contrastive pretraining for CDDG (pretrain pool = source only).
- `p01_method_placeholder.yaml` - traceability binding for the full method +
  ablation arms; status `needs_new_component`.

## How to run a baseline smoke

```bash
python main.py --config configs/experiments/p01/p01_baseline_cwru_dlinear.yaml \
  --override trainer.num_epochs=1 --override data.num_workers=0
```

## Seed policy (paper-level, frozen at PROTOCOL_LOCK)

Default seed list for P01 is `[42, 123, 456, 789, 1024]` (5 seeds, matching the
few-shot design; >=5 for any paper-grade claim). For 3-seed arms use the subset
`[42, 123, 456]`. Set per-run via `--override environment.seed=<s>`.

## Leakage notes (binding)

- DG/CDDG arms: use `data.split.strategy: grouped_metadata` with
  `test_policy: task_defined` so the target domain/system is held out as test.
- Few-shot arms (FS/GFS): grouped_metadata episode-safe grouping is NOT yet
  defined for FS - see `configs/README.md`. Treat few-shot baseline numbers as
  exploratory until episode-safe grouping lands.
- Pretrain arms: the pretraining pool MUST be the source domain only (CWRU
  in-domain or CWRU train split). Pretraining over the target domain is
  forbidden leakage and is explicitly excluded by `target_system_id`.
- Normalization statistics are fit on the train split only (grouped split).

## Source provenance

- Legacy method code (READ-ONLY): `paper/UXFD_paper/1D-2D_fusion_explainable/`
  submodule - `code/models/{fusion_aligned,one_d_branch,two_d_branch,fusion_early}.py`,
  `code/alignment/{physical,semantic,geometric}_alignment.py`,
  `model/Fusion1D2D_ablation.py`.
- Legacy configs (incompatible with maintained CLI; recorded for traceability):
  `paper/UXFD_paper/1D-2D_fusion_explainable/configs/config_{CWRU,XJTU,THU_006}.yaml`
  and `configs/ablation/config_{1D_only,2D_only,no_statistical}.yaml` use the
  pre-5-block `args:` schema and are NOT runnable as-is.
- Maintained engine: PHM-Vibench_fix `main.py --config <yaml>`.

## Non-destructive attestation

- No file under `configs/base/`, `configs/demo/`, or `configs/reference/` was modified.
- No git commit / push was performed.
- The legacy `paper/UXFD_paper/1D-2D_fusion_explainable/` tree remains read-only.

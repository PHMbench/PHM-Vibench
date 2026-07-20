# P08 — HSE-Prompt CDDG experiment configs

Paper-bridge configs for **P08-HSE-Prompt-CDDG**. These are additive configs composed from `configs/base/*` using
registry-style component IDs (`E_*/B_*/H_*` or the registered `HSE_prompt` embedding for the ISFM_Prompt family).
They are the PHM-Vibench_fix-side runnable targets that map 1:1 to the experimental arms in this paper's
`paper/experiments/experiment_plan.md` and `config_bridge.yaml`.

## Source provenance (READ-ONLY)

These configs are derived (not copied verbatim) from the legacy CDDG configs under the **READ-ONLY** submodule
`paper/2025-10_foundation_model_0_metric/configs/CDDG_config/`:

- `experiment_0_cddg_baseline.yaml`  -> `p08_exp0_backbone_baseline.yaml`
- `experiment_1_cddg_hse.yaml`       -> `p08_exp1_hse_direct.yaml`
- `experiment_2_cddg_hse_pretrain.yaml` -> `p08_exp2_hse_pretrain.yaml`
- `experiment_3_cddg_hse_prompt.yaml`-> `p08_exp3_hse_prompt.yaml`  (the proposed method)
- `experiment_4_cddg_ablation.yaml`  -> `p08_exp4_ablation_base.yaml` (skeleton; per-row overrides documented in plan)
- `experiment_6_cddg_backbone.yaml`  -> `p08_exp6_backbone_sweep.yaml`
- `experiment_7_cddg_noise.yaml`     -> `p08_exp7_noise_robustness.yaml`

The legacy configs use the old keyspace (`trainer.max_epochs`, `trainer.devices`, `environment.VBENCH_HOME`) and
ambiguous IDs; the configs here re-encode the same experimental intent against the maintained 5-block model +
`base_configs:` composition + `trainer.num_epochs` keyspace so they pass `python -m scripts.validate_configs`.

## Canonical P08 protocol (locked here for the bridge)

- **Datasets (5, LOSO pool):** `[1, 13, 6, 12, 19]` = CWRU, Ottawa-19, THU, JNU, HUST24.
  - NOTE: the legacy manuscript prose says "Ottawa-23" in some tables; `Ottawa-23` is **Dataset_id 5**, NOT 12/13.
    `Dataset_id 13 = RM_017_Ottawa19`, `Dataset_id 12 = RM_016_JNU`. The CDDG run scripts (`run_experiment0_cddg.sh`,
    `run_experiment3_cddg.sh`) lock the 5-dataset pool to `[1,13,6,12,19]`; the bridge follows that and flags the
    prose mismatch as a P1 blocker (must reconcile against accepted run before promotion).
- **Per-dataset `model.input_dim` overrides** (from `run_experiment0_cddg.sh`): `1->2`, `13->2`, `6->1`, `12->1`, `19->3`.
- **Seed policy:** 5 seeds `[42, 123, 456, 789, 999]` (matches `run_experiment3_cddg.sh` and the manuscript's
  5-seed statistical-rigor claim). The base configs default to `seed: 42`; the other 4 seeds are supplied via
  `--override environment.seed=<s>` at launch.
- **Pipeline:** single-stage CDDG arms use `Pipeline_01_default`; pretrain arms use `Pipeline_02_pretrain_fewshot`
  with a `stages:` list (pretrain -> CDDG finetune).

## Smoke-friendly defaults

Every config here defaults to **1 epoch** so `python main.py --config <yaml>` is a fast smoke test.
Real training budgets are documented in `paper/experiments/experiment_plan.md` and applied at launch via
`--override trainer.num_epochs=<N>` (and `stages[k].trainer.num_epochs=<N>` for two-stage arms).

## Leakage flags (must be resolved at P08-G050 before any claim promotion)

- **LOSO target leakage:** the legacy configs train on the full 5-system pool with `target_domain_num: 1` and do
  not explicitly hold one system out as the *unseen* target. The bridge treats the current configs as
  `status: blocked` for every accuracy claim until P08-G050 produces an accepted leave-one-system-out split.
- **Normalization leakage:** `data.normalization: standardization` is computed on the full dataset; must be
  recomputed on the training pool only (fit-on-train, apply-to-target) before promotion.
- **Model selection on target:** `early_stopping` monitors `val_loss`; if val is drawn from the target domain
  this leaks target labels. Re-derive val split from the source pool only.
- **Hidden target access:** few-shot arms must freeze the support set away from the query/val set per episode
  (GFS sampler). Not yet verified against the data_factory.

## Usage

```bash
# Smoke (1 epoch, default seed 42)
python main.py --config configs/experiments/p08/p08_exp3_hse_prompt.yaml

# Real launch with seed sweep (representative)
for s in 42 123 456 789 999; do
  python main.py --config configs/experiments/p08/p08_exp3_hse_prompt.yaml \
    --override environment.seed=$s \
    --override stages[0].trainer.num_epochs=30 \
    --override stages[1].trainer.num_epochs=20
done
```

See `paper/experiments/experiment_plan.md` and `paper/experiments/config_bridge.yaml` in the destination repo
for the full per-arm plan.

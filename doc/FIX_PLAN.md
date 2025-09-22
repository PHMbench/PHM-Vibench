# Remediation Plan (Repository-Wide Review)

Comprehensive fixes distilled from the full code review to restore stable training pipelines, optional tooling guards, and working tests. Tackle in priority order.

## 1. Preserve Scalar `num_classes`
- Keep the configured scalar on `args_model` and expose the per-dataset map via a new attribute (e.g., `dataset_num_classes`).
- Update models that genuinely rely on the mapping (ISFM variants) to read the new attribute.
- ✅ Validation: run `python main.py --config configs/demo/Single_DG/CWRU.yaml` to ensure classifier heads instantiate.

## 2. Harden Default Trainer
- Wrap `swanlab` import in `try/except`; guard logger construction on module availability.
- Default `args.monitor` to `val_total_loss` when missing before creating callbacks (checkpoint + early stopping).
- ✅ Validation: same smoke run as above plus a CPU-only config to confirm wandb/swanlab flags behave.

## 3. Fix Few-Shot Iteration Control
- Replace `os.environ.get('iterations', 1)` with `int(args_environment.iterations)` (fallback to env if CLI overrides).
- ✅ Validation: run `Pipeline_02_pretrain_fewshot` with `iterations=2` and check both pretraining and few-shot loops execute twice.

## 4. Correct Unified Metric Checkpoint Flow
- In `Pipeline_04_unified_metric`, keep the `load_best_model_checkpoint` return value for evaluation but also store the `best_model_path` from the trainer.
- Reuse the saved path for Stage 2 weight loading and guard against missing checkpoints.
- ✅ Validation: execute `script/unified_metric/test_unified_1epoch.sh` (or equivalent single-epoch config) and confirm Stage 2 runs.

## 5. Align Two-Stage Finetuning Helpers
- Pass the pretrained checkpoint path explicitly into `create_finetuning_config` and update the helper call sites.
- Restore lightweight wrappers (or adjust tests) so `_create_pretraining_config/_create_finetuning_config` references no longer break `--test` mode.
- ✅ Validation: `python src/Pipeline_03_multitask_pretrain_finetune.py --test` and a `--stage complete` run using a minimal config.

## 6. Normalise File IDs in Default Task
- Ensure `IdIncludedDataset` emits integer IDs (or coerce within `_shared_step` before metadata lookup) so `.item()` isn’t invoked on strings.
- ✅ Validation: rerun the DG pipeline and confirm batches process without AttributeError.

## 7. Repair Checkpoint Loader Utility
- In `model_factory.load_ckpt`, unwrap `checkpoint['state_dict']` before filtering, and log skipped keys once.
- ✅ Validation: set a `weights_path` in a demo config and verify parameters change post-load (optionally compare tensors).

## 8. Document & Track
- Update `FIX_PLAN.md` (this file) and optionally add inline TODOs where invasive follow-up refactors are needed.
- After each fix, rerun core smoke tests and consider `pytest test/ -k basic` to ensure imports still succeed.

## Open Question
- Confirm with maintainers whether `create_finetuning_config` should always accept a separate `pretrained_checkpoint` argument or infer it from `finetuning_config`. Hold off on altering the helper signature until clarified.

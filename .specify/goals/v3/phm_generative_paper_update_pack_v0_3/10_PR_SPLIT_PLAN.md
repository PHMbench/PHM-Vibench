# 10. PR Split Plan for v0.3

The branch is too broad for one clean merge.  Use this split plan.

## PR-A: Branch hygiene and process artifact freeze

Scope:

```text
.specify/goals/v2/
specs/002-phm-genbench-frontier/
docs/reports/root_directory_cleanup.md
```

Out of scope:

```text
runtime code
model/task/metric changes
```

Validation:

```bash
python -m scripts.validate_docs
```

## PR-B: Entry path and preflight

Scope:

```text
main.py
test/test_hydra_config_compat.py
test/test_main_config_errors.py
```

Validation:

```bash
python main.py --config configs/demo/00_smoke/dummy_dg.yaml --preflight-only
python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml --preflight-only
python -m pytest test/test_hydra_config_compat.py test/test_main_config_errors.py
```

## PR-C: Core generative runtime

Scope:

```text
Pipeline_06_generative.py
task_factory/task/generative/conditional_flow_matching.py
task_factory/task/generative/rectified_flow.py
task_factory/task/generative/ddpm_epsilon.py
Components/generative/losses
Components/generative/samplers
```

Validation:

```bash
python -m pytest test/generative/test_flow_matching_loss.py
python -m pytest test/generative/test_rectified_flow_loss.py
python -m pytest test/generative/test_ddpm_loss.py
python -m pytest test/generative/test_pipeline_train_sample_eval_smoke.py
```

## PR-D: Models

Scope:

```text
model_factory/generative_model/
model_registry.csv
```

Validation:

```bash
python -m pytest test/generative/test_generative_model_forward.py
python -m pytest test/generative/test_unet1d_length_contract.py
python -m pytest test/generative/test_dit1d_patch_contract.py
```

## PR-E: Metrics and manifest

Scope:

```text
Components/generative/metrics
Components/generative/manifests
task/generative/generative_eval.py
```

Validation:

```bash
python -m pytest test/generative/test_synthetic_data_manifest.py
python -m pytest test/generative/test_metric_status_annotations.py
python -m pytest test/generative/test_leakage_metrics.py
```

## PR-F: Paper configs and scripts

Scope:

```text
configs/paper/phm_generative/
scripts/generative_benchmark_effect.py
scripts/paperpack_generative.py
scripts/generative_submission_draft.py
```

Validation:

```bash
python -m scripts.generative_benchmark_effect \
  --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml \
  --dry-run --allow-missing-data

python -m pytest test/scripts/test_generative_benchmark_effect.py
python -m pytest test/scripts/test_paperpack_generative.py
python -m pytest test/scripts/test_generative_submission_draft.py
```

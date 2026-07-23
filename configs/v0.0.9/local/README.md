# Local Override Configs (Machine-Specific)

Place machine-specific minimal override YAMLs here to adapt paths like `data.data_dir` across devices without editing the main experiment YAMLs.

> ⚠️ **Note (v0.1.0)**: This folder and README describe the legacy v0.0.9-style `local_config` mechanism.  
> For maintained configurations, use `base_configs` and the precedence documented in [`configs/README.md`](../../README.md),  
> not legacy `--local_config` examples.

Legacy lookup order (v0.0.9):
1. Explicit CLI: `--local_config /path/to/override.yaml`
2. Default: `configs/local/local.yaml`

Only include keys you want to override. Example (legacy style):

```yaml
# configs/local/local.yaml
data:
  data_dir: "/home/user/data/PHMbenchdata/PHM-Vibench"
  metadata_file: "metadata_6_11.xlsx"   # optional override if differs

# You may also set trainer/environment fields per host if needed
# trainer:
#   accelerator: "cpu"
# environment:
#   PROJECT_HOME: "/home/user/LQ/B_Signal/Signal_foundation_model/Vbench"
```

In v0.0.9 you could also pass an explicit path via CLI:
- Pipeline_01_default: `--local_config configs/local/local.yaml`
- Pipeline_02_pretrain_fewshot: `--local_config configs/local/local.yaml`
- Pipeline_03_multitask_pretrain_finetune: `--local_config configs/local/local.yaml`
- Pipeline_ID: inherits Pipeline_01_default behavior

For maintained configurations, prefer:
- portable environment fields in the relevant `configs/base/` blocks;
- `configs/local/local.yaml` for untracked machine values; and
- repeatable `--override key=value` arguments for explicit run-time changes.

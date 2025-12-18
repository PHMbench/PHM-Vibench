# Local Override Configs (Machine-Specific)

Place machine-specific minimal override YAMLs here to adapt paths like `data.data_dir` across devices without editing the main experiment YAMLs.

> ⚠️ **Note (v0.1.0)**: This folder and README describe the legacy v0.0.9-style `local_config` mechanism.  
> In v0.1.0, the recommended way to adapt paths is via `base_configs` in YAML (see `configs/readme.md`),  
> not via `--local_config` CLI flags.

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

For v0.1.0, please prefer:
- putting all environment fields (including `PROJECT_HOME`) into `configs/base/environment/base.yaml`, and
- using `base_configs.environment` + `--override` instead of `--local_config`.

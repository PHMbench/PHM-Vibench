# Local Override Configs (Machine-Specific)

Place machine-specific minimal override YAMLs here to adapt paths like `data.data_dir` across devices without editing the main experiment YAMLs.

Lookup order used by all pipelines:
1. Explicit CLI: `--local_config /path/to/override.yaml`
2. Default: `configs/local/local.yaml`

Only include keys you want to override. Example:

```yaml
# configs/local/local.yaml
data:
  data_dir: "/home/user/data/PHMbenchdata/PHM-Vibench"
  metadata_file: "metadata_6_11.xlsx"   # optional override if differs

# You may also set trainer/environment fields per host if needed
# trainer:
#   accelerator: "cpu"
# environment:
#   VBENCH_HOME: "/home/user/LQ/B_Signal/Signal_foundation_model/Vbench"
```

You can also pass an explicit path via CLI:
- Pipeline_01_default: `--local_config configs/local/local.yaml`
- Pipeline_02_pretrain_fewshot: `--local_config configs/local/local.yaml`
- Pipeline_03_multitask_pretrain_finetune: `--local_config configs/local/local.yaml`
- Pipeline_ID: inherits Pipeline_01_default behavior

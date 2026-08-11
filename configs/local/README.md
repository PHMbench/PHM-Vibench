# Explicit machine-local configuration

`configs/local/` is an optional workspace for untracked machine-specific values such as
local data roots, scratch output directories, or device preferences.

## Core rule

PHMFactory does **not** automatically read `configs/local/local.yaml` or any other file in
this directory. A local YAML affects an experiment only when the user supplies it:

```bash
phmfactory preflight \
  --config configs/experiments/my_experiment.yaml \
  --local-config configs/local/my_machine.yaml

phmfactory \
  --config configs/experiments/my_experiment.yaml \
  --local-config configs/local/my_machine.yaml
```

The same explicit file must be present in both preflight and run commands. CLI overrides
still have highest precedence:

```text
base_configs
< experiment YAML
< explicit --local-config YAML
< --override key=value
```

Start from `configs/local/local.sample.yaml`, rename the copy to a machine-specific file,
and keep it untracked. Do not store credentials in experiment manifests or attach them to
bug reports.

## 中文说明

`configs/local/` 用于保存本机路径、临时输出目录和设备偏好等不应提交到 Git 的
配置。PHMFactory **不会自动读取** `configs/local/local.yaml`。只有命令中明确写出
`--local-config` 时，该文件才会参与配置合并：

```bash
phmfactory preflight \
  --config configs/experiments/my_experiment.yaml \
  --local-config configs/local/my_machine.yaml
```

这样可以避免同一条命令在两台机器上因为隐藏的本地文件而执行不同实验。正式运行时
必须再次显式提供同一个本地配置文件；`--override` 的优先级仍然最高。

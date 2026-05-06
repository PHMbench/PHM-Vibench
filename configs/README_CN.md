# 配置系统说明（中文）

本文件保留中文说明入口；英文主入口见 [`README.md`](README.md)。配置系统采用
5-block 模型：

- `environment`
- `data`
- `model`
- `task`
- `trainer`

推荐入口：

```bash
python main.py --config <yaml> [--override key=value ...]
```

## 目录结构

- `base/`：基础模板，按 `data/model/task/trainer/environment` 拆分。
- `demo/`：维护中的轻量示例实验，优先从这里复制模板。
- `experiments/`：本地实验配置。
- `local/`：机器本地覆盖配置，不应承载可复现实验逻辑。
- `reference/`：历史参考配置，不建议作为新实验模板。

## Base 配置模板

常用 base 文件：

| 类别 | 路径示例 | 用途 |
|---|---|---|
| data | `configs/base/data/base_classification.yaml` | 单数据集分类 / DG |
| data | `configs/base/data/base_cross_domain.yaml` | 单数据集 cross-domain |
| data | `configs/base/data/base_cross_system.yaml` | 多系统 CDDG |
| data | `configs/base/data/base_fewshot.yaml` | 单系统 few-shot |
| model | `configs/base/model/backbone_dlinear.yaml` | ISFM + HSE + DLinear |
| model | `configs/base/model/backbone_transformer.yaml` | Transformer baseline |
| task | `configs/base/task/classification.yaml` | 分类 / 简单 DG |
| task | `configs/base/task/dg.yaml` | cross-domain DG |
| task | `configs/base/task/cddg.yaml` | cross-system CDDG |
| task | `configs/base/task/fewshot.yaml` | few-shot |
| task | `configs/base/task/pretrain.yaml` | HSE / ISFM 预训练 |
| trainer | `configs/base/trainer/default_single_gpu.yaml` | 默认单 GPU Trainer |
| environment | `configs/base/environment/base.yaml` | 项目路径与运行环境默认值 |

## Demo 组合方式

Demo YAML 通常通过 `base_configs` 组合基础块，然后在本文件内覆盖少量字段：

```yaml
base_configs:
  environment: configs/base/environment/base.yaml
  data: configs/base/data/base_cross_domain.yaml
  model: configs/base/model/backbone_dlinear.yaml
  task: configs/base/task/dg.yaml
  trainer: configs/base/trainer/default_single_gpu.yaml
```

优先级从低到高：

1. `base_configs.*`
2. demo YAML 自身字段
3. `configs/local/local.yaml` 或 `--local_config`
4. CLI `--override key=value`

## Registry 与 Atlas

- 配置注册表：`configs/config_registry.csv`
- 注册表字段说明：`docs/config_registry_schema.md`
- 生成的人类可读索引：`docs/CONFIG_ATLAS.md`

维护流程：

```bash
python -m scripts.gen_config_atlas
python -m scripts.validate_configs
python -m scripts.config_inspect --config configs/demo/00_smoke/dummy_dg.yaml --override trainer.num_epochs=1
```

新增 demo 或调整 base 时，应同步更新 `configs/config_registry.csv` 并重新生成
`docs/CONFIG_ATLAS.md`。

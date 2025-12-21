# 7 篇 Paper 的 submodule 配置规范（v1）

目标：每篇 paper 的“实验配置 + 个性化变体”都放在各自 submodule 内，主仓库不被 paper 细节弄乱；但运行方式仍严格
遵循 PHM‑Vibench 的 config‑first 模式：`python main.py --config <yaml> --override ...`。

---

## 1) 目录规范（每个 submodule 必须一致）

在 `paper/UXFD_paper/<paper_id>/` 内建议固定：
```
configs/
  vibench/
    min.yaml            # 必须：最小可跑（1 epoch 可通）
    paper_main.yaml      # 建议：论文主结果配置（可多 seed）
    ablations/
      ...                # 可选：消融
  personal/
    ...                  # 可选：个人临时配置（不进入主线）
VIBENCH.md               # 必须：唯一复现入口说明
README.md                # 必须：submodule 顶层职责边界与目录说明
```

README 规则：
- 每次新增/调整一个目录（例如 `configs/vibench/ablations/`），必须同步更新该目录下的 `README.md`
- 目的：让“目录职责边界”不依赖口口相传，避免本科生改动后结构漂移

---

## 2) YAML 写法原则（减少重复、保持 vibench 口径）

### 2.1 必须是 5‑block

每个 vibench config 必须包含：
- `environment`
- `data`
- `model`
- `task`
- `trainer`

### 2.2 优先用 base_configs 继承主仓库模板

建议在 submodule configs 里复用主仓库的 base configs，例如：
- `configs/base/environment/base.yaml`
- `configs/base/data/base_classification.yaml`
- `configs/base/model/backbone_transformer.yaml`（或你们新增的 UXFD base）
- `configs/base/task/*.yaml`
- `configs/base/trainer/default_single_gpu.yaml`

submodule 只覆盖差异项（operators / fusion / fuzzy / explainer 选择 / metadata 要求）。

### 2.3 禁止硬编码绝对路径

所有数据路径用：
- `configs/local/local.yaml` 或
- CLI：`--override data.data_dir=/path/to/...`

避免把机器路径写进 submodule configs（会导致他人无法复现）。

---

## 3) 如何在主仓库运行 submodule config

在主仓库根目录执行（路径指向 submodule 内 YAML）：
```bash
python main.py --config paper/UXFD_paper/<paper_id>/configs/vibench/min.yaml --override trainer.num_epochs=1
```

说明：
- submodule 未 init 时，该路径不存在；但这不应影响主仓库 demos/tests。
- paper 相关运行属于研究流程，不作为主仓库“维护性验证门槛”。

---

## 4) `VIBENCH.md` 必须写清楚什么

每篇 paper 的 `paper/UXFD_paper/<paper_id>/VIBENCH.md` 至少包含：
- 一键命令：指向本 submodule 内的 `configs/vibench/min.yaml`（建议 `--override trainer.num_epochs=1`）
- explain 的 metadata 最小需求（采样率/通道名/工况/domain/窗口等）
- 证据链产物检查：`config_snapshot.yaml`、`artifacts/manifest.json`、`eligibility.json`、`data_metadata_snapshot.json`
- 与 submodule 原始复现脚本的差异（若有）

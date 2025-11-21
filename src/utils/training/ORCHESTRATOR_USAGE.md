# MultiStageOrchestrator / TwoStageOrchestrator 使用说明

本文档说明如何使用新的多阶段调度器来串联多阶段训练（两阶段是特殊情况）。

## 1. 核心入口
- `MultiStageOrchestrator`：通用多阶段调度器，支持任意 `stages=[...]`。
- `TwoStageOrchestrator`：`MultiStageOrchestrator` 的二阶段别名，接口保持 `run_complete()` 返回 `{'stage_1': ..., 'stage_2': ...}`。

## 2. 配置输入格式（推荐优先：单 YAML + stages.overrides）

### 2.1 推荐格式：单 YAML + `stages.overrides`
```python
unified = {
    "stages": [
        {   # stage 1
            "environment": {"stage_name": "pretrain"},
            "data": {...},
            "model": {...},
            "task": {...},
            "trainer": {...},
        },
        {   # stage 2
            "environment": {"stage_name": "finetune"},
            "data": {...},
            "model": {...},
            "task": {...},
            "trainer": {...},
        },
        # ... 可继续增加 stage 3/4/...
    ]
}
orch = MultiStageOrchestrator(unified)
results = orch.run_all_stages()
```

### 2.2 兼容格式：`stage_1` / `stage_2`（二阶段传统配置）
```python
unified = {
    "stage_1": {"data": {...}, "model": {...}, "task": {...}, "trainer": {...}},
    "stage_2": {"data": {...}, "model": {...}, "task": {...}, "trainer": {...}},
}
orch = TwoStageOrchestrator(unified)
summary = orch.run_complete()   # 返回 {'stage_1': {...}, 'stage_2': {...}}
```

> 说明：若传入 dict 且没有 `stages`，会自动把 `stage_1`/`stage_2` 填入 `stages` 列表，并保留 `cfg.stage_1`/`cfg.stage_2` 便于 CLI 覆盖。

## 3. 调用示例

### 3.1 多阶段（3 阶段示例）
```python
from src.utils.training.two_stage_orchestrator import MultiStageOrchestrator

unified = {
    "stages": [
        {"environment": {"stage_name": "pretrain"}, "data": d1, "model": m1, "task": t1, "trainer": tr1},
        {"environment": {"stage_name": "cddg_ft"}, "data": d2, "model": m2, "task": t2, "trainer": tr2},
        {"environment": {"stage_name": "domain_adapt"}, "data": d3, "model": m3, "task": t3, "trainer": tr3},
    ]
}
orch = MultiStageOrchestrator(unified)
results = orch.run_all_stages()
# results 结构：{ "pretrain": {...}, "cddg_ft": {...}, "domain_adapt": {...}, "_ckpt_registry": {...} }
```

### 3.2 二阶段（老格式）
```python
from src.utils.training.two_stage_orchestrator import TwoStageOrchestrator

unified = {
    "stage_1": {"data": d1, "model": m1, "task": t1, "trainer": tr1},
    "stage_2": {"data": d2, "model": m2, "task": t2, "trainer": tr2},
}
orch = TwoStageOrchestrator(unified)
summary = orch.run_complete()
# summary: {'stage_1': {...}, 'stage_2': {...}}
```

### 3.3 双 YAML（P02 传统用法）
```python
from src.utils.config.pipeline_adapters import adapt_p02
from src.utils.training.two_stage_orchestrator import TwoStageOrchestrator

unified = adapt_p02(pretrain_cfg_path, fs_cfg_path, local_config=None)  # 返回 {'stage_1': ..., 'stage_2': ...}
orch = TwoStageOrchestrator(unified)
summary = orch.run_complete()  # 等价两阶段
```

### 3.4 单 YAML + `stages.overrides`（推荐）
1) YAML 示例（单文件）：
```yaml
data: {...}
model: {...}
task: {...}
trainer: {...}
stages:
  - name: pretrain
    environment: {stage_name: "pretrain"}
    overrides:
      task.type: "CDDG"          # 仅写差异
      trainer.max_epochs: 30
  - name: cddg_ft
    environment: {stage_name: "cddg_ft"}
    overrides:
      task.type: "CDDG"
      trainer.max_epochs: 20
      data.num_window: 64
```
2) 组装：
```python
from src.configs.config_utils import load_config
from src.utils.config_utils import apply_overrides_to_config
from src.utils.training.two_stage_orchestrator import MultiStageOrchestrator

base = load_config("my_unified.yaml")
stages_cfg = []
for stage in base.stages:
    cfg = base.copy()
    overrides = getattr(stage, "overrides", {})
    cfg = apply_overrides_to_config(cfg, overrides)
    cfg.environment = getattr(stage, "environment", None)
    stages_cfg.append(cfg)

unified = {"stages": stages_cfg}
orch = MultiStageOrchestrator(unified)
results = orch.run_all_stages()
```

## 4. 返回结果与 checkpoint
- `run_all_stages()` 返回字典：每个 `stage_name` 一个子字典，含 `checkpoint_path/metrics/path` 等；并附 `_ckpt_registry` 记录阶段名到 ckpt 路径的映射。
- `run_complete()` 返回 `{'stage_1': {...}, 'stage_2': {...}}`，适合二阶段视图。

## 5. 校验与错误处理
- Orchestrator 内部使用 `_validate_config_wrapper` 对每个 stage 检查必需字段（data/model/task）。缺关键节会抛错，不再 fallback。
- dry_run 模式：构造 `MultiStageOrchestrator(..., dry_run=True)`，仅走路径与日志，不实际训练。

## 6. 何时选用哪种接口
- 只跑两阶段且保持旧调用习惯：`TwoStageOrchestrator + run_complete()`。
- 多阶段或希望阶段名自定义：`MultiStageOrchestrator + run_all_stages()`（推荐）。
- 双 YAML（P02 遗留）：`adapt_p02(...) -> TwoStageOrchestrator`。
- 单 YAML 多阶段（推荐未来）：使用 `stages` 列表 + overrides 按上述示例组装后交给 `MultiStageOrchestrator`。

## 7. 常见问题
- **为什么需要 `stages` 列表？**  
  统一多阶段入口，减少为每个实验写专用 orchestrator 的重复。
- **还能用 `stage_1.task.lr` 覆盖吗？**  
  可以，`TwoStageOrchestrator` 会保留 `stage_1`/`stage_2` 属性；多阶段时推荐 `stages[i].task.lr`。
- **能否恢复任意阶段？**  
  当前 `_ckpt_registry` 已记录阶段 ckpt，恢复逻辑可在 Pipeline 层根据需要扩展（预留 `resume_stage_name`/`resume_checkpoint` 字段）。

> 提醒：优先采用“单 YAML + stages.overrides”减少重复配置；双 YAML 仅作兼容。

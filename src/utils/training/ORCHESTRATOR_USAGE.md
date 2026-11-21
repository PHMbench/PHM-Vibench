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

### 3.4 公共入口与训练-only阶段

下面是合并到一个完整实验配置中的局部示例；顶层 `data/model/task/trainer/environment`
仍需提供实际任务参数。阶段内 `overrides` 使用嵌套映射，不需要手工再次解析 YAML。

```yaml
pipeline: Pipeline_02_Pretraining_Few_Shot
trainer:
  test_after_fit: true
stages:
  - name: pretrain
    overrides:
      trainer:
        test_after_fit: false
      data:
        load_test: false
  - name: finetune
    overrides: {}
  - name: further_adapt
    overrides: {}
```

完整配置统一使用：

```bash
phmfactory --config path/to/complete_multistage.yaml
```

Pipeline 02 执行全部声明阶段。`trainer.test_after_fit` 未填写时保持原有的评价行为；
显式 `false` 才表示仅训练。`data.load_test: false` 不能与需要评价的阶段并用。
合法 early stopping 属于正常训练结束，不要求机械跑满最大 epoch。

## 4. 阶段结果与权重交接

- `run_all_stages()` 按声明顺序返回每个阶段及既有 `_ckpt_registry`。
- 需评价阶段返回 `status: evaluated`、实际 `checkpoint_path/path` 和来自
  `trainer.test` 的非空有限 `metrics`。多个人口的未命名结果不会截取第一份后继续。
- 仅训练阶段返回 `status: trained_not_evaluated` 和选中的状态，**不含测试 metrics**。
- 每阶段使用独立输出目录，避免短阶段在同一秒生成同名 checkpoint 而相互覆盖。
- 下游继承上一阶段实际选中的状态；必须产出状态的阶段失败时，不沿用更早的状态。
- 这是权重迁移，不是恢复训练：下游重新构建 Task/Trainer 及优化器。原有显式
  模型头匹配规则保持不变，不新增自动重建不兼容 head 的兜底。
- 自动阶段交接与同一阶段的 `model.weights_path` 不能同时指定两个不同来源。
- 公共阶段序列不以阶段下标偏移随机种子；每阶段继承或显式覆盖自己的
  `environment.seed`。这保持原两阶段公共入口的 seed 语义。

## 5. 校验与兼容边界

阶段名必须非空、无首尾空白、唯一且不以下划线开头。配置覆盖、顺序、阶段完成数量
以及必需 checkpoint 会被核对；中间阶段失败不会继续执行依赖它的后继阶段。

`run_complete()` 仅提供一到两阶段的 `stage_1/stage_2` 兼容视图，超过两个阶段请使用
`run_all_stages()`；普通 Pipeline 02 已使用后者。双 YAML 仍需显式选择原有 legacy 模式。

`dry_run=True` 不训练、不创建输出、不返回伪指标，状态是 `not_executed`；它不是
已完成运行。完整训练恢复、数据游标恢复、采样或在线阶段编排不由本功能自动提供。

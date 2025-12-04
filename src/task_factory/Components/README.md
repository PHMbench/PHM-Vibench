# Task Components Library

本目录包含任务所需的可复用组件，目的是解耦“任务逻辑”与“数学计算”。建议所有 Task 通过工厂接口（如 `get_loss_fn` / `get_metrics`）使用组件，避免硬编码。

## 目录与职责
- `loss.py`：Loss 工厂，统一 `get_loss_fn(name)` 接口。
- `contrastive_losses.py`：对比/度量学习损失实现（InfoNCE, SupCon, Triplet, BarlowTwins, VICReg 等）。
- `contrastive_strategies.py`：对比学习策略架构，支持 HSE Prompt 集成和系统感知对比学习，详见 [CONTRASTIVE_STRATEGIES.md](./README_CONTRASTIVE_STRATEGIES.md)。
- `metrics.py`：评估指标工厂，封装 torchmetrics。
- `regularization.py`：正则化工具（L1/L2/Domain penalty/mixup 等）。
- 其他：prompt_contrastive、metric_loss 等高级模块。

## Loss 一览（常用关键字）
> 配置中通过 `task.loss` 指定；对比学习多为 **embedding 输入**，需确保输入格式正确。

### 监督/常规
| Key  | Class                   | 说明                   |
| ---- | ----------------------- | ---------------------- |
| CE   | nn.CrossEntropyLoss     | 多分类                 |
| MSE  | nn.MSELoss              | 回归/预测             |
| BCE  | nn.BCEWithLogitsLoss    | 二分类/多标签         |

### 对比/度量
| Key          | Class            | 说明                           | 典型输入                       |
| ------------ | ---------------- | ------------------------------ | ------------------------------ |
| INFONCE      | InfoNCELoss      | 自监督对比；需正样本对         | features (或 features+labels)  |
| SUPCON       | SupConLoss       | 监督对比；同 label 为正样本    | features + labels              |
| TRIPLET      | TripletLoss      | 三元组损失；需 margin          | features + labels              |
| PROTOTYPICAL | PrototypicalLoss | 原型/度量学习                  | features + labels              |
| BARLOWTWINS  | BarlowTwinsLoss  | 冗余减少；两视图               | z1, z2                         |
| VICREG       | VICRegLoss       | 另一种两视图自监督             | z1, z2                         |

⚠️ 注意事项：
- InfoNCE/SupCon 需要“正样本对”存在：监督对比需 batch 内有重复 label；自监督需双视图或其他配对逻辑。单视图+唯一标签会得到 0 loss。
- Triplet 需要合理的 margin；BarlowTwins/VICReg 需要两视图输入。
- prompt_contrastive 的 `base_loss_type` 只接受文档列出的 key，额外参数请匹配底层 loss。

## 在 HSE 对比预训练任务中通过超参选择 loss

`pretrain/hse_contrastive.py` 通过 `contrastive_strategies.py` 内的策略管理器，支持以下两种配置方式：

### 1. 简单模式：单一对比损失

在 YAML 的 task 段中使用 `contrast_loss` 选择具体 loss 类型：

```yaml
task:
  type: "pretrain"
  name: "hse_contrastive"

  # 选择对比损失类型（关键超参）
  contrast_loss: "INFONCE"         # 或 "SUPCON" / "TRIPLET" / "PROTOTYPICAL" / "BARLOWTWINS" / "VICREG"

  # 通用权重
  contrast_weight: 1.0             # 对比 loss 权重
  classification_weight: 0.0       # 设为 0 表示纯对比预训练；>0 表示对比+CE 混合

  # 具体 loss 的超参（按需）
  temperature: 0.07                # INFONCE / SUPCON 用
  margin: 0.3                      # TRIPLET 用
  barlow_lambda: 5e-3              # BARLOWTWINS 用
```

内部会自动构造：

```python
contrastive_config = {
    "type": "single",
    "loss_type": args_task.contrast_loss,
    "temperature": args_task.temperature,
    "margin": args_task.margin,
    "barlow_lambda": args_task.barlow_lambda,
    ...
}
strategy_manager = create_contrastive_strategy(contrastive_config)
```

### 2. 高级模式：多损失组合 (ensemble)

如果需要组合多种对比损失，可以显式提供 `contrastive_strategy`：

```yaml
task:
  type: "pretrain"
  name: "hse_contrastive"

  contrastive_strategy:
    type: "ensemble"
    losses:
      - loss_type: "INFONCE"
        weight: 0.6
        temperature: 0.07
      - loss_type: "SUPCON"
        weight: 0.4
        temperature: 0.07
```

此时会忽略 `contrast_loss`，直接按 `losses` 列表构建 `EnsembleContrastiveStrategy`。

### 3. 对比 + 分类的组合方式

在 `hse_contrastive` 任务中，对比损失与 CE 分类损失通过权重组合：

- 纯对比预训练：`contrast_weight > 0` 且 `classification_weight = 0`  
- 单阶段“对比 + CE”训练：两者都 > 0  

下游分类阶段仍推荐使用 CDDG/FS/GFS 等专门的 classification 任务，`hse_contrastive` 主要用于预训练阶段 (`type="pretrain", name="hse_contrastive"`)。

## 使用示例（Task 内）
```python
from src.task_factory.Components.loss import get_loss_fn

class MyContrastiveTask(pl.LightningModule):
    def __init__(self, args_task, ...):
        super().__init__()
        self.loss_fn = get_loss_fn(args_task.loss)  # 例如 "INFONCE"

    def training_step(self, batch, batch_idx):
        feats = self.network(batch['x'])
        labels = batch.get('y')
        # 按 loss 类型传参
        if hasattr(self.loss_fn, '__call__'):
            try:
                loss = self.loss_fn(feats, labels)  # SupCon 需要 labels
            except TypeError:
                loss = self.loss_fn(feats)          # InfoNCE 无标签模式
        return loss
```

## Metrics 配置
在 YAML 中通过列表配置，例如：
```yaml
task:
  metrics: ["acc", "f1", "auroc"]
```
在代码中通过 `get_metrics` 取用（详见 `metrics.py`）。

## 规范建议
- 不要在 Task 里硬编码具体 Loss/Metric，统一通过工厂获取。
- 对比学习前确认数据流：是否提供双视图/重复标签；否则 InfoNCE/SupCon 会返回 0。
- 如需新组件，先注册到对应工厂并更新本 README，保持可查性。 

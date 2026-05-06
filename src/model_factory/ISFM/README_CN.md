# ISFM 中文补充说明

英文主入口见 [`README.md`](README.md)。本文件保留原 `readme.md` 中的中文/双语
说明，用于快速理解 ISFM 系列的输入输出约定、模型选择和配置方式。

## 输入 / 输出约定

- 主输入张量形状：`x: [batch_size, L, C]`
  - `L`：时间长度
  - `C`：通道数
- 常见前向调用：

```python
y = model(x, file_id=file_id_batch, task_id="classification", return_feature=False)
```

字段说明：

- `file_id`：来自 DataFactory 的样本 ID，用于从 metadata 查询 `Dataset_id` 与
  `Sample_rate`。
- `task_id`：当前任务类型，常见值包括 `"classification"` 和 `"prediction"`。
- `return_feature=True`：部分模型返回 `(logits, features)`，供对比学习或表示分析使用。

## 系统感知行为

- `M_01_ISFM` / `M_02_ISFM` 会根据 `file_id` 批量解析每个样本的系统信息。
- `H_01_Linear_cla` 按 system id 分组，将同一系统样本送入对应 head。
- `E_01_HSE` / `E_02_HSE_v2` 支持 per-sample `Sample_rate` / `Dataset_id`。
- 如果一个 batch 混合多个系统，优先考虑
  `M_02_ISFM_heterogeneous_batch + H_02_Linear_cla_heterogeneous_batch`。

## 模型选择

| 模型 | 适用场景 | 说明 |
|---|---|---|
| `M_01_ISFM` | 单系统或 sampler 保证单系统 batch | 最稳定的标准 ISFM |
| `M_02_ISFM` | 跨系统泛化、HSE 对比学习、多任务 | 支持系统感知 embedding 与条件向量 |
| `M_02_ISFM_heterogeneous_batch` | 一个 batch 内混合多个系统 | 真正 per-sample system id 处理 |
| `M_03_ISFM` | 研究原型、轻量实验 | 依赖少，便于快速迭代 |

简单决策：

```yaml
单数据集故障诊断:
  推荐: M_01_ISFM

跨系统 / Prompt / HSE 对比学习:
  推荐: M_02_ISFM

异构 batch:
  推荐: M_02_ISFM_heterogeneous_batch

快速研究原型:
  推荐: M_03_ISFM
```

## 配置示例

标准 CDDG 分类：

```yaml
model:
  type: "ISFM"
  name: "M_01_ISFM"
  embedding: "E_01_HSE"
  backbone: "B_04_Dlinear"
  task_head: "H_01_Linear_cla"
  patch_size_L: 256
  patch_size_C: 1
  num_patches: 128
  output_dim: 1024
  num_classes:
    0: 10
```

增强版跨系统设置：

```yaml
model:
  type: "ISFM"
  name: "M_02_ISFM"
  embedding: "E_02_HSE_rec"
  backbone: "B_08_PatchTST"
  task_head: "H_09_multiple_task"
  patch_size_L: 256
  patch_size_C: 1
  num_patches: 128
  output_dim: 1024
  d_model: 256
  n_layers: 4
  dropout: 0.1
```

## 维护约定

- ISFM 组件机器可读索引是 `src/model_factory/ISFM/isfm_components.csv`。
- 新增 `E_*` / `B_*` / `H_*` 组件时，同步更新英文 README、本文和 CSV。
- 中英文文件分工：
  - `README.md`：英文 canonical 说明。
  - `README_CN.md`：中文补充和迁移说明。

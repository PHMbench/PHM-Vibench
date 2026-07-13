-----

# dataset_task 模块说明（简体中文）

本目录存放的是 **面向任务的 Dataset 封装**，用于在「原始信号 + metadata」与 `task_factory` 里的任务模块之间建立桥梁：

- 负责把 `H5DataDict` 与 `metadata` 转换为 `torch.utils.data.Dataset`;
- 根据不同的 `task.type` / `task.name` 构造出对应的 batch 结构（如窗口化、episodic few-shot、预训练掩码等）；
- 被 `src/data_factory/data_factory.py` 动态导入，用于构建最终的 `DataLoader`。

> 完整的 **task ↔ dataset_task 映射表** 已统一收敛到  
> `src/task_factory/task_registry.csv`，建议以该表为唯一信息源维护。

-----

## 📂 目录概览

主要文件与子目录：

| 路径                              | 说明 |
| :-------------------------------- | :--- |
| `Default_dataset.py`              | 通用窗口化 Dataset（滑窗 + 归一化 + 可选加噪），大多数具体 Dataset 继承自此类。 |
| `Dataset_cluster.py`              | 把每个 ID 对应的子 Dataset 聚合成 `IdIncludedDataset`，配合 sampler 使用。 |
| `DG/Classification_dataset.py`    | 域泛化分类任务的 Dataset，对应 `task.type: DG`, `task.name: classification`。 |
| `CDDG/classification_dataset.py`  | 跨数据集域泛化分类 Dataset，对应 `task.type: CDDG`。 |
| `Pretrain/Classification_dataset.py` | 预训练相关任务的 Dataset（分类 / 预测 / 掩码重建等）。 |
| `FS/Classification_dataset.py`    | few-shot 场景下的按样本视角 Dataset，episodic 由 sampler 构建。 |
| `FS/episode_dataset.py`           | 显式 episodic few-shot Dataset，直接返回 support/query 结构。 |
| `GFS/Classification_dataset.py`   | Generalized Few-Shot 分类 Dataset。 |
| `ID/Classification_dataset.py`    | ID 风格任务（如 `ID_task`）使用的 Dataset。 |
| `ID_dataset.py`                   | 配合 `id_data_factory` 的 ID 中心 Dataset。 |

运行时，`data_factory` 通过以下规则动态选择 Dataset：

```python
mod = importlib.import_module(
    f"src.data_factory.dataset_task.{task_type}.{task_name}_dataset"
)
dataset_cls = mod.set_dataset
```

其中 `task_type` / `task_name` 来自配置中的：

```yaml
task:
  type: "DG"
  name: "classification"
```

-----

## 🔗 与 `task_registry.csv` 的统一表格

为了避免「task ↔ dataset」映射分散在多个文件中，  
我们把双方的信息统一整合到：

- `src/task_factory/task_registry.csv`

该 CSV 每一行代表一个支持的组合，列包含：

- `task.type`, `task.name`
- `path`：任务实现（`task_factory` 中的 LightningModule 路径，相对 `src/task_factory`）
- `args`：任务构造函数签名（例如 `(network, args_data, args_model, args_task, args_trainer, args_environment, metadata)`）
- `dataset_path`：对应的 Dataset 实现路径（相对 `src/data_factory`，从 `dataset_task/` 开始）
- `dataset_args`：Dataset 构造函数签名（例如 `(data, metadata, args_data, args_task, mode)`）
- `batch_format`：该任务期望的 batch 结构简要描述
- `notes`：补充说明
- `test_status`：预留给你标记测试状态（如 `passed` / `failed` / `not_tested`）

> 如果你在 `dataset_task/` 下新增了一个 Dataset，同时在 `task_factory` 侧新增了 Task，  
> 推荐只在 `src/task_factory/task_registry.csv` 这一个表中维护映射，而不再重复多份。

-----

## 🆕 新增 dataset_task 的步骤（建议）

1. 在 `src/task_factory/task/` 中先确定好新的 `task.type` / `task.name` 组合，以及该 Task 期望的 batch 结构。
2. 在 `src/data_factory/dataset_task/{task.type}/` 下新增 `{task.name}_dataset.py`，暴露 `set_dataset`，构造函数签名建议沿用：
   - `(data, metadata, args_data, args_task, mode="train")`
3. 在 `src/task_factory/task_registry.csv` 中新增一行：
   - 填写 task 路径 / args，以及 dataset 路径 / dataset_args / batch_format、备注说明。
4. 根据实际情况，在对应子目录（如 `task/pretrain/README.md`、`dataset_task/Pretrain/`）补充更细节的配置说明。

这样，无论是从 Task 侧还是 Dataset 侧，都可以通过这一张 CSV 表找到完整链路：  
`config.task.* → task_factory.* → dataset_task.* → DataLoader → Trainer`。

-----

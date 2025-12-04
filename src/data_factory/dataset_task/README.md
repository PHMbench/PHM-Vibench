-----

# dataset_task Module

This folder hosts **task-oriented dataset wrappers**.  
They sit between the raw data readers (`reader/`) and the task modules (`src/task_factory/task/`), and are responsible for:

- Converting `H5DataDict` + `metadata` into `torch.utils.data.Dataset` objects.
- Shaping each batch so that it matches the expectations of the corresponding task in `task_factory`.
- Encapsulating task-specific windowing, normalization, and sampling strategies.

> 简单理解：不同的 `task.type` / `task.name` 对应不同的 `LightningModule`，  
> 而这里的 `dataset_task` 则负责把底层信号数据整理成该任务所需的 batch 结构。

-----

## 📂 Directory Overview

Key files and directories:

| Path                        | Description |
| :-------------------------- | :---------- |
| `Default_dataset.py`        | Generic window-based dataset (sliding windows, normalization, optional noise). Most specialized datasets subclass this one. |
| `Dataset_cluster.py`        | Wraps per-ID datasets into an `IdIncludedDataset` cluster used by samplers and data factory. |
| `DG/Classification_dataset.py` | Dataset for domain generalization classification (`task.type: DG`, `task.name: classification`). |
| `CDDG/classification_dataset.py` | Dataset for cross-dataset domain generalization classification (`task.type: CDDG`). |
| `Pretrain/Classification_dataset.py` | Dataset for pretraining tasks that still use supervised labels or masked prediction (`task.type: pretrain`). |
| `FS/Classification_dataset.py` | Window-based few-shot dataset (per-sample view) for FS tasks. |
| `FS/Episode_dataset.py`     | Episode-style few-shot dataset (support/query episodic batch). |
| `GFS/Classification_dataset.py` | Dataset for generalized few-shot classification (`task.type: GFS`). |
| `ID/Classification_dataset.py` | Dataset for ID-style tasks, aligned with `ID_task`. |
| `ID_dataset.py`             | ID-centric dataset used by `id_data_factory`, focusing on raw ID access. |

At runtime, `data_factory` chooses the dataset class via:

```python
mod = importlib.import_module(
    f"src.data_factory.dataset_task.{task_type}.{task_name}_dataset"
)
dataset_cls = mod.set_dataset
```

So the key mapping is driven by the same `task.type` / `task.name` pair that `task_factory` uses.

-----

## 🔗 Mapping: task.type / task.name → dataset_task

The following CSV captures the recommended mappings between tasks and dataset wrappers.  
Each row corresponds to one combination of `task.type` and `task.name`, and the dataset under `dataset_task/` that is intended to feed that task.

```csv
id,task.type,task.name,path,args,batch_format,test_status
1,DG,classification,dataset_task/DG/Classification_dataset.py,"(data, metadata, args_data, args_task, mode)","{'x','y','file_id',...}",
2,CDDG,classification,dataset_task/CDDG/classification_dataset.py,"(data, metadata, args_data, args_task, mode)","{'x','y','file_id','domain_id',...}",
3,pretrain,classification,dataset_task/Pretrain/Classification_dataset.py,"(data, metadata, args_data, args_task, mode)","{'x','y','file_id',...}",
4,pretrain,hse_contrastive,dataset_task/Pretrain/Classification_dataset.py,"(data, metadata, args_data, args_task, mode)","{'x','y','file_id','domain_id',...}",
5,pretrain,masked_reconstruction,dataset_task/Pretrain/Classification_dataset.py,"(data, metadata, args_data, args_task, mode)","{'x','mask','file_id',...}",
6,pretrain,prediction,dataset_task/Pretrain/Classification_dataset.py,"(data, metadata, args_data, args_task, mode)","{'x','y','file_id',...}",
7,pretrain,classification_prediction,dataset_task/Pretrain/Classification_dataset.py,"(data, metadata, args_data, args_task, mode)","{'x','y','file_id',...}",
8,FS,prototypical_network,dataset_task/FS/Classification_dataset.py,"(data, metadata, args_data, args_task, mode)","per-sample few-shot view; sampler builds episodes",
9,FS,matching_network,dataset_task/FS/Classification_dataset.py,"(data, metadata, args_data, args_task, mode)","per-sample few-shot view; sampler builds episodes",
10,FS,knn_feature,dataset_task/FS/Classification_dataset.py,"(data, metadata, args_data, args_task, mode)","per-sample feature view for kNN",
11,FS,finetuning,dataset_task/FS/Classification_dataset.py,"(data, metadata, args_data, args_task, mode)","standard supervised few-shot finetuning batches",
12,GFS,classification,dataset_task/GFS/Classification_dataset.py,"(data, metadata, args_data, args_task, mode)","{'x','y','base/novel flags',...}",
13,GFS,matching,dataset_task/GFS/Classification_dataset.py,"(data, metadata, args_data, args_task, mode)","GFS-style episodes / batches",
14,Default_task,Default_task,dataset_task/Default_dataset.py,"(data, metadata, args_data, args_task, mode)","Default windows: {'x','y'}",
15,Default_task,ID_task,dataset_task/ID/Classification_dataset.py,"(data, metadata, args_data, args_task, mode)","ID-based windows: {'x','y','file_id',...}",
```

- `args` 列为当前实现中的构造函数签名，便于你在阅读代码时快速对齐。
- `batch_format` 只给出一个简要的结构提示；详细字段请查对应 task 的 README，例如：
  - `src/task_factory/task/pretrain/README.md`
  - `src/task_factory/task/FS/README.md`
  - `src/task_factory/task/GFS/README.md`
- `test_status` 留空，方便你手动维护每个组合的测试结果（如 `passed` / `failed` / `not_tested`）。

-----

## 🧩 How it works with `data_factory`

1. `data_factory` 依据 `args_task` 过滤出需要的 `Id`（`search_dataset_id` / `search_ids_for_task`）。
2. 通过 reader 和缓存 (`H5DataDict`) 准备好原始信号矩阵。
3. 根据 `task.type` / `task.name` 导入上表对应的 `set_dataset` 并实例化：

```python
dataset_cls = set_dataset  # imported from dataset_task/{task.type}/{task.name}_dataset.py
train_dataset[id] = dataset_cls({id: self.data[id]}, self.target_metadata, self.args_data, self.args_task, 'train')
```

4. 使用 `IdIncludedDataset` + 自定义 sampler 组合成最终的 `DataLoader`，供 `task_factory` 构建的任务模块消费。

-----

## 🆕 Adding a New dataset_task

When you introduce a new `task.type` / `task.name` pair on the task side:

1. Decide the batch structure that the new task expects.
2. Implement a new dataset under `dataset_task/{task.type}/{task.name}_dataset.py` exposing `set_dataset`.
3. Ensure its `__init__` signature follows the existing pattern:  
   `(data, metadata, args_data, args_task, mode="train")`.
4. Add a new row into:
   - `src/task_factory/task_type_name_mapping.csv`
   - `src/data_factory/dataset_task/dataset_task_mapping.csv` (see below)

This keeps the config → task → dataset pipeline explicit and traceable.

-----


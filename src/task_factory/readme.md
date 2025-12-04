

-----

# Task Factory Module (Continue updating)

## 🎯 Purpose

The **Task Factory** is the central orchestrator of the Vibench training pipeline. Its primary role is to assemble a **PyTorch Lightning `LightningModule`**, which encapsulates the entire training, validation, and testing logic. It acts as the glue that connects the neural network (`model`), the data (`DataLoaders`), and the specific training procedures, such as loss calculations, metric logging, and optimization steps.

This factory-based approach allows Vibench to handle diverse tasks—from standard classification to complex domain generalization—by simply swapping out the task configuration.

> 提示：`CLAUDE.md` 仅作导航，最新字段说明与示例请以本 README 为准。

-----

## 📂 Module Structure

The module is composed of a main factory function and a structured set of directories for different task implementations.

| File / Directory | Description |
| :--- | :--- |
| `task_factory.py` | The main entry point. It contains the `task_factory(...)` function that receives the model, all configuration arguments, and metadata to build the final `LightningModule`. |
| `Default_task.py` | Baseline Lightning task wrapper, used as default single-task implementation and as a base class for many custom tasks. |
| `task/` | Subfolders for concrete task families: `DG/`, `CDDG/`, `pretrain/`, `FS/`, `GFS/`, `ID/` (e.g. `ID_task`) and `MT/` (multi-task Lightning modules). Each subfolder contains one or more `LightningModule` implementations. |
| `Components/` | A collection of reusable modules for building tasks, such as specialized loss functions (`loss.py`), performance metrics (`metrics.py`), and regularization techniques. |

-----

## ⚙️ Configuration

The behavior of the `Task_Factory` is controlled via the `task` section in your YAML configuration file.

**Key Configuration Fields:**

  * **`type`**: Specifies the task category. This corresponds to a subfolder within the `src/task_factory/task/` directory (e.g., `DG`, `CDDG`, `FS`, `pretrain`).
  * **`name`**: The name of the Python file within the `type` subfolder that contains the task logic（例如 `classification.py` → `name: "classification"`，`hse_contrastive.py` → `name: "hse_contrastive"`）。
  * **Task-Specific Options**: Any other parameters needed by the task, such as the names of loss functions, metric choices, regularization strengths, or learning algorithm hyperparameters.

**Example Configuration (`.yaml`):**

```yaml
task:
  name: "classification"
  type: "DG"   # 或 "CDDG" / "FS" / "pretrain"

  task_list: ['classification', 'prediction']
  target_domain_num: 1

  loss: "CE" # cross_entropy
  metrics: ["acc"]
  target_system_id: [1,13,6,12,19]

  
  optimizer: "adam"

  lr: 0.0001
  weight_decay: 0.0001

  scheduler: true
  scheduler_type: "reduceonplateau"

  patience: 20

  step_size: 3
  gamma: 0.5

  regularization: 
    l2: 1e-5
    l1: 1e-5
  alpha_prediction: 1

  # prediction args
  mask_ratio: 0.1
  forecast_part: 0.1

  num_systems: 1
  num_domains: 1
  num_labels: 3 # n_way to set num_labels, should be equal to the number of 
  num_support: 1
  num_query: 1
  num_episodes: 5

-----

## 🔖 Common `type` / `name` combinations (v0.1.0)

下面的 CSV 表是当前版本中 **推荐/已有实现** 的 `task.type` 与 `task.name` 组合，一行对应一种可选任务；你只需要在配置里填这两列，就能通过 Task Factory 正确加载对应模块。  
同时给出了 Task 的构造函数 `args`、对应的 `dataset_task` 路径与构造参数，以及预留的 `test_status` 列方便你标记测试情况。  
完整表格维护在 `src/task_factory/task_registry.csv` 中，下面是其结构示意：

```csv
id,task.type,task.name,path,args,dataset_path,dataset_args,batch_format,notes,test_status
1,Default_task,Default_task,Default_task.py,"(network, args_data, args_model, args_task, args_trainer, args_environment, metadata)",dataset_task/Default_dataset.py,"(data, metadata, args_data, args_task, mode)","Default windows: {'x','y'}","Baseline single-task Lightning wrapper",
2,Default_task,ID_task,task/ID/ID_task.py,"(network, args_data, args_model, args_task, args_trainer, args_environment, metadata)",dataset_task/ID/Classification_dataset.py,"(data, metadata, args_data, args_task, mode)","ID-based windows: {'x','y','file_id',...}","ID_dataset / ID_task pipeline with flexible windowing",
3,DG,classification,task/DG/classification.py,"(network, args_data, args_model, args_task, args_trainer, args_environment, metadata)",dataset_task/DG/Classification_dataset.py,"(data, metadata, args_data, args_task, mode)","{'x','y','file_id',...}","Single-dataset domain generalization classification",
4,CDDG,classification,task/CDDG/classification.py,"(network, args_data, args_model, args_task, args_trainer, args_environment, metadata)",dataset_task/CDDG/classification_dataset.py,"(data, metadata, args_data, args_task, mode)","{'x','y','file_id','domain_id',...}","Cross-dataset (CDDG) classification",
...
```
```

- 当你编写 config 时，只要保证：
  - `task.type` 正好是上表中的 `task.type` 字段
  - `task.name` 正好是上表中的 `task.name` 字段  
 其余超参数（如 loss、metrics、mask_ratio 等）会在各自的任务 README 中详细说明（例如 `task/pretrain/README.md`、`task/FS/README.md` 等）。

-----

## 🌊 Workflow

The factory follows a clear, step-by-step process to build the task module.

1.  **Receive Inputs**: The factory is called after the `Model_Factory` has created the neural network. It takes the `network` (`nn.Module`) and all relevant configuration objects (`args_task`, `args_data`, `args_model`, etc.) as input.

2.  **Dynamic Import**: Using the `type` and `name` from the configuration, the factory constructs the import path for the desired task module. For example, a `type` of "DG" and `name` of "classification" resolves to `src.task_factory.task.DG.classification`.

3.  **Instantiation**: The factory imports the `task` class from the selected module and creates an instance of it. It passes the `network`, all necessary configurations, and the dataset `metadata` to the class constructor.

4.  **Return `LightningModule`**: The fully initialized `LightningModule` is returned to the main pipeline.

-----

## 🎁 Returned Object

The `task_factory` function returns a single, powerful object:

  * **A `pytorch_lightning.LightningModule` instance**: This object is now ready for the PyTorch Lightning `Trainer`. It contains all the necessary logic, including:
      * `training_step`: Defines what happens for each batch during training.
      * `validation_step` & `test_step`: Defines the logic for evaluation.
      * `configure_optimizers`: Sets up the optimizer(s) and learning rate scheduler(s).
      * Logging of losses and metrics to the specified logger (e.g., TensorBoard, W\&B).


## TODO 
# 1领域泛化(DG)任务
task:
  name: classification
  type: DG
  simpler: default  # 使用默认的DG选择器
  target_system_id: [RM_001_CWRU]
  source_domain_id: [0, 1, 2]
  target_domain_id: [3, 4]

# 2小样本学习(Few-Shot)任务
task:
  name: classification
  type: few_shot
  simpler: few_shot
  target_system_id: [RM_001_CWRU]
  n_way: 5         # 5类分类
  k_shot: 1        # 每类1个样本用于训练
  n_query: 15      # 每类最多15个样本用于测试
  label_column: Label

# 3. 不平衡数据任务
task:
  name: classification
  type: imbalanced
  simpler: imbalanced
  target_system_id: [RM_001_CWRU]
  imbalance_ratio: 0.1  # 少数类与多数类的比例
  minority_labels: [2, 4]  # 指定少数类标签
  stratify: true  # 使用分层抽样

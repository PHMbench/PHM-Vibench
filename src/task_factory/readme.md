

-----

# Task Factory Module (Continue updating)

## 🎯 Purpose

The **Task Factory** is the central orchestrator of the Vibench training pipeline. Its primary role is to assemble a **PyTorch Lightning `LightningModule`**, which encapsulates the entire training, validation, and testing logic. It acts as the glue that connects the neural network (`model`), the data (`DataLoaders`), and the specific training procedures, such as loss calculations, metric logging, and optimization steps.

This factory-based approach allows Vibench to handle diverse tasks—from standard classification to complex domain generalization—by simply swapping out the task configuration.

-----

## 📂 Module Structure

The module is composed of a main factory function and a structured set of directories for different task implementations.

| File / Directory | Description |
| :--- | :--- |
| `task_factory.py` | The main entry point. It contains the `task_factory(...)` function that receives the model, all configuration arguments, and metadata to build the final `LightningModule`. |
| `Default_task.py` | A baseline implementation for a standard classification task. It serves as a simple, ready-to-use option and a good starting point for creating new tasks. |
| `task/` | A directory containing subfolders for specialized task families. Each subfolder (e.g., `DG/` for Domain Generalization, `FS/` for Few-Shot) contains specific `LightningModule` implementations tailored to that paradigm. In the sub folder, there are multiple PHM tasks including classification, RUL prediction, anomaly detection, etc. 
|



| `Components/` | A collection of reusable modules for building tasks, such as specialized loss functions (`loss.py`), performance metrics (`metrics.py`), and regularization techniques. |

-----

## ⚙️ Configuration

The behavior of the `Task_Factory` is controlled via the `task` section in your YAML configuration file.

**Key Configuration Fields:**

  * **`type`**: Specifies the task category. This corresponds to a subfolder within the `src/task_factory/task/` directory (e.g., `DG`, `FS`, `Pretrain`). Use `"Default_task"` to select the baseline task.
  * **`name`**: The name of the Python file within the `type` subfolder that contains the task logic.
  * **Task-Specific Options**: Any other parameters needed by the task, such as the names of loss functions, metric choices, regularization strengths, or learning algorithm hyperparameters.

**Example Configuration (`.yaml`):**

```yaml
task:
  name: "classification"
  type: 'DG' # CDDG  # FS

  task_list: ['classification', 'prediction']
  target_domain_num: 1

  loss: "CE" # cross_entropy
  metrics: ["acc"]
  target_system_id: [1,5,6,13,19]

  
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
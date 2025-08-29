# Multi-Task PHM Foundation Model Experiment Plan

## 实验目标

配置4个骨干模型（B_04_Dlinear, B_06_TimesNet, B_08_PatchTST, B_09_FNO）同时执行4个任务：

1. **故障诊断（Fault Diagnosis）** - 多分类任务
2. **异常检测（Anomaly Detection）** - 二分类任务 （只有一个预测值0，1）
3. **信号预测（Signal Prediction）** - 时间序列预测任务
4. **剩余寿命预测（RUL Prediction）** - 回归任务

## 实施方案

### 1. 多任务头增强

#### 文件修改：`src/model_factory/ISFM/task_head/multi_task_head.py`

需要增加的功能：

- 添加信号预测任务头（使用H_03_Linear_pred）
- 修改forward函数以支持信号预测（不使用mean pooling）
- 支持task_id='signal_prediction'
- 保留现有的分类、异常检测和RUL预测功能

关键修改点：

```python
# 在__init__中添加
self.signal_prediction_head = self._build_signal_prediction_head()

# 新增函数
def _build_signal_prediction_head(self):
    # 使用H_03_Linear_pred作为信号预测头
    return H_03_Linear_pred(self.args_m)

# 在forward中处理信号预测
elif task_id == 'signal_prediction':
    # 不进行mean pooling，保持序列维度
    return self._forward_signal_prediction(x)  # x shape: (B, L, C)
```

### 2. 配置文件设计

#### 存放位置：`script/Vibench_paper/foundation model/`

创建4个配置文件：

- `multitask_B_04_Dlinear.yaml`
- `multitask_B_06_TimesNet.yaml`
- `multitask_B_08_PatchTST.yaml`
- `multitask_B_09_FNO.yaml`

#### 配置文件模板结构（）：

```yaml
environment:
  VBENCH_HOME: "/home/lq/LQcode/2_project/PHMBench/Vbench"
  project: "MultiTask_Foundation_Model"
  seed: 42
  output_dir: "results/multitask_{model_name}"
  iterations: 3
  wandb: True
  swanlab: True
  notes: 'Multi-task foundation model with {model_name}'

data:
  data_dir: "/mnt/crucial/LQ/PHM-Vibench"
  metadata_file: "metadata_6_11.xlsx"
  batch_size: 128
  num_workers: 32
  window_size: 4096
  stride: 4
  truncate_length: 200
  num_window: 512
  normalization: 'standardization'
  
  # RUL特定数据集配置
  rul_dataset_id: 2  # XJTU bearing dataset for RUL
  
model:
  name: "M_01_ISFM"
  type: "ISFM"
  input_dim: 2
  
  # 架构组件
  embedding: E_01_HSE
  backbone: {backbone_name}  # B_04_Dlinear / B_06_TimesNet / B_08_PatchTST / B_09_FNO
  task_head: multi_task_head  # 使用增强的多任务头
  
  # 模型参数
  num_patches: 128
  patch_size_L: 256
  patch_size_C: 1
  output_dim: 1024
  num_heads: 8
  num_layers: 3
  d_ff: 2048
  dropout: 0.1
  
  # 多任务配置
  classification_head: H_02_distance_cla
  prediction_head: H_03_Linear_pred
  hidden_dim: 512
  activation: "gelu"
  rul_max_value: 2000.0
  
  # 骨干网络特定参数
  # For B_09_FNO
  modes: 32
  width: 128
  
  # For B_06_TimesNet
  e_layers: 2
  factor: 5

task:
  name: "multi_task_phm"
  type: "multi_task"
  
  # 启用的任务
  enabled_tasks: 
    - 'classification'       # 故障诊断
    - 'anomaly_detection'   # 异常检测  
    - 'signal_prediction'   # 信号预测
    - 'rul_prediction'      # RUL预测
  
  # 任务权重
  task_weights:
    classification: 1.0
    anomaly_detection: 0.6
    signal_prediction: 0.7
    rul_prediction: 0.8
  
  # 任务特定配置
  classification:
    loss: "CE"
    num_classes: auto  # 根据数据集自动设置
    label_smoothing: 0.1
  
  anomaly_detection:
    loss: "BCE"
    threshold: 0.5
    class_weights: [1.0, 2.0]
  
  signal_prediction:
    loss: "MSE"
    pred_len: 96
    use_mean_pooling: false  # 关键：不使用mean pooling
  
  rul_prediction:
    loss: "MSE"
    max_rul_value: 2000.0
    normalize_targets: true
    dataset_id: 2  # XJTU dataset
  
  # 目标系统
  target_system_id: [1, 2, 5, 6, 13, 19]
  
  # 优化参数
  optimizer: "adamw"
  lr: 0.0005
  weight_decay: 0.01
  
  # 学习率调度
  scheduler:
    name: "cosine"
    T_max: 100
    eta_min: 1e-6

trainer:
  name: "Default_trainer"
  num_epochs: 150
  gpus: 1
  precision: 16
  accumulate_grad_batches: 2
  gradient_clip_val: 1.0
  
  # 早停
  early_stopping: true
  patience: 25
  monitor: "val_total_loss"
  
  # 检查点
  save_top_k: 3
  save_last: true
  
  # 日志
  log_every_n_steps: 50
```

### 3. 执行脚本

#### 主执行脚本：`script/Vibench_paper/foundation model/run_multitask_experiments.sh`

```bash
#!/bin/bash

# Multi-Task Foundation Model Experiments
# 运行4个骨干模型的多任务实验

echo "=========================================="
echo "Multi-Task PHM Foundation Model Experiments"
echo "=========================================="

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:/home/lq/LQcode/2_project/PHMBench/PHM-Vibench"

# 实验目录
EXPERIMENT_DIR="script/Vibench_paper/foundation model"
RESULTS_DIR="results/multitask_experiments"

# 创建结果目录
mkdir -p $RESULTS_DIR

# 运行B_04_Dlinear实验
echo "Running Multi-Task with B_04_Dlinear..."
python main.py --config "$EXPERIMENT_DIR/multitask_B_04_Dlinear.yaml" \
    --output_dir "$RESULTS_DIR/B_04_Dlinear" \
    2>&1 | tee "$RESULTS_DIR/B_04_Dlinear.log"

# 运行B_06_TimesNet实验  
echo "Running Multi-Task with B_06_TimesNet..."
python main.py --config "$EXPERIMENT_DIR/multitask_B_06_TimesNet.yaml" \
    --output_dir "$RESULTS_DIR/B_06_TimesNet" \
    2>&1 | tee "$RESULTS_DIR/B_06_TimesNet.log"

# 运行B_08_PatchTST实验
echo "Running Multi-Task with B_08_PatchTST..."
python main.py --config "$EXPERIMENT_DIR/multitask_B_08_PatchTST.yaml" \
    --output_dir "$RESULTS_DIR/B_08_PatchTST" \
    2>&1 | tee "$RESULTS_DIR/B_08_PatchTST.log"

# 运行B_09_FNO实验
echo "Running Multi-Task with B_09_FNO..."
python main.py --config "$EXPERIMENT_DIR/multitask_B_09_FNO.yaml" \
    --output_dir "$RESULTS_DIR/B_09_FNO" \
    2>&1 | tee "$RESULTS_DIR/B_09_FNO.log"

echo "=========================================="
echo "All experiments completed!"
echo "Results saved in: $RESULTS_DIR"
echo "=========================================="
```

### 4. 数据集配置 @

#### XJTU数据集（ID=2）用于RUL任务

- 数据集名称：XJTU轴承数据集
- 特点：包含轴承从正常到失效的完整退化过程
- 适合任务：RUL预测
- 数据路径：通过metadata_6_11.xlsx中的Dataset_id=2访问

#### 其他数据集用于多任务

- CWRU (ID=1): 故障诊断，异常检测 信号预测
- THU (ID=6): 故障诊断，异常检测 信号预测
- MFPT (ID=5): 故障诊断，异常检测 信号预测
- IMS (ID=13): 异常检测 信号预测
- HIT (ID=19): 故障诊断，异常检测 信号预测

### 5. 评估指标

#### 各任务评估指标

1. **故障诊断**

   - Accuracy
   - F1-score
   - Confusion Matrix
   - Per-class Precision/Recall
2. **异常检测**

   - ROC-AUC
   - Precision@Recall
   - F1-score
   - False Positive/Negative Rate
3. **信号预测**

   - MSE
   - MAE
   - RMSE
   - R² Score
   - DTW Distance
4. **RUL预测**

   - MSE
   - MAE
   - MAPE
   - Score Function (NASA PHM08)
   - Relative Error

### 6. 实验流程

1. **准备阶段**

   - 修改multi_task_head.py添加信号预测功能
   - 创建4个模型配置文件
   - 准备执行脚本
2. **执行阶段**

   - 运行run_multitask_experiments.sh
   - 监控训练过程（通过wandb/swanlab）
   - 记录训练日志（wandb）
3. **评估阶段**

   - 收集各模型在4个任务上的性能
   - 对比不同骨干网络的效果
   - 分析多任务学习的权衡
4. **结果分析**

   - 生成性能对比表
   - 绘制学习曲线
   - 分析任务间的相互影响

### 7. 预期成果

1. **模型性能对比**

   - 4个骨干网络在各任务上的性能表现
   - 多任务学习vs单任务学习的对比
2. **最佳模型选择**

   - 综合性能最优的骨干网络
   - 各任务的最优配置参数
3. **研究发现**

   - 任务间的协同效应分析
   - 多任务权重优化策略
   - 骨干网络特性对多任务的影响

### 8. 注意事项

1. **内存管理**

   - 多任务训练需要更多GPU内存
   - 可能需要减小batch_size或使用梯度累积
2. **训练策略**

   - 考虑使用任务交替训练
   - 动态调整任务权重
   - 监控各任务的收敛速度
3. **数据平衡**

   - 确保各任务的数据量平衡
   - 处理类别不平衡问题
   - 考虑数据增强策略

## 执行计划

1. ✅ 分析现有多任务配置和管道结构
2. ✅ 识别4个任务所需的任务头
3. ✅ 设计多任务实验配置模式
4. ⏳ 创建增强的多任务头（无mean pooling的信号预测）
5. 每个模块都 进行review
6. ⏳ 在script/Vibench_paper/foundation model/创建4个模型配置
7. ⏳ 在script/Vibench_paper/plan/记录实验计划
8. ⏳ 创建执行脚本
9. ⏳ 配置XJTU数据集（ID 2）用于RUL任务

## 更新日志

- 2025-08-28: 初始计划制定
- 待定: 实施多任务头增强
- 待定: 创建配置文件和脚本
- 待定: 运行实验并分析结果

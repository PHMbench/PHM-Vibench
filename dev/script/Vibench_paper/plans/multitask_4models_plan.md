# Multi-Task PHM Foundation Model Experiment Plan

## 实验目标

配置4个骨干模型（B_04_Dlinear, B_06_TimesNet, B_08_PatchTST, B_09_FNO）同时执行4个任务：

1. **故障诊断（Fault Diagnosis）** - 多分类任务，识别不同故障类型
2. **异常检测（Anomaly Detection）** - 二分类任务，输出单值预测（0=正常，1=异常）
3. **信号预测（Signal Prediction）** - 时间序列预测任务，预测未来信号序列
4. **剩余寿命预测（RUL Prediction）** - 回归任务，预测设备剩余使用寿命

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
from .H_03_Linear_pred import H_03_Linear_pred

class MultiTaskHead(nn.Module):
    def __init__(self, args_m):
        super().__init__()
        # 现有初始化代码保持不变...
        
        # 新增：直接复用H_03_Linear_pred作为信号预测头
        self.signal_prediction_head = H_03_Linear_pred(args_m)
    
    def forward(self, x, system_id=None, task_id=None, return_feature=False, **kwargs):
        """
        统一的forward接口，根据task_id分发到不同任务
        """
        # 信号预测任务：直接传递原始序列，不做mean pooling
        if task_id == 'signal_prediction':
            # H_03_Linear_pred期望输入(B,L,C)，需要shape参数
            shape = kwargs.get('shape', (96, 2))  # 默认预测96步，2通道
            return self.signal_prediction_head(x, shape=shape)
        
        # 其他任务：先进行mean pooling
        if x.dim() == 3:
            x = x.mean(dim=1)  # (B, L, C) -> (B, C)
        
        # 处理共享特征
        shared_features = self.shared_layers(x)
        
        if return_feature:
            return shared_features
        
        # 根据task_id分发到对应的任务头
        if task_id == 'classification':
            return self._forward_classification(shared_features, system_id)
        elif task_id == 'rul_prediction':
            return self._forward_rul_prediction(shared_features)
        elif task_id == 'anomaly_detection':
            return self._forward_anomaly_detection(shared_features)
        elif task_id == 'all' or task_id is None:
            return self._forward_all_tasks(shared_features, x, system_id, **kwargs)
```

#### 实现要点：
- **复用现有模块**：直接使用H_03_Linear_pred，无需重新实现
- **保持简洁**：信号预测直接调用，其他任务走原有逻辑
- **参数传递**：通过kwargs传递shape参数给H_03_Linear_pred
- **输入处理**：信号预测保持(B,L,C)形状，其他任务进行mean pooling

### 2. 配置文件设计

#### 存放位置：`script/Vibench_paper/foundation model/`

创建4个配置文件：

- `multitask_B_04_Dlinear.yaml`
- `multitask_B_06_TimesNet.yaml`
- `multitask_B_08_PatchTST.yaml`
- `multitask_B_09_FNO.yaml`

#### 配置模式说明

**测试模式（当前设置）**：
- `wandb: dryrun` - 日志记录但不上传到服务器
- `swanlab: False` - 关闭SwanLab日志
- `num_epochs: 1` - 快速验证，1个epoch即可
- `early_stopping: false` - 关闭早停，确保完整运行
- `iterations: 1` - 单次运行，减少测试时间

**生产模式（正式实验时切换）**：
- `wandb: True` - 完整日志记录和上传
- `swanlab: False` - 可选开启SwanLab
- `num_epochs: 50-150` - 完整训练周期
- `early_stopping: true` - 开启早停避免过拟合
- `iterations: 3` - 多次运行取平均结果

#### 配置文件模板结构：

```yaml
environment:
  VBENCH_HOME: "/home/lq/LQcode/2_project/PHMBench/Vbench"
  project: "MultiTask_Foundation_Model"
  seed: 42
  output_dir: "results/multitask_{model_name}"
  iterations: 1  # 测试阶段设为1，生产阶段改为3
  wandb: dryrun  # 测试阶段使用dryrun，生产阶段改为True
  swanlab: False  # 测试阶段关闭，生产阶段按需开启
  notes: 'Multi-task foundation model with {model_name} - TEST MODE'

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
  task_head: MultiTaskHead  # 使用增强的多任务头（multi_task_head.py中的MultiTaskHead类）
  
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
  num_epochs: 1  # 测试阶段：1 epoch快速验证，生产阶段：50-150 epochs
  gpus: 1
  precision: 16
  accumulate_grad_batches: 2
  gradient_clip_val: 1.0
  
  # 早停（测试阶段建议关闭）
  early_stopping: false  # 测试阶段关闭，生产阶段开启
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

#### 快速测试脚本：`script/Vibench_paper/foundation model/test_multitask.sh`

```bash
#!/bin/bash
# 快速测试脚本 - 验证多任务功能

echo "=========================================="
echo "Multi-Task Foundation Model - Quick Test"
echo "=========================================="

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:/home/lq/LQcode/2_project/PHMBench/PHM-Vibench"

# 测试目录
EXPERIMENT_DIR="script/Vibench_paper/foundation model"
TEST_RESULTS_DIR="results/test_multitask"

# 创建测试结果目录
mkdir -p $TEST_RESULTS_DIR

# 只测试最简单的模型验证流程
echo "Testing Multi-Task with B_04_Dlinear (Quick Test)..."
python main.py --config "$EXPERIMENT_DIR/multitask_B_04_Dlinear.yaml" \
    --output_dir "$TEST_RESULTS_DIR/B_04_Dlinear_test" \
    2>&1 | tee "$TEST_RESULTS_DIR/B_04_Dlinear_test.log"

if [ $? -eq 0 ]; then
    echo "✅ Test passed! Ready for full experiments."
else
    echo "❌ Test failed! Please check configuration."
    exit 1
fi

echo "=========================================="
echo "Quick test completed successfully!"
echo "Test results saved in: $TEST_RESULTS_DIR"
echo "=========================================="
```

### 4. 数据集配置

#### 任务-数据集映射

| 数据集 | ID | 故障诊断 | 异常检测 | 信号预测 | RUL预测 | 说明 |
|--------|-------|----------|----------|----------|---------|------|
| CWRU   | 1     | ✓        | ✓        | ✓        | -       | 凯斯西储大学轴承数据集 |
| XJTU   | 2     | ✓        | ✓        | ✓        | ✓       | 西安交通大学轴承退化数据集 |
| MFPT   | 5     | ✓        | ✓        | ✓        | -       | 机械故障预防技术数据集 |
| THU    | 6     | ✓        | ✓        | ✓        | -       | 清华大学振动数据集 |
| IMS    | 13    | -        | ✓        | ✓        | -       | IMS轴承数据集 |
| HIT    | 19    | ✓        | ✓        | ✓        | -       | 哈工大振动数据集 |

#### 数据集特点

**XJTU数据集（ID=2）- RUL专用**
- 包含轴承从正常到失效的完整退化过程
- 多个运行到失效的轴承样本
- 适合剩余寿命预测任务
- 通过metadata_6_11.xlsx中的Dataset_id=2访问

**其他数据集 - 多任务通用**
- 包含正常和多种故障状态
- 适合故障诊断、异常检测和信号预测
- 每个数据集包含不同工况和故障类型

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

### 第一阶段：准备工作
1. ✅ **分析现有多任务配置和管道结构**
   - 研究现有的multi_task_head.py实现
   - 了解H_09_multiple_task的架构
   - 分析配置文件结构和参数设置

2. ✅ **识别4个任务所需的任务头**
   - 故障诊断: H_02_distance_cla (distance-based classification)
   - 异常检测: 二分类输出头 (single binary output)
   - 信号预测: H_03_Linear_pred (无mean pooling)
   - RUL预测: 回归输出头 (ReLU activation for positive values)

3. ✅ **设计多任务实验配置模式**
   - 完成配置文件结构设计
   - 确定任务权重和损失函数配置
   - 规划数据集分配策略

### 第二阶段：核心实现
4. ⏳ **创建增强的多任务头**
   - 修改`src/model_factory/ISFM/task_head/multi_task_head.py`
   - 添加信号预测功能（保持序列维度，不使用mean pooling）
   - 支持task_id='signal_prediction'
   - 代码审查确保实现正确性

5. ⏳ **创建4个模型配置文件**
   - 在`script/Vibench_paper/foundation model/`创建配置文件：
     - `multitask_B_04_Dlinear.yaml`
     - `multitask_B_06_TimesNet.yaml`
     - `multitask_B_08_PatchTST.yaml`
     - `multitask_B_09_FNO.yaml`
   - 每个配置文件包含模型特定参数

6. ⏳ **配置数据集映射**
   - 配置XJTU数据集（ID=2）专门用于RUL任务
   - 设置其他数据集用于多任务学习
   - 验证数据加载和预处理流程

### 第三阶段：部署和测试
7. ✅ **完善实验计划文档**
   - 在`script/Vibench_paper/plans/`记录详细实验计划
   - 包含评估指标、预期结果和分析方法

8. ⏳ **创建执行脚本**
   - 创建`script/Vibench_paper/foundation model/run_multitask_experiments.sh`
   - 支持单独运行和批量实验
   - 包含日志记录和错误处理

9. ⏳ **系统测试和验证（渐进式测试策略）**
   
   **Step 1: 单元测试**
   - 测试MultiTaskHead类的实例化
   - 验证各task_id的forward函数调用
   - 确认输入输出shape正确性
   
   **Step 2: 配置验证**
   - 验证YAML配置文件语法正确性
   - 测试各模型参数加载
   - 确认数据集ID映射正确
   
   **Step 3: 单模型测试**
   - 先测试B_04_Dlinear（最简单模型）
   - 使用num_epochs=1, wandb=dryrun模式
   - 验证4个任务都能正常运行
   
   **Step 4: 完整流程验证**
   - 所有4个模型快速测试（1 epoch）
   - 检查日志输出和中间结果
   - 确认GPU内存使用合理

### 第四阶段：实验执行
10. ⏳ **运行基线实验**
    - 执行4个模型的多任务训练
    - 监控训练过程和资源使用
    - 记录中间结果和异常情况

11. ⏳ **结果分析和优化**
    - 收集各模型的性能指标
    - 分析任务间的相互影响
    - 根据结果调整参数设置

12. ⏳ **实验报告和总结**
    - 生成性能对比表和可视化图表
    - 撰写实验结果分析报告
    - 提出改进建议和后续研究方向

## 更新日志

- **2025-08-29-2**: 测试模式配置优化
  - 设置wandb为dryrun模式，便于测试而不产生线上记录
  - 关闭swanlab减少测试复杂度
  - epochs设为1进行快速验证
  - 添加渐进式测试策略（单元测试→配置验证→单模型→完整流程）
  - 创建快速测试脚本test_multitask.sh
  - 明确测试模式和生产模式的参数差异

- **2025-08-29-1**: 实现方案简化
  - 直接复用H_03_Linear_pred作为信号预测头，避免"炫技式"复杂度
  - 保持现有架构不变，最小化代码修改
  - 优化forward逻辑，信号预测保持序列维度，其他任务使用mean pooling

- **2025-08-29**: 计划优化和重构
  - 优化任务描述和数据集映射表
  - 重新组织执行计划，分为4个阶段12个步骤
  - 添加详细的多任务头实现代码示例
  - 完善配置文件模板和技术细节
  - 更新任务-数据集映射关系

- **2025-08-28**: 初始计划制定
  - 确定4个骨干模型和4个任务的实验框架
  - 设计基本的配置文件结构
  - 规划实验流程和评估指标

## 待办事项优先级

### 高优先级（立即执行）
- [ ] **实施多任务头增强**（第二阶段-步骤4）
  - 修改multi_task_head.py添加H_03_Linear_pred支持
  - 实现signal_prediction任务的forward逻辑
  
- [ ] **创建测试配置文件**（第二阶段-步骤5）
  - 优先创建multitask_B_04_Dlinear.yaml（测试模式）
  - 验证配置参数正确性

### 中优先级（测试通过后执行）
- [ ] **快速功能验证**（第三阶段-步骤9）
  - 运行test_multitask.sh脚本
  - 验证4个任务都能正常调用
  
- [ ] **完整配置创建**
  - 创建其余3个模型配置文件
  - 创建完整执行脚本

### 低优先级（验证完成后执行）
- [ ] 配置XJTU数据集映射（第二阶段-步骤6）
- [ ] 运行完整实验（第四阶段-步骤10）
- [ ] 结果分析和报告（第四阶段-步骤11-12）

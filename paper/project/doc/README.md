# HSE-Prompt优化策略实验体系

## 📋 项目概述

本项目提供了HSE-Prompt优化策略的完整实验框架，包括Baseline建立、优化策略验证、扩展验证实验等全套实验流程。该实验体系支持渐进式实验设计，从建立性能基线开始，逐步验证HSE-Prompt优化策略的有效性。

**主要特性**:

- ✅ 渐进式Baseline建立体系
- ✅ 系统化优化策略验证
- ✅ 自动化表格生成工具(根据实验结果)
- ✅ 标准化复现流程
- ✅ 零歧义执行指南

**项目结构**:

```
paper/2025-10_foundation_model_0_metric/
├── configs/           # 实验配置文件
├── scripts/           # 执行脚本
├── docs/              # 文档指南
├── experiments/       # 实验设计文档
├── results/           # 实验结果
└── paper_tables/      # 论文表格
```

📋 实现本README的代码修改可见**详细实施计划**: [doc_code_alignment_plan.md](docs/11_17/doc_code_alignment_plan.md)

以及[phm_vibench_code_analysis_2025_11_17.md](docs/11_17/phm_vibench_code_analysis_2025_11_17.md)

#### ✅ 实现状态更新 (2025-11-18)

**核心功能状态**: 🎉 **完全可用**

- ✅ **CLI --override参数**: 已实现并集成到所有Pipeline
- ✅ **Pipeline_02两阶段训练**: fs_config_path参数已修复
- ✅ **配置覆盖系统**: 支持嵌套配置、多数据类型、错误处理
- ✅ **实验脚本**: run_hse_experiments.sh参数传递完整
- ✅ **命令示例**: 所有README命令现在都可以正常执行

**用户可以直接按照README执行所有实验，无需额外代码修改。**

---

## 🎯 科学研究框架

### 核心研究问题与假设

本研究通过系统的实验设计回答两个核心科学问题：

#### 问题一：基线建立与增量验证 (实验0-3)

**核心问题**: HSE-Prompt方法在统一度量框架下能否显著优于无预训练/无Prompt的基线方法？

**具体假设**:

- **H0**: 无信号处理优化的Backbone+Head基线性能最低 (65-70%)
- **H1**: HSE嵌入直接应用能提升跨域泛化能力 (70-75%)
- **H2**: 无监督预训练能显著改善few-shot学习 (80-85%)
- **H3**: 完整HSE-Prompt方法达到最优性能 (87-92%)

#### 问题二：优化策略普适性验证 (实验4-7)

**核心问题**: HSE-Prompt各组件的贡献大小如何？在不同Backbone/噪声条件下是否仍然成立？

**具体假设**:

- **组件贡献**: Prompt机制 > HSE嵌入 > 预训练策略 > 融合方式
- **Backbone普适性**: 性能提升在主流Backbone上一致
- **噪声鲁棒性**: 优化策略在低信噪比条件下稳定
- **少样本极限**: 方法在极低样本条件下仍有效

### 精确实验矩阵与执行策略

我们的实验采用**精确控制的渐进式设计**，通过明确的训练策略对比来量化HSE-Prompt各组件的贡献。实验设计严格区分数据处理方式和预训练策略，确保可重复的科学验证。

#### 精确实验执行矩阵

| 实验        | 训练策略             | Prompts使用 | 预训练     | Few-shot  | 数据集处理策略                    | Pipeline                     | 运行次数          | 预期性能 | 配置文件                                            |
| ----------- | -------------------- | ----------- | ---------- | --------- | --------------------------------- | ---------------------------- | ----------------- | -------- | --------------------------------------------------- |
| **0** | Backbone+Head基线    | ❌ 禁用     | ❌ 无      | ✅ 5-shot | **5个数据集分别独立训练**   | Pipeline_01_default          | **5次运行** | 65-70%   | `configs/experiment_0_backbone_head.yaml`         |
| **1** | HSE + 直接Few-Shot   | ❌ 禁用     | ❌ 无      | ✅ 5-shot | **单模型同时处理5个数据集** | Pipeline_01_default          | **5次运行** | 70-75%   | `configs/experiment_1_direct_fewshot.yaml`        |
| **2** | HSE + 无Prompt预训练 | ❌ 禁用     | ✅ 无监督  | ✅ 5-shot | **单模型同时处理5个数据集** | Pipeline_02_pretrain_fewshot | **5次运行** | 80-85%   | `configs/experiment_2_unsupervised_pretrain.yaml` |
| **3** | HSE-Prompt预训练     | ✅ 启用     | ✅ HSE引导 | ✅ 5-shot | **单模型同时处理5个数据集** | Pipeline_02_pretrain_fewshot | **5次运行** | 87-92%   | `configs/experiment_3_hse_prompt_pretrain.yaml`   |

#### 🔍 实验执行策略详细说明

**数据集统一配置**: `target_system_id: [1, 5, 6, 12, 19]`

- **系统ID 1**: CWRU (Case Western Reserve University) - 轴承故障诊断
- **系统ID 5**: Ottawa (University of Ottawa) - 轴承故障数据集
- **系统ID 6**: THU (Tsinghua University) - 电机故障诊断
- **系统ID 12**: JNU (Jiangnan University) - 泵设备故障诊断
- **系统ID 19**: HUST (Huazhong University of Science and Technology) - 风机故障诊断

##### 实验0: 独立数据集基线建立

**执行策略**: 5个数据集完全独立训练，建立最基础的性能下限

```bash
# 对每个数据集单独训练Backbone+Head模型 (5次独立运行)
for dataset_id in 1 5 6 12 19; do
    python main.py --config_path configs/experiment_0_backbone_head.yaml \
                   --pipeline Pipeline_01_default \
                   --override task.target_system_id=[$dataset_id]
done
```

**科学意义**: 建立工业故障诊断的最低性能基准，无任何信号处理优化

##### 实验1: 统一模型直接Few-Shot学习

**执行策略**: 单一模型同时处理5个数据集，验证HSE嵌入的直接效果

```bash
# 单模型统一训练，同时处理5个数据集 (1次运行)
python main.py --config_path configs/experiment_1_direct_fewshot.yaml \
               --pipeline Pipeline_01_default \
               --override task.target_system_id=[1,5,6,12,19]
```

**科学意义**: 评估HSE信号嵌入在无预训练情况下的跨域泛化能力

##### 实验2: 无监督预训练策略

**执行策略**: 两阶段训练（无监督预训练 → Few-shot微调），验证预训练效果

```bash
# 两阶段训练：无监督预训练 + Few-shot微调 (1次运行)
python main.py --config_path "paper/2025-10_foundation_model_0_metric/configs/experiment_2_unsupervised_pretrain.yaml" \
               --pipeline "Pipeline_02_pretrain_fewshot" \
               --fs_config_path "paper/2025-10_foundation_model_0_metric/configs/experiment_2_unsupervised_fs.yaml" \
               --override task.target_system_id=[1,5,6,12,19]
```

**科学意义**: 量化无监督预训练对few-shot学习的提升，排除prompt机制影响

##### 实验3: HSE-Prompt完整方法

**执行策略**: 两阶段训练（HSE-Prompt预训练 → Few-shot微调），验证完整方法

```bash
# 两阶段训练：HSE-Prompt预训练 + Few-shot微调 (1次运行)
python main.py --config_path "paper/2025-10_foundation_model_0_metric/configs/experiment_3_hse_prompt_pretrain.yaml" \
               --pipeline "Pipeline_02_pretrain_fewshot" \
               --fs_config_path "paper/2025-10_foundation_model_0_metric/configs/experiment_3_hse_prompt_fs.yaml" \
               --override task.target_system_id=[1,5,6,12,19]
```

**科学意义**: 验证HSE-Prompt方法的完整效果，作为性能上界参考
        
### 优化策略验证

基于实验0-3的基线建立，进一步验证HSE-Prompt各组件的优化效果：

4. **实验4**: 组件消融研究 - 量化各HSE-Prompt组件的贡献大小
5. **实验5**: 少样本梯度扫描 - 验证方法在极低样本条件下的性能
6. **实验6**: Backbone普适性验证 - 测试HSE-Prompt在主流Backbone上的泛化性

**Backbone对比组件**:

- **B_04_Dlinear**: 线性预测模型，轻量级基准
- **B_06_TimesNet**: 时序分析网络，处理时间依赖性
- **B_08_PatchTST**: 补丁时间序列Transformer，当前SOTA
- **B_09_FNO**: 傅里叶神经算子，频域建模方法

**对比控制策略**:

- 相同参数量范围：所有Backbone控制在1-5M参数
- 相同训练配置：学习率、批次大小、训练轮数统一
- 相同计算资源：确保公平比较的环境

7. **实验7**: 噪声鲁棒性评估 - 评估方法在噪声环境下的稳定性

#### 优化策略实验矩阵

| 实验        | 研究目标           | 测试内容                           | Backbone            | Pipeline            | 样本设置                     | 预期性能     | 配置文件                                            |
| ----------- | ------------------ | ---------------------------------- | ------------------- | ------------------- | ---------------------------- | ------------ | --------------------------------------------------- |
| **4** | 组件消融研究       | HSE、Prompt、融合方式单独/组合效果 | B_04_Dlinear        | Pipeline_02_pretrain_fewshot | 5-shot × 5种子              | 量化贡献     | `configs/experiment_4_ablation.yaml`              |
| **5** | 少样本梯度扫描     | shots=[1,3,5,10,15,20]性能变化     | B_04_Dlinear        | Pipeline_02_pretrain_fewshot | 梯度扫描 × 5种子            | 性能曲线     | `configs/experiment_5_fewshot_sweep.yaml`         |
| **6** | Backbone普适性验证 | B_04、B_06、B_08、B_09泛化性测试   | B_04/B_06/B_08/B_09 | Pipeline_02_pretrain_fewshot | 5-shot × 4Backbone × 3种子 | 跨架构一致性 | `configs/experiment_6_backbone_universality.yaml` |
| **7** | 噪声鲁棒性评估     | SNR=[20,10,5,0]dB下的性能稳定性    | B_04_Dlinear        | Pipeline_02_pretrain_fewshot | 5-shot × 4SNR × 3种子      | 抗噪性能     | `configs/experiment_7_noise_robustness.yaml`      |

### 论文表格对应关系

| 表格编号 | 表格标题              | 对应实验       | 核心验证要点                | 性能评估指标        |
| -------- | --------------------- | -------------- | --------------------------- | ------------------- |
| 表1      | Backbone+Head基线性能 | 实验0          | 最低性能基准                | 准确率、F1分数      |
| 表2      | 直接Few-Shot性能对比  | 实验1          | HSE嵌入直接效果             | 跨域泛化能力        |
| 表3      | 预训练策略对比        | 实验2 vs 实验3 | 预训练vs HSE-Prompt提升效果 | 性能提升幅度        |
| 表4      | 组件消融研究结果      | 实验4          | 各组件贡献量化              | 组件重要性排序      |
| 表5      | Backbone普适性验证    | 实验6          | 跨架构泛化性                | 性能一致性分析      |
| 表6      | 少样本梯度扫描结果    | 实验5          | 极限样本条件性能曲线        | 性能衰减率分析      |
| 表7      | 噪声鲁棒性评估结果    | 实验7          | 不同SNR下稳定性             | 抗噪性能指标        |
| 表8      | 实验资源与时间统计    | 全部实验       | 计算资源需求与执行时间      | GPU小时数、Wall时间 |

### 资源配置与时间估算

基于单张NVIDIA RTX 4090 (24GB显存)的计算资源估算：

#### 实验资源配置矩阵

| 实验        | 单次运行GPU时间    | 总运行次数 | 预估GPU小时 | 内存需求 | 批处理大小 | 训练轮数 | Wall时间预估 |
| ----------- | ------------------ | ---------- | ----------- | -------- | ---------- | -------- | ------------ |
| **0** | 0.5小时            | 5次        | 2.5小时     | 8GB      | 32         | 50       | 3小时        |
| **1** | 0.8小时            | 5次        | 4.0小时     | 12GB     | 32         | 50       | 5小时        |
| **2** | 1.2小时            | 5次        | 6.0小时     | 16GB     | 32         | 50       | 8小时        |
| **3** | 1.5小时            | 5次        | 7.5小时     | 16GB     | 32         | 50       | 10小时       |
| **4** | 1.0小时            | 5次        | 5.0小时     | 12GB     | 32         | 50       | 6小时        |
| **5** | 0.6小时×6配置     | 5次        | 18.0小时    | 12GB     | 32         | 50       | 24小时       |
| **6** | 0.8小时×4Backbone | 3次        | 9.6小时     | 12GB     | 32         | 50       | 12小时       |
| **7** | 0.7小时×4SNR      | 3次        | 8.4小时     | 12GB     | 32         | 50       | 10小时       |

#### 总体资源需求

- **总GPU小时数**: 约61小时 (RTX 4090)
- **总Wall时间**: 约78小时 (含中间间隔)
- **峰值内存需求**: 16GB显存
- **存储需求**: 约50GB (模型+结果+日志)

#### 资源优化建议

1. **并行执行**: 实验0可并行执行5个数据集，节省4小时
2. **批次优化**: 根据显存调整batch_size至64，减少训练时间30%
3. **混合精度**: 启用FP16训练，减少50%显存占用，提升20%速度
4. **梯度累积**: 使用梯度累积模拟大批次，平衡内存与性能

展现形式

---

## 🛠️ Vbench配置重写系统详解

PHM-Vibench提供了强大的配置重写功能，支持灵活的实验参数调整和批量实验执行。该系统采用4×4灵活设计，支持多种配置源和重写方式的组合。

### 核心功能特性

#### 4×4灵活性矩阵

- **配置源**: 预设配置 → YAML文件 → Python字典 → ConfigWrapper对象
- **重写方式**: 预设覆盖 → YAML文件 → 字典重写 → ConfigWrapper更新
- **嵌套支持**: 自动处理深层配置结构
- **点号展开**: 支持 `task.target_system_id`风格的深层参数访问

#### 基础重写语法

```python
from src.configs import load_config

# 字典重写支持点号展开和自动类型转换
config = load_config('experiment_1', {
    'task.target_system_id': [1, 5, 6, 12, 19],  # 数据集配置
    'model.d_model': 256,                         # 模型维度
    'task.lr': 0.001,                            # 学习率
    'data.batch_size': 32,                       # 批次大小
    'trainer.max_epochs': 50                      # 训练轮数
})
```

#### 链式更新支持

```python
# 多阶段配置链式更新，支持复杂的实验设计
config = (load_config('experiment_3')
           .update({'model.prompt_dim': 128})           # 第一阶段：prompt维度
           .update({'task.contrast_weight': 0.15})      # 第二阶段：对比学习权重
           .update({'data.batch_size': 32})             # 第三阶段：批次大小
           .update({'trainer.devices': 1}))             # 第四阶段：GPU设备
```

### 实际使用示例

#### 单个实验执行

```bash
# 基础实验执行（使用默认配置）
python main.py --config_path configs/experiment_1_direct_fewshot.yaml

# 参数重写实验（单层参数）
python main.py --config_path configs/experiment_1_direct_fewshot.yaml \
               --override task.target_system_id=[1,5,6,12,19] \
               --override task.lr=0.0005

# 深层参数重写（点号展开）
python main.py --config_path configs/experiment_3_hse_prompt_pretrain.yaml \
               --override model.prompt_dim=128 \
               --override model.training_stage=pretrain \
               --override task.few_shot.shots=[5,10,20]
```

#### 批量参数扫描实验

```bash
# Prompt维度扫描实验
for prompt_dim in 64 128 256; do
    python main.py --config_path configs/experiment_3_hse_prompt_pretrain.yaml \
                   --override model.prompt_dim=$prompt_dim \
                   --override task.target_system_id=[1,5,6,12,19]
    echo "Completed experiment with prompt_dim=$prompt_dim"
done

# 学习率扫描实验
for lr in 0.001 0.0005 0.0001; do
    python main.py --config_path configs/experiment_1_direct_fewshot.yaml \
                   --override task.lr=$lr \
                   --override task.target_system_id=[1,5,6,12,19]
    echo "Completed experiment with lr=$lr"
done
```

#### 数据集组合实验

```bash
# 单数据集实验（实验0风格）
for dataset_id in 1 5 6 12 19; do
    python main.py --config_path configs/experiment_0_backbone_head.yaml \
                   --override task.target_system_id=[$dataset_id]
    echo "Completed dataset $dataset_id baseline"
done

# 数据集子集实验
for dataset_subset in "[1,5]" "[6,12]" "[19]"; do
    python main.py --config_path configs/experiment_1_direct_fewshot.yaml \
                   --override task.target_system_id=$dataset_subset
    echo "Completed subset $dataset_subset experiment"
done
```

#### 🤖 自动化批量实验（规划）

完整的自动化批量实验 & WandB 集成方案，已迁移至规划文档：

- 详见 `paper/2025-10_foundation_model_0_metric/plan/automation_scripts.md`
- 包含：
  - 批量实验执行脚本设计（run_all_experiments）
  - 参数扫描脚本（parameter_sweep）
  - 实验监控脚本（monitor_experiments）
  - 结果收集与报告生成脚本（collect_results）

### 高级配置模式

#### 多阶段Pipeline配置

```python
# 复杂的两阶段实验配置
pretrain_config = load_config('experiment_3_hse_prompt_pretrain', {
    'pipeline.stages.pretraining.enabled': True,
    'pipeline.stages.pretraining.epochs': 30,
    'pipeline.stages.pretraining.learning_rate': 0.0005,
    'pipeline.stages.pretraining.freeze_prompt': False,

    'pipeline.stages.finetuning.enabled': True,
    'pipeline.stages.finetuning.epochs': 20,
    'pipeline.stages.finetuning.learning_rate': 0.0001,
    'pipeline.stages.finetuning.freeze_prompt': True,
    'pipeline.stages.finetuning.backbone_lr_multiplier': 0.1
})
```

#### 条件配置和参数依赖

```python
# 基于条件的动态配置
def create_conditional_config(use_prompts=True, dataset_combo='all'):
    base_config = {
        'task.target_system_id': [1, 5, 6, 12, 19]
    }

    if use_prompts:
        base_config.update({
            'model.use_prompt': True,
            'model.prompt_dim': 128,
            'task.prompt_weight': 0.1
        })
    else:
        base_config.update({
            'model.use_prompt': False,
            'model.prompt_dim': 0,
            'task.prompt_weight': 0.0
        })

    if dataset_combo == 'subset':
        base_config['task.target_system_id'] = [1, 6, 12]  # CWRU, THU, JNU

    return base_config

# 使用条件配置
config_with_prompts = create_conditional_config(use_prompts=True, dataset_combo='all')
config_without_prompts = create_conditional_config(use_prompts=False, dataset_combo='subset')
```

### 配置验证和调试

#### 配置完整性检查

```python
from src.configs import load_config, validate_config

# 加载并验证配置
config = load_config('experiment_1', {
    'task.target_system_id': [1, 5, 6, 12, 19]
})

# 验证配置完整性
is_valid, issues = validate_config(config)
if not is_valid:
    print("❌ Configuration validation failed:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("✅ Configuration validation passed")
```

---

## 🚀 快速开始

### 1. 环境配置

```bash
# 克隆项目
git clone <repository_url>
cd PHM-Vibench/paper/2025-10_foundation_model_0_metric

# 创建环境
conda create -n hse-prompt python=3.9
conda activate hse-prompt

# 安装依赖
pip install -r requirements.txt
export PYTHONPATH="${PYTHONPATH}:$(pwd)/../src"
```

### 2. 运行Baseline实验

```bash
# 运行所有Baseline实验
python scripts/run_all_baseline_experiments.py \
    --config_dir configs/ \
    --output_dir results/baseline_experiments

# 生成基线表格
python scripts/generate_paper_tables.py \
    --results_dir results/baseline_experiments \
    --output_dir results/paper_tables
```

### 3. 运行优化策略验证

```bash
# 消融研究
python scripts/run_optimization_experiments.py \
    --config_path configs/optimization_ablation.yaml

# Backbone普适性验证
python scripts/run_optimization_experiments.py \
    --config_path configs/backbone_universality.yaml

# 噪声鲁棒性评估
python scripts/run_optimization_experiments.py \
    --config_path configs/noise_robustness.yaml
```

### 4 完整自动化脚本系统

```bash
# 进入项目根目录
cd /home/user/LQ/B_Signal/Signal_foundation_model/Vbench

# 🎯 基线实验 (实验0-3) - 1-2天
bash paper/2025-10_foundation_model_0_metric/run_experiment0.sh  # Backbone+Head基线
bash paper/2025-10_foundation_model_0_metric/run_experiment1.sh  # HSE直接Few-Shot
bash paper/2025-10_foundation_model_0_metric/run_experiment2.sh  # 无监督预训练
bash paper/2025-10_foundation_model_0_metric/run_experiment3.sh  # HSE-Prompt完整方法

# 🔬 优化策略验证 (实验4-7) - 3-5天
bash paper/2025-10_foundation_model_0_metric/run_experiment4.sh  # 组件消融研究
bash paper/2025-10_foundation_model_0_metric/run_experiment5.sh  # 少样本梯度扫描
bash paper/2025-10_foundation_model_0_metric/run_experiment6.sh  # Backbone普适性验证
bash paper/2025-10_foundation_model_0_metric/run_experiment7.sh  # 噪声鲁棒性评估

# 📊 完整实验矩阵 (总计约72次独立运行)
# 实验0: 5数据集 × 1种子 = 5次
# 实验1: 1统一模型 × 5种子 = 5次
# 实验2: 1统一模型 × 5种子 × 2阶段 = 5次
# 实验3: 1统一模型 × 5种子 × 2阶段 = 5次
# 实验4: 6组件配置 × 5种子 = 30次
# 实验5: 6shots设置 × 3种子 = 18次
# 实验6: 4Backbone × 3种子 = 12次
# 实验7: 4SNR设置 × 3种子 = 12次

# 💡 完整执行提示:
# - 基线实验优先 (实验0-3) → 优化验证 (实验4-7)
# - 总计约61 GPU小时 (RTX 4090)
# - 所有脚本自动处理日志、结果组织和错误恢复
```

---

## 🚀 零歧义实验执行指南

为确保实验执行的准确性和可重现性，提供详细的执行指南和批量脚本示例。

### 📋 论文实验执行流程

#### 阶段1: 环境验证 (5分钟)

```bash
# 进入项目根目录
cd /home/user/LQ/B_Signal/Signal_foundation_model/Vbench

# 运行环境验证脚本
bash paper/2025-10_foundation_model_0_metric/scripts/validation/quick_validation.sh

# 预期输出: ✅ 所有检查通过，环境准备就绪
```

#### 阶段2: 基线实验建立 (预计1-2小时)

**🎉 一键执行：使用自动化脚本**

```bash
# 进入项目根目录
cd /home/user/LQ/B_Signal/Signal_foundation_model/Vbench

# 实验0: Backbone+Head独立基线 (5个数据集分别训练)
bash paper/2025-10_foundation_model_0_metric/run_experiment0.sh

# 实验1: HSE + 直接Few-Shot学习 (5种子的统一训练)
bash paper/2025-10_foundation_model_0_metric/run_experiment1.sh

# 实验2: 无监督预训练 + Few-shot微调 (两阶段训练)
bash paper/2025-10_foundation_model_0_metric/run_experiment2.sh

# 实验3: HSE-Prompt完整方法 (两阶段训练，包含Prompt机制)
bash paper/2025-10_foundation_model_0_metric/run_experiment3.sh
```

**💡 执行说明：**
- 每个脚本自动处理多种子循环、结果组织和日志记录
- 预期性能递增：实验0 (65-70%) → 实验1 (70-75%) → 实验2 (80-85%) → 实验3 (87-92%)
- 结果保存在 `paper/2025-10_foundation_model_0_metric/results/experiment_X/`

#### 阶段2.5: 优化策略验证 (预计3-5天)

**🎉 一键执行：优化策略自动化脚本**

```bash
# 进入项目根目录
cd /home/user/LQ/B_Signal/Signal_foundation_model/Vbench

# 实验4: 组件消融研究 (量化HSE、Prompt、融合方式贡献)
bash paper/2025-10_foundation_model_0_metric/run_experiment4.sh

# 实验5: 少样本梯度扫描 (shots=[1,3,5,10,15,20]性能曲线)
bash paper/2025-10_foundation_model_0_metric/run_experiment5.sh

# 实验6: Backbone普适性验证 (B_04/B_06/B_08/B_09跨架构测试)
bash paper/2025-10_foundation_model_0_metric/run_experiment6.sh

# 实验7: 噪声鲁棒性评估 (SNR=[20,10,5,0]dB稳定性测试)
bash paper/2025-10_foundation_model_0_metric/run_experiment7.sh
```

**💡 优化实验说明：**
- **实验4**：6种组件配置 × 5种子 = 30次运行，量化各组件贡献度
- **实验5**：6种shots设置 × 3种子 = 18次运行，绘制少样本性能曲线
- **实验6**：4种Backbone × 3种子 = 12次运行，验证架构泛化性
- **实验7**：4种SNR设置 × 3种子 = 12次运行，测试噪声鲁棒性

#### 阶段3: 结果收集与分析 (10分钟)

```bash
# 收集所有实验结果
python paper/2025-10_foundation_model_0_metric/scripts/analysis/collect_results.py \
    --results_dir "results/" \
    --output_dir "paper/2025-10_foundation_model_0_metric/results/summary"

# 生成性能分析报告
python paper/2025-10_foundation_model_0_metric/scripts/analysis/performance_analysis.py \
    --results_dir "paper/2025-10_foundation_model_0_metric/results/summary" \
    --output_dir "paper/2025-10_foundation_model_0_metric/results/analysis"

# 生成论文表格
python paper/2025-10_foundation_model_0_metric/scripts/generate_paper_tables.py \
    --results_dir "paper/2025-10_foundation_model_0_metric/results/summary" \
    --output_dir "paper/2025-10_foundation_model_0_metric/paper_tables"

echo "🎉 所有实验完成，结果已保存到 paper_tables/ 目录"
```

### 📁 结果文件命名规范

#### 标准化文件命名结构

为确保实验结果的组织性和可追溯性，所有实验结果文件采用以下命名规范：

```
results/
├── experiment_0_backbone_head/
│   ├── dataset_1/
│   │   ├── seed_42/
│   │   │   ├── model_checkpoint.pt
│   │   │   ├── training_log.json
│   │   │   ├── metrics.json
│   │   │   └── config_backup.yaml
│   │   └── seed_123/  # ... 其他种子
│   ├── dataset_5/      # ... 其他数据集
│   └── aggregated_results.json
├── experiment_1_direct_fewshot/
│   ├── seed_42/
│   │   ├── model_checkpoint.pt
│   │   ├── training_log.json
│   │   ├── metrics.json
│   │   ├── confusion_matrix.png
│   │   └── config_backup.yaml
│   └── seed_123/      # ... 其他种子
├── experiment_2_unsupervised_pretrain/
│   ├── seed_42/
│   │   ├── model_checkpoint.pt
│   │   ├── training_log.json
│   │   ├── metrics.json
│   │   └── config_backup.yaml
│   └── seed_123/      # ... 其他种子
├── experiment_3_hse_prompt_pretrain/
│   └── seed_42/       # ... 其他种子
├── experiment_4_ablation/
│   └── component_wise_results/
├── experiment_5_fewshot_sweep/
│   ├── shots_1/
│   ├── shots_3/
│   └── shots_5/
├── experiment_6_backbone_universality/
│   ├── B_04_Dlinear/
│   ├── B_06_TimesNet/
│   ├── B_08_PatchTST/
│   └── B_09_FNO/
└── experiment_7_noise_robustness/
    ├── SNR_20dB/
    ├── SNR_10dB/
    └── SNR_5dB/
```

#### 详细命名规范

**基线实验 (实验0-3)**:

- `{experiment_name}_dataset_{dataset_id}_seed_{seed}/metrics.json`
- `{experiment_name}_dataset_{dataset_id}_seed_{seed}_model_checkpoint.pt`
- `{experiment_name}_dataset_{dataset_id}_seed_{seed}_training_log.json`

**优化策略实验 (实验4-7)**:

- **实验4 (消融)**: `experiment_4_ablation_component_{component}_seed_{seed}_metrics.json`
- **实验5 (少样本扫描)**: `experiment_5_fewshot_shots_{shot}_seed_{seed}_metrics.json`
- **实验6 (Backbone普适性)**: `experiment_6_backbone_{backbone}_seed_{seed}_metrics.json`
- **实验7 (噪声鲁棒性)**: `experiment_7_noise_SNR_{snr}_seed_{seed}_metrics.json`

**聚合结果文件**:

- `{experiment_name}_aggregated_summary.json`
- `{experiment_name}_performance_summary.csv`
- `all_experiments_comparison.json`

#### Metrics文件格式

```json
{
  "experiment_name": "experiment_3_hse_prompt_pretrain",
  "dataset_id": 1,
  "seed": 42,
  "model_config": {
    "name": "M_02_ISFM_Prompt",
    "embedding": "E_01_HSE_v2",
    "backbone": "B_04_Dlinear",
    "task_head": "H_01_Linear_cla"
  },
  "training_config": {
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 50,
    "target_system_id": [1, 5, 6, 12, 19]
  },
  "results": {
    "accuracy": 0.8923,
    "f1_macro": 0.8871,
    "precision_macro": 0.8895,
    "recall_macro": 0.8848,
    "training_time_seconds": 5423.7,
    "inference_time_ms": 2.34,
    "memory_usage_mb": 1247
  },
  "cross_dataset_results": {
    "CWRU": {"accuracy": 0.9123, "f1_macro": 0.9087},
    "Ottawa": {"accuracy": 0.8765, "f1_macro": 0.8721},
    "THU": {"accuracy": 0.8989, "f1_macro": 0.8943},
    "JNU": {"accuracy": 0.8834, "f1_macro": 0.8789},
    "HUST": {"accuracy": 0.8896, "f1_macro": 0.8856}
  },
  "timestamp": "2025-01-28T14:23:45Z",
  "git_commit": "abc123def456"
}
```

### 🎯 快速执行命令总结

**🎉 完整自动化实验执行 (8个专用脚本)**:

```bash
# 进入项目根目录
cd /home/user/LQ/B_Signal/Signal_foundation_model/Vbench

# 🎯 基线实验 (实验0-3)
bash paper/2025-10_foundation_model_0_metric/run_experiment0.sh  # Backbone+Head基线
bash paper/2025-10_foundation_model_0_metric/run_experiment1.sh  # HSE直接Few-Shot
bash paper/2025-10_foundation_model_0_metric/run_experiment2.sh  # 无监督预训练
bash paper/2025-10_foundation_model_0_metric/run_experiment3.sh  # HSE-Prompt完整方法

# 🔬 优化策略验证 (实验4-7)
bash paper/2025-10_foundation_model_0_metric/run_experiment4.sh  # 组件消融研究
bash paper/2025-10_foundation_model_0_metric/run_experiment5.sh  # 少样本梯度扫描
bash paper/2025-10_foundation_model_0_metric/run_experiment6.sh  # Backbone普适性验证
bash paper/2025-10_foundation_model_0_metric/run_experiment7.sh  # 噪声鲁棒性评估
```

**📊 专用脚本说明**:
- **实验0**: 独立数据集基线 → 5数据集 × 1种子 = 5次运行
- **实验1**: 统一模型Few-Shot → 1模型 × 5种子 = 5次运行
- **实验2**: 两阶段无监督预训练 → 1模型 × 5种子 × 2阶段 = 5次运行
- **实验3**: 两阶段HSE-Prompt → 1模型 × 5种子 × 2阶段 = 5次运行
- **实验4**: 组件消融研究 → 6配置 × 5种子 = 30次运行
- **实验5**: 少样本梯度扫描 → 6shots × 3种子 = 18次运行
- **实验6**: Backbone普适性 → 4架构 × 3种子 = 12次运行
- **实验7**: 噪声鲁棒性 → 4SNR × 3种子 = 12次运行

**🔧 单个实验调试**:

```bash
# 快速验证实验 (1 epoch, 单数据集)
python main.py \
    --config_path "paper/2025-10_foundation_model_0_metric/configs/experiment_1_direct_fewshot.yaml" \
    --override "task.target_system_id=[1]" \
    --override "trainer.max_epochs=1"

# 完整实验 (50 epochs, 多数据集)
python main.py \
    --config_path "paper/2025-10_foundation_model_0_metric/configs/experiment_1_direct_fewshot.yaml" \
    --override "task.target_system_id=[1,5,6,12,19]"
```

---

## 📊 实验配置详解

### 🔧 实验验证标准

为确保实验结果的可靠性和可重现性，所有实验遵循统一的验证标准：

#### 重复实验设计

- **标准重复次数**: 每个实验配置使用5个不同随机种子
- **种子序列**: [42, 123, 456, 789, 999] - 确保跨实验一致性
- **环境隔离**: 每次重复独立的环境初始化
- **结果报告**: 均值 ± 标准差格式 (mean ± std)

#### 性能指标标准

- **主要指标**: Accuracy, F1-Score (macro平均)
- **辅助指标**: Precision, Recall (macro平均)
- **效率指标**: 训练时间, 推理时间, 内存占用
- **稳定性指标**: 标准差, 性能一致性

#### 结果验证方法

```python
# 标准性能指标计算
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def calculate_metrics(y_true, y_pred):
    """计算标准性能指标"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro')
    }
```

### 统一配置原则

所有实验遵循以下统一配置原则（按 embedding / 模型 设计区分）：

- **Backbone**: 基线与优化实验主干网络统一在 Dlinear 系列上对齐，突出 HSE / HSE-Prompt 优化策略本身的贡献。
- **Embedding 与模型映射**:
  - **实验0（非 HSE 基线）**  
    - 模型: `M_01_ISFM`，Embedding: `E_03_Patch`。  
    - `E_03_Patch` 只从模型配置 `args_m` 读取信息（例如 `window_size`、`patch_size_L`、`input_dim`、`d_model`、`output_dim`），并不再对输入长度做强 assert：  
      - `self.seq_len = data.window_size`，`num_patches = seq_len // patch_size_L`；  
      - 输入形状由 DataLoader 保证为 `(B, L, C)`，其中 `C` 与 `model.input_dim` 对齐。  
    - 实验0在 5 个数据集分别训练，通道数可能不同，因此需要在运行前确认 `model.input_dim` 与各数据集真实通道数匹配（否则应通过 override 单独调整）。
  - **实验1–2（HSE 系列，无 Prompt）**  
    - 模型: `M_01_ISFM`（不带 Prompt）；  
    - Embedding: HSE 系列（`E_01_HSE` / `E_02_HSE_v2`，当前 YAML 统一使用 `E_01_HSE`）；  
    - 实验1：HSE + 直接 Few-Shot；  
    - 实验2：HSE + 无 Prompt 的两阶段预训练（`Pipeline_02_pretrain_fewshot`，pretrain + few-shot，两个阶段都保持 HSE 而不过早引入 Prompt）。
  - **实验3–7（HSE-Prompt 系列）**  
    - 模型: `M_02_ISFM_Prompt`；  
    - Embedding: 统一为 `HSE_prompt`（`src/model_factory/ISFM_Prompt/embedding/HSE_prompt.py`），不再混用旧的 `E_01_HSE_v2`：  
      - 实验3：完整 HSE-Prompt 方法（两阶段：HSE_prompt 预训练 + Few-shot 微调）；  
      - 实验4–7：在同一 HSE-Prompt 架构上做组件消融、少样本梯度、Backbone 普适性、噪声鲁棒性等验证。  
    - `HSE_prompt` 的前向接口为 `HSE_prompt(x, fs, dataset_ids)`，其中 `x` 形状为 `(B, L, C)`；`fs` 和 `dataset_ids` 由 metadata 中的 `Sample_rate`、`Dataset_id` 提供，由 `M_02_ISFM_Prompt._embed` 负责拼装。
- **Head**: 统一使用线性分类头（`H_01_Linear_cla`）或其多任务扩展，方便跨实验比较。
- **数据集**: 5个目标系统 `[1,5,6,12,19]` 统一评估标准。
- **重复验证**: 核心实验采用多种子重复，确保结果具有统计可靠性。

### 数据集配置

**统一数据集配置**: `target_system_id: [1, 5, 6, 12, 19]`

| 系统ID | 数据集 | 机构                                          | 设备类型 |
| ------ | ------ | --------------------------------------------- | -------- |
| 1      | CWRU   | Case Western Reserve University               | 轴承     |
| 5      | Ottawa | University of Ottawa                          | 轴承     |
| 6      | THU    | Tsinghua University                           | 轴承     |
| 12     | JNU    | jiangnan University                           | 轴承     |
| 19     | HUST   | Huazhong University of Science and Technology | 风机     |

### 配置文件结构

```yaml
# 基础配置
experiment_name: "实验名称"
description: "实验描述"

# 数据配置
data:
  target_system_id: [1, 5, 6, 12, 19]  # CWRU, Ottawa, THU, JNU, HUST
  few_shot:
    shots: [1, 2, 3, 4, 5]  # 或 [5] 统一5-shot
  pretraining:
    enabled: true/false
    method: "hse_prompt"/"unsupervised"/None

# 模型配置
model:
  name: "M_01_ISFM"  # 或 M_02_ISFM_Prompt
  # Embedding 选择示意（具体以各 experiment_X YAML 为准）:
  # - 实验0:  E_03_Patch           (Patch-based baseline embedding)
  # - 实验1–2: E_01_HSE / E_02_HSE_v2 (HSE 系列)
  # - 实验3–7: HSE_prompt           (HSE + Prompt)
  embedding: "E_0[X]_[Embedding_Type]"  # 标准命名占位符
  backbone: "B_04_Dlinear"  # 统一backbone
  task_head: "H_01_Linear_cla"  # 多任务分类head

# 训练配置
task:
  training:
    batch_size: 16
    max_epochs: 100
  optimizer:
    name: "adam"
    lr: 0.001
```

### 📁 配置文件标准化规范

为确保实验配置的一致性和可维护性，所有配置文件严格遵循Vbench标准化命名和结构规范。

#### 标准命名规范

**配置文件目录结构**:

```
paper/2025-10_foundation_model_0_metric/
└── configs/
    ├── experiment_0_backbone_head.yaml           # 实验0: Backbone+Head基线
    ├── experiment_1_direct_fewshot.yaml          # 实验1: 直接Few-Shot
    ├── experiment_2_unsupervised_pretrain.yaml   # 实验2: 无监督预训练
    ├── experiment_3_hse_prompt_pretrain.yaml     # 实验3: HSE-Prompt预训练
    ├── experiment_4_ablation.yaml                # 实验4: 组件消融研究
    ├── experiment_5_fewshot_sweep.yaml           # 实验5: 少样本梯度扫描
    ├── experiment_6_backbone_universality.yaml   # 实验6: Backbone普适性验证
    ├── experiment_7_noise_robustness.yaml        # 实验7: 噪声鲁棒性评估
    ├── unified_metric_main.yaml                  # 统一度量学习主要配置
    ├── baseline_direct_fewshot.yaml              # 直接Few-Shot基线
    ├── baseline_unsupervised_pretraining.yaml    # 无监督预训练基线
    ├── baseline_hse_prompt_pretraining.yaml      # HSE-Prompt基线
    └── README.md                                 # 配置使用说明
```

#### Vbench组件命名标准

**✅ 正确的组件命名格式**:

```yaml
model:
  name: "M_02_ISFM_Prompt"           # ✅ 标准: M_XX_组件名称
  embedding: "E_01_HSE_v2"           # ✅ 标准: E_XX_组件名称
  backbone: "B_08_PatchTST"          # ✅ 标准: B_XX_组件名称
  task_head: "H_01_Linear_cla"       # ✅ 标准: H_XX_组件名称
```

**❌ 错误的组件命名格式**:

```yaml
model:
  name: "HSE_Prompt"                 # ❌ 错误: 缺少M_XX_前缀
  embedding: "HSE_v2"                # ❌ 错误: 缺少E_XX_前缀
  backbone: "PatchTST"               # ❌ 错误: 缺少B_XX_前缀
  task_head: "Linear_cla"            # ❌ 错误: 缺少H_XX_前缀
```

#### 标准配置模板

**实验配置模板**:

```yaml
# =============================================================================
# 实验[编号]: [实验描述]
# 目标: [实验目标]
# 预期性能: [性能范围]
# =============================================================================
# Environment Configuration
environment:
  VBENCH_HOME: "/home/user/LQ/B_Signal/Signal_foundation_model/Vbench"
  project: "experiment_[number]_[description]"
  seed: 42
  output_dir: "results/experiment_[number]"
  notes: "[详细的实验描述]"
  iterations: 5  # 统计重复次数
  wandb: true

# Data Configuration
data:
  data_dir: "/home/user/data/PHMbenchdata/PHM-Vibench"
  metadata_file: "metadata.xlsx"
  batch_size: 32
  num_workers: 8
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  normalization: "standardization"
  window_size: 4096
  stride: 5
  num_window: 64
  dtype: "float32"
  pin_memory: true

# Model Configuration (Vbench Standard Components)
model:
  name: "M_0[X]_[Model_Type]"              # 标准模型命名
  type: "[Model_Type]"

  # Architecture components
  embedding: "E_0[X]_[Embedding_Type]"     # 标准嵌入命名
  backbone: "B_0[X]_[Backbone_Type]"       # 标准骨干命名
  task_head: "H_0[X]_[Task_Head_Type]"     # 标准任务头命名

  # Model dimensions
  input_dim: 1
  d_model: 256
  output_dim: 128

  # Transformer configuration
  num_heads: 8
  num_layers: 4
  e_layers: 4
  d_ff: 512
  dropout: 0.1
  activation: "relu"
  factor: 5

  # Experiment-specific configuration
  use_prompt: false  # 根据实验调整
  prompt_dim: 128    # 根据实验调整
  training_stage: "direct_fewshot"  # 根据实验调整

  # Memory optimization
  gradient_checkpointing: true
  mixed_precision: true

# Task Configuration
task:
  name: "[task_name]"
  type: "CDDG"

  # Multi-domain target setup
  target_system_id: [1, 5, 6, 12, 19]  # 统一数据集配置
  target_domain_num: 5

  # Few-shot learning configuration
  few_shot:
    enabled: true
    shots: [5]  # 5-shot学习
    samples_per_class: [5]
    support_query_split: 0.5

  # Training hyperparameters
  loss: "CE"
  metrics: ["acc", "f1", "precision", "recall"]
  optimizer: "adamw"
  lr: 0.001
  weight_decay: 0.0001
  epochs: 50
  early_stopping: true
  es_patience: 15
  backbone_lr_multiplier: 1.0

  # Experiment-specific task configuration
  contrast_loss: "CE"  # 根据实验调整
  contrast_weight: 0.0  # 根据实验调整
  temperature: 0.07
  use_momentum: false
  prompt_weight: 0.0     # 根据实验调整
  use_system_sampling: false

# Trainer Configuration
trainer:
  name: "Default_trainer"

  # PyTorch Lightning configuration
  max_epochs: 50
  gpus: 1
  auto_select_gpus: true
  progress_bar_refresh_rate: 20
  check_val_every_n_epoch: 5
  deterministic: true
  gradient_clip_val: 1.0

  # Experiment settings
  num_runs: 5  # 5 seeds for statistical significance
  save_dir: "results/experiment_[number]"

  # Logging configuration
  use_wandb: true
  save_checkpoints: true
  log_every_n_steps: 10

  # Performance optimization
  accelerator: "gpu"
  devices: 1
  precision: 16
  accumulate_grad_batches: 1

  # Early stopping
  early_stopping: true
  patience: 15

# Pipeline Configuration
pipeline:
  name: "Pipeline_0[X]_[pipeline_type]"

  # Experiment-specific pipeline configuration
  stages:
    # 根据实验需求配置阶段
    [stages_configuration]

# Analysis Configuration
analysis:
  # Statistical analysis
  significance_test:
    method: "wilcoxon"
    alpha: 0.05
    effect_size: "cohen_d"

  # Result collection
  collect_metrics:
    - "accuracy"
    - "f1_score"
    - "precision"
    - "recall"
    - "confusion_matrix"
    - "roc_auc"

  # Visualization
  generate_plots:
    - "confusion_matrix"
    - "learning_curves"
    - "performance_comparison"

  # Export formats
  export_formats: ["json", "csv", "latex"]
```

#### 配置验证清单

**配置文件完整性检查**:

- [ ] `environment` 部分包含所有必需字段
- [ ] `data` 部分包含正确的数据集配置
- [ ] `model` 部分使用Vbench标准组件命名
- [ ] `task` 部分包含 `target_system_id: [1, 5, 6, 12, 19]`
- [ ] `trainer` 部分包含合理的训练参数
- [ ] 所有路径引用使用正确的绝对路径
- [ ] 配置文件格式为有效的YAML

**组件命名验证**:

- [ ] 模型名称以 `M_` 开头
- [ ] 嵌入组件以 `E_` 开头
- [ ] 骨干组件以 `B_` 开头
- [ ] 任务头组件以 `H_` 开头
- [ ] 所有组件名称遵循 Vbench 标准

---

## 📋 实验流程指南

### 阶段1: 基线建立 (1-2天)

**目标**: 建立渐进式性能基线

**步骤**:

1. 运行Backbone+Head基线实验 (表1，实验0)
2. 运行直接Few-Shot实验 (表2，实验1)
3. 运行HSE预训练实验 (表3，实验2-3)

**验证要点**:

- 实验0 < 实验1 < 实验2 < 实验3 的性能递增
- 随shot增加性能递增

### 阶段2: 优化策略验证 (3-5天)

**目标**: 系统验证HSE-Prompt各组件效果

**步骤**:

1. 消融研究实验 (表8，实验5)
2. Backbone普适性验证 (表5，实验6)
3. 噪声鲁棒性评估 (表7，实验7)
4. 少样本学习性能 (表6，实验5)

**验证要点**:

- 各组件贡献量化
- 跨backbone一致性
- 噪声环境稳定性
- 极限样本性能

### 阶段3: 结果整合 (1天)

**目标**: 生成标准化论文表格

**步骤**:

1. 自动生成所有表格
2. 可视化图表生成
3. 复现性检查

---

## 🔧 故障排除

### 常见问题

1. **环境问题**: 依赖版本冲突 → 使用requirements-lock.txt
2. **内存不足**: 减少batch_size → 修改配置文件
3. **数据问题**: 路径错误 → 检查data_dir配置
4. **GPU问题**: CUDA错误 → 检查驱动和版本匹配

### 调试工具

```bash
# 查看实验日志
tail -f baseline_experiment.log

# 检查GPU使用
nvidia-smi

# 验证配置文件
python scripts/validate_config.py --config_path configs/xxx.yaml

# 单步调试
python scripts/debug_experiment.py --config_path configs/xxx.yaml --debug
```

---

## 📚 文档结构

### 配置文档

- [配置文件说明](configs/README.md) - 详细配置参数说明
- [实验设计文档](experiments/) - 实验设计理念和方案

### 执行指南

- [实验执行指南](docs/EXPERIMENT_GUIDE.md) - 详细执行步骤
- [复现指南](docs/REPRODUCTION_GUIDE.md) - 完整复现流程

### 结果文档

- [结果表格模板](results/Result_Tables_Template.md) - 表格格式说明

---

## 🤝 贡献指南

### 提交Issue

- Bug报告: 详细描述问题、环境、复现步骤
- 功能建议: 描述需求、使用场景、预期效果
- 文档改进: 指出文档问题、提供改进建议

### 提交PR

- 代码遵循项目规范
- 添加必要的测试
- 更新相关文档
- 通过CI检查

### 开发环境

```bash
# 安装开发依赖
pip install -r requirements-dev.txt

# 运行测试
pytest tests/

# 代码格式化
black scripts/ configs/
flake8 scripts/ configs/
```

---

## 📋 详细实验计划

🎯 **完整实验设计方案**: 详见 [Experimental_Plan.md](experiments/Experimental_Plan.md)

该文档包含详细的实验设计方案，包括：

- **实验0-8完整设计**: 从基线到高级的渐进式验证
- **技术实现细节**: 每个实验的精确配置和性能预期
- **资源配置规划**: 详细的计算资源和时间需求评估

---

## 📄 许可证

本项目采用MIT许可证，详见[LICENSE](LICENSE)文件。

---

## 📞 联系方式

- **项目维护**: PHM-Vibench Team
- **技术支持**: [GitHub Issues](https://github.com/your-repo/issues)
- **学术合作**: research@example.com

---

## 🙏 致谢

感谢以下开源项目和贡献者：

- PyTorch Lightning
- SciPy
- Hugging Face Transformers
- 以及所有为本项目做出贡献的研究人员

---

**最后更新**: 2025-01-28
**版本**: v1.1
**项目状态**: 活跃开发中

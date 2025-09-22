# 阶段3: 实验执行指南

ContrastiveIDTask实验设计、执行和监控的完整指南。

## 📋 本阶段目标

- [x] 设计系统性实验方案
- [x] 执行单数据集和跨数据集实验
- [x] 进行全面的消融研究
- [x] 监控和优化实验过程

## 🚀 快速开始

### 1. 单数据集实验
```bash
python multi_dataset_runner.py \
    --datasets CWRU \
    --strategy single \
    --config ../examples/config_templates/single_dataset.yaml \
    --output_dir results/single_cwru
```

### 2. 跨数据集域泛化
```bash
python multi_dataset_runner.py \
    --datasets CWRU,XJTU \
    --strategy cross_domain \
    --config ../examples/config_templates/cross_domain.yaml \
    --output_dir results/cross_domain
```

### 3. 消融研究
```bash
python ablation_study.py \
    --config base_config.yaml \
    --parameters temperature,window_size,batch_size \
    --output_dir results/ablation
```

## 🛠️ 核心工具详解

### multi_dataset_runner.py
**主要功能**: 统一的多数据集实验管理器

#### 实验策略
```bash
# 单数据集实验
python multi_dataset_runner.py --strategy single --datasets CWRU

# 跨数据集域泛化 (源→目标)
python multi_dataset_runner.py --strategy cross_domain --datasets CWRU,XJTU

# 多数据集联合训练
python multi_dataset_runner.py --strategy multi_dataset --datasets CWRU,XJTU,PU

# 域自适应实验
python multi_dataset_runner.py --strategy domain_adaptation --datasets CWRU,XJTU
```

#### 高级选项
```bash
# 并行执行多个实验
python multi_dataset_runner.py \
    --datasets CWRU,XJTU,PU,FEMTO \
    --strategy cross_domain \
    --parallel \
    --max_workers 4

# 自动超参数网格搜索
python multi_dataset_runner.py \
    --datasets CWRU \
    --strategy single \
    --grid_search \
    --param_grid config/grid_search.yaml

# 继续中断的实验
python multi_dataset_runner.py --resume --checkpoint_dir results/interrupted_exp/
```

### ablation_study.py
**主要功能**: 系统性超参数消融研究

#### 单参数扫描
```bash
# 温度参数消融
python ablation_study.py \
    --config base_config.yaml \
    --param_sweep temperature 0.01,0.05,0.07,0.1,0.2,0.5 \
    --dataset CWRU \
    --output_dir results/temp_ablation

# 窗口大小消融
python ablation_study.py \
    --config base_config.yaml \
    --param_sweep window_size 128,256,512,1024 \
    --dataset CWRU
```

#### 多参数组合
```bash
# 多参数网格搜索
python ablation_study.py \
    --config base_config.yaml \
    --parameters temperature,window_size,batch_size \
    --max_combinations 50 \
    --optimization_metric accuracy

# 贝叶斯优化
python ablation_study.py \
    --config base_config.yaml \
    --parameters temperature,lr,weight_decay \
    --optimizer bayesian \
    --n_trials 100
```

## 📊 实验设计方案

### 🎯 基础实验矩阵

| 实验类型 | 数据集组合 | 目的 | 预期结果 |
|----------|------------|------|----------|
| Baseline | CWRU单独 | 建立基准性能 | ~75-85% |
| Cross-Domain | CWRU→XJTU | 测试域泛化能力 | ~60-75% |
| Multi-Source | CWRU+XJTU→PU | 多源域预训练 | ~70-80% |
| Few-Shot | CWRU→XJTU(5%) | 少样本适应 | ~50-65% |

### 🧪 消融研究设计

#### 核心参数消融
```yaml
# config/ablation_params.yaml
temperature:
  values: [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5]
  priority: high

window_size:
  values: [64, 128, 256, 512, 1024]
  priority: high

num_window:
  values: [1, 2, 3, 4, 5]
  priority: medium

batch_size:
  values: [8, 16, 32, 64]
  priority: medium
```

#### 架构组件消融
```bash
# 损失函数消融
python ablation_study.py --param_sweep loss_type infonce,simclr,triplet,contrastive

# 采样策略消融
python ablation_study.py --param_sweep sampling_strategy random,sequential,evenly_spaced

# 特征维度消融
python ablation_study.py --param_sweep d_model 32,64,128,256,512
```

## 📈 实验监控与管理

### 🔍 实验状态监控
```bash
# 查看所有实验状态
python multi_dataset_runner.py --status --output_dir results/

# 实时监控训练进度
tail -f results/experiment_name/training.log

# GPU使用监控
watch -n 1 nvidia-smi

# 实验资源监控
python monitor_experiments.py --output_dir results/ --refresh 10
```

### 📊 实验进度可视化
```python
# 启动实验监控界面
from multi_dataset_runner import ExperimentMonitor

monitor = ExperimentMonitor('results/')
monitor.start_web_interface(port=8080)
# 访问 http://localhost:8080 查看进度
```

### 🚨 实验异常处理
```bash
# 自动重启失败实验
python multi_dataset_runner.py --auto_restart --check_interval 600

# 实验健康检查
python experiment_health_check.py --results_dir results/ --fix_issues
```

## 🎯 实验最佳实践

### 🔄 实验版本管理
```bash
# 每个实验保存完整配置
export EXPERIMENT_NAME="cwru_baseline_v1.0"
python multi_dataset_runner.py \
    --config base_config.yaml \
    --experiment_name $EXPERIMENT_NAME \
    --save_config \
    --git_commit

# 实验结果版本控制
git add results/$EXPERIMENT_NAME/
git commit -m "Add experiment: $EXPERIMENT_NAME"
git tag exp-$EXPERIMENT_NAME
```

### 📝 实验记录管理
```python
# experiment_logger.py
class ExperimentLogger:
    def __init__(self, experiment_name):
        self.name = experiment_name
        self.start_time = time.time()

    def log_hyperparams(self, config):
        """记录超参数配置"""

    def log_metrics(self, metrics, step):
        """记录训练指标"""

    def log_artifacts(self, file_paths):
        """记录实验产物"""
```

### 🎲 随机种子管理
```python
# 确保实验可重现
def set_deterministic_training():
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

## 🧰 高级实验技巧

### 并行实验执行
```bash
# 使用GNU parallel执行多个独立实验
cat experiment_list.txt | parallel -j 4 python multi_dataset_runner.py --config {}

# 分布式训练
torchrun --nproc_per_node=4 multi_dataset_runner.py --distributed

# 集群批处理
sbatch --array=1-10 run_ablation_array.sh
```

### 早停和检查点管理
```python
# 智能早停策略
early_stopping_config = {
    'patience': 10,
    'min_delta': 0.001,
    'monitor': 'val_accuracy',
    'mode': 'max'
}

# 检查点保存策略
checkpoint_config = {
    'save_top_k': 3,
    'monitor': 'val_accuracy',
    'every_n_epochs': 5,
    'save_last': True
}
```

### 动态超参数调整
```python
# 学习率调度
scheduler_config = {
    'type': 'cosine_annealing',
    'T_max': 100,
    'eta_min': 1e-6
}

# 温度参数衰减
temperature_schedule = {
    'initial': 0.1,
    'decay_rate': 0.95,
    'decay_steps': 10
}
```

## 📊 实验结果分析

### 实验结果汇总
```bash
# 生成实验汇总报告
python analyze_experiments.py \
    --results_dir results/ \
    --output_report experiment_summary.html

# 导出结果到表格
python export_results.py \
    --results_dir results/ \
    --format csv,json,latex \
    --metrics accuracy,f1_score,precision,recall
```

### 统计显著性检验
```python
# 多实验统计分析
from scipy import stats
from multi_dataset_runner import ResultsAnalyzer

analyzer = ResultsAnalyzer()

# 加载多次运行结果
results_A = analyzer.load_experiment_results('results/method_A/')
results_B = analyzer.load_experiment_results('results/method_B/')

# t-test
t_stat, p_value = stats.ttest_ind(results_A, results_B)
print(f"统计显著性: p-value = {p_value:.4f}")

# 效应量计算
cohen_d = analyzer.compute_effect_size(results_A, results_B)
print(f"Cohen's d: {cohen_d:.4f}")
```

## 🔧 故障排除

### ❌ 训练不收敛
```bash
# 诊断训练问题
python diagnose_training.py --experiment_dir results/problematic_exp/

# 调试建议
python ablation_study.py \
    --config debug_config.yaml \
    --param_sweep lr 1e-4,5e-4,1e-3,5e-3 \
    --debug_mode
```

### ❌ 内存溢出 (OOM)
```python
# 动态批大小调整
def find_optimal_batch_size(initial_size=32):
    for batch_size in [initial_size//2, initial_size//4, initial_size//8]:
        try:
            run_training(batch_size=batch_size)
            return batch_size
        except torch.cuda.OutOfMemoryError:
            continue
    raise RuntimeError("无法找到合适的批大小")
```

### ❌ 实验中断恢复
```bash
# 自动恢复中断的实验
python multi_dataset_runner.py \
    --resume_from results/interrupted_exp/checkpoints/last.ckpt \
    --continue_training
```

## 🎯 进入下一阶段

### 检查清单
- [ ] 基线实验完成且性能合理
- [ ] 跨数据集实验显示域泛化能力
- [ ] 消融研究覆盖关键超参数
- [ ] 实验结果已保存并版本控制

### 实验质量评估
```bash
# 验证实验完整性
python validate_experiments.py --results_dir results/ --check_completeness

# 生成实验质量报告
python experiment_quality_check.py --results_dir results/
```

### 下一步行动
```bash
# 进入结果分析阶段
cd ../04_analysis/

# 开始性能基准测试
python performance_benchmark.py --experiments_dir ../03_experiments/results/
```

## 📚 深入学习

### 实验设计理论
- **对照实验设计**: 确保单一变量控制
- **多重比较校正**: Bonferroni, FDR校正方法
- **效应量计算**: Cohen's d, eta squared
- **置信区间**: Bootstrap方法

### 相关工具和框架
- **Weights & Biases**: 实验跟踪和可视化
- **MLflow**: 机器学习实验管理
- **Optuna**: 超参数优化框架
- **Ray Tune**: 分布式超参数调优

---

**🎉 恭喜！您已掌握实验执行的核心技能。**

好的实验设计是科学研究的基础。通过系统性的实验，您将获得有说服力的研究结果。

让我们进入[结果分析阶段](../04_analysis/README.md)深入挖掘实验洞察。
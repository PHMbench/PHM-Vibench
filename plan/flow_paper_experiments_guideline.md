# 🚀 Flow预训练模块论文级实验指南

## 概述

本指南提供了使用Flow预训练模块进行**发表级研究实验**的完整流程，涵盖从实验设计到论文写作的全过程。

---

## 1. 实验准备 (Experiment Preparation)

### 1.1 环境验证
```bash
# 验证Flow模块设置
python validate_flow_setup.py

# 检查GPU资源
nvidia-smi

# 确认数据完整性
ls -la data/metadata_6_11.xlsx
```

### 1.2 数据集准备

#### 标准数据集配置
```yaml
# 推荐用于论文的数据集组合
datasets:
  train: [CWRU, XJTU, FEMTO]     # 多样化训练集
  val: [THU, SEU]                # 独立验证集  
  test: [IMS, PU]                # 完全独立测试集
```

#### 数据预处理标准
```yaml
preprocessing:
  window_size: 1024              # 标准窗口大小
  stride: 256                    # 25%重叠
  normalization: 'standardization'
  truncate_length: 2000          # 统一序列长度
  sampling_rate: 12000           # 统一采样率
```

### 1.3 计算资源规划
```bash
# 建议资源配置
GPU: RTX 3090/4090 (24GB) × 1-2
RAM: 64GB+
存储: 500GB+ SSD
预估时间: 24-48小时 (完整实验)
```

---

## 2. 基线实验 (Baseline Experiments)

### 2.1 Flow基线模型
```bash
# 标准Flow预训练基线
./run_flow_experiments.sh research --gpu 0 --wandb --notes "Paper_Baseline_Flow"

# 配置: flow_research_experiment.yaml
# - 200 epochs
# - batch_size: 64
# - lr: 5e-4
# - num_steps: 100
```

### 2.2 传统方法对比基线

#### a) CNN-based预训练
```yaml
model:
  name: "ResNet1D"
  layers: [64, 128, 256, 512]
  
task:
  name: "masked_reconstruction" 
  mask_ratio: 0.15
  loss: "MSE"
```

#### b) Transformer预训练
```yaml
model:
  name: "B_08_PatchTST"
  d_model: 512
  n_heads: 8
  n_layers: 6

task:
  name: "masked_reconstruction"
  patch_len: 16
  stride: 8
```

#### c) VAE基线
```yaml
model:
  name: "VAE_Baseline"
  latent_dim: 256
  encoder_layers: [512, 256, 128]
  decoder_layers: [128, 256, 512]
```

### 2.3 评估指标标准
```python
# 下游任务评估指标
primary_metrics = {
    'classification': ['accuracy', 'f1_macro', 'precision', 'recall'],
    'few_shot': ['5_shot_acc', '10_shot_acc', '20_shot_acc'],
    'domain_transfer': ['target_domain_acc', 'adaptation_speed']
}

# 预训练质量指标
pretraining_metrics = {
    'reconstruction': ['mse', 'mae', 'ssim'],
    'representation': ['feature_diversity', 'linear_separability'],
    'efficiency': ['params_count', 'training_time', 'inference_time']
}
```

---

## 3. 消融研究 (Ablation Studies)

### 3.1 Flow组件消融

#### a) 采样步数消融
```bash
# 不同采样步数对比
for steps in 20 50 100 200 500; do
  python run_flow_experiment_batch.py custom \
    --experiments baseline \
    --config_override "task.num_steps=$steps" \
    --notes "Ablation_Steps_$steps" \
    --wandb
done
```

#### b) 噪声调度消融
```bash
# 不同sigma范围
configs=(
  "sigma_min=0.001,sigma_max=1.0"
  "sigma_min=0.01,sigma_max=2.0" 
  "sigma_min=0.0001,sigma_max=0.5"
)

for config in "${configs[@]}"; do
  python run_flow_experiment_batch.py custom \
    --experiments baseline \
    --config_override "$config" \
    --notes "Ablation_Sigma_$config"
done
```

#### c) 时间编码消融
```yaml
# 移除时间编码
model:
  use_time_embedding: false
  
# 不同时间编码方式
time_encoding_types: ['sinusoidal', 'learned', 'none']
```

### 3.2 对比学习权重消融
```bash
# 对比学习权重扫描
weights=(0.0 0.1 0.3 0.5 0.7 1.0)

for w in "${weights[@]}"; do
  ./run_flow_experiments.sh contrastive \
    --config_override "task.contrastive_weight=$w" \
    --notes "Ablation_Contrastive_$w" \
    --wandb
done
```

### 3.3 架构深度消融
```yaml
# 不同模型深度配置
model_configs:
  small:
    hidden_dim: 128
    n_layers: 4
    
  medium:  
    hidden_dim: 256
    n_layers: 6
    
  large:
    hidden_dim: 512
    n_layers: 8
```

---

## 4. 对比实验 (Comparative Experiments)

### 4.1 生成模型对比

#### Flow vs VAE
```bash
# Flow训练
./run_flow_experiments.sh research --notes "Comparison_Flow"

# VAE训练 
python main.py --config configs/comparison/vae_baseline.yaml --notes "Comparison_VAE"

# 对比评估
python scripts/compare_generative_models.py --models flow,vae
```

#### Flow vs Diffusion
```bash
# Diffusion基线
python main.py --config configs/comparison/ddpm_baseline.yaml --notes "Comparison_DDPM"

# 性能对比
python scripts/benchmark_sampling_speed.py --models flow,ddpm
```

### 4.2 预训练方法对比

#### Flow vs Contrastive Learning
```yaml
# 纯对比学习基线
task:
  name: "contrastive_pretrain"
  temperature: 0.1
  projection_dim: 256
  augmentation: ['noise', 'scaling', 'permutation']
```

#### Flow vs Masked Modeling
```yaml  
# MAE-style预训练
task:
  name: "masked_reconstruction"
  mask_ratio: 0.25
  mask_strategy: 'random'
  reconstruction_target: 'original'
```

---

## 5. 泛化性实验 (Generalization Studies)

### 5.1 跨数据集评估

#### 设置1: 单源域→多目标域
```bash
# 训练配置
source_dataset="CWRU"
target_datasets=("XJTU" "THU" "SEU" "IMS")

# Flow预训练
./run_flow_experiments.sh pipeline02 \
  --config_override "data.train_datasets=[$source_dataset]" \
  --notes "CrossDataset_Flow_${source_dataset}"

# 评估所有目标域
for target in "${target_datasets[@]}"; do
  python evaluate_cross_domain.py \
    --source $source_dataset \
    --target $target \
    --model flow_pretrained
done
```

#### 设置2: 多源域→单目标域
```bash
# 多源预训练
python run_multi_source_training.py \
  --sources "CWRU,XJTU,THU" \
  --target "SEU" \
  --model flow \
  --notes "MultiSource_Flow"
```

### 5.2 Few-Shot学习评估
```python
# Few-shot评估协议
def evaluate_few_shot(model, dataset, shots=[1, 5, 10, 20]):
    results = {}
    for n_shot in shots:
        # 随机采样support set
        support_acc = []
        for trial in range(10):  # 10次重复实验
            acc = run_few_shot_trial(model, dataset, n_shot, seed=trial)
            support_acc.append(acc)
        
        results[f"{n_shot}_shot"] = {
            'mean': np.mean(support_acc),
            'std': np.std(support_acc),
            'ci_95': confidence_interval(support_acc)
        }
    return results
```

### 5.3 噪声鲁棒性测试
```python
# 噪声鲁棒性评估
noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
noise_types = ['gaussian', 'uniform', 'salt_pepper']

for noise_type in noise_types:
    for level in noise_levels:
        results = evaluate_with_noise(
            model='flow_pretrained',
            noise_type=noise_type,
            noise_level=level,
            dataset='test_clean'
        )
```

---

## 6. 规模化实验 (Scaling Experiments)

### 6.1 模型大小Scaling
```yaml
# 不同模型规模
model_scales:
  nano:    {hidden_dim: 64,  n_layers: 2}   # ~10K params
  tiny:    {hidden_dim: 128, n_layers: 4}   # ~50K params  
  small:   {hidden_dim: 256, n_layers: 6}   # ~200K params
  medium:  {hidden_dim: 512, n_layers: 8}   # ~1M params
  large:   {hidden_dim: 1024, n_layers: 10} # ~4M params
```

### 6.2 数据量Scaling
```bash
# 不同数据量训练
data_ratios=(0.1 0.25 0.5 0.75 1.0)

for ratio in "${data_ratios[@]}"; do
  python train_with_data_ratio.py \
    --ratio $ratio \
    --model flow \
    --notes "DataScaling_${ratio}"
done
```

### 6.3 训练时长影响
```bash
# 不同训练epoch数对比
epochs=(10 25 50 100 200 500)

for ep in "${epochs[@]}"; do
  ./run_flow_experiments.sh baseline \
    --config_override "task.epochs=$ep" \
    --notes "EpochScaling_$ep"
done
```

---

## 7. 结果收集与分析 (Result Analysis)

### 7.1 自动结果汇总脚本
```python
#!/usr/bin/env python3
# scripts/collect_results.py

import pandas as pd
import json
from pathlib import Path

def collect_experiment_results(experiment_dir="results/"):
    """汇总所有实验结果"""
    results = []
    
    for exp_path in Path(experiment_dir).glob("*/"):
        if exp_path.is_dir():
            metrics_file = exp_path / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    metrics = json.load(f)
                
                results.append({
                    'experiment': exp_path.name,
                    'accuracy': metrics.get('test_accuracy', 0),
                    'f1_score': metrics.get('test_f1', 0),
                    'training_time': metrics.get('training_time', 0),
                    'params_count': metrics.get('model_params', 0)
                })
    
    df = pd.DataFrame(results)
    df.to_csv('experiment_results_summary.csv', index=False)
    return df

# 使用
results_df = collect_experiment_results()
print(results_df.describe())
```

### 7.2 统计显著性检验
```python
# 统计检验脚本
import scipy.stats as stats

def statistical_comparison(method1_scores, method2_scores):
    """比较两种方法的统计显著性"""
    
    # Shapiro-Wilk正态性检验
    _, p1 = stats.shapiro(method1_scores)
    _, p2 = stats.shapiro(method2_scores)
    
    if p1 > 0.05 and p2 > 0.05:
        # 正态分布，使用t检验
        t_stat, p_value = stats.ttest_ind(method1_scores, method2_scores)
        test_type = "t-test"
    else:
        # 非正态分布，使用Mann-Whitney U检验
        u_stat, p_value = stats.mannwhitneyu(method1_scores, method2_scores)
        test_type = "Mann-Whitney U"
    
    # 效应大小 (Cohen's d)
    pooled_std = np.sqrt(((len(method1_scores)-1)*np.var(method1_scores) + 
                         (len(method2_scores)-1)*np.var(method2_scores)) / 
                        (len(method1_scores)+len(method2_scores)-2))
    cohens_d = (np.mean(method1_scores) - np.mean(method2_scores)) / pooled_std
    
    return {
        'test_type': test_type,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'effect_size': cohens_d,
        'effect_magnitude': interpret_cohens_d(cohens_d)
    }
```

### 7.3 学习曲线分析
```python
# 学习曲线绘制
import matplotlib.pyplot as plt
import seaborn as sns

def plot_learning_curves(experiments):
    """绘制多个实验的学习曲线对比"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 训练损失
    for exp_name, metrics in experiments.items():
        axes[0,0].plot(metrics['train_loss'], label=exp_name)
    axes[0,0].set_title('Training Loss')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    
    # 验证准确率
    for exp_name, metrics in experiments.items():
        axes[0,1].plot(metrics['val_accuracy'], label=exp_name)
    axes[0,1].set_title('Validation Accuracy')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Accuracy')
    axes[0,1].legend()
    
    plt.tight_layout()
    plt.savefig('learning_curves_comparison.pdf', dpi=300, bbox_inches='tight')
```

---

## 8. 论文图表生成 (Paper Figures)

### 8.1 性能对比表格生成
```python
# LaTeX表格生成脚本
def generate_latex_table(results_df, caption="", label=""):
    """生成LaTeX格式的结果表格"""
    
    latex_table = f"""
\\begin{{table}}[h]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{lcccc}}
\\toprule
Method & Accuracy (\\%) & F1-Score & Parameters & Time (min) \\\\
\\midrule
"""
    
    for _, row in results_df.iterrows():
        method = row['method'].replace('_', '\\_')
        acc = f"{row['accuracy']:.2f} $\\pm$ {row['acc_std']:.2f}"
        f1 = f"{row['f1_score']:.3f}"
        params = f"{row['params']/1000:.0f}K" if row['params'] < 1e6 else f"{row['params']/1e6:.1f}M"
        time = f"{row['training_time']:.1f}"
        
        latex_table += f"{method} & {acc} & {f1} & {params} & {time} \\\\\n"
    
    latex_table += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    return latex_table

# 使用
table = generate_latex_table(
    results_df, 
    caption="Performance comparison of different pretraining methods on vibration signal classification",
    label="tab:performance_comparison"
)
print(table)
```

### 8.2 消融研究可视化
```python
# 消融研究热力图
def plot_ablation_heatmap(ablation_results):
    """绘制消融研究热力图"""
    
    # 准备数据
    components = ['Flow', 'Contrastive', 'Time_Embed', 'Multi_Scale']
    metrics = ['Accuracy', 'F1_Score', 'Transfer_Acc']
    
    # 创建结果矩阵
    results_matrix = np.array(ablation_results).reshape(len(components), len(metrics))
    
    # 绘制热力图
    plt.figure(figsize=(8, 6))
    sns.heatmap(results_matrix, 
                annot=True, 
                fmt='.3f',
                xticklabels=metrics,
                yticklabels=components,
                cmap='RdYlBu_r',
                center=0.5)
    
    plt.title('Ablation Study Results')
    plt.xlabel('Evaluation Metrics')
    plt.ylabel('Model Components')
    plt.tight_layout()
    plt.savefig('ablation_heatmap.pdf', dpi=300, bbox_inches='tight')
```

### 8.3 t-SNE特征可视化
```python
# 特征空间可视化
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_learned_features(model, dataloader, save_path):
    """可视化学习到的特征表示"""
    
    features = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            # 提取特征表示
            feat = model.extract_features(x)  # 假设模型有此方法
            features.append(feat.cpu().numpy())
            labels.append(y.cpu().numpy())
    
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    # t-SNE降维
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    # 绘制
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                         c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization of Learned Features')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
```

---

## 9. 实验脚本模板 (Experiment Templates)

### 9.1 完整实验Pipeline
```bash
#!/bin/bash
# full_paper_experiments.sh

set -e  # 遇到错误立即停止

# 1. 环境验证
echo "=== 验证实验环境 ==="
python validate_flow_setup.py || exit 1

# 2. 基线实验
echo "=== 运行基线实验 ==="
experiments=("flow_research" "vae_baseline" "contrastive_baseline")

for exp in "${experiments[@]}"; do
    echo "Running $exp..."
    ./run_flow_experiments.sh $exp --wandb --notes "Paper_Baseline_$exp"
    
    # 检查实验是否成功
    if [ $? -ne 0 ]; then
        echo "实验 $exp 失败！"
        exit 1
    fi
done

# 3. 消融研究
echo "=== 消融研究 ==="
bash scripts/run_ablation_studies.sh

# 4. 对比实验  
echo "=== 对比实验 ==="
bash scripts/run_comparative_studies.sh

# 5. 结果汇总
echo "=== 结果汇总 ==="
python scripts/collect_results.py
python scripts/generate_paper_figures.py

echo "=== 所有实验完成！ ==="
```

### 9.2 超参数扫描脚本
```python
#!/usr/bin/env python3
# hyperparameter_sweep.py

import itertools
import subprocess
import yaml

def hyperparameter_sweep():
    """超参数网格搜索"""
    
    # 定义搜索空间
    param_grid = {
        'task.lr': [1e-4, 5e-4, 1e-3],
        'task.flow_lr': [1e-4, 5e-4, 1e-3],
        'task.contrastive_weight': [0.1, 0.3, 0.5],
        'model.hidden_dim': [256, 512],
        'task.num_steps': [50, 100, 200]
    }
    
    # 生成所有参数组合
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    for combination in itertools.product(*param_values):
        params = dict(zip(param_names, combination))
        
        # 构建命令
        overrides = [f"{k}={v}" for k, v in params.items()]
        override_str = ",".join(overrides)
        
        experiment_name = "_".join([f"{k.split('.')[-1]}{v}" for k, v in params.items()])
        
        cmd = [
            "python", "run_flow_experiment_batch.py", "custom",
            "--experiments", "baseline",
            "--config_override", override_str,
            "--notes", f"HyperSweep_{experiment_name}",
            "--wandb"
        ]
        
        print(f"运行: {' '.join(cmd)}")
        subprocess.run(cmd)

if __name__ == "__main__":
    hyperparameter_sweep()
```

---

## 10. 论文写作模板 (Writing Templates)

### 10.1 实验设置段落模板
```latex
\subsection{Experimental Setup}

We evaluate our Flow-based pretraining approach on X industrial vibration datasets, including CWRU bearing dataset~\cite{cwru}, XJTU-SY bearing dataset~\cite{xjtu}, and THU gearbox dataset~\cite{thu}. 

\textbf{Data Preprocessing:} Following standard practices~\cite{previous_work}, we segment each signal into windows of length 1024 with 75\% overlap, resulting in XXX training samples across Y fault categories. All signals are normalized using standardization.

\textbf{Model Configuration:} Our Flow model consists of a Z-layer transformer encoder with hidden dimension D=512. We use T=100 denoising steps during training and S=20 steps for fast sampling during inference. The contrastive learning component uses temperature τ=0.1 and projection dimension P=256.

\textbf{Training Details:} We train all models for E=200 epochs using Adam optimizer with learning rate lr=5×10⁻⁴. The batch size is set to B=64, and we apply early stopping with patience P=20 based on validation loss. All experiments are conducted using PyTorch on NVIDIA RTX 3090 GPUs.

\textbf{Evaluation Protocol:} We assess model performance using K-fold cross-validation (K=5) and report mean accuracy along with 95\% confidence intervals. For few-shot evaluation, we randomly sample N={1,5,10,20} examples per class and repeat each experiment R=10 times.
```

### 10.2 结果讨论要点
```latex
\subsection{Results and Analysis}

\textbf{Main Results:} Table~\ref{tab:main_results} shows that our Flow-based pretraining achieves state-of-the-art performance across all benchmark datasets. Specifically, our method obtains XX.X\% accuracy on CWRU, outperforming the previous best method by Y.Y\% (p<0.01, Cohen's d=Z.Z).

\textbf{Ablation Study:} The ablation results in Table~\ref{tab:ablation} demonstrate the importance of each component. Removing the Flow mechanism leads to A\% performance drop, while disabling contrastive learning reduces accuracy by B\%. This indicates that both generative modeling and contrastive learning contribute synergistically.

\textbf{Cross-Dataset Generalization:} Figure~\ref{fig:cross_domain} illustrates the superior generalization capability of our approach. When trained on dataset X and tested on dataset Y, our method maintains Z\% of its original performance, significantly outperforming baseline methods.

\textbf{Few-Shot Performance:} Our Flow pretraining enables effective few-shot learning as shown in Figure~\ref{fig:few_shot}. With only N=5 examples per class, our method achieves XX\% accuracy, approaching the performance of fully supervised methods.

\textbf{Computational Efficiency:} Despite the iterative sampling process, our method achieves competitive inference speed (X ms per sample) while maintaining superior accuracy. The pretraining phase requires Y hours on a single GPU, making it practically feasible.
```

### 10.3 局限性分析框架
```latex
\subsection{Limitations and Future Work}

While our Flow-based pretraining shows promising results, several limitations should be acknowledged:

\textbf{Dataset Bias:} Our evaluation focuses primarily on bearing fault diagnosis. The generalizability to other types of mechanical systems (e.g., pumps, motors) requires further investigation.

\textbf{Computational Cost:} The iterative denoising process increases inference time compared to single-forward methods. Future work could explore faster sampling techniques or distillation approaches.

\textbf{Hyperparameter Sensitivity:} The performance depends on careful tuning of Flow-specific hyperparameters (σ_min, σ_max, num_steps). More robust automatic hyperparameter selection methods would be beneficial.

\textbf{Theoretical Analysis:} While empirical results are strong, deeper theoretical understanding of why Flow models work well for vibration signals would strengthen the contribution.

Future research directions include: (1) extending to multimodal signals, (2) incorporating physical constraints into the generative model, and (3) developing specialized Flow architectures for time series data.
```

---

## 11. 质量控制检查清单 (Quality Control)

### 11.1 实验前检查
- [ ] **环境配置**
  - [ ] CUDA版本兼容性确认
  - [ ] 依赖库版本锁定
  - [ ] 随机种子固定 (reproducibility)
  
- [ ] **数据准备**
  - [ ] 训练/验证/测试集划分合理
  - [ ] 数据泄露检查 (no data leakage)
  - [ ] 样本平衡性分析

- [ ] **实验设计**
  - [ ] 对照组设置合理  
  - [ ] 变量控制 (只改变一个因素)
  - [ ] 足够的重复实验次数

### 11.2 实验中监控
- [ ] **训练过程**
  - [ ] Loss收敛性检查
  - [ ] 梯度爆炸/消失监控
  - [ ] 内存使用量跟踪
  
- [ ] **验证结果**
  - [ ] 过拟合检测
  - [ ] 模型收敛确认
  - [ ] 中间结果合理性

### 11.3 结果分析检查
- [ ] **统计有效性**
  - [ ] 显著性检验完成
  - [ ] 效应大小计算
  - [ ] 置信区间报告
  
- [ ] **可重现性**
  - [ ] 代码版本记录
  - [ ] 配置文件保存
  - [ ] 环境信息记录

---

## 12. 常见问题与解决方案

### Q1: 实验结果不稳定怎么办？
**A:** 
1. 固定所有随机种子 (Python, NumPy, PyTorch, CUDA)
2. 增加实验重复次数 (建议≥5次)
3. 检查数据加载顺序是否固定
4. 使用确定性算法 (`torch.use_deterministic_algorithms(True)`)

### Q2: 内存不足如何处理？
**A:**
1. 减少batch_size
2. 使用梯度累积 (`accumulate_grad_batches`)
3. 启用mixed precision训练 (`precision=16`)
4. 使用checkpoint技术

### Q3: 训练时间过长怎么优化？
**A:**
1. 使用多GPU训练
2. 减少Flow采样步数
3. 使用更小的模型作为初步验证
4. 采用learning rate warmup加速收敛

### Q4: 如何确保公平对比？
**A:**
1. 使用相同的数据划分
2. 相同的评估指标和协议
3. 相同的计算资源限制
4. 报告所有尝试的超参数组合

---

## 📊 总结

本指南提供了使用Flow预训练模块进行**论文级研究**的完整方法论，从实验设计到结果分析的全流程覆盖。

### 🎯 关键成功要素
1. **严格的实验设计** - 控制变量，合理对照
2. **充分的统计分析** - 显著性检验，效应大小
3. **全面的消融研究** - 理解每个组件的贡献
4. **robust的评估协议** - 多数据集，多指标验证
5. **可重现的实验流程** - 详细记录，版本控制

### 🚀 开始研究实验
```bash
# 快速开始
git clone <your-repo>
cd PHM-Vibench-flow
python validate_flow_setup.py
bash full_paper_experiments.sh
```

**预期产出**: 高质量的实验结果，可发表的图表，完整的消融研究，以及robust的统计分析。
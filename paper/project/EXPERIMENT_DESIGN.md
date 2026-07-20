# MOE Explainable 完整实验设计

> 制定时间: 2026-03-17
> 优先级: P2 (第二批投稿)

---

## 🎯 实验目标

用**物理同构专家 + 可审计路由**证明MoE的解释性与稳定性

---

## 📋 实验总览

| 实验类型 | 实验名称 | 数据集 | Seed数 | GPU预估 |
|----------|----------|--------|--------|---------|
| **基准** | CWRU/XJTU 5-seed | 2 | 5 | 10h |
| **消融** | 专家数3/5/8 | CWRU | 3 | 4h |
| **稳定性** | 多seed稳定性分析 | CWRU | 5 | 3h |
| **路由** | 路由可解释性分析 | 3数据集 | 3 | 4h |
| **泛化** | 跨域路由迁移 | 3数据集 | 3 | 6h |

**总计GPU**: 27小时

---

## 实验1: 基准实验 (CWRU/XJTU 5-seed)

### 目标
多seed验证稳定性，报告CV和95% CI

### 数据集配置
```yaml
datasets:
  - name: CWRU
    config: configs/unified_baseline/config_MoE.yaml
    classes: 10

  - name: XJTU
    config: configs/unified_baseline/config_MoE.yaml
    classes: 15
```

### MoE配置
```yaml
model:
  name: PhysicsConstrainedMoE
  num_experts: 5
  router:
    type: StatisticalRouter
    features: [mean, std, kurtosis, skewness, rms]
  experts:
    - LowPassExpert
    - HighPassExpert
    - BandPassExpert
    - EnvelopeExpert
    - HarmonicExpert
```

### 评估指标
```yaml
metrics:
  performance:
    - accuracy
    - f1_macro

  stability:
    - mean±std
    - 95% CI
    - CV (变异系数)
    - 若CV>10%需给原因

  routing:
    - routing_entropy: 路由熵
    - path_signature: 路径签名
    - expert_distribution: 专家激活分布
```

### 执行脚本
```bash
#!/bin/bash
# scripts/run_baseline_moe.sh

DATASETS=(CWRU XJTU)
SEEDS=(42 123 456 789 1024)

for dataset in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "Running: $dataset + seed$seed"

    CUDA_VISIBLE_DEVICES=6 python main.py \
      --config_dir configs/unified_baseline/config_MoE.yaml \
      --dataset $dataset \
      --seed $seed \
      --output_dir paper/UXFD_paper/MOE_explainable/results/${dataset}_seed${seed}/
  done
done

# 稳定性分析
python scripts/analyze_stability.py \
  --input results/CWRU_seed* results/XJTU_seed* \
  --output results/stability_analysis/
```

### 预期输出
```
results/
├── CWRU_seed42/
│   ├── run_meta.yaml
│   ├── metrics.json
│   ├── routing_entropy.json
│   ├── path_signature.npy
│   └── expert_activations.npy
├── CWRU_seed{123,456,789,1024}/
├── XJTU_seed{42,123,456,789,1024}/
└── stability_analysis/
    ├── cv_analysis.json
    └── stability_report.md
```

### 表格模板
```latex
% Table: Multi-Seed Stability
\begin{table}[ht]
\centering
\caption{Multi-Seed Stability Analysis (5-seed)}
\label{tab:moe_stability}
\begin{tabular}{lcccccc}
\toprule
Dataset & Mean±Std & 95\% CI & CV & Best & Worst & $\Delta$ \\
\midrule
CWRU & 85.2±3.1 & [82.1, 88.3] & 3.6\% & 89.1 & 81.5 & 7.6 \\
XJTU & 82.7±4.2 & [78.5, 86.9] & 5.1\% & 88.3 & 77.2 & 11.1 \\
\bottomrule
\end{tabular}
\end{table}

% 若CV>10%，需添加原因分析
\textbf{Note:} CV values are within acceptable range (<10\%).
If CV>10\%, we provide improvement strategies in Section X.X.
```

---

## 实验2: 消融实验 (专家数3/5/8)

### 目标
专家数 vs 性能 vs 稳定性

### 消融配置
```yaml
expert_configurations:
  - num_experts: 3
    experts: [LowPass, HighPass, BandPass]
    params: 1.5M
    expected: 中等性能，高稳定性

  - num_experts: 5
    experts: [LowPass, HighPass, BandPass, Envelope, Harmonic]
    params: 2.1M
    expected: 高性能，中等稳定性

  - num_experts: 8
    experts: [LowPass, HighPass, BandPass, Envelope, Harmonic, Wavelet, FFT, I]
    params: 3.2M
    expected: 最高性能，低稳定性
```

### 消融矩阵
| Experts | 参数量 | 预期性能 | 预期CV |
|---------|--------|----------|--------|
| 3 | 1.5M | 83% | <3% |
| 5 | 2.1M | 85% | <4% |
| 8 | 3.2M | 86% | <6% |

### 执行脚本
```bash
#!/bin/bash
# scripts/run_expert_ablation.sh

NUM_EXPERTS=(3 5 8)
SEEDS=(42 123 456)

for num in "${NUM_EXPERTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "Running: ${num}experts + seed$seed"

    CUDA_VISIBLE_DEVICES=6 python main.py \
      --config_dir configs/unified_baseline/config_MoE_${num}experts.yaml \
      --seed $seed \
      --output_dir paper/UXFD_paper/MOE_explainable/results/ablation_${num}experts_seed${seed}/
  done
done

# 生成消融曲线
python scripts/plot_expert_ablation.py \
  --input results/ablation_*experts/ \
  --output manuscript/figures/expert_ablation.pdf
```

### 预期表格
```latex
% Table: Expert Ablation
\begin{table}[ht]
\centering
\caption{Expert Number Ablation Study (mean±std, 3-seed)}
\label{tab:expert_ablation}
\begin{tabular}{ccccc}
\toprule
\# Experts & Accuracy (\%) & Params & CV (\%) & Routing Entropy \\
\midrule
3 & 83.5±2.8 & 1.5M & 2.8 & 0.82 \\
5 & 85.2±3.1 & 2.1M & 3.6 & 1.12 \\
8 & 86.1±4.5 & 3.2M & 5.2 & 1.35 \\
\bottomrule
\end{tabular}
\end{table}
```

### 关键发现
```
1. 3专家: 最稳定但性能最低
2. 5专家: 性能-稳定性最佳平衡
3. 8专家: 性能最高但稳定性下降
4. 路由熵随专家数增加而增加
```

---

## 实验3: 稳定性分析

### 目标
深入分析CV，提出改进策略

### 稳定性分析指标
```yaml
stability_metrics:
  - coefficient_of_variation: CV = σ/μ × 100%
  - confidence_interval: 95% CI = μ ± 1.96×σ/√n
  - seed_consistency: 种子间性能一致性
  - routing_stability: 路由决策稳定性
```

### 改进策略
```yaml
improvement_strategies:
  - name: Routing Regularization
    method: 添加路由熵正则化
    expected: 降低CV 1-2%

  - name: Expert Initialization
    method: 优化专家初始化
    expected: 降低CV 2-3%

  - name: Learning Rate Schedule
    method: 使用余弦退火
    expected: 降低CV 1-2%
```

### 执行脚本
```bash
#!/bin/bash
# scripts/run_stability_improvement.sh

STRATEGIES=(baseline routing_reg expert_init lr_schedule)
SEEDS=(42 123 456 789 1024)

for strategy in "${STRATEGIES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "Running: $strategy + seed$seed"

    CUDA_VISIBLE_DEVICES=6 python main.py \
      --config_dir configs/stability/config_${strategy}.yaml \
      --seed $seed \
      --output_dir results/stability_improvement/${strategy}_seed${seed}/
  done
done

# 对比分析
python scripts/compare_stability_strategies.py \
  --input results/stability_improvement/ \
  --output manuscript/tables/stability_improvement.tex
```

### 预期表格
```latex
% Table: Stability Improvement Strategies
\begin{table}[ht]
\centering
\caption{Stability Improvement Strategies (5-seed)}
\label{tab:stability_improvement}
\begin{tabular}{lccc}
\toprule
Strategy & Mean±Std & CV (\%) & Improvement \\
\midrule
Baseline & 85.2±3.1 & 3.6 & - \\
+ Routing Reg & 86.1±2.5 & 2.9 & -0.7\% \\
+ Expert Init & 87.3±2.1 & 2.4 & -1.2\% \\
+ LR Schedule & 86.8±2.3 & 2.7 & -0.9\% \\
\midrule
\textbf{Combined} & \textbf{88.5±1.8} & \textbf{2.0} & \textbf{-1.6\%} \\
\bottomrule
\end{tabular}
\end{table}
```

---

## 实验4: 路由可解释性分析

### 目标
量化路由的可解释性

### 路由指标
```yaml
routing_metrics:
  - name: Routing Entropy
    formula: H(g) = -Σg_i log(g_i)
    meaning: 路由不确定性
    range: [0, log(K)]

  - name: Path Signature
    formula: σ = sign(g(x))
    meaning: 激活模式
    values: binary vector

  - name: Expert IoU
    formula: IoU = |E_i ∩ E_j| / |E_i ∪ E_j|
    meaning: 专家协同性
    range: [0, 1]

  - name: Routing Stability
    formula: ρ(g(x), g(x+ε))
    meaning: 扰动后路由一致性
    range: [-1, 1]
```

### 执行脚本
```bash
#!/bin/bash
# scripts/analyze_routing.sh

DATASETS=(CWRU XJTU FEMTO)

for dataset in "${DATASETS[@]}"; do
  echo "Analyzing routing on $dataset"

  python scripts/analyze_routing.py \
    --model_path results/${dataset}_seed42/best_model.pth \
    --dataset $dataset \
    --output_dir results/routing_analysis/${dataset}/
done

# 生成路由可视化
python scripts/visualize_routing.py \
  --input results/routing_analysis/ \
  --output manuscript/figures/routing_visualization.pdf
```

### 预期输出
```
results/routing_analysis/
├── CWRU/
│   ├── routing_entropy_distribution.png
│   ├── expert_activation_heatmap.png
│   ├── path_signature_analysis.json
│   └── routing_stability_scores.json
├── XJTU/
└── FEMTO/

manuscript/figures/
├── routing_entropy_distribution.pdf
├── expert_activation_heatmap.pdf
└── path_signature_visualization.pdf
```

### 专家激活模式
```yaml
expected_patterns:
  - fault_type: 内圈故障
    dominant_expert: HighPassExpert
    activation_weight: >0.6
    reason: 高频特征主导

  - fault_type: 外圈故障
    dominant_expert: LowPassExpert
    activation_weight: >0.6
    reason: 低频特征主导

  - fault_type: 滚动体故障
    dominant_expert: EnvelopeExpert
    activation_weight: >0.6
    reason: 包络特征主导

  - fault_type: 复合故障
    dominant_expert: 多专家协同
    activation_weight: 分布式
    reason: 多特征融合
```

---

## 实验5: 跨域路由迁移

### 目标
验证路由模式在跨域场景的泛化

### 跨域设置
```yaml
transfer_scenarios:
  - name: In-domain
    train: CWRU
    test: CWRU
    focus: 基线路由模式

  - name: Cross-domain C→X
    train: CWRU
    test: XJTU
    focus: 路由泛化性

  - name: Cross-domain C→F
    train: CWRU
    test: FEMTO
    focus: 路由迁移性

  - name: Cross-domain C→I
    train: CWRU
    test: IMS
    focus: 专家激活模式保持
```

### 执行脚本
```bash
#!/bin/bash
# scripts/run_routing_transfer.sh

CUDA_VISIBLE_DEVICES=6 python scripts/run_routing_transfer.py \
  --train CWRU \
  --test XJTU,FEMTO,IMS \
  --output_dir results/routing_transfer/

# 分析路由迁移
python scripts/analyze_routing_transfer.py \
  --input results/routing_transfer/ \
  --output manuscript/tables/routing_transfer.tex
```

### 预期表格
```latex
% Table: Routing Transfer
\begin{table}[ht]
\centering
\caption{Cross-domain Routing Transfer}
\label{tab:routing_transfer}
\begin{tabular}{lcccc}
\toprule
Scenario & Acc (\%) & Routing Entropy & Path IoU & Expert Pattern \\
\midrule
In-domain (CWRU) & 85.2 & 1.12 & 1.00 & Preserved \\
Cross (C→X) & 72.3 & 1.35 & 0.78 & Partially \\
Cross (C→F) & 68.5 & 1.42 & 0.71 & Changed \\
Cross (C→I) & 71.8 & 1.38 & 0.75 & Partially \\
\bottomrule
\end{tabular}
\end{table}
```

### 关键发现
```
1. 路由熵在跨域场景增加 (1.12 → 1.35+)
2. 路径IoU下降表明路由模式变化
3. 专家激活模式部分保持但有所改变
4. 物理同构设计有助于跨域泛化
```

---

## 📁 结果归档结构

```
MOE_explainable/
├── results/
│   ├── CWRU_seed{42,123,456,789,1024}/
│   ├── XJTU_seed{42,123,456,789,1024}/
│   ├── ablation_3experts_seed*/
│   ├── ablation_8experts_seed*/
│   ├── stability_improvement/
│   ├── routing_analysis/
│   └── routing_transfer/
├── manuscript/
│   ├── tables/
│   │   ├── moe_stability.tex
│   │   ├── expert_ablation.tex
│   │   ├── stability_improvement.tex
│   │   ├── routing_interpretability.tex
│   │   └── routing_transfer.tex
│   └── figures/
│       ├── expert_ablation.pdf
│       ├── routing_entropy_distribution.pdf
│       ├── expert_activation_heatmap.pdf
│       └── routing_transfer_visualization.pdf
└── EXPERIMENT_DESIGN.md (本文件)
```

---

## ✅ 完成标准

- [ ] 10个基准实验 (2数据集 × 5seed)
- [ ] 9个消融实验 (3配置 × 3seed)
- [ ] 20个稳定性改进实验 (4策略 × 5seed)
- [ ] 9个路由分析实验 (3数据集 × 3seed)
- [ ] 12个跨域迁移实验 (4场景 × 3seed)
- [ ] CV分析完成 (若>10%有改进方案)
- [ ] 所有表格/图表生成

---

## 📊 GPU时间表

| 周次 | 实验 | GPU | 预计完成 |
|------|------|-----|----------|
| Week1 | 基准5-seed | 10h | Day 1-2 |
| Week1 | 专家消融 | 4h | Day 3 |
| Week2 | 稳定性改进 | 3h | Day 4 |
| Week2 | 路由分析 | 4h | Day 5 |
| Week2 | 跨域迁移 | 6h | Day 6-7 |

---

## ⚠️ 稳定性要求

**若CV>10%，必须提供**:
1. 原因分析 (初始化/路由/数据)
2. 改进策略 (至少2种)
3. 改进后结果对比

**当前目标**: CV < 5%

---

_制定: PHM研究总控智能体 | 2026-03-17_

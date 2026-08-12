# Explainable_FD_Toolkit 完整实验设计

> 制定时间: 2026-03-17
> 优先级: P1 (第一批投稿)

---

## 🎯 实验目标

建立故障诊断可解释性的**统一评估基准**，为领域提供标准化的benchmark

---

## 📋 实验总览

| 实验类型 | 实验名称 | 数据集 | 模型数 | Seed数 | GPU预估 |
|----------|----------|--------|--------|--------|---------|
| **基准** | 多模型Benchmark | CWRU+XJTU | 5 | 3 | 10h |
| **小样本** | K-shot诊断 | CWRU | 3 | 5 | 5h |
| **泛化** | 跨数据集泛化 | 5数据集 | 3 | 3 | 8h |
| **竞争** | Captum/SHAP/LIME对比 | CWRU | 3 | 1 | 2h |

**总计GPU**: 25小时

---

## 实验1: 多模型Benchmark (基准实验)

### 目标
≥5模型 × ≥2解释方法 × ≥2数据集 × ≥3seed

### 模型矩阵
```yaml
models:
  - name: ResNet18
    type: CNN
    params: 11.2M
    reason: 通用CNN基线

  - name: TSPN
    type: Transparent
    params: 2.1M
    reason: 透明信号处理基线

  - name: Transformer
    type: Attention
    params: 3.8M
    reason: 注意力基线

  - name: SincNet
    type: FrequencyCNN
    params: 0.5M
    reason: 频域专用基线

  - name: 1D-CNN
    type: SimpleCNN
    params: 1.2M
    reason: 简单1D基线
```

### 解释方法矩阵
```yaml
methods:
  - name: Intrinsic
    type: intrinsic
    description: 注意力权重/门控值

  - name: GradCAM
    type: posthoc
    description: 梯度加权类激活映射

  - name: IntegratedGradients
    type: posthoc
    description: 积分梯度归因
```

### 数据集矩阵
```yaml
datasets:
  - name: CWRU
    samples: 10000
    classes: 10
    sampling_rate: 12kHz
    signal_length: 4096

  - name: XJTU
    samples: 8000
    classes: 15
    sampling_rate: 25.6kHz
    signal_length: 4096
```

### 评估指标
```yaml
metrics:
  performance:
    - accuracy
    - f1_macro
    - precision_macro
    - recall_macro

  explainability:
    - faithfulness (Del@10, AOPC)
    - stability (Spearman, IoU)
    - efficiency (comp_time_ms)
    - coverage (activation_ratio)
```

### 执行脚本
```bash
#!/bin/bash
# scripts/run_full_benchmark.sh

MODELS=(resnet18 tspn transformer sincnet 1dcnn)
METHODS=(intrinsic gradcam)
DATASETS=(CWRU XJTU)
SEEDS=(42 123 456)

for model in "${MODELS[@]}"; do
  for method in "${METHODS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        echo "Running: $model + $method + $dataset + seed$seed"

        CUDA_VISIBLE_DEVICES=0 python scripts/run_benchmark.py \
          --model $model \
          --method $method \
          --dataset $dataset \
          --seed $seed \
          --output_dir results/benchmark/${model}_${method}_${dataset}_seed${seed}/

        # 验证输出schema
        python scripts/validate_schema.py \
          --run_dir results/benchmark/${model}_${method}_${dataset}_seed${seed}/
      done
    done
  done
done

# 汇总结果
python scripts/aggregate_benchmark.py \
  --input results/benchmark/ \
  --output manuscript/tables/
```

### 预期输出
```
results/benchmark/
├── resnet18_intrinsic_CWRU_seed42/
│   ├── run_meta.yaml
│   ├── metrics.json
│   ├── results.csv
│   └── figures/
├── ... (60个实验目录)

manuscript/tables/
├── benchmark_main.tex          # 主性能表
├── explainability_eval.tex     # 可解释性评估表
└── statistical_tests.tex       # 统计检验结果

manuscript/figures/
├── benchmark_radar.pdf         # 雷达图
├── benchmark_heatmap.pdf       # 热力图
└── benchmark_bar_comparison.pdf # 柱状对比图
```

### 表格模板
```latex
% Table: Main Performance Benchmark
\begin{table*}[ht]
\centering
\caption{Multi-model Benchmark on PHM-Vibench (mean±std, 3-seed)}
\label{tab:benchmark_main}
\small
\begin{tabular}{l|cc|cc|c}
\toprule
\multirow{2}{*}{Model} & \multicolumn{2}{c|}{CWRU} & \multicolumn{2}{c|}{XJTU} & \multirow{2}{*}{Params} \\
& Acc (\%) & F1 & Acc (\%) & F1 & \\
\midrule
ResNet18 & 92.1±1.2 & 0.915 & 88.3±2.1 & 0.875 & 11.2M \\
TSPN & \textbf{94.2±0.8} & \textbf{0.935} & \textbf{91.5±1.5} & \textbf{0.908} & 2.1M \\
Transformer & 93.5±1.0 & 0.928 & 89.7±1.8 & 0.889 & 3.8M \\
SincNet & 91.8±1.5 & 0.912 & 87.2±2.3 & 0.865 & 0.5M \\
1D-CNN & 90.5±1.8 & 0.898 & 85.9±2.5 & 0.852 & 1.2M \\
\bottomrule
\end{tabular}
\end{table*}
```

---

## 实验2: K-shot小样本诊断

### 目标
验证解释方法在小样本场景的稳定性和有效性

### K-shot设置
```yaml
k_shots:
  - k: 1
    samples_per_class: 1
    total_train: 10
    description: 极端小样本

  - k: 5
    samples_per_class: 5
    total_train: 50
    description: 少样本

  - k: 10
    samples_per_class: 10
    total_train: 100
    description: 中等样本

  - k: full
    samples_per_class: all
    total_train: ~8000
    description: 全量样本
```

### 模型选择
```yaml
models:
  - ResNet18  # 代表CNN
  - TSPN      # 代表透明网络
  - 1D-CNN    # 代表简单模型
```

### 实验矩阵
- 模型: 3
- K-shot: 4 (1, 5, 10, full)
- Seed: 5 (稳定性分析)
- **总计**: 3 × 4 × 5 = 60次实验

### 执行脚本
```bash
#!/bin/bash
# scripts/run_few_shot.sh

MODELS=(resnet18 tspn 1dcnn)
K_SHOTS=(1 5 10 full)
SEEDS=(42 123 456 789 1024)

for model in "${MODELS[@]}"; do
  for k in "${K_SHOTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      echo "Running: $model + k=$k + seed$seed"

      CUDA_VISIBLE_DEVICES=0 python scripts/run_few_shot.py \
        --model $model \
        --dataset CWRU \
        --k_shot $k \
        --seed $seed \
        --output_dir results/few_shot/${model}_k${k}_seed${seed}/
    done
  done
done

# 生成小样本曲线
python scripts/plot_few_shot_curve.py \
  --input results/few_shot/ \
  --output manuscript/figures/few_shot_curve.pdf
```

### 预期结果
```
manuscript/figures/few_shot_curve.pdf:
  - X轴: K (1, 5, 10, full)
  - Y轴: Accuracy
  - 曲线: 3条 (ResNet18, TSPN, 1D-CNN)
  - 误差条: mean±std (5-seed)

manuscript/tables/few_shot_results.tex:
  - 性能表: 各K值的mean±std
  - 稳定性表: CV vs K
```

### 关键发现预期
1. TSPN在小样本场景更稳定 (低CV)
2. ResNet18需要更多样本才能收敛
3. 解释方法在K≥5时趋于稳定

---

## 实验3: 跨数据集泛化

### 目标
验证解释方法在跨域场景的泛化能力

### 泛化场景
```yaml
scenarios:
  - name: in_domain
    train: CWRU
    test: CWRU
    description: 域内基线

  - name: cross_domain_C2X
    train: CWRU
    test: XJTU
    description: 跨域CWRU→XJTU

  - name: cross_domain_X2C
    train: XJTU
    test: CWRU
    description: 跨域XJTU→CWRU

  - name: multi_source_2toF
    train: [CWRU, XJTU]
    test: FEMTO
    description: 多源→FEMTO

  - name: multi_source_2toI
    train: [CWRU, XJTU]
    test: IMS
    description: 多源→IMS
```

### 模型选择
```yaml
models:
  - ResNet18      # CNN代表
  - TSPN          # 透明网络代表
  - Transformer   # Attention代表
```

### 泛化指标
```yaml
metrics:
  - accuracy
  - domain_gap: acc_train - acc_test
  - faithfulness_transfer: faithfulness on target domain
  - stability_transfer: stability on target domain
```

### 执行脚本
```bash
#!/bin/bash
# scripts/run_generalization.sh

MODELS=(resnet18 tspn transformer)
SCENARIOS=(in_domain cross_domain_C2X cross_domain_X2C multi_source_2toF multi_source_2toI)
SEEDS=(42 123 456)

for model in "${MODELS[@]}"; do
  for scenario in "${SCENARIOS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      echo "Running: $model + $scenario + seed$seed"

      CUDA_VISIBLE_DEVICES=0 python scripts/run_generalization.py \
        --model $model \
        --scenario $scenario \
        --seed $seed \
        --output_dir results/generalization/${model}_${scenario}_seed${seed}/
    done
  done
done

# 分析域间差距
python scripts/analyze_domain_gap.py \
  --input results/generalization/ \
  --output manuscript/tables/domain_gap.tex
```

### 预期表格
```latex
% Table: Cross-domain Generalization
\begin{table}[ht]
\centering
\caption{Cross-domain Generalization Performance}
\label{tab:domain_gap}
\begin{tabular}{lccc}
\toprule
Scenario & ResNet18 & TSPN & Transformer \\
\midrule
In-domain (CWRU) & 92.1 & 94.2 & 93.5 \\
Cross (C→X) & 75.3 & 82.1 & 78.5 \\
Domain Gap & 16.8 & 12.1 & 15.0 \\
\midrule
In-domain (XJTU) & 88.3 & 91.5 & 89.7 \\
Cross (X→C) & 72.8 & 79.6 & 76.2 \\
Domain Gap & 15.5 & 11.9 & 13.5 \\
\bottomrule
\end{tabular}
\end{table}
```

---

## 实验4: 竞争对比

### 目标
与Captum/SHAP/LIME的系统性对比

### 对比维度
```yaml
dimensions:
  - name: Speed
    metric: comp_time_ms
    unit: ms/sample

  - name: Stability
    metric: stability_score
    unit: [0, 1]

  - name: Faithfulness
    metric: del_10
    unit: [0, 1]

  - name: Engineering Friendliness
    metric: api_complexity
    unit: lines of code
```

### 对比方法
```yaml
methods:
  - name: Captum
    library: captum
    methods: [IntegratedGradients, Saliency, DeepLift]

  - name: SHAP
    library: shap
    methods: [KernelSHAP, DeepSHAP]

  - name: LIME
    library: lime
    methods: [LimeTabular]

  - name: Ours
    library: explainable_fd_toolkit
    methods: [UnifiedExplainer]
```

### 执行脚本
```bash
#!/bin/bash
# scripts/run_competitive_comparison.sh

METHODS=(captum shap lime ours)
DATASET=CWRU
SEED=42

for method in "${METHODS[@]}"; do
  echo "Running: $method comparison"

  CUDA_VISIBLE_DEVICES=0 python scripts/run_competitive_comparison.py \
    --method $method \
    --dataset $DATASET \
    --seed $SEED \
    --output_dir results/competitive_comparison/${method}/
done

# 生成对比报告
python scripts/generate_comparison_report.py \
  --input results/competitive_comparison/ \
  --output manuscript/tables/competitive_comparison.tex
```

### 预期表格
```latex
% Table: Competitive Comparison
\begin{table}[ht]
\centering
\caption{Comparison with Existing XAI Libraries}
\label{tab:competitive}
\begin{tabular}{lcccc}
\toprule
Method & Speed (ms) & Stability & Faithfulness & API LOC \\
\midrule
Captum & 125.3 & 0.72 & 0.81 & 45 \\
SHAP & 892.5 & 0.68 & 0.79 & 62 \\
LIME & 456.8 & 0.65 & 0.75 & 58 \\
\textbf{Ours} & \textbf{15.2} & \textbf{0.88} & \textbf{0.92} & \textbf{23} \\
\bottomrule
\end{tabular}
\end{table}
```

---

## 📁 结果归档结构

```
Explainable_FD_Toolkit/
├── results/
│   ├── benchmark/
│   │   ├── resnet18_intrinsic_CWRU_seed42/
│   │   │   ├── run_meta.yaml
│   │   │   ├── metrics.json
│   │   │   ├── results.csv
│   │   │   └── figures/
│   │   └── ... (60个实验)
│   ├── few_shot/
│   │   └── ... (60个实验)
│   ├── generalization/
│   │   └── ... (45个实验)
│   └── competitive_comparison/
│       └── ... (4个方法)
├── manuscript/
│   ├── tables/
│   │   ├── benchmark_main.tex
│   │   ├── explainability_eval.tex
│   │   ├── few_shot_results.tex
│   │   ├── domain_gap.tex
│   │   └── competitive_comparison.tex
│   └── figures/
│       ├── benchmark_radar.pdf
│       ├── few_shot_curve.pdf
│       └── generalization_heatmap.pdf
└── EXPERIMENT_DESIGN.md (本文件)
```

---

## ✅ 完成标准

- [ ] 60个基准实验完成
- [ ] 60个小样本实验完成
- [ ] 45个泛化实验完成
- [ ] 4个竞争对比完成
- [ ] 所有表格/图表生成
- [ ] 结果通过schema验证

---

## 📊 GPU时间表

| 周次 | 实验 | GPU | 预计完成 |
|------|------|-----|----------|
| Week1 | 基准Benchmark | 10h | Day 1-2 |
| Week1 | 小样本 | 5h | Day 3 |
| Week2 | 泛化 | 8h | Day 4-5 |
| Week2 | 竞争对比 | 2h | Day 6 |

---

_制定: PHM研究总控智能体 | 2026-03-17_

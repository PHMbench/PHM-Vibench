# 1D-2D Fusion Explainable 完整实验设计

> 制定时间: 2026-03-17
> 优先级: P1 (第一批投稿)

---

## 🎯 实验目标

验证**三层对齐融合**方法的性能提升来自可检验的跨模态一致性

---

## 📋 实验总览

| 实验类型 | 实验名称 | 数据集 | Seed数 | GPU预估 |
|----------|----------|--------|--------|---------|
| **基准** | CWRU/XJTU 3-seed | 2 | 3 | 6h |
| **消融** | 融合机制消融 | CWRU | 3 | 4h |
| **小样本** | K-shot融合对比 | CWRU | 5 | 4h |
| **泛化** | 跨数据集+噪声 | 3+噪声 | 3 | 6h |
| **解释** | 三层对齐评估 | CWRU | 1 | 2h |

**总计GPU**: 22小时

---

## 实验1: 基准实验 (CWRU/XJTU 3-seed)

### 目标
统一协议下多数据集多seed验证

### 数据集配置
```yaml
datasets:
  - name: CWRU
    config: paper/UXFD_paper/1D-2D_fusion_explainable/configs/config_CWRU.yaml
    classes: 10
    samples: 10000

  - name: XJTU
    config: paper/UXFD_paper/1D-2D_fusion_explainable/configs/config_XJTU.yaml
    classes: 15
    samples: 8000
```

### 评估指标
```yaml
metrics:
  performance:
    - accuracy
    - f1_macro
    - precision
    - recall

  fusion:
    - 1d_contribution: 1D分支贡献权重
    - 2d_contribution: 2D分支贡献权重
    - alignment_score: 三层对齐分数

  explainability:
    - faithfulness (Del@10)
    - stability (Spearman)
    - cross_modal_consistency
```

### 执行脚本
```bash
#!/bin/bash
# scripts/run_baseline.sh

DATASETS=(CWRU XJTU)
SEEDS=(42 123 456)

for dataset in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "Running: $dataset + seed$seed"

    CUDA_VISIBLE_DEVICES=0 python main.py \
      --config_dir paper/UXFD_paper/1D-2D_fusion_explainable/configs/config_${dataset}.yaml \
      --seed $seed \
      --output_dir paper/UXFD_paper/1D-2D_fusion_explainable/results/${dataset}_seed${seed}/
  done
done

# 汇总结果
python scripts/aggregate_baseline.py \
  --input results/CWRU_seed* results/XJTU_seed* \
  --output manuscript/tables/baseline_results.tex
```

### 预期输出
```
results/
├── CWRU_seed42/
│   ├── run_meta.yaml
│   ├── metrics.json
│   ├── best_model.pth
│   └── fusion_analysis/
│       ├── 1d_contribution.json
│       ├── 2d_contribution.json
│       └── alignment_score.json
├── CWRU_seed123/
├── CWRU_seed456/
├── XJTU_seed42/
├── XJTU_seed123/
└── XJTU_seed456/
```

### 表格模板
```latex
% Table: Baseline Results
\begin{table}[ht]
\centering
\caption{1D-2D Fusion Baseline Performance (mean±std, 3-seed)}
\label{tab:fusion_baseline}
\begin{tabular}{lcccc}
\toprule
Dataset & Accuracy (\%) & F1 & 1D Contrib & 2D Contrib \\
\midrule
CWRU & 95.7±0.8 & 0.952 & 0.48±0.05 & 0.52±0.05 \\
XJTU & 93.2±1.2 & 0.925 & 0.45±0.06 & 0.55±0.06 \\
\bottomrule
\end{tabular}
\end{table}
```

---

## 实验2: 消融实验 (融合机制消融)

### 目标
验证各组件的贡献

### 消融配置
```yaml
configurations:
  - name: Full
    components: [1D, 2D, Statistical, Physical-Align, Semantic-Align, Geometric-Align]
    description: 完整模型

  - name: 1D-only
    components: [1D, Statistical]
    description: 仅1D分支

  - name: 2D-only
    components: [2D, Statistical]
    description: 仅2D分支

  - name: No-statistical
    components: [1D, 2D, Physical-Align, Semantic-Align, Geometric-Align]
    description: 无统计特征

  - name: No-alignment
    components: [1D, 2D, Statistical]
    description: 无三层对齐

  - name: Simple-fusion
    components: [1D, 2D, Statistical, Concat]
    description: 简单拼接融合
```

### 消融矩阵
| 配置 | 1D | 2D | Stat | Physical | Semantic | Geometric | 预期Δ |
|------|----|----|------|----------|----------|-----------|-------|
| Full | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 基线 |
| 1D-only | ✓ | ✗ | ✓ | - | - | - | -7% |
| 2D-only | ✗ | ✓ | ✓ | - | - | - | -8% |
| No-stat | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | -2% |
| No-align | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | -5% |
| Simple-fusion | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | -8% |

### 执行脚本
```bash
#!/bin/bash
# scripts/run_ablation.sh

CONFIGS=(1D_only 2D_only no_statistical no_alignment simple_fusion)
SEEDS=(42 123 456)

for config in "${CONFIGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "Running ablation: $config + seed$seed"

    CUDA_VISIBLE_DEVICES=0 python main.py \
      --config_dir paper/UXFD_paper/1D-2D_fusion_explainable/configs/ablation/config_${config}.yaml \
      --seed $seed \
      --output_dir paper/UXFD_paper/1D-2D_fusion_explainable/results/ablation/${config}_seed${seed}/
  done
done

# 生成消融曲线
python scripts/plot_ablation_results.py \
  --input results/ablation/ \
  --output manuscript/figures/ablation_curve.pdf
```

### 预期表格
```latex
% Table: Ablation Study
\begin{table}[ht]
\centering
\caption{Ablation Study on CWRU (mean±std, 3-seed)}
\label{tab:ablation}
\begin{tabular}{lccc}
\toprule
Configuration & Accuracy (\%) & $\Delta$ Acc & Alignment Score \\
\midrule
Full (Ours) & 95.7±0.8 & - & 0.92±0.03 \\
1D-only & 88.5±2.1 & -7.2 & - \\
2D-only & 87.9±2.3 & -7.8 & - \\
No-statistical & 93.5±1.5 & -2.2 & 0.88±0.04 \\
No-alignment & 90.2±1.8 & -5.5 & 0.75±0.05 \\
Simple-fusion & 88.1±2.5 & -7.6 & 0.68±0.06 \\
\bottomrule
\end{tabular}
\end{table}
```

---

## 实验3: 小样本实验 (K-shot融合对比)

### 目标
验证融合方法在小样本场景的优势

### K-shot设置
```yaml
k_shots: [1, 5, 10, full]
models:
  - Fusion1D2D (ours)
  - 1D-only
  - 2D-only
  - TSPN (baseline)
```

### 实验矩阵
- 模型: 4
- K-shot: 4
- Seed: 5
- **总计**: 4 × 4 × 5 = 80次实验

### 执行脚本
```bash
#!/bin/bash
# scripts/run_few_shot_fusion.sh

MODELS=(Fusion1D2D 1D_only 2D_only TSPN)
K_SHOTS=(1 5 10 full)
SEEDS=(42 123 456 789 1024)

for model in "${MODELS[@]}"; do
  for k in "${K_SHOTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      echo "Running: $model + k=$k + seed$seed"

      CUDA_VISIBLE_DEVICES=0 python scripts/run_few_shot_fusion.py \
        --model $model \
        --k_shot $k \
        --seed $seed \
        --output_dir results/few_shot/${model}_k${k}_seed${seed}/
    done
  done
done
```

### 预期结果
```
manuscript/figures/few_shot_fusion_curve.pdf:
  - 展示融合 vs 单模态的小样本性能
  - 融合优势随K增加的变化趋势

关键发现:
  - K=1: 融合优势最大 (+15%)
  - K=5: 融合优势中等 (+8%)
  - K=10: 融合优势稳定 (+5%)
  - K=full: 融合优势保持 (+3%)
```

---

## 实验4: 泛化性实验 (跨数据集+噪声鲁棒性)

### 跨数据集泛化
```yaml
scenarios:
  - {train: CWRU, test: CWRU, type: in_domain}
  - {train: CWRU, test: XJTU, type: cross_domain}
  - {train: CWRU, test: THU_006, type: cross_domain}
```

### 噪声鲁棒性
```yaml
noise_levels:
  - {snr: clean, description: 无噪声}
  - {snr: 20, description: 轻度噪声}
  - {snr: 10, description: 中度噪声}
  - {snr: 5, description: 重度噪声}
  - {snr: 0, description: 极重度噪声}
```

### 执行脚本
```bash
#!/bin/bash
# scripts/run_generalization_fusion.sh

# 跨数据集泛化
CUDA_VISIBLE_DEVICES=0 python scripts/run_cross_domain.py \
  --train CWRU \
  --test XJTU,THU_006 \
  --output results/generalization/

# 噪声鲁棒性
for snr in clean 20 10 5 0; do
  echo "Testing SNR=$snr"

  CUDA_VISIBLE_DEVICES=0 python main.py \
    --config_dir configs/noise/config_snr_${snr}.yaml \
    --output_dir results/noise_robustness/snr_${snr}/
done

# 生成噪声鲁棒性曲线
python scripts/plot_noise_robustness.py \
  --input results/noise_robustness/ \
  --output manuscript/figures/noise_robustness.pdf
```

### 预期表格
```latex
% Table: Cross-domain Generalization
\begin{table}[ht]
\centering
\caption{Cross-domain and Noise Robustness}
\label{tab:generalization}
\begin{tabular}{lcccc}
\toprule
\multirow{2}{*}{Scenario} & \multicolumn{2}{c}{Accuracy (\%)} & \multirow{2}{*}{Domain Gap} \\
\cmidrule{2-3}
& Ours & Baseline & \\
\midrule
In-domain (CWRU) & 95.7 & 92.1 & - \\
Cross (C→X) & 82.3 & 75.8 & 13.4 \\
Cross (C→THU) & 78.5 & 71.2 & 17.2 \\
\midrule
Clean & 95.7 & 92.1 & - \\
SNR=20dB & 93.2 & 88.5 & 2.5 \\
SNR=10dB & 88.6 & 81.3 & 7.1 \\
SNR=5dB & 79.4 & 68.7 & 10.7 \\
\bottomrule
\end{tabular}
\end{table}
```

---

## 实验5: 解释评估 (三层对齐评估)

### 目标
量化评估三层对齐的可解释性

### 评估指标
```yaml
tri_level_metrics:
  physical_layer:
    - energy_conservation: 能量守恒度
    - frequency_alignment: 频谱对齐度

  semantic_layer:
    - cross_modal_consistency: 跨模态一致性
    - feature_alignment: 特征对齐度

  geometric_layer:
    - topology_preservation: 拓扑保持度
    - distance_preservation: 距离保持度

  overall:
    - alignment_score: 综合对齐分数 [0,1]
```

### 执行脚本
```bash
#!/bin/bash
# scripts/evaluate_tri_level_alignment.sh

CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_tri_level_alignment.py \
  --model_path results/CWRU_seed42/best_model.pth \
  --dataset CWRU \
  --output_dir results/tri_level_eval/

# 生成三层对齐可视化
python scripts/visualize_tri_level_alignment.py \
  --input results/tri_level_eval/ \
  --output manuscript/figures/tri_level_alignment.pdf
```

### 预期输出
```
results/tri_level_eval/
├── physical_alignment.json
│   - energy_conservation: 0.95
│   - frequency_alignment: 0.92
├── semantic_alignment.json
│   - cross_modal_consistency: 0.88
│   - feature_alignment: 0.90
├── geometric_alignment.json
│   - topology_preservation: 0.91
│   - distance_preservation: 0.89
└── overall_alignment.json
    - alignment_score: 0.92
```

---

## 📁 结果归档结构

```
1D-2D_fusion_explainable/
├── results/
│   ├── CWRU_seed{42,123,456}/
│   ├── XJTU_seed{42,123,456}/
│   ├── ablation/
│   │   ├── 1D_only_seed*/
│   │   ├── 2D_only_seed*/
│   │   ├── no_statistical_seed*/
│   │   ├── no_alignment_seed*/
│   │   └── simple_fusion_seed*/
│   ├── few_shot/
│   ├── generalization/
│   ├── noise_robustness/
│   └── tri_level_eval/
├── manuscript/
│   ├── tables/
│   │   ├── baseline_results.tex
│   │   ├── ablation.tex
│   │   ├── few_shot.tex
│   │   └── generalization.tex
│   └── figures/
│       ├── ablation_curve.pdf
│       ├── few_shot_curve.pdf
│       ├── noise_robustness.pdf
│       └── tri_level_alignment.pdf
└── EXPERIMENT_DESIGN.md (本文件)
```

---

## ✅ 完成标准

- [ ] 6个基准实验 (2数据集 × 3seed)
- [ ] 18个消融实验 (6配置 × 3seed)
- [ ] 80个小样本实验 (4模型 × 4K × 5seed)
- [ ] 15个泛化实验 (5场景 × 3seed)
- [ ] 5个噪声鲁棒性实验
- [ ] 三层对齐评估完成
- [ ] 所有表格/图表生成

---

## 📊 GPU时间表

| 周次 | 实验 | GPU | 预计完成 |
|------|------|-----|----------|
| Week1 | 基准 | 6h | Day 1 |
| Week1 | 消融 | 4h | Day 2 |
| Week2 | 小样本 | 4h | Day 3-4 |
| Week2 | 泛化+噪声 | 6h | Day 5-6 |
| Week2 | 解释评估 | 2h | Day 7 |

---

_制定: PHM研究总控智能体 | 2026-03-17_

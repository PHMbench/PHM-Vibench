# Explainable_FD_Toolkit 实验补充计划

> 生成时间: 2026-03-17
> 基于: CORE.md, PAPER2_COMPLETION_SUMMARY.md

---

## 📊 当前资产盘点

### ✅ 已完成
- Benchmark结果 (benchmark_12_02, benchmark_12_04_test)
- Demo结果 (demo_signals.npy, final_demo_report.md)
- 统一评估协议

### ⚠️ 缺失 (投稿必备)
- 竞争对比 (Captum/SHAP/LIME)
- ≥2个完整工业demo
- 多模型×多方法×多数据集系统benchmark

---

## 🎯 实验清单

### P0 - 投稿必备 (必须完成)

#### 实验1: 竞争对比实验
**目标**: 与Captum/SHAP/LIME的系统性对比

**评估维度**:
| 维度 | 指标 | Captum | SHAP | LIME | Ours |
|------|------|--------|------|------|------|
| 速度 | 解释耗时(ms) | | | | |
| 稳定性 | 扰动一致性 | | | | |
| 忠实度 | Deletion AOPC | | | | |
| 工程友好度 | API复杂度评分 | | | | |

**执行命令**:
```bash
# 运行竞争对比实验
python scripts/run_competitive_comparison.py \
  --methods captum,shap,lime,ours \
  --datasets CWRU,XJTU \
  --output results/competitive_comparison/

# 生成对比报告
python scripts/generate_comparison_report.py \
  --input results/competitive_comparison/ \
  --output manuscript/tables/competitive_comparison.tex
```

**预期输出**:
- `results/competitive_comparison/metrics.json`
- `manuscript/tables/competitive_comparison.tex`
- `manuscript/figures/competitive_comparison.png`

**GPU资源**: 2 GPU小时

---

#### 实验2: 工业 Demo
**目标**: ≥2个端到端demo，记录延迟/失败率

**Demo场景**:
1. **Demo A**: CWRU轴承故障诊断
2. **Demo B**: XJTU轴承全生命周期监测

**执行命令**:
```bash
# Demo A: CWRU
python scripts/run_industrial_demo.py \
  --dataset CWRU \
  --model resnet18 \
  --output results/demo_cwru/

# Demo B: XJTU
python scripts/run_industrial_demo.py \
  --dataset XJTU \
  --model tspn \
  --output results/demo_xjtu/
```

**预期输出**:
- 延迟分布 (含P95)
- 失败率统计
- 英文图表 + 报告

**GPU资源**: 1 GPU小时

---

#### 实验3: 多模型×多方法×多数据集 Benchmark
**目标**: ≥5模型 × ≥2解释方法 × ≥2数据集 × ≥3seed

**模型矩阵**:
- ResNet18
- TSPN
- Transformer
- SincNet
- 1D-CNN

**解释方法**:
- Intrinsic (注意力权重)
- Post-hoc (GradCAM)

**数据集**:
- CWRU
- XJTU

**执行命令**:
```bash
# 批量运行benchmark
for model in resnet18 tspn transformer sincnet 1dcnn; do
  for method in intrinsic gradcam; do
    for dataset in CWRU XJTU; do
      for seed in 42 123 456; do
        python scripts/run_benchmark.py \
          --model $model \
          --method $method \
          --dataset $dataset \
          --seed $seed \
          --output results/benchmark/${model}_${method}_${dataset}_seed${seed}/
      done
    done
  done
done

# 汇总结果
python scripts/aggregate_benchmark.py \
  --input results/benchmark/ \
  --output manuscript/tables/benchmark_main.tex
```

**预期输出**:
- 主性能表 (mean±std + 95% CI)
- 解释评估表
- 统计检验结果

**GPU资源**: 15 GPU小时

---

### P1 - 加分项 (建议完成)

#### 实验4: 扩展数据集验证
**数据集**: FEMTO, IMS, THU

**GPU资源**: 10 GPU小时

#### 实验5: 可视化案例库
**目标**: 为论文提供高质量可视化

**GPU资源**: 1 GPU小时

---

## 📋 依赖检查

### 脚本存在性
```bash
# 检查关键脚本
ls scripts/run_benchmark_standalone.py
ls scripts/run_unified_explain_eval.py
ls scripts/validate_schema.py
```

### 数据存在性
```bash
# 检查数据集
ls /path/to/CWRU/
ls /path/to/XJTU/
```

### 环境依赖
- Python 3.8+
- PyTorch 1.12+
- Captum, SHAP, LIME

---

## 📊 结果模板

### 表1: 主性能表
| 模型 | CWRU Acc | XJTU Acc | 参数量 | 推理时间 |
|------|----------|----------|--------|----------|
| ResNet18 | - | - | 11.2M | - |
| TSPN | - | - | 2.1M | - |
| ... | ... | ... | ... | ... |

### 表2: 解释评估表
| 方法 | Faithfulness | Stability | Efficiency |
|------|--------------|-----------|------------|
| Intrinsic | - | - | - |
| GradCAM | - | - | - |

---

## 🚀 执行顺序

1. **Day 1-2**: 运行竞争对比实验 (P0)
2. **Day 3**: 运行工业Demo (P0)
3. **Day 4-7**: 运行多模型Benchmark (P0)
4. **Day 8-10**: 扩展数据集 (P1)
5. **Day 11**: 生成可视化案例 (P1)

**总GPU预估**: 29 GPU小时

---

## ✅ 完成标准

- [ ] 竞争对比表完整，所有指标有数据
- [ ] ≥2个Demo有完整延迟/失败率报告
- [ ] 主性能表有≥3seed统计
- [ ] 所有结果符合schema_v1
- [ ] 表格/图表可直接用于论文

---

_生成: PHM研究总控智能体 | 2026-03-17_

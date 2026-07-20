# Paper_fuzzy_XFD 实验补充计划

> 生成时间: 2026-03-17
> 基于: CORE.md, paper_blueprint.md

---

## 📊 当前资产盘点

### ✅ 已完成
- 配置文件 (config_FuzzyLogic_v2.yaml)
- 模糊规则可视化素材 (FuzzyLogic_explainable/results/)

### ⚠️ 缺失
- 多seed实验结果
- 规则级解释评估
- 安全关键失败案例

---

## 🎯 实验清单

### P0 - 投稿必备 (必须完成)

#### 实验1: 多数据集 + 多Seed实验
**目标**: CWRU+XJTU，≥3seed，输出mean±std + 95% CI

**执行命令**:
```bash
# CWRU 3-seed
for seed in 42 123 456; do
  CUDA_VISIBLE_DEVICES=0 python main.py \
    --config_dir configs/unified_baseline/config_FuzzyLogic_v2.yaml \
    --dataset CWRU \
    --seed $seed \
    --output_dir outputs/Fuzzy_CWRU_seed${seed}/
done

# XJTU 3-seed
for seed in 42 123 456; do
  CUDA_VISIBLE_DEVICES=0 python main.py \
    --config_dir configs/unified_baseline/config_FuzzyLogic_v2.yaml \
    --dataset XJTU \
    --seed $seed \
    --output_dir outputs/Fuzzy_XJTU_seed${seed}/
done
```

**预期输出**:
- 性能表: mean±std, 95% CI
- 规则激活统计

**GPU资源**: 6 GPU小时

---

#### 实验2: 规则级解释评估
**目标**: Faithfulness + Stability + Sparsity + Efficiency

**评估指标**:
| 指标 | 定义 | 实现方法 |
|------|------|----------|
| Faithfulness | Del@k, AOPC | 规则/特征遮挡 |
| Stability | 扰动下激活规则一致性 | 扰动实验 |
| Sparsity | 激活规则数、覆盖率 | 统计分析 |
| Efficiency | 推理耗时 | 时间统计 |

**执行命令**:
```bash
# 规则级解释评估
python scripts/evaluate_rule_explainability.py \
  --model_path outputs/Fuzzy_CWRU_seed42/best_model.pth \
  --dataset CWRU \
  --output results/rule_eval/
```

**GPU资源**: 2 GPU小时

---

#### 实验3: 安全关键失败案例
**目标**: 2-3个高风险误判样本深度分析

**输出要求**:
- 触发规则
- 隶属度曲线/数值
- 决策路径
- 可截图入论文

**执行命令**:
```bash
# 识别失败案例
python scripts/identify_failure_cases.py \
  --model_path outputs/Fuzzy_CWRU_seed42/best_model.pth \
  --dataset CWRU \
  --output results/failure_cases/

# 深度分析失败案例
python scripts/analyze_failure_case.py \
  --case_id 1 \
  --output results/failure_analysis/case1/
```

**GPU资源**: 0.5 GPU小时

---

### P1 - 加分项

#### 实验4: 扩展数据集验证
**数据集**:
- MFPT/THU: 规则更贴近工程可读
- Ottawa23: 规则稳定性压力测试
- SEU: 复合故障失败案例

**重点**: 安全关键失败案例从复杂工况中抽取，更能体现兜底价值

**GPU资源**: 6 GPU小时

---

## 📋 依赖检查

### 配置文件检查
```bash
# 确认配置存在
ls configs/unified_baseline/config_FuzzyLogic_v2.yaml
```

### 脚本检查
```bash
# 确认评估脚本
ls scripts/evaluate_rule_explainability.py
ls scripts/identify_failure_cases.py
ls scripts/analyze_failure_case.py
```

---

## 📊 结果模板

### 表1: 主性能表
| 数据集 | Seed 42 | Seed 123 | Seed 456 | Mean±Std | 95% CI |
|--------|---------|----------|----------|----------|--------|
| CWRU | - | - | - | - | - |
| XJTU | - | - | - | - | - |

### 表2: 规则级解释评估表
| 指标 | CWRU | XJTU |
|------|------|------|
| Del@10 | - | - |
| AOPC | - | - |
| Stability (规则一致性) | - | - |
| Sparsity (平均激活规则数) | - | - |
| Efficiency (推理耗时ms) | - | - |

### 表3: 安全关键失败案例
| 案例 | 真实标签 | 预测标签 | 触发规则 | 隶属度 | 决策路径 |
|------|----------|----------|----------|--------|----------|
| 1 | - | - | - | - | - |
| 2 | - | - | - | - | - |
| 3 | - | - | - | - | - |

### 图1: 隶属度曲线
- 失败案例的隶属度可视化

### 图2: 决策路径图
- 失败案例的决策路径追踪

---

## 🚀 执行顺序

1. **Day 1-2**: CWRU 3-seed实验
2. **Day 3-4**: XJTU 3-seed实验
3. **Day 5**: 规则级解释评估
4. **Day 6**: 安全关键失败案例分析
5. **Day 7**: 汇总结果，生成表格/图表
6. **Day 8-10**: 扩展数据集 (P1)

**总GPU预估**: 14.5 GPU小时 (P0: 8.5h, P1: 6h)

---

## ✅ 完成标准

- [ ] CWRU+XJTU各有≥3seed结果
- [ ] 规则级解释评估指标完整
- [ ] ≥2个安全关键失败案例有完整分析
- [ ] 表格/图表可直接用于论文
- [ ] 失败案例可截图入论文

---

_生成: PHM研究总控智能体 | 2026-03-17_

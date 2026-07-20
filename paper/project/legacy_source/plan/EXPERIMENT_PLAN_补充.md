# TII_operator_attention 实验补充计划

> 生成时间: 2026-03-17
> 基于: README.md, Operator_Attention_Theory_Analysis.md

---

## 📊 当前资产盘点

### ✅ 已完成
- 完整理论分析文档 (Operator_Attention_Theory_Analysis.md)
- 合成信号验证框架设计
- 验证结果 (synthetic_signal_examples.png, results/validation/)
- 概念验证 (工业数据 ~20% acc)

### ⚠️ 缺失
- 合成信号系统验证
- 定理证明完善
- 可解释性度量量化

---

## 🎯 实验清单

### P0 - 投稿必备 (必须完成)

#### 实验1: 合成信号验证实验 (最重要)
**目标**: 8类合成信号，验证算子选择的物理合理性

**合成信号类型**:
| 类型 | 描述 | 预期激活算子 |
|------|------|--------------|
| 1. 单频正弦 | 纯频信号 | FFT |
| 2. 双频叠加 | 两个频率 | FFT |
| 3. 调幅信号 | AM调制 | HT |
| 4. 调频信号 | FM调制 | HT, WF |
| 5. 脉冲信号 | 瞬态冲击 | LNO |
| 6. 噪声叠加 | 含噪信号 | MA, I |
| 7. 小波成分 | 小波特征 | WF |
| 8. 复合信号 | 多特征混合 | 多算子协同 |

**执行命令**:
```bash
# 生成8类合成信号
python code/generate_synthetic_signals.py \
  --output data/synthetic_signals/

# 运行算子注意力测试
python code/synthetic_verification.py \
  --input data/synthetic_signals/ \
  --output results/synthetic_verification/ \
  --verbose

# 生成验证报告
python code/generate_verification_report.py \
  --input results/synthetic_verification/ \
  --output manuscript/synthetic_verification_report.md
```

**预期输出**:
- 每类信号的算子注意力权重分布
- 物理合理性评分
- 热图/一致性评分/对照理论预期

**GPU资源**: 1 GPU小时

---

#### 实验2: 定理验证实验
**目标**: 验证Theorem 1-2，输出可并入LaTeX的证明与附录

**Theorem 1: 通用逼近能力**
- 验证方法: 合成信号逼近实验
- 输出: 逼近误差曲线

**Theorem 2: 物理一致性保证**
- 验证方法: 能量守恒/时频不确定性验证
- 输出: 物理约束满足度

**执行命令**:
```bash
# Theorem 1验证
python experiments/validate_theorem_1.py \
  --output results/theorems/theorem1/

# Theorem 2验证
python experiments/validate_theorem_2.py \
  --output results/theorems/theorem2/

# 生成LaTeX附录
python experiments/generate_theorem_appendix.py \
  --input results/theorems/ \
  --output manuscript/theorem_appendix.tex
```

**GPU资源**: 1 GPU小时

---

#### 实验3: 可解释性度量实验
**目标**: 量化OAS/OSS/OCS指标

**评估指标**:
| 指标 | 定义 | 公式 |
|------|------|------|
| OAS (算子激活度) | 激活算子比例 | OAS = (1/K)Σ 1[α_i > τ] |
| OSS (算子专一度) | 注意力集中度 | OSS = 1 - H({α})/log(K) |
| OCS (算子一致性) | 相似信号权重一致性 | 基于KL散度 |

**执行命令**:
```bash
python experiments/evaluate_interpretability_metrics.py \
  --dataset CWRU \
  --output results/interpretability_metrics/
```

**GPU资源**: 0.5 GPU小时

---

### P2 - 工业数据补充 (可选)

#### 实验4: 工业数据概念验证
**目标**: 配置矩阵对比，重点分析可解释性而非性能

**配置矩阵**:
| 配置 | L1正则 | 说明 |
|------|--------|------|
| A | 0 | 无正则 |
| B | 1e-6 | 轻正则 |
| C | 1e-5 | 中正则 |

**重点**: 输出权重分布与解释稳定性对照，非准确率

**GPU资源**: 2 GPU小时

---

## 📋 依赖检查

### 脚本检查
```bash
# 确认关键脚本
ls code/generate_synthetic_signals.py
ls code/synthetic_verification.py
ls experiments/validate_theorem_1.py
ls experiments/validate_theorem_2.py
```

### 理论文档检查
```bash
# 确认理论分析完整
ls Operator_Attention_Theory_Analysis.md
ls bare_jrnl_new_sample4.tex
```

---

## 📊 结果模板

### 表1: 合成信号验证
| 信号类型 | 预期算子 | 实际激活算子 | 一致性评分 |
|----------|----------|--------------|-----------|
| 单频正弦 | FFT | - | - |
| 双频叠加 | FFT | - | - |
| 调幅信号 | HT | - | - |
| ... | ... | ... | ... |

### 表2: 定理验证
| 定理 | 验证方法 | 结论 | 证据 |
|------|----------|------|------|
| T1: 通用逼近 | 逼近实验 | - | - |
| T2: 物理一致性 | 约束检验 | - | - |

### 表3: 可解释性度量
| 指标 | CWRU | XJTU | 合成信号 |
|------|------|------|----------|
| OAS | - | - | - |
| OSS | - | - | - |
| OCS | - | - | - |

### 图1: 算子注意力热图
- 8类信号的算子权重分布热图

### 图2: 物理合理性验证
- 预期 vs 实际算子选择对比

### 图3: 定理验证曲线
- 逼近误差曲线
- 物理约束满足度

---

## 🚀 执行顺序

1. **Day 1**: 生成合成信号 (P0)
2. **Day 2**: 运行合成信号验证 (P0)
3. **Day 3**: 定理验证实验 (P0)
4. **Day 4**: 可解释性度量实验 (P0)
5. **Day 5**: 生成报告和LaTeX附录 (P0)
6. **Day 6-7**: 工业数据补充 (P2)

**总GPU预估**: 4.5 GPU小时 (P0: 2.5h, P2: 2h)

---

## 📝 论文适配说明

### 目标期刊
- IEEE Transactions on Signal Processing (理论方法)
- SIAM Journal on Applied Mathematics

### 论文重点
1. **理论贡献**: 算子空间理论、物理约束注意力、可解释性度量
2. **实验重点**: 合成信号验证 > 工业数据性能
3. **价值主张**: 理论创新而非SOTA性能

### 与其他子项目的区分
- 不负责统一可解释性平台 (Explainable_FD_Toolkit)
- 不替代MoE/1D-2D/Fuzzy，而是互补 (算子级 vs 路径级/规则级)
- 全局理论整合由Neuralsymbolic_theory负责

---

## ✅ 完成标准

- [ ] 8类合成信号验证完成
- [ ] Theorem 1-2验证完成
- [ ] 可解释性度量OAS/OSS/OCS有数据
- [ ] LaTeX附录可直接并入论文
- [ ] 物理合理性分析清晰

---

_生成: PHM研究总控智能体 | 2026-03-17_

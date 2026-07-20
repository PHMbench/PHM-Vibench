# Neuralsymbolic_theory 实验补充计划

> 生成时间: 2026-03-17
> 基于: CORE.md, README.md

---

## 📊 当前资产盘点

### ✅ 已完成
- 验证demo脚本 (run_validation_demo.py, simple_validation_demo.py)
- Manuscript草稿 (paper.md)
- 理论框架文档

### ⚠️ 缺失
- 命题验证实验不完整
- 跨方法映射验证
- 多数据集论证

---

## 🎯 实验清单

### P0 - 投稿必备 (必须完成)

#### 实验1: 命题验证实验
**目标**: 每个命题至少1个可复现实验脚本 + 输出图表

**命题1: 符号约束提升可信度**
- 假设: 引入符号约束后，模型输出的可信度（一致性/可解释性）提升
- 验证方法: 对比有/无符号约束的可解释性指标

**执行命令**:
```bash
python experiments/validate_proposition_1.py \
  --with_constraint True \
  --output results/propositions/prop1_with/

python experiments/validate_proposition_1.py \
  --with_constraint False \
  --output results/propositions/prop1_without/
```

---

**命题2: 物理同构增强鲁棒性 (重点)**
- 假设: 物理同构设计提升跨域/跨工况鲁棒性
- 验证方法: 跨数据集泛化实验 + 扰动鲁棒性实验

**执行命令**:
```bash
python experiments/validate_proposition_2.py \
  --source_dataset CWRU \
  --target_datasets XJTU,FEMTO,IMS \
  --output results/propositions/prop2/
```

---

**命题3: 性能-解释性帕累托边界**
- 假设: 存在性能-解释性权衡，但可通过优化设计逼近帕累托前沿
- 验证方法: 多模型对比，绘制帕累托曲线

**执行命令**:
```bash
python experiments/validate_proposition_3.py \
  --models resnet,tspn,moe,fuzzy,operator \
  --output results/propositions/prop3/
```

**GPU资源**: 8 GPU小时

---

#### 实验2: 跨方法映射验证
**目标**: 至少覆盖 Paper1/4/5 的代表机制，映射验证"可运行"

**映射关系**:
| 子项目 | 机制 | 映射到理论层 |
|--------|------|--------------|
| Paper1 (1D-2D) | 跨模态对齐 | 语义层映射 |
| Paper4 (MoE) | 路由选择 | 符号层映射 |
| Paper5 (Fuzzy) | 规则推理 | 推理层映射 |

**执行命令**:
```bash
python code/validate_mapping.py \
  --paper1_path ../1D-2D_fusion_explainable/ \
  --paper4_path ../MOE_explainable/ \
  --paper5_path ../Paper_fuzzy_XFD/ \
  --output results/mapping_validation/
```

**GPU资源**: 2 GPU小时

---

#### 实验3: 多数据集论证
**目标**: CWRU+XJTU用于命题泛化，失败案例写成"边界条件/反例"

**执行命令**:
```bash
python experiments/validate_across_datasets.py \
  --datasets CWRU,XJTU \
  --output results/cross_dataset/
```

**GPU资源**: 4 GPU小时

---

### P1 - 加分项

#### 实验4: 边界条件分析
**目标**: 明确命题成立/失败的条件

**GPU资源**: 2 GPU小时

---

## 📋 依赖检查

### 脚本检查
```bash
# 确认验证脚本
ls experiments/validate_proposition_1.py
ls experiments/validate_proposition_2.py
ls experiments/validate_proposition_3.py
ls code/validate_mapping.py
```

### 依赖项目检查
```bash
# 确认其他子项目存在
ls ../1D-2D_fusion_explainable/
ls ../MOE_explainable/
ls ../Paper_fuzzy_XFD/
```

---

## 📊 结果模板

### 表1: 命题验证总结
| 命题 | 验证状态 | 支持证据 | 边界条件 |
|------|----------|----------|----------|
| P1: 符号约束→可信度 | - | - | - |
| P2: 物理同构→鲁棒性 | - | - | - |
| P3: 性能-解释性帕累托 | - | - | - |

### 表2: 跨方法映射验证
| 子项目 | 映射成功 | 关键发现 |
|--------|----------|----------|
| Paper1 (1D-2D) | - | - |
| Paper4 (MoE) | - | - |
| Paper5 (Fuzzy) | - | - |

### 图1: 命题验证图
- 每个命题的关键证据可视化

### 图2: 跨方法映射图
- 四层架构与子项目的映射关系

---

## 🚀 执行顺序

1. **Day 1-3**: 命题验证实验 (P0)
2. **Day 4**: 跨方法映射验证 (P0)
3. **Day 5-6**: 多数据集论证 (P0)
4. **Day 7**: 汇总结果，生成表格/图表
5. **Day 8**: 边界条件分析 (P1)

**总GPU预估**: 16 GPU小时 (P0: 14h, P1: 2h)

---

## ✅ 完成标准

- [ ] 三个命题各有验证实验
- [ ] 跨方法映射验证完成
- [ ] 多数据集论证完成
- [ ] 失败案例写成边界条件
- [ ] 表格/图表可直接用于论文

---

_生成: PHM研究总控智能体 | 2026-03-17_

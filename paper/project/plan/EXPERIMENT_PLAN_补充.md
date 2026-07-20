# MOE_explainable 实验补充计划

> 生成时间: 2026-03-17
> 基于: CORE.md, MoE_Paper_Progress_Summary.md

---

## 📊 当前资产盘点

### ✅ 已完成
- 配置文件 (config_MoE.yaml, config_MoE_5experts.yaml, config_MoE_8experts.yaml)
- 历史分析报告 (moe_analysis_report.txt)
- 可视化素材 (expert_activation_heatmap.png)

### ⚠️ 缺失
- 多seed实验结果
- 专家消融系统实验
- 路由可解释性量化评估

---

## 🎯 实验清单

### P0 - 投稿必备 (必须完成)

#### 实验1: 多Seed稳定性实验
**目标**: ≥5seed，输出mean±std + 95% CI，若CV>10%需给原因

**配置**: experts=5 (默认)

**执行命令**:
```bash
# CWRU 5-seed
for seed in 42 123 456 789 1024; do
  CUDA_VISIBLE_DEVICES=0 python main.py \
    --config_dir configs/unified_baseline/config_MoE.yaml \
    --seed $seed \
    --output_dir outputs/MoE_CWRU_seed${seed}/
done

# XJTU 5-seed
for seed in 42 123 456 789 1024; do
  CUDA_VISIBLE_DEVICES=0 python main.py \
    --config_dir configs/unified_baseline/config_MoE.yaml \
    --dataset XJTU \
    --seed $seed \
    --output_dir outputs/MoE_XJTU_seed${seed}/
done
```

**预期输出**:
- 性能表: mean±std, 95% CI
- CV分析: 若CV>10%，分析原因

**GPU资源**: 10 GPU小时 (5 seed × 2 dataset × 1h)

---

#### 实验2: 专家消融实验
**目标**: experts=3/5/8的性能-参数-稳定性曲线

**执行命令**:
```bash
# 3 experts
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config_dir configs/unified_baseline/config_MoE_3experts.yaml \
  --output_dir outputs/MoE_3experts/

# 5 experts (已在实验1完成)
# 8 experts
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config_dir configs/unified_baseline/config_MoE_8experts.yaml \
  --output_dir outputs/MoE_8experts/
```

**预期输出**:
- 消融曲线图: 性能 vs 参数量 vs 稳定性
- 最优专家数分析

**GPU资源**: 4 GPU小时

---

#### 实验3: 路由可解释性实验
**目标**: 路由熵 + 路径签名 + 专家激活分布 + 解释稳定性

**评估指标**:
| 指标 | 定义 | 评估方法 |
|------|------|----------|
| 路由熵 | H(g) = -Σg_i log(g_i) | 路由分布熵 |
| 路径签名 | σ = sign(g(x)) | 激活模式 |
| 专家激活分布 | P(E_i) | 统计分布 |
| 解释稳定性 | 路由一致性 | 扰动实验 |

**执行命令**:
```bash
# 路由可解释性评估
python scripts/evaluate_routing_interpretability.py \
  --model_path outputs/MoE_CWRU_seed42/best_model.pth \
  --dataset CWRU \
  --output results/routing_eval/
```

**GPU资源**: 2 GPU小时

---

### P1 - 加分项

#### 实验4: 扩展数据集验证
**数据集**: FEMTO, IMS, THU

**重点**: 路由可解释性"迁移"实验，展示专家激活模式是否保持物理含义

**GPU资源**: 6 GPU小时

#### 实验5: 稳定性改进对照
**目标**: 至少2种稳定性改进策略对比

**GPU资源**: 4 GPU小时

---

## 📋 依赖检查

### 配置文件检查
```bash
# 确认配置存在
ls configs/unified_baseline/config_MoE.yaml
ls configs/unified_baseline/config_MoE_3experts.yaml
ls configs/unified_baseline/config_MoE_5experts.yaml
ls configs/unified_baseline/config_MoE_8experts.yaml
```

### 脚本检查
```bash
# 确认评估脚本
ls scripts/evaluate_routing_interpretability.py
```

---

## 📊 结果模板

### 表1: 多Seed性能表
| 数据集 | Seed 42 | Seed 123 | Seed 456 | Seed 789 | Seed 1024 | Mean±Std | 95% CI | CV |
|--------|---------|----------|----------|----------|-----------|----------|--------|-----|
| CWRU | - | - | - | - | - | - | - | - |
| XJTU | - | - | - | - | - | - | - | - |

### 表2: 专家消融表
| Experts | 准确率 | 参数量 | 稳定性(CV) |
|---------|--------|--------|-----------|
| 3 | - | - | - |
| 5 | - | - | - |
| 8 | - | - | - |

### 表3: 路由可解释性表
| 指标 | CWRU | XJTU |
|------|------|------|
| 路由熵 | - | - |
| 路径签名一致性 | - | - |
| 解释稳定性 | - | - |

### 图1: 专家消融曲线
- X轴: 专家数
- Y轴: 准确率/参数量/稳定性

### 图2: 路由可视化
- 专家激活热图
- 路径签名分布

---

## 🚀 执行顺序

1. **Day 1-3**: CWRU 5-seed实验
2. **Day 4-6**: XJTU 5-seed实验
3. **Day 7**: 专家消融实验
4. **Day 8**: 路由可解释性实验
5. **Day 9**: 汇总结果，生成表格/图表
6. **Day 10-12**: 扩展数据集 (P1)

**总GPU预估**: 26 GPU小时 (P0: 16h, P1: 10h)

---

## ✅ 完成标准

- [ ] CWRU+XJTU各有≥5seed结果
- [ ] CV分析完成，若>10%有改进方案
- [ ] 专家消融曲线完整
- [ ] 路由可解释性指标完整
- [ ] 表格/图表可直接用于论文

---

_生成: PHM研究总控智能体 | 2026-03-17_

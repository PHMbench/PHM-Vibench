# Paper 4 (MOE_explainable) 执行计划 - 2024-12-14

> **目标**：从当前85%完成度的原型系统推进到论文级成果，重点解决实验一致性和数据可信度问题
> **范围**：Paper/MOE_explainable项目，完成P0-P2阶段任务
> **时间线**：24-72小时（P0）→ 1-2周（P1）→ 1个月（P2）

## 一、现状分析

### 1.1 已完成工作（✅）
- **核心架构**：物理专家模块（低频、谐波、包络）+ 统计路由器
- **代码实现**：完整的MoE模型系统，支持3/5/8专家配置
- **配置文件**：包含不同专家数量和seed20的配置
- **基础可视化**：训练曲线、专家权重分布、路径签名分析

### 1.2 关键阻塞（⚠️）
- **数据路径问题**：`THU_018_basic`路径配置未验证
- **实验一致性**：多seed结果未验证，准确率口径不统一
- **稳定性问题**：不同训练轮次结果波动较大

## 二、执行计划

### 阶段1：P0任务 - 基础验证与稳定（24-72小时）

#### 1.1 数据路径验证与修复
- [x] 检查现有配置文件中的数据路径设置
- [ ] 验证 `THU_018_basic` 数据集路径是否可访问
- [ ] 修复路径问题（如果存在）
- [ ] 测试最小可运行配置：
  ```bash
  CUDA_VISIBLE_DEVICES=1 python main_com.py \
    --config_dir configs/unified_baseline/config_MoE_3experts.yaml
  ```

#### 1.2 最佳配置复现（seed20）
- [ ] 定位seed20配置文件：`config_MoE_*_seed20.yaml`
- [ ] 执行训练并记录完整命令和输出
- [ ] 保存训练曲线、loss/accuracy图表
- [ ] 记录最终性能指标（准确率、参数量、训练时间）

#### 1.3 诊断脚本开发
- [ ] 创建 `scripts/diagnose_moe_stability.py`
  - 自动检测数据路径问题
  - 验证模型前向传播
  - 检查梯度流和参数更新
- [ ] 生成诊断报告，标识潜在问题点

### 阶段2：P1任务 - 深度实验与对比（1-2周）

#### 2.1 专家数量消融实验
- [ ] 准备实验矩阵：
  | 专家数 | 配置文件 | Seed集 | 目标指标 |
  |--------|----------|--------|----------|
  | 3      | MoE_3experts | [20,42,2024] | 准确率/参数量 |
  | 5      | MoE_5experts | [20,42,2024] | 准确率/参数量 |
  | 8      | MoE_8experts | [20,42,2024] | 准确率/参数量 |

- [ ] 批量执行实验脚本：
  ```bash
  ./scripts/run_moe_ablation.sh
  ```
- [ ] 生成性能对比图表：
  - 专家数 vs 准确率曲线
  - 专家数 vs 参数量曲线
  - 稳定性箱线图（3 seeds）

#### 2.2 稳定性改进策略验证
- [ ] 策略1：改进初始化
  - Xavier/Kaiming初始化对比
  - 专家偏置初始化（避免初始负载不均）

- [ ] 策略2：学习率调度优化
  - 余弦退火 vs StepLR
  - 专家专用学习率（router lr < experts lr）

- [ ] 策略3：门控正则化
  - 添加负载均衡损失
  - 专家使用率正则项

- [ ] 记录每种策略的改进效果

#### 2.3 统一基线表生成
- [ ] 创建 `scripts/generate_baseline_table.py`
  - 自动读取所有实验结果
  - 计算均值±标准差
  - 生成95%置信区间
  - 导出LaTeX表格格式

- [ ] 锁定发布数据：
  - 最终准确率：mean ± std (CI)
  - 参数量（精确到K）
  - Seed集合：[20, 42, 2024]
  - 训练配置（epochs, early stopping）

### 阶段3：P2任务 - 论文级产出（1个月）

#### 3.1 高级可视化开发
- [ ] 专家激活热力图
  - 时间序列 × 专家矩阵
  - 故障类型 × 专家偏好图

- [ ] 路由熵分析
  - 信息熵随epoch变化
  - 不同故障类型的路由确定性

- [ ] 路径签名可视化
  - 信号→专家→决策的完整路径
  - 物理解释性标注

#### 3.2 论文素材准备
- [ ] 图表集（paper quality）：
  - Fig 1: MoE架构图
  - Fig 2: 专家数量消融结果
  - Fig 3: 稳定性对比（箱线图）
  - Fig 4: 专家激活模式
  - Fig 5: 路由熵分析

- [ ] 表格集：
  - Table 1: 与基线方法对比
  - Table 2: 消融实验结果
  - Table 3: 稳定性统计（多seed）

- [ ] 代码仓库整理：
  - 清理实验代码
  - 添加示例notebook
  - 编写README

## 三、文件结构与交付物

### 3.1 需要创建的文件
```
Paper/MOE_explainable/plan/12_14/
├── codex/
│   ├── plan_moe_explainable_12_14.md      # 本计划文件
│   ├── p0_completion_report.md            # P0阶段完成报告
│   ├── p1_progress_report.md              # P1阶段进度报告
│   └── final_delivery_checklist.md        # 最终交付检查清单
├── scripts/
│   ├── diagnose_moe_stability.py          # 稳定性诊断脚本
│   ├── run_moe_ablation.sh                # 批量消融实验脚本
│   └── generate_baseline_table.py         # 基线表生成脚本
└── results/
    ├── figures/                           # 论文级图表
    ├── tables/                            # 结果表格
    └── logs/                              # 实验日志
```

### 3.2 需要修改的文件
- `configs/unified_baseline/config_MoE_*.yaml` - 确保数据路径正确
- `main_com.py` - 可能需要添加MoE特定的日志记录
- `Paper/MOE_explainable/README.md` - 更新最新结果和运行指南

## 四、风险与应对

### 4.1 技术风险
- **风险**：多seed实验结果差异过大
- **应对**：增加seed数量，使用统计显著性检验

### 4.2 时间风险
- **风险**：消融实验耗时过长
- **应对**：并行运行不同配置，使用checkpoint恢复

### 4.3 数据风险
- **风险**：数据集访问问题
- **应对**：准备备选数据集（CWRU, XJTU）

## 五、执行检查点

### Day 1-2：P0验证
- [ ] 数据路径可访问
- [ ] seed20配置可复现
- [ ] 诊断脚本正常运行

### Day 7：P1中期
- [ ] 完成3/5/8专家消融实验
- [ ] 至少验证2种稳定性策略
- [ ] 初步基线表生成

### Day 14：P1完成
- [ ] 所有P1任务完成
- [ ] 生成进度报告

### Day 30：P2完成
- [ ] 论文素材齐全
- [ ] 代码仓库整理完成
- [ ] 准备投稿

## 六、成功标准

1. **可复现性**：所有实验结果可通过命令行一键复现
2. **统计严谨性**：提供多seed统计和置信区间
3. **物理解释性**：专家激活模式符合物理直觉
4. **论文质量**：图表和表格达到期刊投稿标准

---

**执行优先级**：
1. 立即开始：验证数据路径和seed20复现
2. 并行执行：稳定性诊断 + 配置文件检查
3. 顺序执行：消融实验 → 可视化 → 论文素材

**关键命令记录**：
```bash
# 基础运行
CUDA_VISIBLE_DEVICES=1 python main_com.py --config_dir configs/unified_baseline/config_MoE_3experts_seed20.yaml

# 批量实验
for experts in 3 5 8; do
  for seed in 20 42 2024; do
    CUDA_VISIBLE_DEVICES=1 python main_com.py \
      --config_dir configs/unified_baseline/config_MoE_${experts}experts_seed${seed}.yaml
  done
done
```

## 七、执行日志

### 2024-12-14 执行开始
- [x] 创建计划目录结构
- [x] 复制计划文件到对应位置
- [ ] 接下来：验证数据路径配置
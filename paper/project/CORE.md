# Paper 4（唯一核心文件）：Physics‑Constrained MoE Explainable FD（顶刊口径）

> Autoresearch root normalization: the maintained paper path is `paper/UXFD_paper/MOE_explainable`, and the maintained execution root is `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench_fix`. Historical `Paper/...` references below are legacy aliases.


> 本文件是 `paper/UXFD_paper/MOE_explainable/` 的唯一“总控核心文件”。
> 目标：用“物理同构专家 + 可审计路由”把 MoE 的解释性与稳定性做成可复现证据链：不仅报告准确率，还要证明“哪个专家为何被激活”且在多 seed/多数据集下稳定。

---

## 0. 一句话定位

将故障诊断 MoE 从黑盒门控升级为路径级可解释系统：专家设计与物理机理对齐，路由决策可审计，并用多 seed 统计、专家消融与路由一致性验证其稳定与泛化。

## 0.5 Innovation Contract

- Maintained innovation authority: `innovation_contract.md`
- New-gate review must bind all innovation claims, `CWRU/XJTU/THU_006`, and `>=0.98` in-domain passes through this file before the paper can return to `completed`.

---

## 1. 顶刊硬性需求（必须满足）

### 1.1 稳定性（必须）
- ≥3 seed（建议≥5）输出 `mean±std` + `95%CI`；
- 报告 CV；若 CV>10% 必须给原因与改进对策对照。

### 1.2 专家消融（必须）
- experts=3/5/8 的性能‑参数‑稳定性曲线（同口径、同输出格式）。

### 1.3 可解释评估（必须）
- 路由熵、路径签名、专家激活分布；
- 解释稳定性/一致性对齐统一协议。

---

## 2. 当前仓库证据（已存在，可复用）

- `paper/UXFD_paper/MOE_explainable/results/moe_analysis_report.txt`
- `paper/UXFD_paper/MOE_explainable/results/expert_activation_heatmap.png`
- `paper/UXFD_paper/MOE_explainable/results/path_signature_visualization.png`

---

## 3. 唯一复现入口（对外口径）

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/MOE_explainable/configs/vibench/min.yaml --override trainer.num_epochs=1 --override data.num_workers=0
```

专家消融（配置已存在）：
```bash
CUDA_VISIBLE_DEVICES=0 python paper/UXFD_paper/MOE_explainable/scripts/run_expert_ablation_probe.py --output-dir paper/UXFD_paper/MOE_explainable/results/autoresearch/<run_id>/expert_ablation --datasets CWRU --expert-counts 3 5 8
```

---

## 4. 论文骨架（写什么 + 证据是什么）

- Method：专家物理同构设计、统计特征路由、可审计解释（路径签名/熵/激活谱）
- Results：
  - 主性能（多数据集+多seed）
  - 专家消融曲线（3/5/8）
  - 稳定性改进对照（至少2策略）
  - 路由解释一致性（稳定性/一致性）

---

## 5. 执行计划与预期结果（唯一计划入口）

- 最完整执行计划：`paper/UXFD_paper/MOE_explainable/plan/12_15/codex/EXECUTION_PLAN_12_15.md`
- 预期结果矩阵：`paper/UXFD_paper/MOE_explainable/plan/12_15/codex/EXPECTED_RESULTS_12_15.md`
- P0 任务包（执行官入口）：`paper/UXFD_paper/MOE_explainable/plan/12_15/codex/AGENT_TASKS_P0.md`

---

## 6. 历史资料整合索引（只作背景/实现细节）

- 进展与历史实验：`paper/UXFD_paper/MOE_explainable/doc/`
- 旧蓝图（已合并）：`paper/UXFD_paper/MOE_explainable/paper_blueprint.md`

---

## 7. 数据集覆盖矩阵（按 Vibench Dataset_id 扩展）

> Dataset_id↔Name 映射见：`data/vibench_dataset_catalog.md`。

### 7.1 本 Paper 建议最小覆盖（顶刊最低要求）
- In-domain：`RM_001_CWRU`（1）、`RM_002_XJTU`（2）

### 7.2 本 Paper 建议扩展覆盖（突出“路由泛化/专家可迁移”）
- 物理/采样条件多样性：`RM_003_FEMTO`（3）、`RM_006_THU`（6）、`RM_007_MFPT`（7）
- 齿轮/复合场景补强：`RM_004_IMS`（4）与/或 `RM_027_PU`（20）

### 7.3 写作口径（如何在论文里描述）
- 主表：至少 CWRU + XJTU 的多 seed；
- 路由可解释性“迁移”实验：优先 IMS/PU（展示专家激活模式是否保持物理含义）。

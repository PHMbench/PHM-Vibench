# Paper 1（唯一核心文件）：1D-2D Fusion Explainable Fault Diagnosis（顶刊口径）

> Autoresearch root normalization: the maintained paper path is `paper/UXFD_paper/1D-2D_fusion_explainable`, and the maintained execution root is `.` (the nested ViBench repository root). Historical `Paper/...` references below are legacy aliases.


> 本文件是 `paper/UXFD_paper/1D-2D_fusion_explainable/` 的唯一“总控核心文件”。
> 目标：让最严格审稿人可以逐条核验 **每个关键结论** 的证据链（数据→配置→命令→日志→结果真源→图表→正文表述）。  
> 范围：本论文只讨论 **1D–2D 融合 + 三层对齐的可解释方法**；统一评估协议/工具链由 Paper2（Toolkit）负责。

---

## 0. 论文一句话定位

以“物理–语义–几何”三层对齐实现可解释的 1D 时序 + 2D 时频融合诊断，并在 PHM‑Vibench 多数据集（至少 CWRU + XJTU）上用多 seed 统计与解释性定量评估证明：性能提升来自可检验的跨模态一致性而非黑盒拼接。

## 0.5 Innovation Contract

- Maintained innovation authority: `innovation_contract.md`
- New-gate review must bind all innovation claims, `CWRU/XJTU/THU_006`, and `>=0.98` in-domain passes through this file before the paper can return to `completed`.

---

## 1. 顶刊硬性需求（必须满足）

### 1.1 数据与评估口径（统一）
- 数据：PHM‑Vibench 多数据集验证（至少 **CWRU + XJTU**；建议扩展到更多 Vibench 数据集以增强外推可信度）。
- 统计：至少 **3-seed**，报告 `mean±std` 与 `95% CI`。
- 解释评估：按统一协议输出（不是只有可视化）：
  - Faithfulness：Deletion/Occlusion（Del@k / AOPC）
  - Stability：扰动一致性（Spearman/IoU 等）
  - Efficiency：解释耗时与资源

### 1.3 统一输出 schema（必须）
- 所有运行输出必须符合 Paper2 schema v1：`paper/UXFD_paper/Explainable_FD_Toolkit/schema/SCHEMA_V1.md`
- 每个 `<RUN_DIR>` 必须包含：
  - `run_meta.yaml`（schema_version=`paper2_schema_v1`）
  - `metrics.json`（schema_version=`paper2_schema_v1`）
  - 并通过：`python paper/UXFD_paper/Explainable_FD_Toolkit/scripts/validate_schema.py --run_dir <RUN_DIR>`

### 1.2 证据链（审稿人可验收）
每个结果表/图必须满足：
- 可追溯到 `run_meta.yaml`（含数据集、seed、git hash/版本、超参摘要）
- 可追溯到 `metrics.json`（主指标 + explainability 指标）
- 可追溯到生成命令（可复制粘贴运行）

---

## 2. 当前仓库证据（已存在，可复用）

### 2.1 已有结果素材（用于写作/对齐）
- `paper/UXFD_paper/1D-2D_fusion_explainable/results/analysis_report.txt`
- `paper/UXFD_paper/1D-2D_fusion_explainable/results/performance_comparison.png`
- `paper/UXFD_paper/1D-2D_fusion_explainable/results/attention_weights.png`
- `paper/UXFD_paper/1D-2D_fusion_explainable/results/contribution_heatmap.png`

### 2.2 现有风险证据（必须正视）
- 历史稳定性测试产出“全失败”（说明入口/配置/环境存在不一致）：  
  - `paper/UXFD_paper/1D-2D_fusion_explainable/experiments/stability_test/stability_test_summary.md`

---

## 3. 唯一复现入口（对外口径）

> 目标：未来所有文档只引用此处命令；如需变更，先更新本文件，再更新 README。

### 3.1 单数据集（对齐统一基线口径）
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/1D-2D_fusion_explainable/configs/vibench/min.yaml --override trainer.num_epochs=1 --override data.num_workers=0
```

### 3.2 多数据集（本 Paper 配置）
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/1D-2D_fusion_explainable/configs/config_CWRU.yaml
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/1D-2D_fusion_explainable/configs/config_XJTU.yaml
```

---

## 4. 方法与论文骨架（写什么 + 证据是什么）

### 4.1 Method 核心要点（必须写清楚）
- 输入：时域信号（1D）与时频表示（2D）；
- 融合：双分支/渐进融合；
- 对齐：三层对齐（物理/语义/几何）；
- 解释：模态贡献、对齐一致性、跨模态一致性。

### 4.2 结果章节最小交付（顶刊最低配置）
- Table（主性能）：CWRU + XJTU（≥3 seed，含 CI）
- Table（解释评估）：faithfulness + stability + efficiency（同样含统计）
- Figure：主结果误差条；faithfulness 曲线；stability 扰动曲线；2个成功案例 + 1个失败案例

---

## 5. 执行计划（唯一计划文件）

- 详细执行计划：`paper/UXFD_paper/1D-2D_fusion_explainable/plan/12_15/codex/EXECUTION_PLAN_12_15.md`
- 预期结果矩阵：`paper/UXFD_paper/1D-2D_fusion_explainable/plan/12_15/codex/EXPECTED_RESULTS_12_15.md`
- P0 任务包（执行官入口）：`paper/UXFD_paper/1D-2D_fusion_explainable/plan/12_15/codex/AGENT_TASKS_P0.md`

---

## 6. 历史文档整合索引（只做证据/背景，不做口径真源）

> 原则：历史文档不删，但“对外口径”只认本 CORE + 真源结果表。

- 研究与技术背景：`paper/UXFD_paper/1D-2D_fusion_explainable/doc/`
- 历史计划：`paper/UXFD_paper/1D-2D_fusion_explainable/plan/11_26/`、`paper/UXFD_paper/1D-2D_fusion_explainable/plan/12_14/`
- 旧蓝图（已合并到本 CORE 的需求/验收）：`paper/UXFD_paper/1D-2D_fusion_explainable/paper_blueprint.md`

---

## 7. 数据集覆盖矩阵（按 Vibench Dataset_id 扩展）

> Dataset_id↔Name 映射见：`data/vibench_dataset_catalog.md`。

### 7.1 本 Paper 建议最小覆盖（顶刊最低要求）
- In-domain：`RM_001_CWRU`（1）、`RM_002_XJTU`（2）

### 7.2 本 Paper 建议扩展覆盖（提升说服力）
- 时频/机械工况多样性：`RM_003_FEMTO`（3）、`RM_004_IMS`（4）
- 变转速压力测试：`RM_005_Ottawa23`（5）

### 7.3 写作口径（如何在论文里描述）
- 主表（Table 2）：至少 CWRU + XJTU；
- 泛化/鲁棒性补强（Table 5/Appendix）：优先加入 FEMTO/IMS/Ottawa23。

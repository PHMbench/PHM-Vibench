# Paper 2（唯一核心文件）：Explainable FD Toolkit（顶刊口径）

> Autoresearch root normalization: the maintained paper path is `paper/project`, and the maintained execution root is the repository root (`.`). Historical `Paper/...` references below are legacy aliases.

> 本文件是 `paper/UXFD_paper/Explainable_FD_Toolkit/` 的唯一“总控核心文件”。
> 目标：把故障诊断可解释性从“零散可视化”升级为可复现的基础设施论文：统一 API + 统一评估协议 + 一键 benchmark + 竞争对比（Captum/SHAP/LIME）+ 工业 demo。

---

## 0. 一句话定位

Explainable FD Toolkit 是面向故障诊断的“可解释性操作系统”：用统一接口与评估协议，使不同模型/不同解释方法在 PHM‑Vibench 多数据集上可公平对比、可复现、可发布，并支持工程报告与 demo 输出。

## 0.5 Innovation Contract

- Maintained innovation authority: `innovation_contract.md`
- New-gate review must bind innovation claims, required datasets, and comparison coverage through this file before the project can return to `completed`.

---

## 1. 顶刊硬性需求（必须满足）

### 1.1 统一接口（可引用、稳定）
- `SignalData / ModelPlugin / ExplainabilityMethod` 三类接口稳定；
- 输出格式统一：`run_meta.yaml`、`metrics.json`、`results.csv`、关键图表；
- 协议绑定：`Paper/doc/12_14/codex/explainability_eval_protocol.md`。
- 统一输出 schema（Paper1–6共用）：`paper/UXFD_paper/Explainable_FD_Toolkit/schema/SCHEMA_V1.md`（schema_version=`paper2_schema_v1`）。

### 1.2 Benchmark（可复现、可扩展）
- ≥5 个模型 × ≥2 类解释方法（intrinsic + post-hoc）；
- ≥2 个数据集（至少 CWRU + XJTU）；并建议扩展到更多 Vibench 数据集以增强基准价值（见 `data/vibench_dataset_catalog.md`）；
- ≥3-seed 或等价统计口径（CI/显著性）。

### 1.3 竞争对比（必须）
- Captum/SHAP/LIME：速度、稳定性、忠实度、工程友好度对比表（脚本可复现）。

### 1.4 工业 demo（必须）
- ≥2 个端到端 demo（脚本 + 英文图表 + 报告），并记录延迟/失败率。

---

## 2. 唯一复现入口（对外口径）

```bash
# 工具包独立benchmark
python paper/UXFD_paper/Explainable_FD_Toolkit/scripts/run_benchmark_standalone.py

# 统一基线解释评估（对齐主仓库模型）
python paper/UXFD_paper/Explainable_FD_Toolkit/scripts/run_unified_explain_eval.py
```

建议输出收口到：
- `paper/UXFD_paper/Explainable_FD_Toolkit/benchmark_results/`
- `paper/UXFD_paper/Explainable_FD_Toolkit/results/`
- 以及各 paper 的 `outputs/`（必须包含 `run_meta.yaml` 与 `metrics.json` 并通过 schema 校验）。

---

## 3. 结果真源与“禁止写死数字”规则

### 3.1 规则
- README/论文正文 **禁止直接写死** “某模型=xx%”；必须引用真源表（CSV/JSON）或由脚本生成的表格文件。

### 3.2 真源输出（统一 schema 建议）
- `run_meta.yaml`：dataset_id、seed、config、git hash、环境摘要、运行命令
- `metrics.json`：主性能 + explainability 指标（faithfulness/stability/efficiency/…）
- `results.csv`：逐样本或逐折明细（若需要统计检验）

---

## 4. 论文骨架（写什么 + 证据是什么）

- Method：接口设计（抽象边界）、评估协议（指标定义+实现）、报告生成与demo流水线
- Results：
  - 多模型×多方法×多数据集 benchmark 主表
  - Captum/SHAP/LIME 对比表（含资源与时间）
  - 工业 demo 的可复现报告与延迟统计

---

## 5. 执行计划与预期结果（唯一计划入口）

- 最完整执行计划：`paper/UXFD_paper/Explainable_FD_Toolkit/plan/12_15/codex/EXECUTION_PLAN_12_15.md`
- 预期结果矩阵：`paper/UXFD_paper/Explainable_FD_Toolkit/plan/12_15/codex/EXPECTED_RESULTS_12_15.md`
- P0 任务包（执行官入口）：`paper/UXFD_paper/Explainable_FD_Toolkit/plan/12_15/codex/AGENT_TASKS_P0.md`

---

## 6. 历史文档整合索引（只作背景/实现细节）

- 设计与使用文档：`paper/UXFD_paper/Explainable_FD_Toolkit/doc/`
- 结果与可视化：`paper/UXFD_paper/Explainable_FD_Toolkit/results/`、`paper/UXFD_paper/Explainable_FD_Toolkit/benchmark_results/`
- 旧蓝图（已合并到本 CORE 的需求/验收）：`paper/UXFD_paper/Explainable_FD_Toolkit/paper_blueprint.md`

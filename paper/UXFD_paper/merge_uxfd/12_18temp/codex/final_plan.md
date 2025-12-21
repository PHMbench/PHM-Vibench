# UXFD 合并 Final Plan（最终版）

本文件是 UXFD 合并到 PHM‑Vibench 的**唯一最终计划**（Final SSOT）。目标是做到：
- 7 篇 paper **完全解耦**（配置/个性化实验都在各自 submodule 内）
- 主仓库只沉淀 **可复用通用能力**（model_factory / explain_factory / report+collect）
- 运行方式严格遵循 vibench：`python main.py --config <yaml> [--override ...]`

> 相关支撑文档（按职责拆分，按需阅读）  
> - @`paper/UXFD_paper/merge_uxfd/12_21/codex/step_by_step_ops.md`（本科生可照做的逐步操作）  
> - @`paper/UXFD_paper/merge_uxfd/12_21/codex/submodule_config_conventions.md`（submodule configs 规范）  
> - @`paper/UXFD_paper/merge_uxfd/12_21/codex/VIBENCH_MAPPING_TEMPLATE.md`（每篇 paper 的 VIBENCH.md 模板）  
> - @`paper/UXFD_paper/merge_uxfd/12_21/codex/model_alignment_plan.md`（模型范式不偏离上游的论证）  
> - @`paper/UXFD_paper/merge_uxfd/12_21/codex/model_collection_integration_plan.md`（对比模型移植到 X_model/baselines）  
> - @`paper/UXFD_paper/merge_uxfd/12_21/codex/manifest_to_csv_spec.md`（manifest.json → CSV 汇总规范）  

---

## 0) 硬约束（不满足就不合并）

- 单一入口不变：`python main.py --config <yaml> [--override ...]`
- 5‑block 不变：`environment/data/model/task/trainer`
- 不新增第 6 个一级 block：扩展开关统一挂 `trainer.extensions.*`
- 主仓库 demos/tests 不依赖任何 UXFD submodule 初始化
- 主仓库 `requirements.txt` 为硬上限；Explain/Agent 额外依赖必须 optional import + 可审计降级（不能 ImportError 崩溃）
- **README 同步**：每次新增/调整一个目录（文件夹），必须同时新增/更新该目录下的 `README.md`，说明该目录的职责边界、入口与产物

---

## 1) 目标产物（你最后要看到什么）

### 1.1 每个 paper submodule 内（paper 自己负责）

在 `paper/UXFD_paper/<paper_id>/` 内必须存在：
- `configs/vibench/min.yaml`：最小可跑（1 epoch）
- `VIBENCH.md`：唯一复现入口（命令 + metadata 要求 + 产物检查）

目录规范：@`paper/UXFD_paper/merge_uxfd/12_21/codex/submodule_config_conventions.md`

### 1.2 主仓库内（平台能力区，通用复用）

必须提供：
- **通用 UXFD 模块**（1D/2D 算子、fusion、fuzzy、operator_attention、TSPN_UXFD）：`src/model_factory/X_model/UXFD/**`
- **对比 baselines**（来自 `model_collection`，只保留 forward + 适配）：`src/model_factory/X_model/baselines/**`
- **解释框架**：`src/explain_factory/**`（继承/移植既有解释方法；以 eligibility+artifacts 为核心）
- **证据链索引**：每次 run 生成 `artifacts/manifest.json`
- **汇总脚本**：读取 manifest 批量导出 CSV（论文表输入）

---

## 2) 目录与职责边界（最重要的解耦点）

### 2.1 Paper submodules（个性化/配置/论文资产区）

- 路径：`paper/UXFD_paper/<paper_id>/`
- 放：configs、ablation、论文写作资产、paper 专属脚本
- 不放：会影响主仓库维护性的通用模块（通用算子/通用解释/通用基线）

### 2.2 主仓库（通用能力区）

- 路径：`src/**`
- 放：可复用模型组件（算子/融合/规则/注意力/2D 时频）、通用 baselines、通用 explain/report/collect
- 不放：7 篇 paper 的完整配置文件与个性化实验（全部留在 submodule）

---

## 3) 模型整合策略（不偏离上游 UXFD 模型范式）

核心原则：尽量贴近上游
`/home/user/LQ/B_Signal/Unified_X_fault_diagnosis/model/{Signal_processing.py, Signal_processing_2D.py, TSPN.py}`。

### 3.1 默认路线：`TSPN_UXFD`（保持上游层级结构）

`TSPN_UXFD` 必须保持：
- `SignalProcessingLayer → FeatureExtractorlayer → Classifier`
- 输入 `(B,L,C)`，与 vibench 数据形状一致

并在不改变数学逻辑的情况下补齐：
- Registry：稳定 `operator_id`（算子可插拔、可追溯）
- Adapter：明确 layout（解决 `BLC/BCL/BTFC` 混用）
- HookStore：统一收集中间量，为 explain_factory 提供证据（不允许各算子自行落盘）

详细论证与接入方式：@`paper/UXFD_paper/merge_uxfd/12_21/codex/model_alignment_plan.md`

### 3.2 UXFD 通用模块的主仓库落位（建议结构）

在 `src/model_factory/X_model/UXFD/` 内按主题组织：
- `signal_processing_1d/`：对齐上游 `Signal_processing.py`
- `signal_processing_2d/`：对齐上游 `Signal_processing_2D.py`（输出常见为 `BTFC`）
- `fusion/`：对齐 `Fusion1D2D*.py`
- `fuzzy/`：对齐 `FuzzyLogic*.py`
- `operator_attention/`：对齐 `operator_attention*.py`
- `tspn/`：`TSPN_UXFD` + HookStore/Adapters/Registry

### 3.3 对比模型（model_collection）整理到 `X_model/baselines/`

- 来源：`/home/user/LQ/B_Signal/Unified_X_fault_diagnosis/model_collection/`
- 目标：`src/model_factory/X_model/baselines/`
- 原则：只保留“模型 + forward + vibench 适配”，不带 dataset/trainer/脚本

落地计划：@`paper/UXFD_paper/merge_uxfd/12_21/codex/model_collection_integration_plan.md`

---

## 4) Explain/Report/Collect（证据链闭环）

### 4.1 explain_factory：继承现有可解释方法（不要重造轮子）

Explain 的最小承诺：
- 永远产出 `artifacts/data_metadata_snapshot.json`
- explain 开启时永远产出 `artifacts/explain/eligibility.json`（即使不可用也要 reasons，不崩溃）

解释方法来源：
- 优先移植/封装上游已有方法，例如 `model_collection/GradCAM_XFD.py`（以 explainer 形式接入）
- 梯度/IG/occlusion 尽量用纯 PyTorch 实现；第三方库必须 optional import

解释开关建议放在：
```yaml
trainer:
  extensions:
    explain:
      enable: true
      explainer: "router_weights"  # 或 "gradients" / "timefreq" / "gradcam_xfd" / "fuzzy_rules"
```

### 4.2 report：每次 run 固定产出 `manifest.json`

`artifacts/manifest.json` 作为 run 的索引（SSOT），供 collect 与论文脚本稳定消费。

### 4.3 collect：把很多份 manifest 扁平化成 CSV（更清晰）

collect 的最终输出是 CSV（而不是让人读 json）：
- `reports/uxfd_runs.csv`：一行一个 run
- `reports/uxfd_explain.csv`：可选，解释统计/明细

字段与扁平化规则（固定）：@`paper/UXFD_paper/merge_uxfd/12_21/codex/manifest_to_csv_spec.md`

---

## 5) 执行拆分（Work Packages，按依赖顺序）

### WP0：Submodule 落位与入口文档
- 目标：`paper/UXFD_paper/<paper_id>/VIBENCH.md` + `configs/vibench/min.yaml` 完整落在各自 submodule
- 参考模板：@`paper/UXFD_paper/merge_uxfd/12_21/codex/VIBENCH_MAPPING_TEMPLATE.md`

### WP1：主仓库 UXFD 通用模块整理（X_model/UXFD）
- 目标：把 1D/2D/融合/规则/注意力等“通用模块”搬到主仓库并有序组织

### WP2：对比 baselines 整理（X_model/baselines）
- 目标：把 `model_collection` 的对比模型变成 vibench 可加载的 baselines

### WP3：explain_factory 落地（继承既有方法）
- 目标：eligibility + artifacts + 最少一两个 explainer 跑通

### WP4：report+manifest + collect→CSV
- 目标：每次 run 生成 manifest；脚本扫 runs 输出 CSV

### WP5：agent_factory（TODO-only，可选）
- 目标：结构化蒸馏落盘（默认不接 LLM，不依赖网络）

执行细化步骤：@`paper/UXFD_paper/merge_uxfd/12_21/codex/step_by_step_ops.md`

---

## 6) Definition of Done（最终验收）

- 主仓库命令全部通过（不依赖 submodule）：
  - `python main.py --config configs/demo/00_smoke/dummy_dg.yaml`
  - `python -m scripts.validate_configs`
  - `python -m pytest test/`
- 任意 1 个 submodule 的最小配置能跑通：
  - `python main.py --config paper/UXFD_paper/<paper_id>/configs/vibench/min.yaml --override trainer.num_epochs=1`
- 输出目录包含证据链：
  - `config_snapshot.yaml`
  - `artifacts/manifest.json`
  - `artifacts/data_metadata_snapshot.json`
  - `artifacts/explain/eligibility.json`（若 explain.enable=true）
- collect 能生成 CSV：
  - `reports/uxfd_runs.csv`（最少包含 paper_id/run_id/metrics_path 等列）
- 本次改动涉及的每个目录都有同步的 `README.md`（新目录必须有；旧目录必须更新到与计划一致）

# UXFD Merge Backlog（从 12_22 迁移的未完成 TODO）

目的：把之前分散在 `12_22/*`、`final_plan.md` 等处的“未完成事项”集中到 12/23，形成一个清晰的执行清单。

范围说明：
- 这里只列 **UXFD merge 相关** 的 TODO（WP0–WP5），不把主仓库无关的历史 TODO 一并搬进来。
- 代码移植以 “先跑通（Copy + Adapter）→ 再优化” 为策略（review 结论）。

上游路径约定（用于 TODO 中引用）：

```bash
export UXFD_UPSTREAM=/home/user/LQ/B_Signal/Unified_X_fault_diagnosis
```

---

## P0（阻塞项：必须先完成）

### WP0：Submodules（先做 1 篇 Pilot，再复制到其余 6 篇）

- [ ] 选择 Pilot paper（默认建议）：`paper/UXFD_paper/1D-2D_fusion_explainable` `[复杂度: 低]`
- [ ] 在 Pilot submodule 内新增：
  - [ ] `paper/UXFD_paper/<pilot>/configs/vibench/min.yaml` `[复杂度: 中]`
  - [ ] `paper/UXFD_paper/<pilot>/VIBENCH.md` `[复杂度: 低]`
  - [ ] `paper/UXFD_paper/<pilot>/configs/vibench/README.md`（若新建目录）`[复杂度: 低]`
- [ ] Pilot 验证（至少 1 epoch + 产物闭环）：
  - [ ] `python main.py --config paper/UXFD_paper/<pilot>/configs/vibench/min.yaml --override trainer.num_epochs=1` `[复杂度: 低]`
  - [ ] 检查 `<run_dir>/artifacts/manifest.json` 是否存在 `[复杂度: 低]`
  - [ ] `python -m scripts.collect_uxfd_runs --input <output_root> --out_dir reports/` `[复杂度: 低]`
- [ ] 复制模板到剩余 6 个 submodule（每篇至少先落一个能跑的 `min.yaml`，paper 个性化后续再扩展）`[复杂度: 中]`

备注：
- submodule 内改动需要在 submodule 仓库提交（本地 commit 即可），主仓库只更新 gitlink。
- 模板与规范：
  - `paper/UXFD_paper/merge_uxfd/12_21/codex/VIBENCH_MAPPING_TEMPLATE.md`
  - `paper/UXFD_paper/merge_uxfd/12_21/codex/submodule_config_conventions.md`

---

## P1（高优先：让 Pilot 真正跑通所需“血肉”）

### WP1：UXFD 通用组件真正移植（`src/model_factory/X_model/UXFD_component/**`）

目标：补齐当前 “骨架已立，血肉未填” 的部分，让 Pilot config 在真实 Context 下可复用验证。

- [ ] 迁移策略固定：**Copy + Adapter**（先跑通，不追求一次性重构）
- [ ] 去重与统一配置（解决 `X_model/*.py` 与 `X_model/UXFD/**` 的双份逻辑）：
  - [ ] 明确规则：`X_model/UXFD_component/**` 只放可复用组件；`X_model/*.py` 只放 model_factory entrypoints/shims `[复杂度: 中]`
  - [ ] 将 `TSPN_UXFD.py` 从 alias 升级为真正 UXFD 入口（默认关闭扩展能力，保持行为不变）`[复杂度: 中]`
  - [ ] 将 `Signal_processing.py` / `Feature_extract.py` 逐步改为 shim re-export（真实实现迁到 `X_model/UXFD_component/signal_processing/sp_1d` 与 `X_model/UXFD_component/feature_extractor`）`[复杂度: 中]`
  - [ ] （可选）新增 `model.name: UXFD_component` dispatcher（`X_model/UXFD_component/__init__.py` 导出 `Model`），用 `model.preset` 统一 TSPN/baselines `[复杂度: 高]`
- [ ] signal_processing 目录结构（按你的要求：覆盖所有 SP，而不只 1D）：
  - [ ] `src/model_factory/X_model/UXFD_component/signal_processing/adapters/`（输入形状归一化为 `BLC/BTFC`）`[复杂度: 中]`
  - [ ] `src/model_factory/X_model/UXFD_component/signal_processing/sp_1d/`（对齐上游 `Signal_processing.py`）`[复杂度: 中]`
  - [ ] `src/model_factory/X_model/UXFD_component/signal_processing/sp_2d/`（对齐上游 `Signal_processing_2D.py`）`[复杂度: 中]`
- [ ] 组件优先级（建议按 Pilot 依赖排序）：
  1) [ ] `${UXFD_UPSTREAM}/model/Fusion1D2D.py` / `Fusion1D2D_simple.py` → `UXFD/fusion_routing/` `[复杂度: 中]`
  2) [ ] `${UXFD_UPSTREAM}/model/Signal_processing_2D.py`（至少 1 条 BTFC 可用路径）→ `UXFD/signal_processing/sp_2d/` `[复杂度: 中]`
  3) [ ] `feature_extractor_2d`（T/F 双分支 + rfft(magnitude)）→ `UXFD/feature_extractor/` `[复杂度: 中]`
  4) [ ] `${UXFD_UPSTREAM}/model/FuzzyLogic*.py`（若 Pilot 需要）→ `UXFD/logic_inference/` 或 `UXFD/fusion_routing/` `[复杂度: 高]`
  5) [ ] `${UXFD_UPSTREAM}/model/operator_attention.py` / `OperatorAttention_*`（若 Pilot 需要）→ `UXFD/fusion_routing/` `[复杂度: 中]`
  6) [ ] `${UXFD_UPSTREAM}/model/MoE*.py`（若 Pilot 需要）→ `UXFD/fusion_routing/` `[复杂度: 高]`
- [ ] 数据集输入形状适配（best-effort adapter）：
  - [ ] 1D：`(B,C,L)/(B,L)/(B,L,1)` → `BLC`
  - [ ] 2D：`(B,C,T,F)/(B,T,F)/(B,T,F,1)` → `BTFC`
  - [ ] 无法判别语义时给出可读错误信息（后续可复用到 eligibility/report）
- [ ] 复杂值 FFT 统一约定：`torch.fft.*` → `abs()`（magnitude only）

### WP2：TSPN_UXFD HookStore Wrapper（Explainability 的关键前置）

目标：不修改 `src/model_factory/X_model/TSPN.py`，通过 wrapper 捕获中间证据，服务 WP3。

- [ ] 新增 `src/model_factory/X_model/UXFD_component/tspn/` wrapper（组合或继承）`[复杂度: 中]`：
  - [ ] 捕获 SP 输出 / FE 输出 / Fusion 输出 / Attention weights / Router weights（以 best-effort 为准）
  - [ ] 写 `artifacts/explain/hookstore.json`（旁路写文件，不影响训练）
  - [ ] 不改变 forward 计算结果
- [ ] 在 `src/model_factory/model_registry.csv` 增加条目（例如 `TSPN_UXFD_HOOKED`），供 paper config 选择 `[复杂度: 低]`

---

## P2（中优先：对比实验、解释闭环增强）

### WP3：explain_factory 真正执行（产出 `summary.json`）

当前状态：eligibility 已写出，但 explainer 没有被 pipeline/trainer 调用。

- [ ] 至少接入 1 个可跑 explainer（优先 torch-only）`[复杂度: 中]`：
  - [ ] `gradcam_xfd`（或 `timefreq`/`router_weights`）
  - [ ] 产出：`artifacts/explain/summary.json`（并写回 manifest 字段）
- [ ] 明确 explainer 的 required meta schema（采样率/窗长/stride 等）`[复杂度: 中]`

### WP4：report/collect 工具稳定性（补单测）

当前状态：manifest→CSV 已可用，但缺少测试保障。

- [ ] 给 `scripts/collect_uxfd_runs.py` 增加最小单元测试（构造假的 `run/artifacts/manifest.json`，断言 CSV 列/值）`[复杂度: 低]`
- [ ] （可选）输出第二张 explain 明细表：`reports/uxfd_explain.csv`

### WP5：baselines 扩展（torch-only 优先）

- [ ] 来源：`${UXFD_UPSTREAM}/model_collection/*.py`
- [ ] 先做 inventory（列出有哪些 baselines、哪些需要额外依赖）`[复杂度: 低]`
  - [ ] `ls ${UXFD_UPSTREAM}/model_collection` 并生成清单（写入 `paper/UXFD_paper/merge_uxfd/12_23/TODO_BACKLOG.md` 附录即可）
- [ ] torch-only 优先（先保证能 import + forward）`[复杂度: 中]`
  - [ ] ResNet / SincNet / WKN / EELM / F_EQL（若存在）
  - [ ] TFN / MCN / Physics_informed_PDN（先跑通最小 forward，再决定是否注册）
- [ ] 依赖额外库的 baseline（默认不注册，避免破坏主仓库可运行性）`[复杂度: 中]`
  - [ ] CI_GNN（典型依赖 `torch_geometric`）
- [ ] 统一适配接口（全部接受 `x: (B,L,C)` 输出 logits），并提供对应 `BASE_*` entrypoint 让 model_factory 直接 import `[复杂度: 中]`
- [ ] 按 `paper/UXFD_paper/merge_uxfd/12_21/codex/model_collection_integration_plan.md` 继续迁移 baseline `[复杂度: 中]`
- [ ] 对需要额外依赖的模型（如 `CI_GNN`/`torch_geometric`）保持 optional-import 或不注册
- [ ] 每个 baseline 的最小 vibench config 仍放在各自 paper submodule 内（不污染主仓库 configs）

---

## P3（可选：不阻塞当前里程碑）

### Agent（TODO-only 蒸馏落盘）

- [ ] 仅做 TODO-only evidence 落盘（默认不接 LLM、不开网络）`[复杂度: 中]`
- [ ] 等 explain/report schema 稳定后再考虑 agent 的自动整理

---

## 索引（原始出处，便于追溯）

- `paper/UXFD_paper/merge_uxfd/12_22/status_review_and_todos.md`
- `paper/UXFD_paper/merge_uxfd/12_22/upstream_gap_analysis_and_plan.md`
- `paper/UXFD_paper/merge_uxfd/12_18temp/codex/final_plan.md`

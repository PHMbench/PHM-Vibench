# UXFD 合并落地操作手册（给本科生的逐步修改指南）

本手册把 `paper/UXFD_paper/merge_uxfd/12_18temp/codex/init_plan.md` 的 PR0–PR5 变成“照做就能完成”的步骤清单。

约束提醒（必须遵守）：
- 单一入口不变：`python main.py --config <yaml> [--override ...]`
- 5‑block 不变：`environment/data/model/task/trainer`
- 不新增第 6 个一级 block：扩展开关挂在 `trainer.extensions.*`（或兼容位置）
- 主仓库 tests/demos 不依赖 paper submodule 初始化
- `requirements.txt` 为硬上限：Explain/Agent 额外依赖必须 optional import + 可审计降级

---

## 0. 开始前检查（先做，避免把仓库改坏）

### 0.0 README 规则（必须遵守）
只要你在某个 PR 里新增/调整了一个目录（文件夹），就必须同时新增/更新该目录下的 `README.md`：
- 写清楚该目录的职责边界（放什么/不放什么）
- 给出最小入口（如何被 vibench 加载/如何跑）
- 说明产物与输出路径（如 artifacts/manifest、eligibility、CSV）

### 0.1 基础自检（你必须能跑通）
在仓库根目录执行：
```bash
python main.py --config configs/demo/00_smoke/dummy_dg.yaml
python -m scripts.validate_configs
python -m pytest test/
```

如果这里不通过，先不要做 UXFD 合并；先把环境/依赖问题解决。

### 0.2 你将要新增的“主仓库能力区”目录（记住这些）
- 模型：`src/model_factory/X_model/uxfd_tspn/`
- UXFD 通用算子/模块（建议分主题整理）：`src/model_factory/X_model/UXFD/`
- Baseline 对比模型（来自 model_collection）：`src/model_factory/X_model/baselines/`
- Explain：`src/explain_factory/`
- Agent（TODO-only）：`src/agent_factory/`
- 映射文档（放在各自 submodule 内）：`paper/UXFD_paper/<paper_id>/VIBENCH.md`
- 每篇 paper 的配置文件（放在各自 submodule 内）：`paper/UXFD_paper/<paper_id>/configs/vibench/`
- 汇总脚本（collect）：`scripts/collect_uxfd_runs.py`

### 0.3 你不该碰的东西（除非导师说可以）
- 不改 `main.py` CLI 语义
- 不新增 YAML 第 6 个一级 block
- 不把新配置写进 `configs/reference/`（这是 legacy）
- 不让 tests 依赖 `paper/UXFD_paper/*` submodule 已初始化

---

## PR0：UXFD Paper submodule “落位 + 入口文档”

目标：先把 paper 资产以 submodule 形式放好，同时主仓库在 submodule 未 init 时也能正常运行。

### PR0-1 创建目录结构（只创建入口与占位）
1) 创建目录：`paper/UXFD_paper/`
2) 在该目录新增两个文件：
   - `paper/UXFD_paper/README.md`：7 篇 paper 的索引页（只放链接与说明边界）
   - `paper/UXFD_paper/README_SUBMODULE.md`：submodule 初始化说明（参考 `paper/README_SUBMODULE.md` 的写法）
3) 创建 7 个 submodule 目录名（先空目录也行，后续再 init submodule）：
   - `paper/UXFD_paper/fusion_1d2d/`
   - `paper/UXFD_paper/xfd_toolkit/`
   - `paper/UXFD_paper/llm_xfd_toolkit/`
   - `paper/UXFD_paper/moe_xfd/`
   - `paper/UXFD_paper/fuzzy_xfd/`
   - `paper/UXFD_paper/nesy_theory/`
   - `paper/UXFD_paper/op_attention_tii/`

### PR0-2 映射文档位置（重要：放在各自 submodule 内，不污染主仓库 docs）

约定：每篇 paper 的“vibench 映射/一键复现说明”都写到对应 submodule 里，例如：
- `paper/UXFD_paper/<paper_id>/VIBENCH.md`

原因：
- 主仓库 `docs/` 保持干净，避免混入 7 篇 paper 的细节
- paper 的复现口径与其 README/实验资产在同一 repo 内，最不容易漂移

操作建议：
1) submodule 初始化后进入该 submodule 目录
2) 新增 `VIBENCH.md`（可直接复制 `paper/UXFD_paper/merge_uxfd/12_21/codex/VIBENCH_MAPPING_TEMPLATE.md`）
3) 同步更新该 submodule 内相关目录的 `README.md`（如新增了 `configs/vibench/` 等目录）
3) 在 submodule 内提交并 push
4) 回到主仓库提交 submodule 指针更新

### PR0-3 验收
```bash
python main.py --config configs/demo/00_smoke/dummy_dg.yaml
python -m pytest test/
```

---

## PR1：TSPN_UXFD（贴近上游）+ Registry/HookStore/Adapters（模型骨架）

目标：尽量不偏离上游 `/home/user/LQ/B_Signal/Unified_X_fault_diagnosis/model/{TSPN,Signal_processing,Signal_processing_2D}.py`，
把 UXFD 通用模块整理进主仓库 `model_factory`，并先跑通最小 forward。

### PR1-0 选择实现路线（本科生默认选 A）

**路线 A（推荐，最贴近上游）**：做 `TSPN_UXFD`，保持 `SignalProcessingLayer → FeatureExtractorlayer → Classifier`：
1) 复用/轻改主仓库现有 `src/model_factory/X_model/TSPN.py`、`src/model_factory/X_model/Signal_processing.py`
2) 追加上游 2D 模块：新增 `src/model_factory/X_model/UXFD/signal_processing_2d/`（移植 `Signal_processing_2D.py`）
3) 把 1D‑2D fusion、fuzzy、operator_attention 等“通用算子”移植到 `src/model_factory/X_model/UXFD/**`
4) 增加 HookStore（统一收集中间量，供 explain_factory 用）
5) 注册模型名 `TSPN_UXFD` 到 `src/model_factory/model_registry.csv`

**路线 B（可选，后续需要更复杂 operator_graph 才做）**：GraphConfigurableModel（更通用的图容器）。
下面 PR1-1～PR1-8 的目录骨架更偏路线 B；如果你先走路线 A，可先跳过这些实现细节。

### PR1-1 新增模型目录骨架
创建目录：
```
src/model_factory/X_model/uxfd_tspn/
  __init__.py
  graph_model.py
  operator_registry.py
  hook_store.py
  operators/
    __init__.py
    base.py
  adapters/
    __init__.py
    permute.py
```

### PR1-2 写 `BaseOperator`（最小接口）
在 `src/model_factory/X_model/uxfd_tspn/operators/base.py` 定义：
- `forward(x, meta) -> x'`
- `input_layout` / `output_layout`
- `capabilities()`
- `explain_hooks()`

### PR1-3 写 LayoutSpec 与 AdapterOperators（先解决 BLC<->BCL）
在 `src/model_factory/X_model/uxfd_tspn/adapters/permute.py` 实现 `PermuteAdapter`：
- 输入输出 layout 明确（`BLC`→`BCL` 或反向）
- 只做 `permute`，不要混入业务逻辑

### PR1-4 写 OperatorRegistry（稳定 `operator_id`）
在 `src/model_factory/X_model/uxfd_tspn/operator_registry.py`：
- 注册表：`operator_id -> operator_class`
- 禁止用类名自动生成 id（id 必须是稳定字符串）

建议命名例子：
- `SP_1D/FFT`
- `FE/Mean`
- `ROUTER/OP_ATT`

### PR1-5 写 HookStore（统一收集中间量）
在 `src/model_factory/X_model/uxfd_tspn/hook_store.py`：
- 最小结构：`store[name] = {"tensor": t, "semantic": "...", "layout": "...", "axes": [...] }`
- GraphModel forward 里统一写入（operator 不允许自己落盘）

### PR1-6 写 GraphConfigurableModel（解析 operator_graph、自动插 adapter、传 meta、收集 hooks）
在 `src/model_factory/X_model/uxfd_tspn/graph_model.py`：
1) 从 config 读取 `model.operator_graph`
2) 按 stage 顺序拼接 operators
3) 如果前后 layout 不一致：
   - 能用 adapter 修复则自动插入
   - 否则报错，并在错误信息里打印 layout chain（便于排查）
4) forward 时把 `meta` 一路传下去
5) forward 后得到 HookStore（供 explain_factory 使用）

### PR1-7 注册到主仓库模型 registry
编辑 `src/model_factory/model_registry.csv`，新增一行，例如：
- `model.type=X_model`
- `model.name=TSPN_UXFD`（或 `GraphModel_UXFD`）
- `module_path=src/model_factory/X_model/uxfd_tspn/graph_model.py`（确保导出 `Model` 类或符合现有 factory 约定）

### PR1-8 写最小单测（只测 forward）
在 `test/` 下新增一个最小测试：
- 构造一个 operator_graph（比如 `PermuteAdapter + Linear` 的 dummy operator）
- forward 输入 `(B,L,C)` 不报错即可

### PR1 验收
```bash
python -m pytest test/
python -m scripts.config_inspect --config configs/demo/00_smoke/dummy_dg.yaml --override trainer.num_epochs=1
```

---

## PR1b：把 `model_collection` 的对比模型整理进 `X_model/baselines/`

目标：把 `/home/user/LQ/B_Signal/Unified_X_fault_diagnosis/model_collection/` 的对比模型做成“可被 vibench 加载”的
baseline，方便后续在各 paper submodule 里写对比实验 configs。

### PR1b-1 建目录与命名（不要污染现有 CNN/Transformer）
创建目录：
```
src/model_factory/X_model/baselines/
```

建议模型注册名（避免重名）：
- `BASE_ResNet_MC`
- `BASE_SincNet`
- `BASE_WKN`
- `BASE_TFN`
- `BASE_MCN`

### PR1b-2 逐个移植（只保留模型 forward）
对每个 baseline：
1) 从上游文件抽取模型定义（不要带 trainer/dataset）
2) 包一层 `Model` 类，输入统一为 `(B,L,C)`（内部 permute 适配）
3) 输出统一为 `(B,num_classes)`
4) 注册到 `src/model_factory/model_registry.csv`

参考落位规范：`paper/UXFD_paper/merge_uxfd/12_21/codex/model_collection_integration_plan.md`

### PR1b 验收
用 dummy 数据跑任意一个 baseline（先 1 epoch）：
```bash
python main.py --config configs/demo/00_smoke/dummy_dg.yaml --override model.name=BASE_ResNet_MC trainer.num_epochs=1
```

---

## PR2：每篇 paper 的配置文件放在各自 submodule（彻底解耦，不污染主 configs）

目标：7 篇 paper 的所有配置文件都保存在各自 submodule 内；主仓库只提供通用 base_configs 与通用组件。

### PR2-1 在每个 submodule 内创建 vibench configs 目录
对每个 paper submodule（进入 submodule 目录后）创建：
```
configs/vibench/
```

并至少提供 1 个“最小可跑配置”（5‑block）：
- 示例路径：`paper/UXFD_paper/<paper_id>/configs/vibench/min.yaml`

写法建议（减少重复）：
- 在 YAML 里使用 `base_configs` 引用主仓库的 `configs/base/**` 与 demo 模板（只覆盖 paper 差异）

### PR2-2 更新 submodule 内的 `VIBENCH.md`
每篇 paper 必须在 submodule 内提供唯一入口：
```
paper/UXFD_paper/<paper_id>/VIBENCH.md
```
至少包含：
- 一键命令（指向 submodule 内的 config）：
  - `python main.py --config paper/UXFD_paper/<paper_id>/configs/vibench/min.yaml --override trainer.num_epochs=1`
- Explain 所需 metadata 最小需求（用于 ExplainReady）
- 证据链产物（manifest/eligibility/snapshot/metrics）检查路径

### PR2 验收（最关键）
挑 1 篇 paper 先跑通（submodule config）：
```bash
python main.py --config paper/UXFD_paper/op_attention_tii/configs/vibench/min.yaml --override trainer.num_epochs=1
python -m scripts.validate_configs
```

---

## PR3：Explain Factory（metadata_reader + ExplainReady + 最小解释器）

目标：即使只安装 core requirements，也要能产出 `eligibility.json`；解释不可用时必须“可审计降级”。

### PR3-1 新增统一 batch 解包函数（meta 穿透）
新增文件：`src/utils/batch.py`
实现：`unpack_batch(batch) -> (x, y, meta)`，支持：
- `(x, y)`
- `(x, y, meta)`
- `{"x":..., "y":..., "meta":...}`

### PR3-2 新增 explain_factory 目录
创建：
```
src/explain_factory/
  __init__.py
  metadata_reader.py
  eligibility.py
  explainers/
    __init__.py
    router_weights.py
    gradients.py
    gradcam_xfd.py
    timefreq.py
    fuzzy_rules.py
```

### PR3-3 metadata_reader（来源优先级 + fallback）
实现优先级：
1) batch 内的 meta（来自 `unpack_batch`）
2) dataset 的 `get_metadata()`（若存在）
3) fallback 默认（并标记 `meta_source="default"`）

必须能输出：
- `artifacts/data_metadata_snapshot.json`
  - `meta_source` / `degraded` / `missing_keys`

### PR3-4 ExplainReady（门控 + reasons）
实现：`ExplainReady(model, x, meta) -> (ok, reasons)`
- `reasons` 必须包含 `code/message/suggestion`
- 落盘：`artifacts/explain/eligibility.json`

### PR3-5 最小解释器（先做两个）
1) `router_weights`：如果 HookStore 有 router/operator 权重 → 直接输出（内生解释）
2) `gradients`：如果 torch 可用 → 输出时间×通道归因（后验解释）

继承现有方法（避免重复造轮子）：
3) `gradcam_xfd`：优先参考并移植 `/home/user/LQ/B_Signal/Unified_X_fault_diagnosis/model_collection/GradCAM_XFD.py`
4) `timefreq`：对齐 `Signal_processing_2D.py` 的时频输出（BTFC）做可视化与频带归因
5) `fuzzy_rules`：对齐 `FuzzyLogic*.py` 的规则命中/隶属度/推理链输出（若模型启用）

所有额外依赖（例如 captum/shap）必须 optional import；没装也不许崩溃。

### PR3-6 接入点：用 `trainer.extensions.explain.enable` 控制
你需要在 trainer 的某个稳定阶段调用 explain（常见选择）：
- validation/test 结束后（更适合解释结果落盘与汇总）
- 或每 N steps（不建议一开始就做，太慢）

要求：
- `enable=false`：不影响主流程
- `enable=true`：无论能不能解释，至少产出 `eligibility.json` + `data_metadata_snapshot.json`

并新增解释器选择：
- `trainer.extensions.explain.explainer: router_weights | gradients | timefreq | fuzzy_rules | ...`

### PR3 验收
```bash
python main.py --config configs/demo/00_smoke/dummy_dg.yaml --override trainer.num_epochs=1
python -m pytest test/
```
检查输出目录中是否出现：
- `artifacts/data_metadata_snapshot.json`
- `artifacts/explain/eligibility.json`（即使解释器不可用也必须有）

---

## PR4：Collect/Report（生成 manifest + 汇总脚本）

目标：固定证据链格式，paper 脚本只读 `manifest.json` 就能找到所有产物。

### PR4-1 生成 `artifacts/manifest.json`
在 run 结束时生成（建议写在 trainer 的 on_fit_end/on_test_end）：
- `paper_id/preset_version/run_id`
- `config_snapshot/metrics/figures_dir`
- `data_metadata_snapshot/eligibility/explain_dir/distilled_dir`
- 可选：`layout_chain/hooks_index`

### PR4-2 汇总脚本
新增：`scripts/collect_uxfd_runs.py`
- 读取多个 run 目录下的 `manifest.json`
- 生成总表：CSV/JSON（按 paper_id/dataset/domain/model 聚合）

建议输出两个 CSV（更清晰）：
- `reports/uxfd_runs.csv`：一行一个 run（manifest 扁平化后直接落表）
- `reports/uxfd_explain.csv`：解释相关的明细/统计（可选）

#### PR4-2.1 CSV 扁平化规则（写死在脚本里，避免后续漂移）
1) 规定一套稳定列名（优先使用 `prefix/field`）：
   - `paper_id,run_id,preset_version,manifest_path,config_snapshot`
   - `metrics_path,figures_dir,explain_dir,distilled_dir`
   - `meta_source,degraded,missing_keys`
   - `explain_ok,explainer_id,explain_reasons,explain_summary_path`
   - `metric/acc,metric/f1,metric/auc,...`（按你们输出的 metrics 实际 key 展开）
2) dict 用 `prefix/field` 展开；缺失填空字符串
3) list/复杂结构（例如 reasons 列表）用 JSON 字符串写入单元格；或另出明细 CSV
4) 所有路径列写相对路径（相对 run 目录或相对仓库根），避免机器差异

### PR4 验收
对一个跑过的 run：
```bash
python -m scripts.collect_uxfd_runs --input save/ --output save/uxfd_summary.csv
```
并确认生成的 `reports/uxfd_runs.csv`（或你指定的输出路径）能直接用 Excel 打开阅读。

---

## PR5：agent_factory（TODO-only 蒸馏落盘）

目标：不接 LLM，只把“可用于后续写论文/写解释”的结构化信息落盘。

### PR5-1 新增目录与模板
创建：
```
src/agent_factory/
  __init__.py
  todo_only.py
  templates/
    distilled_frontmatter.md
```

落盘规范：
- `distilled/<paper_id>/<run_id>/evidence.json`
  - 摘要：metrics + explain_summary + 关键配置摘要（operator_graph 摘要）

### PR5 验收
开启 `trainer.extensions.agent.enable=true`（仍不调用 LLM），检查 `distilled/` 目录结构是否生成。

---

## 最终总验收（DoD）

当你完成 PR0–PR5，至少满足：
1) 主仓库 demos/tests 全部通过（不依赖 submodule）
2) 任意一个 submodule 内的 `paper/UXFD_paper/<paper_id>/configs/vibench/*.yaml` 能跑通 1 epoch
3) 输出目录包含：
   - `config_snapshot.yaml`
   - `artifacts/data_metadata_snapshot.json`
   - `artifacts/explain/eligibility.json`
   - `artifacts/manifest.json`

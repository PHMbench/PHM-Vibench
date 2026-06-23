# PHM-Vibench Overall Optimization Report Outline

This report outline is a writing scaffold for the next optimization pass. It uses four heading
levels:

- Level 1: optimization domain.
- Level 2: problem cluster.
- Level 3: concrete optimization item.
- Level 4: required content unit.

The report should separate essential complexity from accidental complexity. Essential complexity
includes signal models, HSE/InfoNCE math, dataset constraints, and benchmark protocol. Accidental
complexity includes silent fallback, duplicated glue, unclear contracts, dependency mixing, and
documentation drift.


## Current State Snapshot

- Branch context: current work is on `lq_merge_UXFD`, with multiple pre-existing local changes.
- Maintained execution path: `python main.py --config <yaml>` with explicit top-level `pipeline`.
- Verified gates from the latest pass: 32 targeted tests passed, config validation passed, config inspect passed, and Hydra dummy smoke wrote a valid manifest.
- Known non-report blockers: legacy TFN README image links break docs validation; existing non-report whitespace remains in `src/model_factory/X_model/MWA_CNN.py`.
- Report purpose: turn optimization findings into an executable register, not to redefine signal-model or benchmark science.

## Optimization Register

| ID | Priority | Status | Area | Deliverable | Gate |
|---|---|---|---|---|---|
| 1.1.1 | P0 | Done | 主入口 | `main.py` strict config entry | `test_main_strictness` + smoke |
| 1.1.2 | P0 | Done | 主入口 | top-level `pipeline` required | `config_inspect` pipeline_import |
| 1.2.1 | P1 | Done | 配置 | documented merge precedence | `config_inspect` field sources |
| 1.2.2 | P1 | Done | 配置 | Hydra boundary and registry status | `gen_config_atlas` diff gate |
| 2.1.1 | P0 | Done | P01 | linear training flow via run contract | P01 dummy smoke |
| 2.1.2 | P0 | Done | P01 | shared run contract helpers | `test_run_contract_helper` |
| 2.2.1 | P0 | Done | P02 | explicit `pipeline_mode` dispatch | `test_pipeline_02_modes` |
| 2.2.2 | P0 | Done | P02 | no silent fallback | P02 mode tests + source scan |
| 2.3.1 | P1 | Done | P02 | stage checkpoint and result contract | P02 stage contract test |
| 3.1.1 | P0 | Done | Loss | unlabeled paired-view InfoNCE | `test_infonce_pairing` |
| 3.1.2 | P0 | Done | Loss | supervised InfoNCE label checks | contrastive loss tests |
| 3.2.1 | P0 | Done | Loss | strategy compute fail-fast | `test_hse_contrastive_failfast` |
| 3.2.2 | P0 | Done | HSE | HSE tensor invariant checks | `test_hse_contrastive_failfast` |
| 4.1.1 | P0 | Done | Artifacts | required manifest schema | `test_run_artifacts_contract` |
| 4.1.2 | P0 | Done | Artifacts | resolved config snapshot | smoke manifest inspection |
| 4.1.3 | P0 | Done | Artifacts | metrics CSV discovery including legacy | `test_run_artifacts_contract` |
| 4.2.1 | P0 | Done | Artifacts | single sidecar helper path | run artifact helper test |
| 4.2.2 | P1 | Done | Artifacts | explain eligibility fail-loud | enabled-explain failure test |
| 5.1.1 | P0 | Done | CI | active config tools workflow | workflow under `.github/workflows` |
| 5.1.2 | P0 | Done | CI | core/test dependency gate | workflow install command |
| 5.2.1 | P0 | Done | Tests | targeted unit gates | 22 targeted tests passed |
| 5.2.2 | P1 | Partial | Tests | demo smoke matrix | dummy smoke passed; P02 current-branch rerun pending |
| 5.2.3 | P1 | Done | Tests | docs/static check separation | known legacy doc/diff blockers |
| 6.1.1 | P1 | Done | Deps | core/test/dev/gui split | requirements file scan |
| 6.1.2 | P1 | Done | Deps | optional logger behavior | smoke without wandb/swanlab |
| 6.2.1 | P2 | Done | Frontend | Streamlit marked experimental | import-skip smoke |
| 6.2.2 | P2 | Done | Frontend | `app/` to `frontend/` migration | docs path scan |
| 7.1.1 | P1 | Done | Docs | README/AGENTS command alignment | docs path scan |
| 7.1.2 | P1 | Done | Docs | repo index and generated atlas discipline | atlas gate passed after generation |
| 7.2.1 | P1 | Done | Release | release command checklist | Command Matrix |
| 7.2.2 | P1 | Done | Release | residual risk register | known blockers documented |
| 8.1.1 | P1 | Done | Report | fixed four-level item template | heading count check |
| 8.2.1 | P1 | Done | Report | priority labels for every item | metadata coverage check |

# 1. 主入口与 Config-First 工作流优化

一级标题需要说明本优化域的维护价值：它决定用户如何启动实验、CI 如何验证配置、父仓如何复现结果。这里应明确 `python main.py --config <yaml>` 是维护主路径，其他入口只能作为兼容或实验路径存在。

## 1.1 入口契约收紧

二级标题需要聚合“入口行为不清晰”这一类问题，包括默认 demo、隐式 pipeline、兼容参数、错误信息等。

### 1.1.1 `python main.py --config <yaml>` 作为唯一维护主路径

Priority: `P0`  
Status: `Done`  
Evidence: `main.py` strict config entry  
Gate: `test_main_strictness` + smoke

三级标题需要写清一个可独立检查的优化项：主入口必须由显式 config 驱动，不能靠默认路径启动。

#### 现状

- 当前维护目标是 config-first，用户通过 `python main.py --config <yaml>` 启动实验。
- `--config_path` 可保留为兼容参数，但不能高于 `--config`。
- README、AGENTS、demo README 需要使用同一条命令口径。

#### 问题

- 如果入口仍存在默认 demo，会隐藏配置缺失或错误路径。
- 如果文档同时推荐多个入口，用户和 CI 无法判断哪个才是受维护路径。
- 入口模糊会放大下游 pipeline fallback 和 artifact contract 漂移。

#### 改法

- 缺失 `--config` 和 `--config_path` 时直接退出，错误信息包含“缺少显式配置”。
- `--config` 优先，`--config_path` 只作为兼容输入。
- 文档只把 `python main.py --config <yaml>` 作为主命令。

#### 验收

- 单测覆盖 `main([])` 或缺失 config 的退出行为。
- smoke demo 使用 `python main.py --config configs/demo/00_smoke/dummy_dg.yaml` 成功。
- README、AGENTS、configs README 不再描述隐式默认 demo。

### 1.1.2 顶层 `pipeline:` 必填

Priority: `P0`  
Status: `Done`  
Evidence: top-level `pipeline` required  
Gate: `config_inspect` pipeline_import

三级标题需要写清 YAML 文件如何声明 pipeline，以及缺失时如何 fail-fast。

#### 现状

- 每个 maintained config 应声明顶层 `pipeline:`。
- `scripts.config_inspect` 可展示 pipeline import target。

#### 问题

- 隐式 pipeline 会让同一个 YAML 在不同入口下产生不同解释。
- 缺失 pipeline 如果被默认为 `Pipeline_01_default`，会造成 silent behavior change。

#### 改法

- 主入口读取 YAML 时检查其必须是 mapping。
- 顶层 `pipeline` 必须是非空字符串。
- pipeline import 失败应报出模块名和配置路径。

#### 验收

- 单测覆盖缺失 pipeline、非 mapping YAML、无效 pipeline。
- `python -m scripts.config_inspect --config <yaml>` 的 `pipeline_import` 为 PASS。

## 1.2 配置解析与覆盖顺序

二级标题需要说明配置合并的固定顺序，避免多套 config 系统互相覆盖。

### 1.2.1 Base configs、local override、CLI override 的优先级

Priority: `P1`  
Status: `Done`  
Evidence: documented merge precedence  
Gate: `config_inspect` field sources

三级标题需要写清配置数据流，避免“同名字段在哪里生效”成为隐式约定。

#### 现状

- 配置采用五块模型：`environment/data/model/task/trainer`。
- demo config 可通过 `base_configs` 复用基础块。
- CLI dot override 是运行时最高优先级。

#### 问题

- 如果 local override 和 CLI override 顺序不清，实验复现会失败。
- 如果配置字段既出现在 data 又出现在 task，容易产生双源真相。

#### 改法

- 报告中明确合并顺序：base configs -> config overrides -> local override -> CLI override。
- 对高风险重复字段列清单，例如 batch size、num workers、device。
- 对每个 demo 给出最小 override 示例。
- `configs/README.md` 固定上述顺序，并要求用 `config_inspect` 查看 field sources。

#### 验收

- `scripts.config_inspect` 输出 field sources。
- `scripts.validate_configs` 对 registry 中 active demo 全部通过。

### 1.2.2 Hydra 配置与现有 YAML 配置的边界

Priority: `P1`  
Status: `Done`  
Evidence: Hydra boundary and registry status  
Gate: `gen_config_atlas` diff gate

三级标题需要说明 Hydra 是新增矩阵还是主路径替代，防止重复维护两套真相。

#### 现状

- 当前仓库已有 `configs/hydra/` 与传统 demo YAML。
- `configs/config_registry.csv` 中已有 Hydra demo 条目。
- `docs/CONFIG_ATLAS.md` 由 registry 生成。

#### 问题

- 如果 Hydra demo 进入 registry，但 atlas 未同步，会造成文档漂移。
- 如果传统 YAML 与 Hydra YAML 同时声明同一 demo，报告必须说明谁是维护入口。

#### 改法

- 报告中把 Hydra 作为 config matrix 扩展说明，不默认替代传统 smoke。
- 每个 Hydra demo 必须有 registry 行、atlas 条目、最小命令。
- 明确哪些 Hydra demo 已 sanity_ok，哪些只是草稿。
- `configs/README.md` 明确 Hydra 仍解析成相同五块 config 运行契约。

#### 验收

- `python -m scripts.gen_config_atlas && git diff --exit-code docs/CONFIG_ATLAS.md` 无差异。
- Hydra demo registry 行的 pipeline、owner code、output pattern 完整。

# 2. Pipeline 优化

一级标题需要说明 pipeline 是训练行为的边界：它不应承载重复胶水，而应串联清晰的数据流。

## 2.1 P01 默认训练流水线

二级标题需要聚合默认训练路径中的重复逻辑、run_dir 处理、manifest 写入、数据工厂关闭等问题。

### 2.1.1 训练主流程线性化

Priority: `P0`  
Status: `Done`  
Evidence: linear training flow via run contract  
Gate: P01 dummy smoke

三级标题需要描述 P01 的目标执行顺序，让读者一屏内理解输入到输出。

#### 现状

- P01 负责常规 DG/CDDG/Fewshot 等单阶段训练。
- 当前较好的流程是：加载配置 -> 准备 run context -> 构建 stack -> fit/test -> 写结果与 manifest。

#### 问题

- 如果 pipeline 内部反复拼接 run_dir、logger_name、artifact path，会形成重复胶水。
- 如果 data/model/task/trainer 构建散落在 pipeline 和 helper 中，读者需要跨文件追踪状态。

#### 改法

- P01 只保留主流程调用。
- run_dir、config snapshot、data metadata、manifest 交给 run contract/helper。
- 保留必要的日志输出，但不重复解释每个底层构建细节。

#### 验收

- P01 smoke demo 通过。
- 成功 run 后存在 `config_snapshot.yaml`、`test_result_0.csv`、`artifacts/manifest.json`。
- manifest 的 `metrics_path` 指向 test result。

### 2.1.2 Run contract 统一

Priority: `P0`  
Status: `Done`  
Evidence: shared run contract helpers  
Gate: `test_run_contract_helper`

三级标题需要说明 run contract 的职责边界。

#### 现状

- 当前仓库已有 `src/utils/training/run_contract.py`，可承接 run context、stack 构建、结果写入。
- `src/explain_factory/run_artifacts.py` 负责 sidecar artifact。

#### 问题

- 如果 pipeline、run_contract、run_artifacts 都各自判断 manifest enable，会产生三处规则。
- 如果 P01/P02 写不同格式 test result，父仓消费会变复杂。

#### 改法

- run_contract 负责训练 run 的统一写入时机。
- run_artifacts 只负责 config snapshot、metadata snapshot、eligibility 这类 sidecar。
- manifest writer 只负责构造父仓可消费 JSON。

#### 验收

- 单测覆盖 run_contract helper。
- artifact contract 单测覆盖 sidecar + manifest。

## 2.2 P02 预训练/少样本流水线

二级标题需要聚合 P02 中最容易产生 silent behavior change 的模式选择、stage 编排和 checkpoint 传递。

### 2.2.1 `pipeline_mode` 显式分发

Priority: `P0`  
Status: `Done`  
Evidence: explicit `pipeline_mode` dispatch  
Gate: `test_pipeline_02_modes`

三级标题需要列清三种模式，避免靠 YAML 结构猜测行为。

#### 现状

- P02 已有 `single`、`staged`、`legacy` 三种模式。
- `pipeline_mode` 缺失时应直接报错。

#### 问题

- 如果只根据 `stages` 是否存在分支，单阶段和 staged YAML 容易误判。
- 如果 `fs_config_path` 与 `single/staged` 同时出现，应视为配置冲突。

#### 改法

- `single`：禁止 `fs_config_path`，禁止 `stages`。
- `staged`：要求非空 `stages`，禁止 `fs_config_path`。
- `legacy`：要求 `fs_config_path`。

#### 验收

- 单测覆盖缺失 mode、未知 mode、mode 与参数冲突。
- P02 HSE single demo 通过。

### 2.2.2 删除静默 fallback

Priority: `P0`  
Status: `Done`  
Evidence: no silent fallback  
Gate: P02 mode tests + source scan

三级标题需要说明 P02 失败策略：错误上抛，不回退。

#### 现状

- P02 的目标是 fail-fast，不再在 orchestrator 失败后偷偷跑 legacy stage。

#### 问题

- fallback 会让错误配置看似训练成功，污染 benchmark 结果。
- fallback 后产物路径和 manifest 可能不符合父仓消费契约。

#### 改法

- 删除 broad try/except fallback。
- 保留兼容 legacy 的显式 `pipeline_mode=legacy`。
- 错误信息包含 mode、config path、冲突字段。

#### 验收

- 单测证明 orchestrator 抛错不会触发 legacy fallback。
- demo matrix 脚本检查源码中不存在 silent fallback 模式。

## 2.3 多阶段 Orchestrator

二级标题需要聚合 staged workflow 的 checkpoint、stage summary 和产物契约。

### 2.3.1 Stage 结果与 checkpoint 契约

Priority: `P1`  
Status: `Done`  
Evidence: stage checkpoint and result contract  
Gate: P02 stage contract test

三级标题需要说明每个 stage 结束后必须有什么可验证结果。

#### 现状

- orchestrator 需要在多阶段之间传递 checkpoint。
- 每个 stage 都应写 run manifest。

#### 问题

- 如果 stage 只返回 Python 对象，不写文件契约，外部无法消费。
- checkpoint 缺失如果被跳过，会造成后续 stage 语义不清。

#### 改法

- 每个 stage 结束后写 `test_result_*.csv` 和 manifest。
- checkpoint 缺失时按 stage 类型 fail-fast 或显式标记 skipped。
- stage summary 记录 run_dir、metrics、checkpoint path。
- 单测固定 checkpoint 注入、fit/test 调用、metrics/manifest 写入和 data close。

#### 验收

- staged config 的 summary 可定位每个 stage run_dir。
- 缺失 checkpoint 的单测或 dry-run 能明确报错。

# 3. Loss 与训练关键路径优化

一级标题需要说明这里属于科学核心路径，不能为了“少代码”改变 loss 语义，只能显化不变量和删除吞错。

## 3.1 InfoNCE 正样本构造

二级标题需要聚合无标签路径、有标签路径和数值稳定性。

### 3.1.1 无标签路径 paired-view 约定

Priority: `P0`  
Status: `Done`  
Evidence: unlabeled paired-view InfoNCE  
Gate: `test_infonce_pairing`

三级标题需要明确无标签 InfoNCE 的 batch layout。

#### 现状

- 无标签 InfoNCE 需要由两视图构造正样本。
- 当前约定应为 `[view1..., view2...]`。

#### 问题

- 如果无标签路径没有正样本，loss 可能变成 0 或 NaN。
- 如果奇数 batch 被接受，pairing 语义必然错误。

#### 改法

- 要求 2D features。
- 无标签 batch size 必须为偶数。
- 正样本 mask 使用上下半 batch 对应样本。

#### 验收

- 单测：偶数 paired views 有非零 loss 且可 backward。
- 单测：奇数 batch 抛 `ValueError`。

### 3.1.2 有标签路径监督正样本

Priority: `P0`  
Status: `Done`  
Evidence: supervised InfoNCE label checks  
Gate: contrastive loss tests

三级标题需要明确 labels 约束和无正样本处理。

#### 现状

- 有标签 InfoNCE 使用同标签非自身样本作为正样本。

#### 问题

- labels 长度不匹配会广播出错误 mask。
- 无正样本时返回 0 会掩盖数据划分问题。

#### 改法

- labels 必须是一维并与 batch 等长。
- 无正样本时 fail-fast 或给出明确策略；报告需记录实际选择。

#### 验收

- 单测覆盖 labels length mismatch。
- 单测覆盖无正样本场景。

## 3.2 Contrastive Strategy 与 HSE 训练

二级标题需要聚合 strategy manager、HSE task、classification + contrastive 混合路径。

### 3.2.1 Strategy compute 不吞错

Priority: `P0`  
Status: `Done`  
Evidence: strategy compute fail-fast  
Gate: `test_hse_contrastive_failfast`

三级标题需要指出哪些错误必须上抛。

#### 现状

- contrastive strategy 是训练关键路径。
- 初始化失败或 compute 失败应中止训练。

#### 问题

- `except Exception -> zero loss` 会让模型训练看似成功但实际跳过核心目标。

#### 改法

- strategy init 失败抛 `RuntimeError`。
- loss compute 失败带 loss type、feature shape、labels 状态上抛。

#### 验收

- 单测构造非法 InfoNCE 输入，strategy 不返回 0。
- HSE demo 训练日志出现非零 contrastive loss。

### 3.2.2 HSE 输入不变量

Priority: `P0`  
Status: `Done`  
Evidence: HSE tensor invariant checks  
Gate: `test_hse_contrastive_failfast`

三级标题需要写清训练前必须检查的 tensor 约束。

#### 现状

- HSE contrastive 依赖 logits、features、labels、system_ids 等输入。

#### 问题

- label 越界、NaN/Inf features、未初始化 strategy 都会造成 silent bad run。

#### 改法

- 检查 logits 维度、label range、feature finite、labels required、strategy initialized。
- metadata/system_id 推断可 best-effort，但训练 loss 不可 best-effort。

#### 验收

- 单测覆盖非法 label range。
- 单测覆盖 strategy 未初始化。

# 4. Artifact Contract 与父仓消费优化

一级标题需要说明父仓、paper submodule、报告脚本都只应该读取固定产物接口，而不是猜目录结构。

## 4.1 Run 产物固定接口

二级标题需要聚合 manifest、config snapshot、metrics、metadata snapshot。

### 4.1.1 `artifacts/manifest.json`

Priority: `P0`  
Status: `Done`  
Evidence: required manifest schema  
Gate: `test_run_artifacts_contract`

三级标题需要列清 manifest schema。

#### 现状

- manifest 是父仓消费入口。
- collector 和 frontend 都依赖 manifest。

#### 问题

- 如果 manifest 字段缺失，父仓需要猜 `test_result` 或 config 路径。
- 如果 optional 字段不存在，消费者需要额外分支。

#### 改法

- 必填字段：`run_id`、`run_dir`、`stage`、`timestamp`、`seed`、`git_sha`、`config_snapshot`、`metrics_path`、`data_metadata_snapshot`。
- 可选字段：`predictions_path`、`figures_dir`、`explain_dir`、`eligibility`、`distilled_dir`。
- 可选字段不存在时保留空字符串。

#### 验收

- manifest contract 单测覆盖必填字段。
- required mode 下缺失 metrics 或 data metadata 会报错。

### 4.1.2 `config_snapshot.yaml`

Priority: `P0`  
Status: `Done`  
Evidence: resolved config snapshot  
Gate: smoke manifest inspection

三级标题需要说明 snapshot 是复现实验的证据。

#### 现状

- 运行时需要保存 resolved config。

#### 问题

- 只保存原始 config path 无法复现 CLI override 后的真实配置。

#### 改法

- 每个 run_dir 写 `config_snapshot.yaml`。
- snapshot 发生在 run context 准备阶段。

#### 验收

- smoke run 的 manifest `config_snapshot` 指向存在文件。

### 4.1.3 `test_result_*.csv`

Priority: `P0`  
Status: `Done`  
Evidence: metrics CSV discovery including legacy  
Gate: `test_run_artifacts_contract`

三级标题需要说明 metrics 文件命名和兼容策略。

#### 现状

- 新路径使用 `test_result_<iteration>.csv`。
- legacy 可能产生 `test_result.csv`。

#### 问题

- 只支持一种命名会断开旧产物收集。

#### 改法

- manifest 优先选择 `test_result_*.csv`。
- 找不到时兼容 `test_result.csv`。

#### 验收

- 单测覆盖 legacy `test_result.csv` fallback。

## 4.2 Artifact glue 去重

二级标题需要聚合 P01/P02/orchestrator 的重复 sidecar 逻辑。

### 4.2.1 Sidecar 写入入口统一

Priority: `P0`  
Status: `Done`  
Evidence: single sidecar helper path  
Gate: run artifact helper test

三级标题需要说明哪个模块负责哪类 artifact。

#### 现状

- `run_contract` 负责 run 级写入时机。
- `run_artifacts` 负责 sidecar 文件。
- `manifest` 负责父仓 JSON。

#### 问题

- 如果每个 pipeline 自己写 sidecar，字段漂移会很快出现。

#### 改法

- P01/P02/orchestrator 只调用 run_contract 或 run_artifacts。
- 不在 pipeline 中手写 manifest enable、rank 判断、field assembly。

#### 验收

- 搜索 pipeline 文件，不应出现重复的 manifest 字段拼装。

### 4.2.2 Explain eligibility 失败策略

Priority: `P1`  
Status: `Done`  
Evidence: explain eligibility fail-loud  
Gate: enabled-explain failure test

三级标题需要说明 explain 未启用与启用后的不同行为。

#### 现状

- explain 仍是可选能力。

#### 问题

- 启用 explain 时吞掉 eligibility 写入错误，会让 frontend/paper 误以为 explain 不适用。

#### 改法

- explain 未启用：manifest `eligibility` 允许为空。
- explain 启用：eligibility 写入失败应报错。

#### 验收

- 单测覆盖 explain disabled 时 manifest 合法。
- 单测覆盖 explain enabled 写入成功与写入失败直接抛错。

# 5. CI、测试与验收矩阵优化

一级标题需要说明 CI 是防止 config/document/artifact 漂移的最小守门人。

## 5.1 CI 从 TODO 激活

二级标题需要聚合 GitHub workflow、依赖安装和核心命令。

### 5.1.1 Config tools workflow

Priority: `P0`  
Status: `Done`  
Evidence: active config tools workflow  
Gate: workflow under `.github/workflows`

三级标题需要写清 CI 应跑哪些命令。

#### 现状

- `.github/workflows_TODO` 中的 config CI 应迁入 `.github/workflows`。
- 当前仓库已有 core CI 和 config tools CI。

#### 问题

- TODO workflow 不会执行。
- CI 如果只跑局部测试，无法覆盖入口和 artifact contract。

#### 改法

- workflow 安装 core + test 依赖。
- 顺序执行 validate configs、gen atlas diff、config inspect、pytest。

#### 验收

- CI 文件位于 `.github/workflows/`。
- 本地等价命令通过。

### 5.1.2 CI 依赖最小化

Priority: `P0`  
Status: `Done`  
Evidence: core/test dependency gate  
Gate: workflow install command

三级标题需要说明为什么 GUI 不进核心 CI。

#### 现状

- 依赖已拆为 core/test/dev/gui 等文件。

#### 问题

- GUI 依赖进入 CI 会增加安装时间和不稳定性。

#### 改法

- CI 使用 `requirements-core.txt` 和 `requirements-test.txt`。
- Streamlit smoke 可以 import-skip，不作为必过门禁。

#### 验收

- CI workflow 不安装 GUI requirements。

## 5.2 测试分层

二级标题需要说明哪些测试属于快速门禁，哪些属于 demo 验收。

### 5.2.1 单元测试

Priority: `P0`  
Status: `Done`  
Evidence: targeted unit gates  
Gate: 22 targeted tests passed

三级标题需要列出关键单测面。

#### 现状

- 当前已有 main strictness、P02 mode、InfoNCE、HSE fail-fast、artifact contract、collector 测试。

#### 问题

- 如果只跑 demo，失败定位慢。

#### 改法

- 单元测试固定入口、模式分发、loss 不变量、manifest schema。

#### 验收

- targeted tests 全部通过。

### 5.2.2 Demo smoke

Priority: `P1`  
Status: `Partial`  
Evidence: demo smoke matrix  
Gate: dummy smoke passed; P02 current-branch rerun pending

三级标题需要列出最小 demo 矩阵。

#### 现状

- dummy smoke 可作为离线最小验证。
- P02 HSE demo 可验证 contrastive path。

#### 问题

- 只跑 unit test 无法证明 Lightning 路径产物完整。

#### 改法

- 最小矩阵：dummy DG、P02 HSE single、P02 CDDG。
- 每项记录命令、退出码、manifest path、metrics path。

#### 验收

- demo 成功后 manifest 字段完整。

### 5.2.3 文档与静态检查

Priority: `P1`  
Status: `Done`  
Evidence: docs/static check separation  
Gate: known legacy doc/diff blockers

三级标题需要记录哪些检查是主线、哪些失败是历史残留。

#### 现状

- `validate_docs` 可能被 legacy/untracked 文档坏链挡住。
- `git diff --check` 可能被非本轮文件尾随空格挡住。

#### 问题

- 把历史残留混入本轮，会让优化报告无法收口。

#### 改法

- 报告明确本轮触达文件的 `diff --check` 状态。
- 单独列 residual risks。
- 新增 maintained docs path scan，避免旧 frontend 入口回流。

#### 验收

- 本轮触达文件无 whitespace error。
- legacy 文档坏链记录为非本轮阻塞。

# 6. 依赖与前端边界优化

一级标题需要说明依赖和前端是运行体验问题，不应污染核心训练契约。

## 6.1 Requirements 拆分

二级标题需要聚合 runtime、test、dev、GUI 的边界。

### 6.1.1 Core/Test/Dev/GUI 分层

Priority: `P1`  
Status: `Done`  
Evidence: core/test/dev/gui split  
Gate: requirements file scan

三级标题需要写清每个 requirements 文件的职责。

#### 现状

- `requirements-core.txt` 用于训练和配置工具。
- `requirements-test.txt` 用于 pytest。
- `requirements-gui.txt` 用于 Streamlit。
- `requirements.txt` 可作为开发工作站 umbrella install。

#### 问题

- 单文件混装会让 CI、服务器训练、GUI 调试互相拖累。

#### 改法

- README 写清不同安装方式。
- CI 使用最小依赖。

#### 验收

- CI workflow 使用 core/test。
- GUI 文档使用 gui requirements。

### 6.1.2 Optional logger 依赖

Priority: `P1`  
Status: `Done`  
Evidence: optional logger behavior  
Gate: smoke without wandb/swanlab

三级标题需要说明 wandb/swanlab 未安装时的行为。

#### 现状

- 训练可以在缺失 wandb/swanlab 时继续。

#### 问题

- 如果 logger import 失败中断 smoke，会破坏离线验收。

#### 改法

- 缺失 optional logger 时 warning 并跳过。
- 真正的训练错误不能被 logger fallback 掩盖。

#### 验收

- 无 wandb/swanlab 环境下 dummy smoke 仍通过。

## 6.2 Streamlit / Frontend 定位

二级标题需要聚合 frontend 的实验状态、旧 app 删除和文档入口。

### 6.2.1 Experimental，不作为 validation gate

Priority: `P2`  
Status: `Done`  
Evidence: Streamlit marked experimental  
Gate: import-skip smoke

三级标题需要明确 Streamlit 的维护级别。

#### 现状

- Frontend 用于查看 configs、runs、artifacts。

#### 问题

- 如果把前端作为核心 gate，会引入 GUI 依赖不稳定性。

#### 改法

- README、app usage 和 frontend README 标注 Streamlit experimental。
- 测试使用 import-skip。

#### 验收

- core CI 不依赖 streamlit。

### 6.2.2 `app/` 到 `frontend/` 的迁移状态

Priority: `P2`  
Status: `Done`  
Evidence: `app/` to `frontend/` migration  
Gate: docs path scan

三级标题需要说明旧路径和新路径。

#### 现状

- 旧 `app/` 文件已被删除或迁移。
- 新入口是 `frontend/streamlit_app.py`。

#### 问题

- 文档如果仍指向 `streamlit_app.py` 或 `app/gui.py` 会误导用户。

#### 改法

- 文档统一指向 `streamlit run frontend/streamlit_app.py`。
- reports 中记录迁移状态和残留。
- 单测扫描维护文档，禁止回到 `streamlit run streamlit_app.py` 或 `app/gui.py`。

#### 验收

- `rg "streamlit_app.py|app/gui.py"` 结果只保留正确路径或历史文档。

# 7. 文档、Registry 与发布前收口

一级标题需要说明文档不是装饰，而是维护路径和验收证据。

## 7.1 README、AGENTS、Repo Index 对齐

二级标题需要聚合入口说明、阅读顺序、demo 矩阵。

### 7.1.1 维护主路径说明

Priority: `P1`  
Status: `Done`  
Evidence: README/AGENTS command alignment  
Gate: docs path scan

三级标题需要写清 README 与 AGENTS 的最低一致性。

#### 现状

- AGENTS 已要求 config-first 和无隐式 fallback。

#### 问题

- README 的历史 TODO 和旧命令会与新入口冲突。

#### 改法

- README 顶部聚焦维护主路径。
- 历史长文移入 archive 或明确标记 legacy。
- README、AGENTS、configs README 与 frontend docs 的入口命令已由 path scan 固定。

#### 验收

- README、AGENTS、configs README 的命令一致。

### 7.1.2 Repo index 与 Config atlas

Priority: `P1`  
Status: `Done`  
Evidence: repo index and generated atlas discipline  
Gate: atlas gate passed after generation

三级标题需要说明读仓库和读配置的入口。

#### 现状

- `docs/REPO_INDEX.md` 指导子系统阅读。
- `docs/CONFIG_ATLAS.md` 由 registry 生成。

#### 问题

- 如果手改 atlas，registry 和文档会漂移。

#### 改法

- 报告声明 atlas 只由脚本生成。
- 新 demo 先入 registry，再生成 atlas。
- docs README 和 reports README 指向 registry/atlas/report 各自职责。

#### 验收

- `gen_config_atlas` 后无 diff。

## 7.2 发布前检查单

二级标题需要把 release 前必跑项和已知残留风险分开。

### 7.2.1 必跑命令

Priority: `P1`  
Status: `Done`  
Evidence: release command checklist  
Gate: Command Matrix

三级标题需要列出发布前命令。

#### 现状

- 当前已有可用的 config validation、config inspect、pytest、demo matrix 脚本。

#### 问题

- 如果发布检查靠口头记忆，容易漏掉 artifact contract。

#### 改法

- 固定命令清单：
  - `python -m pytest -q test/`
  - `python -m scripts.validate_configs`
  - `python -m scripts.gen_config_atlas && git diff --exit-code docs/CONFIG_ATLAS.md`
  - `python -m scripts.config_inspect --config configs/demo/00_smoke/dummy_dg.yaml --override trainer.num_epochs=1`
  - demo matrix 或最小 smoke。
- 报告末尾 Command Matrix 固定 expected result 与 failure meaning。

#### 验收

- 每条命令记录退出码和关键输出。

### 7.2.2 已知残留风险

Priority: `P1`  
Status: `Done`  
Evidence: residual risk register  
Gate: known blockers documented

三级标题需要列出不属于本轮但会影响检查的残留项。

#### 现状

- legacy collection 中可能存在文档坏链。
- 某些非本轮文件可能有 trailing whitespace。
- paper submodule 可能处于 modified/untracked 状态。

#### 问题

- 如果不隔离残留风险，会误判本轮优化质量。

#### 改法

- 报告中单独列 residual risks。
- 每个风险写清路径、失败命令、是否阻塞发布。
- 本轮不清理 paper submodules、legacy TFN 文档和非本轮 model whitespace。

#### 验收

- 本轮 touched files 的静态检查通过。
- 残留风险有明确 owner 或后续 issue。

# 8. 报告落地要求

一级标题需要说明这份报告最终应该怎么写、怎么被审阅、怎么进入发布流程。

## 8.1 报告格式

二级标题需要固定报告形状，避免写成散文。

### 8.1.1 每个三级标题必须有四个四级标题

Priority: `P1`  
Status: `Done`  
Evidence: fixed four-level item template  
Gate: heading count check

三级标题需要明确模板约束。

#### 现状

- 当前优化内容跨代码、配置、CI、文档多个面。

#### 问题

- 如果每项写法不同，审阅者无法快速比较风险和验收。

#### 改法

- 每个三级标题统一使用 `现状 / 问题 / 改法 / 验收`。
- 只有高风险项额外增加 `风险`，不要替代四个固定单元。

#### 验收

- 报告中每个三级优化项都能独立转成 issue 或 commit。

## 8.2 优先级标注

二级标题需要说明如何决定先做什么。

### 8.2.1 P0/P1/P2 分级

Priority: `P1`  
Status: `Done`  
Evidence: priority labels for every item  
Gate: metadata coverage check

三级标题需要给出分级标准。

#### 现状

- 当前仓库已经有多条并行改动，必须避免大爆炸式重构。

#### 问题

- 没有优先级会导致 artifact、CI、frontend、paper 同时推进，风险过高。

#### 改法

- P0：correctness、fail-fast、artifact contract、CI gate。
- P1：依赖拆分、文档同步、demo matrix。
- P2：frontend polish、legacy cleanup、paper integration。

#### 验收

- 每个优化项在报告中标注 P0/P1/P2。
- P0 项必须有单测或 smoke 验收。

## Command Matrix

| Gate | Command | Expected Result | Failure Meaning |
|---|---|---|---|
| Targeted unit tests | `python -m pytest -q test/test_main_strictness.py test/test_pipeline_02_modes.py test/test_infonce_pairing.py test/test_hse_contrastive_failfast.py test/test_run_artifacts_contract.py test/test_run_contract_helper.py test/test_demo_matrix_script.py test/test_frontend_docs_paths.py test/test_collect_uxfd_runs.py` | all pass | entry, P02, loss, artifact, demo, or docs path regression |
| Config schema | `python -m scripts.validate_configs` | all registry configs pass | config registry/schema drift |
| Atlas sync | `python -m scripts.gen_config_atlas && git diff --exit-code docs/CONFIG_ATLAS.md` | no diff | registry and generated docs are out of sync |
| Config inspect | `python -m scripts.config_inspect --config configs/demo/00_smoke/dummy_dg.yaml --override trainer.num_epochs=1` | sanity table PASS | smoke config cannot be resolved or imported |
| P01 smoke | `python main.py --config configs/demo/00_smoke/dummy_dg.yaml --override trainer.num_epochs=1 --override data.num_workers=0 --override trainer.device=cpu` | writes manifest and metrics | training path or run contract regression |
| Docs check | `python -m scripts.validate_docs` | pass or known residual only | doc links drifted outside accepted residuals |
| Whitespace check | `git diff --check <touched-files>` | pass | report or patch introduced whitespace errors |

## Residual Risk Register

| Risk | Current Evidence | Impact | Owner Decision |
|---|---|---|---|
| Legacy TFN README broken image links | `validate_docs` reports missing `Doc/Figures/*.png` under legacy collection | blocks full docs gate | fix legacy assets or exclude legacy collection explicitly |
| Existing MWA_CNN trailing whitespace | full `git diff --check` reports two lines in `MWA_CNN.py` | blocks full diff check | clean in a separate model cleanup commit |
| P02 current-branch full demo rerun pending | dummy smoke passed; P02 targeted tests passed | medium release confidence gap | run P02 HSE single and CDDG before release |
| Frontend migration paths may still appear in historical docs | `app/` deleted, `frontend/` active | user confusion if old docs are surfaced | update maintained docs, leave archives marked historical |
| Paper submodules are dirty/untracked | git status shows modified/untracked paper entries | release packaging ambiguity | decide whether paper state is in or out of main PR |

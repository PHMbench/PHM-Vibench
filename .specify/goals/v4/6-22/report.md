# PHM-Vibench 仓库现状报告（2026-06-22）

Date: 2026-06-22
生成方式: 10-agent 并行领域分析（a1–a10）+ 主控综合
范围: 仓库全局快照 + 下一步建议
分支: `Feature_factory-update`（vs `main`）

> 本报告由 10 个领域分析 teammate 并行产出、主控综合而成。每个结论尽量挂可核验证据（`file:line` / git log / 命令输出）。报告遵循 v4 既有约定（`Date:` 头、GOAL ID 保留英文），语言为中文。

---

## 1. 执行摘要（TL;DR）

**一句话现状**：仓库的「生成式 PHM 基准」骨架与流程治理已成熟落地，但**论文提交被一个单一门控设计卡死**——所有 36 条实验链都跑完了，却因为指标门控「零容忍」被整体降级为 exploratory，`benchmark_valid_row_count = 0`，状态 `NOT_SUBMISSION_READY`。

**整体健康度信号**

| 维度 | 信号 | 说明 |
|------|------|------|
| 流程治理 | 🟢 成熟 | v1→v4 四代演进，八轴评审 + 两 agent handoff 已固化 |
| 配置系统 | 🟢 成熟 | SSOT 四件套通顺，`validate_configs` 22/22 通过 |
| 数据工厂 | 🟢 成熟（带刺） | 6 数据集 reader 完整，但有硬编码路径、无显式泄漏 guard |
| 生成式模型/任务 | 🟡 成长中 | CFM/RF/DDPM 已实装；4 个 SOTA 方法仅 README 规划无代码 |
| 基准证据链 | 🔴 阻塞 | 门控零容忍致 `benchmark_valid=0`（详见 §3.6） |
| 论文生产 | 🟡 成长中 | 骨架 ~30–40%，589 指标缺口，readiness gate 正确拦提交 |
| 测试 / CI | 🟡 成长中 | generative 测试 99 个，但 **CI 未激活**、工厂层无单测 |

**三件最关键的事**

1. **主阻塞可被「一行门控放宽 + 白名单」解锁**：`benchmark_valid=0` 的根因是 `Pipeline_06_generative.py:475-476` 的 `not_computable == 0` 硬门槛，而非指标缺失。指标实际已算 82.4%（136 中 24 个天然不可算）。这是高杠杆点。
2. **SOTA roster 与 config 错位**：`demo/10_generative/` 有 8 个生成式 smoke config（含 MeanFlow/Drifting/TFM/OT-NFM），但这 4 个方法在代码里只有共享 stub（`_experimental_one_step.py`），**无 method-specific 损失**——config 跑的是 RectifiedFlowLoss，却声称方法保真度。这是 v4 Wave 1 的核心待办。
3. **流程刚升级但尚未受压**：GOAL-V4-010/011（reviewer gate + two-agent handoff）刚于 2026-06-18 落地（90/100 `PASS_WITH_WARNINGS`），但 review 自己警告「future agent could skip the independent Agent B review」——流程依赖自觉，且 `.specify/goals/v4/` 与 `reviews/v4/` 当前仍 untracked。

---

## 2. 仓库全景（规模 & 活跃度）

**规模快照**

| 区域 | 规模 | 出处 |
|------|------|------|
| `src/` | ~330 Python 文件，7 个 Pipeline 编排器 | `find src -name '*.py' \| wc -l` |
| `configs/` | 110 YAML（base 30 / demo 15 / experiments 11 / paper 9 / v0.0.9 遗留 43 / 其他 2；`reference/` 空） | a5 |
| `config_registry.csv` | 49 行（~48 条目），与 `CONFIG_ATLAS.md`（662 行）同步 | a5 |
| `scripts/` | 15 文件；`validate_docs.py` 3252 行、`generative_benchmark_effect.py` 1867 行 | a8/a6 |
| `test/` | 21 文件，216 测试函数（generative 99 + 根级/smoke 117） | a8 |
| `specs/002-phm-genbench-frontier/` | 42+ md（reviews/handoffs/paper/contracts/checklists） | 主控探查 |
| 生成式模型 | `generative_model/` 339 行（4 backbone + 2 条件组件） | a1 |
| 生成式任务组件 | `Components/generative/` ~1455 行；losses 核心 251 行（CFM 75 / RF 78 / DDPM 68 / ScoreSDE 36） | a1/a2 |
| 数据 reader | `data_factory/reader/` 22 个 `RM_*.py`，~1580 行 | a4 |
| TODO/FIXME/XXX | 全仓 120 处（15+ 集中在 `task_factory/task/pretrain/README.md`） | a10 |

**活跃度脉络**（近 40 commit 主题）
- **v4 流程治理**（最新）：reviewer gate + two-agent handoff（GOAL-V4-010/011）
- **论文证据基建**（近两周）：v0.3 long-run 工具、status 脚本、submission readiness gate、evidence pack
- **生成式方法 & 基准**（持续）：6 数据集矩阵、`in_channels` 钉死（`11a9ad5`）、stage ledger 消费 + 基准 promote 门控（`785b13e`）
- **测试 & 校验**（持续）：generative 测试大幅扩充

**未提交变更**（`git status`）：`.specify/goals/v4/`、`specs/002-phm-genbench-frontier/handoffs/2026-06-18-...md`、`specs/002-phm-genbench-frontier/reviews/v4/` 三处 untracked——v4 流程成果尚未入库。

---

## 3. 十领域现状详盘

> 每节统一：现状 / 成熟度 / 风险 / 关键文件。证据见各 agent 原始回传。

### 3.1 生成式模型架构 — 🟡 成长中

**现状**：4 个 backbone（`phm_unet1d` 68 行 / `phm_dit1d` 70 行 / `mamba1d_backbone` 68 行 / `phm_cfm_mlp1d`）+ 2 个条件组件（`condition_encoder` 61 行 / `film` 18 行），共 339 行。所有 backbone 标注 `stateless = True`，遵循 `[N, C, L]` 契约。CFM 达 smoke-runtime，DDPM/RectifiedFlow 达 exploratory-runtime，ScoreSDE/Mamba 为 research-only。

**成熟度依据**：核心 backbone 已可跑，但实验性方法与条件覆盖不全。

**风险**：
- **实验性方法无代码**：MeanFlow/Drifting/TFM/OT-NFM 在 `generative_model/README.md:119-123` 声称支持，但代码中仅出现于注释（a1）。
- **Mamba 状态无关性未验证**：`mamba1d_backbone.py:21-32` 的 `use_true_mamba=true` 分支未通过 sampler 状态无关性测试。
- **条件覆盖与 paper 声称不符**：`condition_encoder.py:44-46` 仅接受 `fault_label`/`domain_id`，`load`/`rpm` 被显式排除——与 paper「工况条件」叙事不一致。

**关键文件**：`src/model_factory/generative_model/{phm_unet1d,phm_dit1d,mamba1d_backbone,condition_encoder,film}.py`

### 3.2 生成式任务·损失·采样器 — 🟡 成长中

**现状**：4 损失已实装（CFM/RF/DDPM 完整；ScoreSDE 仅 36 行 research skeleton）；3 采样器（Euler ODE / DDPM reverse / Annealed Langevin smoke）；scheduler（beta/alpha/alpha_bar）；metrics ~1187 行，6 类全 eval-only。MeanFlow/Drifting/TFM/OT-NFM 4 个 task 共享 `_experimental_one_step.py`（55 行），复用 `RectifiedFlowLoss`，标注 `promotion_required_for_benchmark_valid: True`。

**成熟度依据**：CFM/RF/DDPM 有 loss+sampler+task 三件套；4 个 SOTA 方法无 method-specific 损失。

**风险**：
- **goal_sota 明令禁止复用**：`goal_sota.md:132`「Do not reuse RectifiedFlowLoss while claiming MeanFlow fidelity」——当前正是这种复用状态。
- **ScoreSDE sampler 仅 smoke**：`samplers/score_sde.py` 只有 annealed Langevin，缺 predictor-corrector。
- 泄漏防护已落地：`utility_protocol.py:6` `FORBIDDEN_SYNTHETIC_SOURCE_SPLIPS = {"val","valid","validation","test","target_test"}`；metrics 确认 eval-only（`generative_eval.py:167`）。

**关键文件**：`src/task_factory/Components/generative/{losses,samplers,schedulers,metrics,manifests}/*`、`task/generative/_experimental_one_step.py`

### 3.3 生成式流水线 Pipeline_06 — 🟡 成长中

**现状**：824 行，`train → sample → eval` 三阶段（paperpack 由外部脚本），通过 `task.generative.mode` 选阶段，stage ledger 串联。证据框架完整：normalization artifact（`:287`）、synthetic manifest（`:657`）、eval evidence manifest（`:776`）、stage ledger 写入点（`:579/685/787`）。

**成熟度依据**：阶段齐全、证据字段齐备，但 benchmark-valid 闭环未形成（见 §3.6）。

**风险**：
- **validity_status 默认 exploratory 且无自动升级**：`:673` 默认 `"exploratory"`，无 promote 逻辑。
- **condition sampling 验证弱**：`_condition_sampling_split_verified`（`:214-218`）仅 `train_distribution` policy 检查 metadata split，其他 policy 直接返回 True。
- **preflight 不校验生成式特定项**：`main.py:91-104` 的 `preflight()` 只验 5 个 section + schema，不查 `checkpoint_path`/`condition_sampling_policy`。
- **stage ledger 路径推断脆弱**：`_stage_ledger_path`（`:396-405`）依赖目录名匹配 `STAGE_NAMES`，有推断失败风险。

**关键文件**：`src/Pipeline_06_generative.py`、`main.py`、`scripts/generative_benchmark_effect.py`

### 3.4 数据工厂 — 🟢 成熟（带刺）

**现状**：reader/dataset_task/samplers 三层，覆盖 22+ 数据集（含 6 基准）。6 数据集 reader 成熟（CWRU/XJTU/FEMTO/UNSW/JUST/PU）。H5 缓存 → 并行读取（`max_workers=32`）→ `IdIncludedDataset` → Sampler。splits 集中于 `ID/Get_id.py`，`train_val_ids`/`test_ids` 分离清晰，`target_domain_num` 动态划分。

**成熟度依据**：核心链路跑通，task_type 覆盖 7 种。

**风险**：
- **硬编码绝对路径 8 处**（违反 CLAUDE.md）：`RM_027_PU.py:45-48`、`RM_024_JUST.py:38-41`、`RM_020_DIRG.py:35-38` 等测试块含 `/home/user/...`。
- **泄漏防护无显式 guard**：未发现 val/test/target_test 混入训练源的实际行为，但也**没有 assert/raise 形式的显式断言**——仅靠 `utility_protocol.py` 的 forbidden split 集合（防御纵深不足）。
- **stub reader**：`RM_025_KAIST.py`、`RM_026_HUST23.py` 为 0 字节占位符。

**关键文件**：`src/data_factory/{data_factory.py,ID/Get_id.py,samplers/Get_sampler.py,reader/RM_*.py}`

### 3.5 配置系统 — 🟢 成熟

**现状**：SSOT 链路（registry → atlas → inspect → validate）四件套通顺：`validate_configs` 22/22 通过、`gen_config_atlas` 可再生、`config_inspect` 可溯源。6 数据集矩阵 `configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml` 按 `11a9ad5` 钉死 per-dataset `in_channels`（CWRU=2/XJTU=2/FEMTO=2/UNSW=6/JUST=7/PU=3），支持 CFM/RF/DDPM。`demo/10_generative/` 有 8 个 smoke config。local override 走 `configs/local/local.yaml`。

**成熟度依据**：配置即实验契约（Constitution I）落地扎实，`in_channels` 钉死防漂移。

**风险**：
- **v0.0.9 遗留 43 YAML 无迁移/删除计划**，可能误导模板选择。
- **`reference/` 目录为空**（0 文件），CLAUDE.md 标其 legacy 但用途需澄清。
- **config-impl 错位**：`demo/10_generative/` 有 MeanFlow/Drifting/TFM/OT-NFM 的 smoke config，但对应方法无 method-specific 实装（与 §3.2 互证）。

**关键文件**：`configs/config_registry.csv`、`docs/CONFIG_ATLAS.md`、`scripts/{config_inspect,gen_config_atlas,validate_configs}.py`、`configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml`

### 3.6 基准治理与证据链 — 🔴 主阻塞点

**现状（根因已定位）**：36 条实验链全部达 `COMPLETE_CHAIN`（6 数据集 × 3 方法 × 2 seed），paperpack 目录 36 个完整，stage ledger 四阶段全成功，config/protocol hash、normalization artifact、condition counts、leakage checks 等证据字段均有消费逻辑。但 `benchmark_valid_row_count = 0`（2,490 行全 exploratory）。

**根因**：**门控判 fail，不是指标未实现/未计算**。
- 指标实际覆盖率 82.4%：136 个指标中 24 个 `not_computable`（如单通道数据无法算 cross-channel coherence、envelope spectrum、`spectral_fault_characteristic_peak_error` 等）。
- 但 `Pipeline_06_generative.py:475-476` 要求 `not_computable == 0` 才 `eligible=true`——**零容忍、无白名单、无豁免**。
- `generative_benchmark_effect.py:812-818` 将 `eval:metric_status_ok` 缺失作为降级理由写入 `benchmark_status_reason` → 全部 manifest `promotion.eligible=false`、`missing=["metric_status_ok"]`。

**与论文侧口径的差异（重要）**：`evidence_gaps.md` 从论文视角称「无可计算的质量/效用指标」，但实际 manifest 显示指标已大量计算（82.4%）。**真实阻塞是门控设计，而非计算缺失**——这直接影响下一步策略（改门控 ≪ 重跑实验）。

**最小解锁路径**（任选其一，供决策）：
1. `metric_status_ok` 从「全部指标」改为「核心指标白名单」；
2. 允许 `not_computable` 配额（如 ≤10%）；
3. 增加豁免条款（单通道数据豁免 cross-channel 类指标）。

**关键文件**：`src/Pipeline_06_generative.py:450-495`（`:475-476` 门控核心）、`scripts/generative_benchmark_effect.py:792-840`（`_promotion_status`）、`specs/002-phm-genbench-frontier/contracts/generative-benchmark-contract.md:89-100`（9 项证据要求）、`.specify/memory/constitution.md:60-72`（Principle III）

### 3.7 论文生产流水线 — 🟡 成长中

**现状**：`PAPER_DRAFT.md` 章节结构完整但内容 ~30–40%，状态 `NOT_SUBMISSION_READY`。`evidence_gaps.md` 列 4 条核心缺口 + 按数据集缺失计数（6 数据集各 74–114，**总计 589**）。`submission_readiness.md:3` `NOT_SUBMISSION_READY`。`paperpack_generative.py`（658 行）打包、`generative_submission_draft.py`（417 行）实现 readiness gate（已正确拦提交）。v4 review 90/100 `PASS_WITH_WARNINGS`。

**成熟度依据**：基础设施（目录/脚本/review/gate）齐备并运行，但实验血肉缺失。

**风险**：
- 589 指标缺口（论文视角）——与 §3.6 的根因联动：门控解锁后大部分可转为 benchmark-valid。
- readiness gate 双刃：正确拦提交，但也意味着当前无法产出 submission_ready。
- v4 review 指出当前评审为本地 Codex 自审，**非独立 Agent B 交叉评审**（`reviews/v4/2026-06-18-...:76`）。

**关键文件**：`specs/002-phm-genbench-frontier/paper/{PAPER_DRAFT,evidence_gaps,submission_readiness}.md`、`scripts/{paperpack_generative,generative_submission_draft}.py`

### 3.8 测试与验证 — 🟡 成长中

**现状**：三层架构（配置验证 + 单测 + 文档校验）。21 文件 / 216 函数。generative 测试 99 个（覆盖 DDPM/DiffusionTS/RF/TimeFlow/ScoreSDE/Euler sampler/normalization/condition sampling/benchmark effect）。`validate_configs.py`（99 行，loader + pydantic 双路）。preflight 在 `main.py:91`。

**成熟度依据**：验证机制实装，但结构性缺口明显。

**风险**：
- **CI 未激活**：`.github/workflows/` 为空，`.github/workflows_TODO/config_tools_ci.yml` 存在但未启用——**无 PR 保护**。
- **`validate_docs.py` 单体膨胀**：3252 行（`validate_configs.py` 的 33 倍），24 个 check 函数硬编码 PHM-GenBench 特定校验。
- **工厂层单测缺失**：data/model/task/trainer factory 核心逻辑无单测覆盖。
- **pytest 无分层**：无 `conftest.py`、无 marker、无 fixture 共享，无法选择性跑 slow/GPU。

**关键文件**：`test/generative/{test_benchmark_effect.py(1304行),test_six_dataset_submission.py(796行),test_generative_backbones.py}`、`scripts/{validate_configs,validate_docs}.py`、`.github/workflows_TODO/config_tools_ci.yml`

### 3.9 流程治理 / handoff / review — 🟢 成熟

**现状**：v1→v4 四代演进——v1（18 goals，543 行）基础框架 → v2（13 goals，1711 行）生成式 → v3 paper pack → v4（12 核心 GOAL-V4-*）paper-first 治理。累计 44 个 goal 文件。2026-06-18 完成 GOAL-V4-010（REVIEWER-GATE）+ GOAL-V4-011（TWO-AGENT-HANDOFF-PROTOCOL），评审 90/100 `PASS_WITH_WARNINGS`。

**GOAL-V4-010/011 完成内容**：安装 V4 reviewer gate + two-agent handoff，绑定 `reviewer.md` 八轴标准（method impl/fidelity/smoke/paper integration/claim safety/scope discipline/provenance/next action）、要求 canonical evidence-root 字段、记录 per-goal review 路径。

**S1–S6 lane**（`handoff.md:12-17`）：S1 Baseline Contract（CFM/RF/DDPM+共享合约）/ S2 Score SDE / S3 MeanFlow / S4 Drifting & TFM / S5 OT-NFM / S6 Paper Integration。

**执行顺序**（`goal.md:104-123`）：Wave 0（ROSTER-LOCK + PAPER-SCOPE）→ Wave 1（7 个 SOTA 并行 101–106）→ Wave 2（SMOKE-MATRIX + EVIDENCE-PACK）→ Wave 3（FINAL-DRAFT + REVIEWER-PASS）。

**风险**：
- **流程绕过风险**：`handoff.md:107-108` 自警「future agent could skip the independent Agent B review unless enforced per goal」——流程依赖自觉。
- **untracked**：`.specify/goals/v4/` 与 `reviews/v4/` 仍未 `git add`（流程卫生）。
- **评审非独立**：当前为本地 Codex 自审。

**关键文件**：`.specify/goals/v4/{goal,goal_sota,paper_ready,handoff,reviewer}.md`、`specs/002-phm-genbench-frontier/{reviews/v4,handoffs}/`

### 3.10 文档网络 / 遗留 / 技术债 — 🟡 成长中

**现状**：双语（EN/CN）核心文档齐全（`README/CLAUDE/AGENTS/CONTRIBUTING` 均有 `_CN` 版），14 个 `src/*/CLAUDE.md` 模块深读，`CONFIG_ATLAS.md`（662 行）↔ registry 同步。`Pipeline_ID.py` 8 行 stub（转发 default）。巨型文件：`validate_docs.py` 3252、`generative_benchmark_effect.py` 1867、`Pipeline_03` 907、`Pipeline_06` 824。

**成熟度依据**：文档网络完备，遗留清理与巨型文件待治。

**风险**：
- **遗留配置债**：`configs/v0.0.9/`（43 YAML）+ `reference/`（空）无迁移/删除计划。
- **TODO 集中**：120 处，15+ 集中在 `task_factory/task/pretrain/README.md`（block/temporal/frequency masking、对比学习、多尺度预训练等未实现）。
- **巨型文件可维护性**：`validate_docs.py` 3252 行需插件化拆分；`Pipeline_03/06` 缺子模块抽象。

**关键文件**：`docs/CONFIG_ATLAS.md`、`configs/{v0.0.9,reference}/`、`scripts/validate_docs.py`、`src/task_factory/task/pretrain/README.md`

---

## 4. 横切风险登记表

| # | 风险项 | 影响领域 | 严重度 | 证据/出处 | 建议 |
|---|--------|---------|--------|-----------|------|
| R1 | 指标门控零容忍致 `benchmark_valid=0` | §3.6 / §3.7 | 🔴 高 | `Pipeline_06:475-476`；`generative_benchmark_effect.py:812-818` | 放宽门控（白名单/配额/豁免），最小杠杆解锁 |
| R2 | 4 个 SOTA 方法无 method-specific 损失（复用 RF Loss） | §3.1 / §3.2 | 🔴 高 | `_experimental_one_step.py:54`；`goal_sota.md:132` | Wave 1 各 lane 实装专属损失 |
| R3 | config-impl 错位（smoke config 跑无实装方法） | §3.5 / §3.2 | 🟠 中高 | `demo/10_generative/` 8 config vs stub | 与 R2 同步处理；过渡期 config 标注 exploratory |
| R4 | CI 未激活，无 PR 保护 | §3.8 | 🟠 中 | `.github/workflows/` 空，`workflows_TODO/` 存在 | 启用 `config_tools_ci.yml` |
| R5 | 流程可被绕过（Agent B review 靠自觉） | §3.9 | 🟠 中 | `handoff.md:107-108` | 加 per-goal 强制校验钩子 |
| R6 | 评审非独立（本地 Codex 自审） | §3.7 / §3.9 | 🟠 中 | `reviews/v4/2026-06-18-...:76` | 关键 goal 用独立 Agent B |
| R7 | condition_encoder 排除 load/rpm，与 paper 声称不符 | §3.1 | 🟠 中 | `condition_encoder.py:44-46` | 扩展条件或修正 paper 叙事 |
| R8 | 数据工厂硬编码绝对路径 8 处 | §3.4 | 🟡 低中 | `RM_027_PU.py:45-48` 等 | 迁入 `configs/local/` |
| R9 | 泄漏防护无显式 guard 断言 | §3.4 | 🟡 低中 | 仅 `utility_protocol.py:6` forbidden set | 加 assert/raise 纵深 |
| R10 | 工厂层无单元测试 | §3.8 | 🟠 中 | `test/` 无 factory 测试，无 conftest | 补 data/model/task/trainer 单测 |
| R11 | `validate_docs.py` 3252 行单体 | §3.8 / §3.10 | 🟡 低中 | `wc -l` | 插件化拆分 |
| R12 | v0.0.9 遗留 43 YAML 无迁移计划 | §3.5 / §3.10 | 🟡 低 | `find configs/v0.0.9` | 定迁移/删除期限 |
| R13 | v4 成果 untracked（goals/v4 + reviews/v4） | §3.9 | 🟡 低 | `git status --short` | 下个 commit 显式 add |

---

## 5. 下一步建议（对齐 v4 路线）

> 顺序按「解锁价值 / 降低风险」优先级排，不替用户决策目标归属，仅给建议。

### 立即（本周）
1. **GOAL-V4-000-CLAIM-FREEZE（已规划，待启动）**：按 `handoff.md:149-150`，锁定论文为 implementation-complete / exploratory-evidence 位置，移除所有 unsupported claims，在 `benchmark_valid_row_count=0` 期间保持探索性措辞。**这是 claim 安全的前提，应先于任何性能声明。**
2. **解锁 R1（最高杠杆）**：在 GOAL-V4-000 之外开一个独立小 goal，放宽 `metric_status_ok` 门控（推荐「核心指标白名单 + 单通道豁免」），把 82.4% 已算指标转为 benchmark-valid。预期可把 `benchmark_valid_row_count` 从 0 推到大部分实验——这是把论文从 `NOT_SUBMISSION_READY` 推向 ready 的最短路径。
3. **把 v4 成果入库（R13）**：`git add .specify/goals/v4/ specs/002-phm-genbench-frontier/{handoffs,reviews/v4}/`，避免流程资产遗失。

### 近期（v4 Wave 1）
4. **SOTA roster 补齐（R2/R3）**：按 S1–S6 lane 并行实装——
   - S2 Score SDE：补 predictor-corrector sampler（当前仅 annealed Langevin smoke）；
   - S3 MeanFlow / S4 Drifting & TFM / S5 OT-NFM：各写 method-specific 损失，**停止单纯复用 `RectifiedFlowLoss`**（`goal_sota.md:132` 禁令）；过渡期对应 smoke config 标 `validity_status: exploratory`。
5. **CI 激活（R4）**：`.github/workflows_TODO/config_tools_ci.yml` → `.github/workflows/`，跑 `validate_configs → registry/atlas → config_inspect → pytest`。

### 中期（Wave 2–3）
6. **证据补强**：门控解锁后，补齐剩余可算指标，跑满 6 数据集 × 完整方法矩阵 × 多 seed，产出真正 benchmark-valid 的 quality + utility 双指标。
7. **测试补强（R10）**：补工厂层单测 + `conftest.py` + marker 分层（slow/GPU）。
8. **流程硬化（R5/R6）**：为 reviewer gate 加 per-goal 强制钩子；关键 goal 用独立 Agent B 交叉评审。

### 技术债（穿插）
9. `validate_docs.py` 插件化拆分（R11）；v0.0.9 迁移/删除（R12）；硬编码路径迁入 local（R8）；泄漏 guard 显式化（R9）；condition_encoder 扩展 load/rpm 或修正 paper 叙事（R7）。

---

## 6. 附录：分析方法与数据来源

**方法**：主控先并行探查（`.specify/goals/v4` 格式 / 仓库地图 / 近期 momentum），确认交付契约后，在单条消息内并行拉取 10 个 `general-purpose` teammate（`a1`–`a10`），每个负责一个互不重叠领域，统一返回中文五字段结构化发现（现状 / 成熟度 / 风险 / 关键文件 / 可引用证据）。主控去重归并、抽取横切风险、对齐 v4 路线后落盘。

**10 个 agent 分工**：

| agent | 领域 | 主要证据来源 |
|-------|------|-------------|
| a1 | 生成式模型架构 | `generative_model/*.py` 行数、`stateless` 标注、git log |
| a2 | 任务·损失·采样器 | `Components/generative/*` 行数、`_experimental_one_step.py:54`、`utility_protocol.py:6` |
| a3 | Pipeline_06 | `Pipeline_06_generative.py:26/287/579/673/736`、`main.py:91` |
| a4 | 数据工厂 | `reader/RM_*.py` 1580 行、`grep /home/user`、`Get_id.py:76` |
| a5 | 配置系统 | `find configs/*.yaml`=110、`validate_configs` 22/22、`six_dataset_benchmark_matrix.yaml` |
| a6 | 基准治理（主阻塞） | `Pipeline_06:475-476`、`generative_benchmark_effect.py:792-840/812-818`、`benchmark_effect_manifest.json:103` |
| a7 | 论文生产 | `paper/{PAPER_DRAFT,evidence_gaps,submission_readiness}.md`、`reviews/v4/2026-06-18` |
| a8 | 测试与验证 | `find test/`=21、`grep def test_`=216、`.github/workflows` 空 |
| a9 | 流程治理 | `.specify/goals/{v1,v2,v3,v4}`、`handoff.md:107-149`、`goal.md:104-123` |
| a10 | 文档/遗留/技术债 | `ls *_CN.md`、`grep TODO`=120、`wc -l validate_docs.py`=3252 |

**关键交叉印证**
- R2/R3：a1（README 声称 4 方法无代码）+ a2（4 task 共享 stub 复用 RF Loss）+ a5（demo 有 8 smoke config）三向互证 config-impl 错位。
- benchmark_valid=0：a3（validity_status 默认 exploratory 无升级）+ a6（门控零容忍根因）+ a7（论文 589 缺口）三层定位，a6 给出最精确根因。
- 流程绕过：a7（评审非独立）+ a9（handoff 自警）互证。

**已知口径差异**：论文侧 `evidence_gaps.md` 称「无可计算指标」（589 缺），但实际 manifest（a6）显示已算 82.4%——本报告以 a6 的 manifest 实测为准，论文口径偏保守。

---

*报告结束。如需把某条建议落为具体 GOAL-V4-* 任务，可在 `.specify/goals/v4/` 新建对应 goal 文件并走 handoff.md 的 Builder Start 流程。*

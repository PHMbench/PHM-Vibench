# PHM-Vibench Bug 与冗余优化清查报告

日期：2026-04-16

## 报告目标

本报告面向后续维护者、review agent 和 autoresearch 父仓集成方，用于系统清查 PHM-Vibench 中仍需优化的 bug、冗余实现、legacy 路径、重复文档和测试缺口。

本报告不直接替代 issue 或 PR，而是提供可拆分为 issue/PR 的四级标题结构。每个问题项都必须落到证据、影响、动作和验收。

## 优先级定义

- P0：可能导致 silent wrong、错误结果合法化、训练目标为空、父仓无法消费结果。
- P1：显著增加 first-run、debug 或集成成本，但通常不会直接产出错误科研结果。
- P2：历史兼容、研究草稿、文档重复、长期治理项。

## 四级标题通用填写规则

### 一级标题：审计大模块

#### 一级标题需要包含的内容

一级标题用于划分报告的核心审计维度，例如配置系统、Pipeline、Artifact、测试、legacy 目录、文档索引。

每个一级标题下必须说明：

- 本模块覆盖的系统边界。
- 本模块为什么影响 `T_first-run`、`T_debug`、`P_silent_wrong` 或 `C_integration`。
- 本模块内 P0/P1/P2 问题数量与总体判断。
- 本模块是否已存在部分修复。

### 二级标题：问题域或子系统

#### 二级标题需要包含的内容

二级标题用于定位一个可审计的问题域，例如 Hydra/legacy 双轨、P02 多模式、manifest 契约、test TODO 分流。

每个二级标题下必须说明：

- 涉及的主要路径或模块。
- 当前维护状态：maintained、compatibility、legacy、draft、archive。
- 是否影响主路径 `python main.py --config <yaml>`。
- 是否已经被 preflight、demo matrix、CI 或单测覆盖。

### 三级标题：具体问题项

#### 三级标题需要包含的内容

三级标题必须能直接转成一个 issue、ticket 或 PR。标题应是问题而不是泛泛类别。

每个三级标题下必须说明：

- 问题现象。
- 影响范围。
- 根因判断。
- 推荐处理方式。
- 优先级。
- 预计验证方式。

### 四级标题：执行字段

#### 四级标题需要包含的内容

每个三级问题项下固定使用以下四个四级标题，确保另一个 agent 可以直接执行：

- `#### 现象与证据`：写明文件、命令、测试、日志或文档依据；区分已验证事实和推断风险。
- `#### 影响与风险`：说明影响的是 first-run、debug、silent wrong 还是父仓集成。
- `#### 优化动作`：明确删除、归档、迁移、合并、加 preflight、加测试还是更新文档。
- `#### 验收标准`：给出可执行命令、测试、产物或文档检查结果。

# 1. 总览：当前问题面与优化目标

## 1.1 主路径已经收敛，但历史路径仍很多

### 1.1.1 主路径与历史路径并存导致维护认知成本高

#### 现象与证据

当前维护主路径已经收敛为：

- 入口：`python main.py --config <yaml> [--override key=value ...]`
- 配置：`configs/hydra/` 与 `configs/demo/`
- 预检：`src/configs/preflight.py`
- 运行契约：`src/utils/training/run_contract.py`
- 产物：`artifacts/manifest.json` 与 `test_result_*.csv`
- 验证：`scripts/run_demo_matrix.sh --mode smoke`

同时仓库还保留：

- `configs/reference/`
- `configs/v0.0.9/`
- `src/configs/deprecated/`
- `test/todo_test/`
- `docs/past/`
- `src/model_factory/X_model/legacy_collection/`
- 多个 paper 子模块和子模块局部结果目录

#### 影响与风险

这是 P1 风险。主路径本身可运行，但历史路径数量多，容易导致新人或 agent 从旧配置、旧 README、旧测试进入，绕过 preflight 和 manifest contract。

#### 优化动作

建立维护状态标签并在报告、README、AGENTS、REPO_INDEX 中一致使用：

- maintained：CI 或 demo matrix 覆盖的主路径。
- compatibility：短期兼容但不推荐新实验使用。
- legacy：历史保留，只允许迁移时读取。
- draft：研究草稿，不作为验收依据。
- archive：只做证据链或历史参考。

#### 验收标准

- `README.md` 只推荐 maintained 主路径。
- `AGENTS.md` 明确禁止从 legacy 目录复制模板。
- `docs/REPO_INDEX.md` 能引导 agent 避免递归读大目录。
- `configs/config_registry.csv` 能标识 maintained / legacy / draft 状态。

## 1.2 优化目标函数

### 1.2.1 优化应优先降低 silent wrong，而不是优先扩功能

#### 现象与证据

科研 benchmark 的主要失败模式不是立即报错，而是：

- 错误配置被 fallback 到另一条路径。
- loss 为 0 但训练继续。
- 数据路径不对但仍跑出结果。
- manifest 缺失或 metrics 指向不稳定。
- 父仓读取 trainer 内部细节而不是读取稳定产物。

#### 影响与风险

这是 P0 风险。silent wrong 会产出看似合法但不可复现、不可比较的科研结果。

#### 优化动作

所有后续优化按以下顺序排序：

1. 训练关键错误 fail-fast。
2. 配置和数据路径 preflight。
3. manifest 与 metrics 必填。
4. demo matrix 与 CI 小门禁。
5. legacy/draft 归档。

#### 验收标准

任何 P0 问题必须绑定至少一个测试或 demo matrix gate。没有测试的 P0 问题不得标为完成。

# 2. 配置系统与实验语义冗余

## 2.1 Hydra 与 legacy YAML 双轨

### 2.1.1 同一实验语义存在多份配置来源

#### 现象与证据

维护配置分布在：

- `configs/hydra/experiments/`
- `configs/demo/`
- `configs/base/`
- `configs/reference/`
- `configs/v0.0.9/`

Hydra 迁移已经建立了目标范式，但旧 YAML 仍在兼容窗口中。

#### 影响与风险

这是 P1 风险。同一实验的 Hydra 版、demo 版、reference 版和 v0.0.9 版可能字段不同，导致用户修改了错误的配置源。

#### 优化动作

在报告和文档中定义配置优先级：

1. `configs/hydra/experiments/`：目标组合配置。
2. `configs/demo/`：兼容维护 demo。
3. `configs/experiments/`：本地实验变体。
4. `configs/reference/`：legacy reference，不可作为模板。
5. `configs/v0.0.9/`：历史版本，不纳入主路径。

#### 验收标准

- `python -m scripts.validate_configs` 覆盖 maintained configs。
- `docs/CONFIG_ATLAS.md` 对 maintained configs 可追踪。
- README Quick Start 不引用 `configs/reference/` 或 `configs/v0.0.9/`。

## 2.2 本地路径协议

### 2.2.1 数据根路径仍需持续防止绝对路径回流

#### 现象与证据

当前已推荐使用：

- repo dummy data：相对路径 `data`
- 真数据：`${PHM_VIBENCH_DATA:-data}`
- 本地私有路径：`configs/local/`

历史文档和旧 config 中仍可能存在机器本地绝对路径。

#### 影响与风险

这是 P1 风险。绝对路径会破坏可移植性、CI 和父仓集成。

#### 优化动作

新增或修改 demo 时必须满足：

- 不提交 `/home/...`、`C:\...` 或机器名路径。
- 真数据路径通过环境变量或 CLI override 注入。
- 本地路径只允许进入未跟踪 local override。

#### 验收标准

建议增加检查命令：

```bash
rg -n "/home/|C:\\\\|/Users/" configs README.md docs src
```

命中结果必须是明确 legacy 文档或示例说明，不得出现在 maintained demo。

## 2.3 preflight 覆盖边界

### 2.3.1 preflight 已覆盖基础字段，但 task/model 深层语义还需分阶段扩展

#### 现象与证据

当前 preflight 覆盖：

- required sections
- pipeline declaration/import
- P02 `pipeline_mode`
- `output_dir`
- `data_dir`
- `metadata_file`

尚未系统覆盖：

- model 输入维度与数据通道一致性。
- task 特定必填字段。
- trainer device 与实际硬件可用性。
- registry target 与配置字段交叉校验。

#### 影响与风险

这是 P1/P0 混合风险。基础路径错误已 fail-fast，但模型/任务深层语义错误可能仍延迟到训练阶段才暴露。

#### 优化动作

不要一次性做大 schema。按风险扩展：

- P0：任务必须字段、loss pairing、数据 metadata 强依赖。
- P1：模型输入 shape、trainer device、registry target。
- P2：可选性能参数、UI 展示字段。

#### 验收标准

- `python -m scripts.config_inspect --config <yaml>` 的 sanity 报告包含失败原因和修复建议。
- 新增深层 preflight 时必须有 `test/test_preflight.py` 覆盖。

# 3. Pipeline 与运行时冗余

## 3.1 Pipeline 运行流程重复

### 3.1.1 多个 Pipeline 曾重复实现 run dir、fit/test、manifest 流程

#### 现象与证据

重复流程包括：

- run dir 创建
- `logger_name/run_dir` 注入
- config snapshot
- data/model/task/trainer 构建
- fit/test
- `test_result_*.csv`
- manifest 写出

当前已通过 `src/utils/training/run_contract.py` 抽出共享 helper，但仍需在后续改动中禁止重新复制粘贴这些流程。

#### 影响与风险

这是 P1 风险。重复实现会导致某个 pipeline 漏写 manifest、漏写 config snapshot 或错误处理不一致。

#### 优化动作

将 `run_contract.py` 定义为运行契约边界。新增 pipeline 时优先使用：

- `prepare_run_context`
- `build_training_stack`
- `write_test_result_and_manifest`

#### 验收标准

- 新增 pipeline 不手写 manifest JSON。
- 新增 pipeline 不重复实现 test result CSV 写出。
- `test/test_run_contract_helper.py` 覆盖 helper 行为。

## 3.2 fallback 与吞错

### 3.2.1 训练关键路径不得 warning 后继续

#### 现象与证据

历史风险包括：

- adapter 失败后 fallback legacy。
- loss 失败后返回 0。
- metric 失败后返回空指标。
- test 失败后吞错继续返回结果。

当前 P03/P04/P02/HSE 主风险已处理，但仓库中其他模块仍有 `fallback` 字符串，需要按上下文区分训练关键路径和 optional 路径。

#### 影响与风险

这是 P0 风险。训练关键 fallback 会让错误配置产出错误结果。

#### 优化动作

分类治理：

- 训练关键路径：raise。
- optional artifact：degraded。
- legacy wrapper 内部兼容：必须在 registry 和 README 标注。

#### 验收标准

`scripts/run_demo_matrix.sh` 中的 silent fallback gate 通过。新增训练关键 fallback 必须有合理注释和测试证明不影响结果语义。

## 3.3 P02 多模式复杂度

### 3.3.1 `single | staged | legacy` 需要保持显式语义

#### 现象与证据

P02 现在通过 `pipeline_mode` 分流：

- `single`
- `staged`
- `legacy`

legacy 仍用于 dual-YAML 兼容。

#### 影响与风险

这是 P1 风险。如果用户没有显式 mode，可能把 staged 配置当 single 或把 legacy 当维护主路径。

#### 优化动作

文档和 preflight 固定语义：

- `single`：单阶段训练。
- `staged`：统一多阶段配置。
- `legacy`：短期兼容，必须显式传 `--fs_config_path`。

#### 验收标准

- 缺 `pipeline_mode` 报错。
- `staged` 缺 `stages` 报错。
- `legacy` 缺 `--fs_config_path` 报错。
- 覆盖测试保留在 `test/test_pipeline_02_modes.py`。

# 4. Artifact、Manifest 与父仓集成

## 4.1 manifest 契约

### 4.1.1 父仓只能依赖 manifest 和 metrics，不应读取 trainer 内部

#### 现象与证据

当前标准产物：

```text
<run_dir>/
  config_snapshot.yaml
  test_result_*.csv
  artifacts/
    manifest.json
    data_metadata_snapshot.json
```

manifest 必填字段包括：

- `run_id`
- `stage`
- `timestamp`
- `run_dir`
- `config_snapshot`
- `metrics_path`
- `seed`
- `git_sha`
- `data_metadata_snapshot`

#### 影响与风险

这是 P0/P1 边界问题。manifest 缺失会破坏父仓集成；字段缺失会造成结果不可审计。

#### 优化动作

`src/trainer_factory/extensions/manifest.py` 继续作为 manifest schema 和 writer 的唯一实现位置。

#### 验收标准

`test/test_run_artifacts_contract.py` 必须覆盖：

- 必填字段存在。
- 缺 metrics 时失败。
- optional explain 缺失时不失败。

## 4.2 optional artifact 边界

### 4.2.1 explain、prediction、distilled 只能 degraded，不能替代主结果

#### 现象与证据

optional 产物包括：

- explain eligibility
- explain summary
- prediction dump
- figures
- distilled outputs

#### 影响与风险

这是 P1 风险。如果 optional artifact 失败导致主训练失败，会降低可用性；如果 optional 失败隐藏 manifest 缺失，则会破坏集成。

#### 优化动作

明确边界：

- manifest required
- metrics required
- config snapshot required
- data metadata snapshot required
- explain optional
- prediction optional
- figures optional

#### 验收标准

缺 optional explain 仍可产出 manifest；缺 manifest 或 metrics 必须失败。

# 5. 测试体系冗余与缺口

## 5.1 maintained tests 与 parked tests

### 5.1.1 `test/`、`test/todo_test/` 和历史测试职责不清

#### 现象与证据

仓库存在：

- maintained tests：`test/`
- parked tests：`test/todo_test/`
- 历史说明：`test/README.md`
- optional streamlit / X_model smoke tests

#### 影响与风险

这是 P1 风险。测试边界不清会导致 CI 过重，或把草稿测试误认为主路径门禁。

#### 优化动作

定义测试层级：

- Unit contract：快速、必须进 CI。
- Config/preflight：快速、必须进 CI。
- Smoke runtime：只跑 dummy data。
- Optional integration：需要真实数据或重依赖，手动运行。
- Parked TODO：不进 CI，必须说明转正条件。

#### 验收标准

CI 不依赖真实数据；full demo matrix 只在 `PHM_VIBENCH_DATA` 存在时运行。

## 5.2 P0 bug 回归测试

### 5.2.1 每个 P0 bug 必须有回归测试

#### 现象与证据

当前已有关键测试：

- strict main：`test/test_main_strictness.py`
- preflight：`test/test_preflight.py`
- P02 mode：`test/test_pipeline_02_modes.py`
- InfoNCE pairing：`test/test_infonce_pairing.py`
- HSE fail-fast：`test/test_hse_contrastive_failfast.py`
- manifest：`test/test_run_artifacts_contract.py`
- run contract：`test/test_run_contract_helper.py`
- demo matrix：`test/test_demo_matrix_script.py`

#### 影响与风险

这是 P0 质量门禁。没有测试的 P0 bug 很容易在重构时回流。

#### 优化动作

每个新 P0 issue 必须在报告表中绑定测试文件或新增测试计划。

#### 验收标准

P0 清单中不得出现“无测试覆盖”的已完成项。

# 6. Legacy、历史目录与重复文档

## 6.1 legacy config 归档

### 6.1.1 `configs/reference/` 与 `configs/v0.0.9/` 仍可能被误用

#### 现象与证据

这些目录包含旧实验、旧本地覆盖机制和旧 pipeline 配置。

#### 影响与风险

这是 P1/P2 风险。误用 legacy 配置可能绕过新 preflight 或 manifest 契约。

#### 优化动作

保留但降权：

- README 顶部标 legacy。
- 不在 Quick Start 引用。
- 有价值配置迁移到 Hydra experiments。
- 未迁移配置不进入 demo matrix。

#### 验收标准

`rg -n "configs/reference|configs/v0.0.9" README.md AGENTS.md docs` 的命中只能是说明其 legacy 状态，不应作为推荐入口。

## 6.2 frontend 与旧 app 目录

### 6.2.1 `frontend/` 与旧 `app/` 迁移状态需明确

#### 现象与证据

当前仓库状态显示旧 `app/` 多文件删除，新 `frontend/` 存在迁移痕迹。README 中说明 Streamlit 仍不是 validation gate。

#### 影响与风险

这是 P1 风险。用户可能混淆 CLI 主路径和 GUI 实验控制台。

#### 优化动作

文档固定：

- CLI 是权威运行入口。
- `frontend/` 是 research console。
- GUI 不作为核心 CI 门禁。
- 旧 `app/` 如果保留，只能作为迁移说明或 archive。

#### 验收标准

README、docs/app_usage.md、AGENTS 对 Streamlit 入口描述一致。

## 6.3 X_model legacy wrappers

### 6.3.1 legacy wrapper 与 optional dependency 状态需要表格化

#### 现象与证据

`src/model_factory/X_model/` 下有多个 legacy wrappers 和 `legacy_collection/`，部分依赖可选包，例如 `torch_geometric`。

#### 影响与风险

这是 P1/P2 风险。用户可能把 legacy wrapper 当主模型，或在未安装可选依赖时触发导入失败。

#### 优化动作

在 `src/model_factory/X_model/README.md` 或 registry 中明确：

- maintained
- compatibility wrapper
- optional dependency
- archive only
- not registered

#### 验收标准

每个 registry 中的 X_model 都有状态说明、依赖说明和至少一个 smoke 或 optional 标记。

# 7. 数据层与采样层问题

## 7.1 data_factory fallback 残留

### 7.1.1 reader 或 dataset task 不应被默认替代

#### 现象与证据

`src/data_factory/__init__.py` 中仍有 default fallback 注释或逻辑痕迹。data_factory、dataset_task、reader、sampler 中也存在 TODO。

#### 影响与风险

这是 P0/P1 风险。错误 reader 被默认替代可能直接导致 silent wrong。

#### 优化动作

建议后续审查：

- reader name 不存在时 fail-fast。
- dataset task 不匹配时 fail-fast。
- 只有 dummy smoke 明确允许默认 reader。

#### 验收标准

新增测试覆盖错误 reader/task，不允许悄悄跑 default data。

## 7.2 few-shot / GFS sampler TODO

### 7.2.1 未实现采样协议不能暴露为可用配置

#### 现象与证据

FS、DG README 和 samplers 中存在未实现 feature：

- multi-scale episodes
- cross-domain episodes
- hierarchical few-shot
- domain regularization variants

#### 影响与风险

这是 P1 风险。配置暴露未实现能力会让用户误以为实验语义已被支持。

#### 优化动作

未实现功能必须二选一：

- 从 maintained config 中移除。
- 启用时明确 raise `NotImplementedError`。

#### 验收标准

`scripts.validate_configs` 不接受 maintained demo 中出现未实现开关。

# 8. 模型与任务层 bug 风险

## 8.1 contrastive objective

### 8.1.1 无标签 InfoNCE pairing 是必须保留测试的 P0 修复点

#### 现象与证据

无标签 InfoNCE 已按 SimCLR two-view pairing 修复，并对奇数 batch 抛错。

#### 影响与风险

这是 P0 风险。若回退到全零 positive mask，HSE pretrain 可能训练但目标为空。

#### 优化动作

保留测试并纳入 CI：

- `test/test_infonce_pairing.py`
- `test/test_hse_contrastive_failfast.py`
- demo matrix 的 HSE contrastive signal gate

#### 验收标准

2N batch loss 大于 0；奇数 batch 必须抛错；HSE signal 测试必须通过。

## 8.2 task / metrics 兼容层

### 8.2.1 多任务、RUL、ID metrics 仍有 parked 测试需要转正

#### 现象与证据

`test/todo_test/` 中存在 task-specific metrics、RUL validation、batch metadata 等 parked tests。

#### 影响与风险

这是 P1 风险。多任务指标语义不统一会影响论文表格和跨任务比较。

#### 优化动作

分批转正：

1. silent wrong 类 metrics。
2. task-specific metrics correctness。
3. heavy integration。

#### 验收标准

每批转正后 CI 仍保持小门禁可接受，heavy tests 不默认进入 core CI。

# 9. 文档与索引冗余

## 9.1 README / AGENTS / REPO_INDEX 职责

### 9.1.1 多入口文档容易重复维护

#### 现象与证据

当前入口包括：

- `README.md`
- `AGENTS.md`
- `CLAUDE.md`
- `docs/REPO_INDEX.md`
- `docs/README.md`
- `configs/README.md`
- `docs/CONFIG_ATLAS.md`

#### 影响与风险

这是 P1 风险。重复说明会在后续变更中不一致。

#### 优化动作

固定职责：

- README：用户路径。
- AGENTS：agent 操作规则。
- REPO_INDEX：只读导航。
- CONFIG_ATLAS：生成型配置索引。
- archives：完成项目证据链。

#### 验收标准

文档只保留最短必要说明，重复内容改成链接。

## 9.2 paper 子模块索引

### 9.2.1 paper 目录必须避免默认递归读取

#### 现象与证据

`paper/` 下包含多个 submodule、论文草稿、结果目录和子模块局部 agent 产物。

#### 影响与风险

这是 P1 风险。默认递归读取会浪费上下文，甚至误改子模块。

#### 优化动作

继续维护：

- `paper/README.md`
- `paper/README_SUBMODULE.md`
- `docs/REPO_INDEX.md` 中的避免递归规则

#### 验收标准

agent 处理 paper 请求时先读 paper index，再进入指定 submodule。

# 10. 后续工作拆解表

## 10.1 P0 修复与防回归

### 10.1.1 P0 清单

#### 现象与证据

| ID | 模块 | 问题 | 当前状态 | 证据 |
|---|---|---|---|---|
| P0-01 | main/config | 缺 config 或 pipeline 不应默认跑 demo | 已修复，需保留测试 | `test/test_main_strictness.py` |
| P0-02 | preflight | data/metadata 缺失必须 trainer 前失败 | 已修复，需扩展深层字段 | `test/test_preflight.py` |
| P0-03 | P02 | 多模式必须显式 | 已修复，需文档继续强调 | `test/test_pipeline_02_modes.py` |
| P0-04 | contrastive | 无标签 InfoNCE 不能零 loss | 已修复，需保留 gate | `test/test_infonce_pairing.py` |
| P0-05 | artifacts | manifest/metrics 必须稳定 | 已修复，需所有新 pipeline 复用 helper | `test/test_run_artifacts_contract.py` |

#### 影响与风险

这些问题均可能导致 silent wrong 或父仓无法消费。

#### 优化动作

P0 后续只允许做强化：

- 扩展 preflight。
- 增加回归测试。
- 纳入 demo matrix。
- 不增加隐式 fallback。

#### 验收标准

P0 表中每项必须有测试或 gate。

## 10.2 P1 清理与收敛

### 10.2.1 P1 清单

#### 现象与证据

| ID | 模块 | 问题 | 推荐动作 | 验收 |
|---|---|---|---|---|
| P1-01 | configs | Hydra/demo/reference/v0.0.9 双轨多轨 | 标状态并逐步迁移 | atlas 与 README 一致 |
| P1-02 | Pipeline | 重复运行流程 | 继续使用 `run_contract.py` | 新 pipeline 无手写 manifest |
| P1-03 | tests | maintained 与 parked 混杂 | 明确 CI 与 TODO 边界 | core CI 不跑 heavy tests |
| P1-04 | frontend | app/frontend 迁移痕迹 | 文档统一入口 | README 与 app_usage 一致 |
| P1-05 | X_model | legacy wrapper 状态不清 | registry 标状态 | optional dependency 明确 |

#### 影响与风险

P1 主要增加 first-run、debug 和 review 成本。

#### 优化动作

按文档和 registry 先治理认知边界，再逐步删减代码。

#### 验收标准

新用户不需要读 legacy 目录就能完成 smoke、inspect、run、consume。

## 10.3 P2 长期归档

### 10.3.1 P2 清单

#### 现象与证据

| ID | 模块 | 问题 | 推荐动作 | 验收 |
|---|---|---|---|---|
| P2-01 | docs/past | 历史文档较多 | 保留但不作为入口 | docs README 不推荐直接读 |
| P2-02 | configs deprecated | 旧 config manager | 仅迁移时读取 | import 不进入主路径 |
| P2-03 | paper submodules | 子模块结果和草稿多 | index-first | 不默认递归搜索 |
| P2-04 | test todo | parked tests 未分类 | 按风险分批转正 | TODO README 清晰 |

#### 影响与风险

P2 不直接影响当前 smoke，但会影响长期维护和 agent 上下文成本。

#### 优化动作

归档优先，删除谨慎。删除前必须确认没有 registry、README、CI 引用。

#### 验收标准

archive/index 可追溯，主路径不依赖 P2 内容。

# 11. 报告维护规则

## 11.1 新增问题项规则

### 11.1.1 新问题必须使用四级字段

#### 现象与证据

新增问题不得只写一句 TODO。

#### 影响与风险

没有证据和验收的问题无法交给 agent 或工程师执行。

#### 优化动作

新增问题必须包含：

- 现象与证据
- 影响与风险
- 优化动作
- 验收标准

#### 验收标准

任何三级标题下都能找到四个固定四级标题。

## 11.2 完成问题项规则

### 11.2.1 完成必须绑定验证结果

#### 现象与证据

已完成项必须写明测试、命令或产物。

#### 影响与风险

没有验证的完成状态不可审计。

#### 优化动作

完成项必须补：

- 修改摘要
- 测试命令
- 产物路径
- 剩余风险

#### 验收标准

reviewer 可只看报告判断该项是否完成。

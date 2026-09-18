# PHMFactory Model Catalogue Integration Plan v2

保存自 2026-09-18 对话中的 Plan v2，整理重复表述和聊天界面标记，保留 B−1—B05 的组织、核心术语、验收和停止边界。本文件是来源方案，不宣称其中的任务已经执行；本轮执行边界见 ../README.md。

## 0. 总目标重新定义

本项目的完成条件不是“实现 187 个模型”，而是：

$$
\text{Completion}=\text{Candidate Disposition}_{100\%}\land\text{Scientific Coverage}\land\text{Verified Implementations}\land\text{Maintainability}.
$$

Candidate Disposition：187 个候选全部得到明确结论。
Scientific Coverage：Benchmark 所需 task、capability、inductive bias 被充分覆盖。
Verified Implementations：进入 runtime 的实现有对应框架、算法和 PHM 证据。
Maintainability：依赖、CI、资源和维护成本在预算内。

模型数量、GitHub star、论文年份、Python 文件、registry 行和 forward smoke 均不能单独作为集成理由。

# B−1｜Scientific Scope & Integration Governance Freeze

不实现新模型。

## B−1.1 Benchmark Coverage Matrix

每个 Task 回答：Benchmark question、Required model capability、Simple baseline、Strong general baseline、Distinct inductive bias、PHM-specific baseline（若需要）、Resource-efficient baseline（若重要）、Foundation/pretrained baseline（若科学问题需要）。模型名称最后确定。

Scientific Inclusion Gate：

```text
G1 Benchmark 中存在明确科学角色
G2 相比现有模型提供可解释的新信息
G3 官方源码或可审计参考实现存在
G4 license 足够明确
G5 可以落入冻结的 Task contract
G6 fidelity 可以验证
G7 依赖和运行成本在维护预算内
```

不满足者可处置为低科学价值、重复覆盖、维护成本过高、证据不足、研究用途或外部服务。处置不等于失败。

## B−1.2 Candidate Identity

$$
\text{candidate\_id}=(\text{canonical name},\text{variant},\text{task},\text{source ref}).
$$

同名不同任务、不同实现和不同来源不能混为一项；共享 backbone 可以只实现一次，不同 Task 的证据分别记录。

## B−1.3 状态模型

```yaml
lifecycle: DISCOVERED | AUDITED | QUEUED | IMPLEMENTING | VERIFIED | RETIRED
disposition: EXISTING | CORE_NATIVE | OPTIONAL_ADAPTER | RESEARCH_ONLY | EXTERNAL_ONLY | DUPLICATE | NOT_ADOPTED
reason_code: NONE | LOW_SCIENTIFIC_VALUE | DUPLICATE_COVERAGE | LICENSE | SOURCE_UNAVAILABLE | TASK_UNAVAILABLE | MAINTENANCE_COST | RESOURCE_EXCEEDED | FIDELITY_UNVERIFIABLE | EVIDENCE_INSUFFICIENT
```

生命周期、最终处置、原因分别记录，不再混合成一个枚举。

## B−1.4 Catalogue 与 Registry

MODEL_CATALOGUE.csv 是研究处置表，包含 candidate_id、canonical_name、variant、task_family、scientific_role、paper、source_repository、source_ref、code_license、weights_license、lifecycle、disposition、reason_code、evidence_level、resource_profile、last_reviewed、notes。

原方案设想 model_registry.csv 仅保留 model_id、candidate_id、import_path、supported_tasks、dependency_extra、config_contract、support_tier，不重复论文、许可、来源和处置。

该部分是目标职责提案，必须先核验仓库现有解析实现，不能以保存方案为由直接迁移运行时 registry。

## B−1.5 Evidence Ladder

E0：论文、官方源码、固定来源版本、代码许可和权重许可。

E1：构造、输入输出、dtype、device、serialization、config 和 CLI。

E2：算法保真。优先官方固定权重及输入数值对齐，其次关键中间量对齐，再次独立等式、极限或算子参考；shape-only 不足。

```yaml
fidelity_oracle:
reference_source:
reference_commit:
reference_input:
comparison_target:
tolerance:
known_deviation:
```

E3：真实 PHM 数据切片、冻结 split、有限指标、资源记录、简单基线 sanity comparison、无 target leakage。

```text
INTEGRATED = E0 + E1 + E2
BENCHMARK_READY = E0 + E1 + E2 + E3
```

不得用 Dummy 的 E1 代替 E3。

## B−1.6 Integration Profiles

A — Native trainable：forward、backward、optimizer update、serialization、fidelity oracle、CLI fit/test；PHM 科学比较使用 E3。

B — Optional pretrained：lazy import、缺依赖失败、不默认下载、明确 revision、本地权重加载、离线推断、输出合同、来源、Task 适配。声称支持微调才要求 backward。

C — Statistical/non-gradient：fit/predict、状态保存、随机性、边界、参考等式或上游对齐。不制造 optimizer 测试。

D — External service：仅记录 EXTERNAL_ONLY、API、terms/privacy 和复现边界，不进入 core model registry。

## B−1.7 Checkpoint Provenance

原方案区分：禁止通用 artifact ledger、hash chain、receipt、attestation；允许外部 checkpoint 的最小 model_id、revision、filename、SHA256、license、source URL。该条是保存的提案，不授权本轮新建校验系统。

## B−1.8 Resource Budget

先读真实 T_component、T_public_smoke、M_peak、package_import_time、optional_dependency_size，再定预算，不编数值。

```text
PR CI：contract + focused fidelity + tiny smoke
Nightly：optional dependency matrix + representative PHM runs
Release：installed wheel + representative task/model matrix
```

超预算可移为 optional、research 或不采用，而不是持续扩大 core CI。

## B−1.9 Independent Review

```text
Author context → implementation + tests
Fresh reviewer context → source + diff + tests + artifacts，只读
Merge gate → CI green + reviewer 无未处置独立 P0/P1 + threads 有明确处置
```

不能把作者自查冒充独立审查。

## B−1 Acceptance

冻结 coverage、inclusion gate、candidate identity、正交状态、证据级别、验收 profiles、资源预算依据和 catalogue/registry 职责；不新增模型。

验收：不给定模型名单，仍能判断接入、可选适配、研究、外部、重复或不采用。

# B00｜Deconstruct PR #266

#266 不继续扩大，也不直接作为可合并的大集成包。保存方案时的参考事实为 21 commits、48 files、Draft；开始执行 B00 时重新核验。

## Diff inventory

```text
D0 DISCARD
D1 SHARED_PREREQUISITE
D2 TASK_CONTRACT
D3 MODEL_SPECIFIC
D4 TEST_CONFIG_DOC
```

逐逻辑 hunk 而非只按文件名分类，明确共享 helper、Task、Data adapter、模型、测试、配置、文档和 CI 的实际依赖。

## Dependency DAG

共享前置 → Task contract → model-specific changes；对应测试、配置、文档随其责任项。不能让模型 PR 暗含另一个未合并模型 PR，也不复制共享 helper。

## Superseded mapping

每个旧修改记录 new owner、new PR、保留或丢弃原因。保留项有明确去向后才能关闭 #266 为 superseded。不在 B−1 中执行此事。

# B01｜Representative Core Pilot

每个已冻结 Task 从科学角色中选：简单基线、现代强基线、不同归纳偏置。名称只有通过 Inclusion Gate 后才能确定。

验证链条：candidate audit → E0 → implementation → E1 → E2 → E3 → 分批合并。

默认一模型一 PR。只有同一核心实现、同一上游、参数变体、无独立依赖、同一 fidelity oracle 时允许 2–3 项同批。TimesNet、iTransformer、TSLANet 不因都是分类模型就自动同批。

# B02｜Benchmark-required Models

只补论文表格必需项、coverage 缺口、architecture representative 或必要 PHM comparator。新增模型前回答：缺少它，哪项 Benchmark 结论会实质变弱？回答不了则不采用，而不是按目录顺序搬运。

# B03｜New Task Expansion

```text
Task specification → simplest reference baseline → one real model
→ validate scientific semantics → expand
```

Task spec：Required inputs、Optional inputs、Forbidden information、Target construction、Model output、Loss ownership、Calibration ownership、Metrics、Randomness、Serialization。

需要未来协变量、时间标记、概率输出、自回归采样或非规则采样的模型，必须由匹配 Task 支持，不能删去核心输入后仍称保真。

# B04｜Optional / Foundation Models

默认一模型一 PR，使用 Profile B，明确模型及权重来源，默认不下载。core import 不联网、不下载、不导入可选重依赖。没有真实权重不得用随机初始化冒充预训练模型。

# B05｜Catalogue Closure

逐原始候选形成处置，不能只检查 implemented=187。

```text
EXISTING
CORE_NATIVE
OPTIONAL_ADAPTER
RESEARCH_ONLY
EXTERNAL_ONLY
DUPLICATE
NOT_ADOPTED（必须有原因）
```

NOT_ADOPTED + EVIDENCE_INSUFFICIENT 是当前 release 的明确决定，不是编造确定性；记录检索证据、日期和原因，有新证据才重开。

同时满足每个目标 Task 代表性充分、必要偏置有覆盖、比较对象可执行、没有数量型堆积、依赖和 CI 在预算内。

$$
\text{CatalogueClosed}\land\text{ScientificCoverage}\land\text{EvidenceComplete}\land\text{MaintenanceBudgetSatisfied}.
$$

# Final Gate

最后是 catalogue/docs closure，而不是再创建全模型总实现 PR。

检查所有处置闭合、进入 runtime 的新支持范围有 E0–E2、声称 Benchmark-ready 的配置有 E3、可选权重来源明确、没有 skip 计通过、没有重复实现、预算达标、installed wheel 通过、独立审查无未处置 P0/P1。

现在只执行 B−1。不执行 B00，不修改或合并 #266，不向 dev 合并。

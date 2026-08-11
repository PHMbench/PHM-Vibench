# g01_scientific_validity — 14-Reviewer Dossier

- Repository: `PHMbench/PHM-Vibench`
- Baseline: `dev@7b604a06802b2053611430916d278ee807c6d772`
- Review branch: `reviews/2026-08-11`
- Review count: 14
- Source-code modifications: none


<a id="r01-scientific-contract-and-claim-alignment"></a>
## G01 · R01 — Scientific Contract and Claim Alignment

**Verdict:** `REQUEST_CHANGES`

**Verified facts**

- `PIPE-02`：当前所有 maintained demo 仍为 smoke_only，没有 baseline_valid（VERIFIED，P1）。
- `DATA-11`：不可满足的 target_domain_num 被自动缩减或退化为 train-only（VERIFIED，P0）。
- `PIPE-01`：部分 compatibility/experimental Pipeline 默认可运行，无显式 opt-in（VERIFIED，P1）。

### [G01-R01-F01] 当前所有 maintained demo 仍为 smoke_only，没有 baseline_valid

- Severity: P1
- Status: VERIFIED
- Executed semantics: `sanity_ok` demo 的 `protocol_status` 全部为 `smoke_only`。
- Scientific impact: 仓库尚无一条真实数据、split、evaluation、estimator 全闭合的 baseline。
- Minimal correction: 选择一个普通 classification candidate，完成 reader、split、deterministic evaluation 和 estimator 后，在同一 promotion PR 中晋级。
- Acceptance test: 真实端到端运行产生 checkpoint 和非空有限 metrics，claim 精确到 evaluation unit。

**Minimal PR order**

1. `fix(protocol): fail on impossible target-domain requests` — 保护 `DATA-11` 的主要不变量；只运行其 focused acceptance test。
2. `chore(protocol): freeze explicit source and target domains` — 保护 `DATA-12` 的主要不变量；只运行其 focused acceptance test。

---

<a id="r02-config-runtime-truth"></a>
## G01 · R02 — Configuration Authority and Runtime Dispatch

**Verdict:** `REQUEST_CHANGES`

**Verified facts**

- `CFG-01`：显式实验配置缺少 pipeline 时被静默改写为 Pipeline 01（VERIFIED，P0）。
- `CFG-02`：非法 override 值解析失败后被当作普通字符串继续执行（VERIFIED，P1）。
- `TRAIN-06`：Pipeline 02 在运行前补齐多项 trainer 属性（VERIFIED，P0）。

### [G01-R02-F01] 显式实验配置缺少 pipeline 时被静默改写为 Pipeline 01

- Severity: P0
- Status: VERIFIED
- Executed semantics: `analyze_config()` 使用 `data.get('pipeline', DEFAULT_PIPELINE)`，自动选择 `Pipeline_01_Fault_Diagnosis`。
- Scientific impact: 可能把一个未定义实验直接变成普通故障分类实验，使 requested semantics 与 executed semantics 不相等。
- Minimal correction: 仅当用户完全未提供 config source、明确进入产品默认 quickstart 时选择默认配置；显式 YAML 缺 `pipeline` 直接 `ValueError`。
- Acceptance test: 自定义 YAML 缺 `pipeline` 时失败；无 `--config` 的默认入口仍选择维护的默认配置。

**Minimal PR order**

1. `fix(config): reject missing pipeline in explicit experiments` — 保护 `CFG-01` 的主要不变量；只运行其 focused acceptance test。
2. `fix(trainer): require explicit training budget and device` — 保护 `TRAIN-01` 的主要不变量；只运行其 focused acceptance test。

---

<a id="r03-data-reader-metadata-contract"></a>
## G01 · R03 — Reader, Metadata and Raw-to-Tensor Semantics

**Verdict:** `REQUEST_CHANGES`

**Verified facts**

- `DATA-10`：非法或缺失监督标签被静默过滤（VERIFIED，P0）。
- `DATA-09`：未指定 target_system_id 时返回全部 metadata（VERIFIED，P0）。
- `TASK-02`：指标 data_name 只取 batch 第一个 file_id（VERIFIED，P0）。

### [G01-R03-F01] 非法或缺失监督标签被静默过滤

- Severity: P0
- Status: VERIFIED
- Executed semantics: 筛选函数直接删除这些行并重置索引。
- Scientific impact: 类别分布、文件数和域覆盖被改变。
- Minimal correction: 监督任务发现非法标签时失败并列出 IDs；若用户要排除，必须在 config/metadata 中显式选择。
- Acceptance test: NaN/-1 标签触发可操作错误。

**Minimal PR order**

1. `fix(protocol): require explicit system population` — 保护 `DATA-09` 的主要不变量；只运行其 focused acceptance test。
2. `fix(data): reject invalid supervised labels` — 保护 `DATA-10` 的主要不变量；只运行其 focused acceptance test。

---

<a id="r04-split-sampling-transform-leakage"></a>
## G01 · R04 — Split, Sampling, Transform and Leakage

**Verdict:** `REQUEST_CHANGES`

**Verified facts**

- `DATA-11`：不可满足的 target_domain_num 被自动缩减或退化为 train-only（VERIFIED，P0）。
- `DATA-08`：未知 task type 会退化为 train=test=all IDs（VERIFIED，P0）。
- `PIPE-02`：当前所有 maintained demo 仍为 smoke_only，没有 baseline_valid（VERIFIED，P1）。

### [G01-R04-F01] 不可满足的 target_domain_num 被自动缩减或退化为 train-only

- Severity: P0
- Status: VERIFIED
- Executed semantics: 代码用 `min(requested, max_testable)`；可能把全部域放入训练并留下空测试集。
- Scientific impact: 请求的 DG/CDDG 问题被改写。
- Minimal correction: 在 ID selection 边界验证域数；不足时明确失败并报告可用域。
- Acceptance test: 不可满足的域请求立即失败。

**Minimal PR order**

1. `fix(protocol): reject unknown task types` — 保护 `DATA-08` 的主要不变量；只运行其 focused acceptance test。
2. `fix(protocol): fail on impossible target-domain requests` — 保护 `DATA-11` 的主要不变量；只运行其 focused acceptance test。

---

<a id="r05-model-device-shape-determinism"></a>
## G01 · R05 — Model, Device, Shape and Determinism

**Verdict:** `REQUEST_CHANGES`

**Verified facts**

- `MODEL-03`：HSE 在 eval 模式仍随机选择 patch（VERIFIED，P0）。
- `MODEL-04`：HSE 在 patch 过大时重复时间和通道（VERIFIED，P0）。
- `TASK-04`：类别数由 max(Label)+1 推断但未验证标签连续性（VERIFIED，P0）。

### [G01-R05-F01] HSE 在 eval 模式仍随机选择 patch

- Severity: P0
- Status: VERIFIED
- Executed semantics: `E_01_HSE.forward()` 无条件调用 `torch.randint` 生成 L/C 起点。
- Scientific impact: 预测与 metric 依赖未声明的采样随机性。
- Minimal correction: train 使用随机 patch；eval 使用确定性的 evenly-spaced 或固定位置。
- Acceptance test: `model.eval()` 同输入重复两次输出完全一致；train 模式仍允许随机。

**Minimal PR order**

1. `fix(hse): make evaluation patch selection deterministic` — 保护 `MODEL-03` 的主要不变量；只运行其 focused acceptance test。
2. `fix(hse): reject patches larger than input` — 保护 `MODEL-04` 的主要不变量；只运行其 focused acceptance test。

---

<a id="r06-task-loss-metric-estimator"></a>
## G01 · R06 — Task, Loss, Metric and Estimator

**Verdict:** `REQUEST_CHANGES`

**Verified facts**

- `TASK-05`：regularization 设备探测会消耗参数生成器的第一个参数（VERIFIED，P0）。
- `TASK-04`：类别数由 max(Label)+1 推断但未验证标签连续性（VERIFIED，P0）。
- `MODEL-05`：HSE contrastive 在 validation/test 中仍生成随机增强视图（VERIFIED，P0）。

### [G01-R06-F01] regularization 设备探测会消耗参数生成器的第一个参数

- Severity: P0
- Status: VERIFIED
- Executed semantics: 先对 `self.parameters()` generator 执行 `next(iter(params))` 取设备，随后再遍历剩余参数。
- Scientific impact: 第一个模型参数不参与正则化，实际 objective 与配置不一致。
- Minimal correction: 一次性把 trainable parameters 转为 list，再从首元素取设备并计算全部参数。
- Acceptance test: 两参数 toy model 的 L1/L2 精确等于手算总和，首参数包含在内。

**Minimal PR order**

1. `fix(hse): make evaluation patch selection deterministic` — 保护 `MODEL-03` 的主要不变量；只运行其 focused acceptance test。
2. `fix(hse-task): make validation augmentation deterministic` — 保护 `MODEL-05` 的主要不变量；只运行其 focused acceptance test。

---

<a id="r07-trainer-checkpoint-evaluation-lifecycle"></a>
## G01 · R07 — Trainer, Checkpoint and Evaluation Lifecycle

**Verdict:** `REQUEST_CHANGES`

**Verified facts**

- `TRAIN-04`：Pipeline 02 吞掉 trainer.test 异常并返回空 metrics 成功（VERIFIED，P0）。
- `TRAIN-05`：Pipeline 02 资源清理不在 finally，且 data 未关闭（VERIFIED，P1）。
- `MODEL-03`：HSE 在 eval 模式仍随机选择 patch（VERIFIED，P0）。

### [G01-R07-F01] Pipeline 02 吞掉 trainer.test 异常并返回空 metrics 成功

- Severity: P0
- Status: VERIFIED
- Executed semantics: `run_pretrain` 与 `run_adapt` 使用 `except Exception: pass`，返回 `metrics={}`。
- Scientific impact: 训练阶段可被错误报告为完成，结果无法用于比较。
- Minimal correction: 传播原始 test 异常；空 test result 也显式失败。
- Acceptance test: test 抛异常/返回空 list 时 stage 失败；有效 mapping 才成功。

**Minimal PR order**

1. `fix(hse): make evaluation patch selection deterministic` — 保护 `MODEL-03` 的主要不变量；只运行其 focused acceptance test。
2. `fix(hse-task): make validation augmentation deterministic` — 保护 `MODEL-05` 的主要不变量；只运行其 focused acceptance test。

---

<a id="r08-module-decoupling-replaceability"></a>
## G01 · R08 — Module Decoupling and Replaceability

**Verdict:** `REQUEST_CHANGES`

**Verified facts**

- `TASK-01`：Default_task 自动把缺失 task_id 补为 classification（VERIFIED，P0）。
- `TASK-02`：指标 data_name 只取 batch 第一个 file_id（VERIFIED，P0）。
- `CFG-01`：显式实验配置缺少 pipeline 时被静默改写为 Pipeline 01（VERIFIED，P0）。

### [G01-R08-F01] Default_task 自动把缺失 task_id 补为 classification

- Severity: P0
- Status: VERIFIED
- Executed semantics: `batch.setdefault('task_id', 'classification')`。
- Scientific impact: 非分类或错误 batch 被改写为分类任务。
- Minimal correction: 普通 classification adapter 显式写 task_id，Task 缺键时失败；或对仅分类 Task 根本不需要该键。
- Acceptance test: 缺 task identity 的多任务路径失败；普通单任务路径合同明确。

**Minimal PR order**

1. `fix(config): reject missing pipeline in explicit experiments` — 保护 `CFG-01` 的主要不变量；只运行其 focused acceptance test。
2. `fix(task): remove implicit classification task_id` — 保护 `TASK-01` 的主要不变量；只运行其 focused acceptance test。

---

<a id="r09-user-experience-cli-results-docs"></a>
## G01 · R09 — User Experience, CLI and Result Discoverability

**Verdict:** `REQUEST_CHANGES`

**Verified facts**

- `UX-01`：普通结果、run attestation 和 artifacts manifest 形成多套结果入口（VERIFIED，P1）。
- `TASK-03`：未知 metric 仅打印 warning 并跳过（VERIFIED，P0）。
- `CFG-04`：公共运行成功受 hash/attestation/evidence 后处理支配（VERIFIED，P1）。

### [G01-R09-F01] 普通结果、run attestation 和 artifacts manifest 形成多套结果入口

- Severity: P1
- Status: VERIFIED
- Executed semantics: CSVLogger/test_result/all_results、`.phmfactory/runs/.../run_manifest.json`、`artifacts/manifest.json` 和 evidence registry 并存。
- Scientific impact: 不直接改变数值，但增加误用旧/错误文件的机会。
- Minimal correction: 普通 run 只保留 checkpoint、metrics、logs 和最小状态；解释性产物位于显式子目录。
- Acceptance test: smoke 完成后用户关键结果只有一个清晰位置。

**Minimal PR order**

1. `fix(metrics): reject unsupported metric names` — 保护 `TASK-03` 的主要不变量；只运行其 focused acceptance test。
2. `cleanup(runtime): remove governance from experiment success` — 保护 `CFG-04` 的主要不变量；只运行其 focused acceptance test。

---

<a id="r10-data-factory-boundary-and-user-contract"></a>
## G01 · R10 — Data Factory Boundary and User Contract

**Verdict:** `REQUEST_CHANGES`

**Verified facts**

- `DATA-06`：Same_system_Sampler 静默跳过缺 metadata 的样本（VERIFIED，P0）。
- `DATA-04`：cache 可复用性只看 ID，不看 reader 语义（VERIFIED，P0）。
- `DATA-07`：训练 drop_last=True 可能让小 system 完全消失（VERIFIED，P0）。

### [G01-R10-F01] Same_system_Sampler 静默跳过缺 metadata 的样本

- Severity: P0
- Status: VERIFIED
- Executed semantics: 构造 `indices_per_system` 时直接 `continue`。
- Scientific impact: 系统、类别或文件可能不参与训练。
- Minimal correction: 缺 metadata 或 system key 立即失败，列出 file_id 和缺失字段。
- Acceptance test: dataset sample count 等于 sampler represented sample count；缺 key 失败。

**Minimal PR order**

1. `fix(data): invalidate cache on reader-semantic changes` — 保护 `DATA-04` 的主要不变量；只运行其 focused acceptance test。
2. `fix(sampler): preserve selected sample coverage` — 保护 `DATA-06` 的主要不变量；只运行其 focused acceptance test。

---

<a id="r11-model-factory-resolution-and-construction"></a>
## G01 · R11 — Model Factory Resolution and Construction

**Verdict:** `REQUEST_CHANGES`

**Verified facts**

- `TASK-04`：类别数由 max(Label)+1 推断但未验证标签连续性（VERIFIED，P0）。
- `MODEL-02`：Trainer 将所有非 cpu 设备请求转换为 auto（VERIFIED，P0）。
- `MODEL-01`：Task 根据 gpus 与 CUDA 可用性主动把网络移到 GPU（VERIFIED，P0）。

### [G01-R11-F01] 类别数由 max(Label)+1 推断但未验证标签连续性

- Severity: P0
- Status: VERIFIED
- Executed semantics: metrics 和 model helper 使用最大标签加一。
- Scientific impact: 例如标签 `{1,2}` 被解释为三类，head 与 metric 维度错误。
- Minimal correction: 监督分类要求 label set 精确等于 `0..K-1`，否则失败并要求修 metadata 或显式 mapping。
- Acceptance test: 非连续/非零起始标签失败；合法标签得到正确 K。

**Minimal PR order**

1. `fix(device): make Trainer the sole device authority` — 保护 `MODEL-01` 的主要不变量；只运行其 focused acceptance test。
2. `fix(trainer): honor explicit device selection` — 保护 `MODEL-02` 的主要不变量；只运行其 focused acceptance test。

---

<a id="r12-task-factory-semantics-and-objective-boundary"></a>
## G01 · R12 — Task Factory and Objective Boundary

**Verdict:** `REQUEST_CHANGES`

**Verified facts**

- `TASK-02`：指标 data_name 只取 batch 第一个 file_id（VERIFIED，P0）。
- `TASK-05`：regularization 设备探测会消耗参数生成器的第一个参数（VERIFIED，P0）。
- `TASK-01`：Default_task 自动把缺失 task_id 补为 classification（VERIFIED，P0）。

### [G01-R12-F01] 指标 data_name 只取 batch 第一个 file_id

- Severity: P0
- Status: VERIFIED
- Executed semantics: `first_file_id = file_ids[0]`，随后所有 loss/metric key 使用第一个 Name。
- Scientific impact: mixed-dataset batch 的指标会被错误归类。
- Minimal correction: 在 Task 边界验证 batch 的 Name 唯一；不唯一则明确拒绝或使用真正支持 mixed-dataset 的 task。
- Acceptance test: 混合 Name batch 立即失败并列出 names。

**Minimal PR order**

1. `fix(hse-task): make validation augmentation deterministic` — 保护 `MODEL-05` 的主要不变量；只运行其 focused acceptance test。
2. `fix(task): remove implicit classification task_id` — 保护 `TASK-01` 的主要不变量；只运行其 focused acceptance test。

---

<a id="r13-trainer-factory-device-checkpoint-callbacks"></a>
## G01 · R13 — Trainer Factory, Device and Callbacks

**Verdict:** `REQUEST_CHANGES`

**Verified facts**

- `TRAIN-04`：Pipeline 02 吞掉 trainer.test 异常并返回空 metrics 成功（VERIFIED，P0）。
- `MODEL-02`：Trainer 将所有非 cpu 设备请求转换为 auto（VERIFIED，P0）。
- `MODEL-01`：Task 根据 gpus 与 CUDA 可用性主动把网络移到 GPU（VERIFIED，P0）。

### [G01-R13-F01] Pipeline 02 吞掉 trainer.test 异常并返回空 metrics 成功

- Severity: P0
- Status: VERIFIED
- Executed semantics: `run_pretrain` 与 `run_adapt` 使用 `except Exception: pass`，返回 `metrics={}`。
- Scientific impact: 训练阶段可被错误报告为完成，结果无法用于比较。
- Minimal correction: 传播原始 test 异常；空 test result 也显式失败。
- Acceptance test: test 抛异常/返回空 list 时 stage 失败；有效 mapping 才成功。

**Minimal PR order**

1. `fix(device): make Trainer the sole device authority` — 保护 `MODEL-01` 的主要不变量；只运行其 focused acceptance test。
2. `fix(trainer): honor explicit device selection` — 保护 `MODEL-02` 的主要不变量；只运行其 focused acceptance test。

---

<a id="r14-adversarial-meta-review-factory-arbitration"></a>
## G01 · R14 — Adversarial Meta-Review and Factory Arbitration

**Verdict:** `REQUEST_CHANGES`

**Verified facts**

- `PIPE-02`：当前所有 maintained demo 仍为 smoke_only，没有 baseline_valid（VERIFIED，P1）。
- `TRAIN-06`：Pipeline 02 在运行前补齐多项 trainer 属性（VERIFIED，P0）。
- `TRAIN-04`：Pipeline 02 吞掉 trainer.test 异常并返回空 metrics 成功（VERIFIED，P0）。

### [G01-R14-F01] 当前所有 maintained demo 仍为 smoke_only，没有 baseline_valid

- Severity: P1
- Status: VERIFIED
- Executed semantics: `sanity_ok` demo 的 `protocol_status` 全部为 `smoke_only`。
- Scientific impact: 仓库尚无一条真实数据、split、evaluation、estimator 全闭合的 baseline。
- Minimal correction: 选择一个普通 classification candidate，完成 reader、split、deterministic evaluation 和 estimator 后，在同一 promotion PR 中晋级。
- Acceptance test: 真实端到端运行产生 checkpoint 和非空有限 metrics，claim 精确到 evaluation unit。

**Minimal PR order**

1. `fix(config): reject missing pipeline in explicit experiments` — 保护 `CFG-01` 的主要不变量；只运行其 focused acceptance test。
2. `fix(protocol): fail on impossible target-domain requests` — 保护 `DATA-11` 的主要不变量；只运行其 focused acceptance test。

---

# g02_fail_fast — 14-Reviewer Dossier

- Repository: `PHMbench/PHM-Vibench`
- Baseline: `dev@7b604a06802b2053611430916d278ee807c6d772`
- Review branch: `reviews/2026-08-11`
- Review count: 14
- Source-code modifications: none

<a id="r01-scientific-contract-and-claim-alignment"></a>
## G02 · R01 — Scientific Contract and Claim Alignment

**Verdict:** `REQUEST_CHANGES`

**Verified facts**

- `DATA-11`：不可满足的 target_domain_num 被自动缩减或退化为 train-only（VERIFIED，P0）。
- `DATA-08`：未知 task type 会退化为 train=test=all IDs（VERIFIED，P0）。
- `CFG-01`：显式实验配置缺少 pipeline 时被静默改写为 Pipeline 01（VERIFIED，P0）。

### [G02-R01-F01] 不可满足的 target_domain_num 被自动缩减或退化为 train-only

- Severity: P0
- Status: VERIFIED
- Executed semantics: 代码用 `min(requested, max_testable)`；可能把全部域放入训练并留下空测试集。
- Scientific impact: 请求的 DG/CDDG 问题被改写。
- Minimal correction: 在 ID selection 边界验证域数；不足时明确失败并报告可用域。
- Acceptance test: 不可满足的域请求立即失败。

**Minimal PR order**

1. `fix(config): reject missing pipeline in explicit experiments`。
2. `fix(protocol): reject unknown task types`。

---

<a id="r02-config-runtime-truth"></a>
## G02 · R02 — Configuration Authority and Runtime Dispatch

**Verdict:** `REQUEST_CHANGES`

**Verified facts**

- `CFG-01`：显式实验配置缺少 pipeline 时被静默改写为 Pipeline 01（VERIFIED，P0）。
- `CFG-02`：非法 override 值解析失败后被当作普通字符串继续执行（VERIFIED，P1）。
- `TRAIN-06`：Pipeline 02 在运行前补齐多项 trainer 属性（VERIFIED，P0）。

### [G02-R02-F01] 显式实验配置缺少 pipeline 时被静默改写为 Pipeline 01

- Severity: P0
- Status: VERIFIED
- Executed semantics: `analyze_config()` 使用 `data.get('pipeline', DEFAULT_PIPELINE)`，自动选择 `Pipeline_01_Fault_Diagnosis`。
- Scientific impact: requested semantics 与 executed semantics 不相等。
- Minimal correction: 显式 YAML 缺 `pipeline` 直接 `ValueError`；仅产品默认 quickstart 可选择默认配置。
- Acceptance test: 自定义 YAML 缺 `pipeline` 时失败。

**Minimal PR order**

1. `fix(config): reject missing pipeline in explicit experiments`。
2. `fix(trainer): require explicit training budget and device`。

---

<a id="r03-data-reader-metadata-contract"></a>
## G02 · R03 — Reader, Metadata and Raw-to-Tensor Semantics

**Verdict:** `REQUEST_CHANGES`

**Verified facts**

- `DATA-02`：通用 metadata reader 猜测 delimiter 与 encoding（VERIFIED，P0）。
- `DATA-04`：cache 可复用性只看 ID，不看 reader 语义（VERIFIED，P0）。
- `MODEL-04`：HSE 在 patch 过大时重复时间和通道（VERIFIED，P0）。

### [G02-R03-F01] 通用 metadata reader 猜测 delimiter 与 encoding

- Severity: P0
- Status: VERIFIED
- Executed semantics: `smart_read_csv(auto_detect=True)` 统计逗号/制表符并依次尝试 UTF-8、GBK、Latin-1、`sep=None`。
- Scientific impact: “能读成 DataFrame”不保证 ID、标签或域字段语义正确。
- Minimal correction: 按文件扩展名和显式 delimiter/encoding 读取；解析失败保留原始异常。
- Acceptance test: 错误 delimiter/encoding 立即失败。

**Minimal PR order**

1. `fix(data): make metadata format explicit`。
2. `fix(data): invalidate cache on reader-semantic changes`。

---

<a id="r04-split-sampling-transform-leakage"></a>
## G02 · R04 — Split, Sampling, Transform and Leakage

**Verdict:** `REQUEST_CHANGES`

**Verified facts**

- `DATA-08`：未知 task type 会退化为 train=test=all IDs（VERIFIED，P0）。
- `DATA-11`：不可满足的 target_domain_num 被自动缩减或退化为 train-only（VERIFIED，P0）。
- `TASK-07`：window-level global micro estimator 尚未被代码显式固定（INFERRED，P1）。

### [G02-R04-F01] 未知 task type 会退化为 train=test=all IDs

- Severity: P0
- Status: VERIFIED
- Executed semantics: `search_ids_for_task()` 打印 warning，并把全部 keys 同时用于 train_val 和 test。
- Scientific impact: 构造完全泄漏的评价协议。
- Minimal correction: 未知 task type 直接 `ValueError`，列出已支持类型。
- Acceptance test: 拼错 task.type 在数据构造前失败。

**Minimal PR order**

1. `fix(protocol): reject unknown task types`。
2. `fix(protocol): fail on impossible target-domain requests`。

---

<a id="r05-model-device-shape-determinism"></a>
## G02 · R05 — Model, Device, Shape and Determinism

**Verdict:** `REQUEST_CHANGES`

**Verified facts**

- `MODEL-04`：HSE 在 patch 过大时重复时间和通道（VERIFIED，P0）。
- `MODEL-03`：HSE 在 eval 模式仍随机选择 patch（VERIFIED，P0）。
- `TRAIN-04`：Pipeline 02 吞掉 trainer.test 异常并返回空 metrics 成功（VERIFIED，P0）。

### [G02-R05-F01] HSE 在 patch 过大时重复时间和通道

- Severity: P0
- Status: VERIFIED
- Executed semantics: 使用 `repeat()` 扩展时间轴或复制通道。
- Scientific impact: 输入信号与传感器结构被改变。
- Minimal correction: `patch_size_L > L` 或 `patch_size_C > C` 直接 `ValueError`。
- Acceptance test: 不兼容 patch 配置在首次 forward 失败。

**Minimal PR order**

1. `fix(hse): make evaluation patch selection deterministic`。
2. `fix(hse): reject patches larger than input`。

---

<a id="r06-task-loss-metric-estimator"></a>
## G02 · R06 — Task, Loss, Metric and Estimator

**Verdict:** `REQUEST_CHANGES`

**Verified facts**

- `TASK-05`：regularization 设备探测会消耗参数生成器的第一个参数（VERIFIED，P0）。
- `TASK-06`：未知 regularization 类型被警告后跳过（VERIFIED，P0）。
- `MODEL-05`：HSE contrastive 在 validation/test 中仍生成随机增强视图（VERIFIED，P0）。

### [G02-R06-F01] regularization 设备探测会消耗参数生成器的第一个参数

- Severity: P0
- Status: VERIFIED
- Executed semantics: 先对 `self.parameters()` generator 执行 `next(iter(params))`，随后再遍历剩余参数。
- Scientific impact: 第一个模型参数不参与正则化。
- Minimal correction: 一次性把 trainable parameters 转为 list，再取设备并计算全部参数。
- Acceptance test: 两参数 toy model 的 L1/L2 精确等于手算总和。

**Minimal PR order**

1. `fix(regularization): include every trainable parameter`。
2. `fix(hse-task): make validation augmentation deterministic`。

---

<a id="r07-trainer-checkpoint-evaluation-lifecycle"></a>
## G02 · R07 — Trainer, Checkpoint and Evaluation Lifecycle

**Verdict:** `REQUEST_CHANGES`

**Verified facts**

- `TRAIN-04`：Pipeline 02 吞掉 trainer.test 异常并返回空 metrics 成功（VERIFIED，P0）。
- `TRAIN-05`：Pipeline 02 资源清理不在 finally，且 data 未关闭（VERIFIED，P1）。
- `MODEL-05`：HSE contrastive 在 validation/test 中仍生成随机增强视图（VERIFIED，P0）。

### [G02-R07-F01] Pipeline 02 吞掉 trainer.test 异常并返回空 metrics 成功

- Severity: P0
- Status: VERIFIED
- Executed semantics: `run_pretrain` 与 `run_adapt` 使用 `except Exception: pass`，返回 `metrics={}`。
- Scientific impact: evaluation failure + stage success。
- Minimal correction: 传播原始 test 异常；空 test result 也显式失败。
- Acceptance test: test 抛异常/返回空 list 时 stage 失败。

**Minimal PR order**

1. `fix(pipeline02): fail when stage evaluation fails`。
2. `fix(pipeline02): close data and loggers in finally`。

---

<a id="r08-module-decoupling-replaceability"></a>
## G02 · R08 — Module Decoupling and Replaceability

**Verdict:** `REQUEST_CHANGES`

**Verified facts**

- `TASK-01`：Default_task 自动把缺失 task_id 补为 classification（VERIFIED，P0）。
- `ARCH-01`：共享 classification runtime 特判 Multitask 并修改 Data/Model 配置（VERIFIED，P1）。
- `TRAIN-06`：Pipeline 02 在运行前补齐多项 trainer 属性（VERIFIED，P0）。

### [G02-R08-F01] Default_task 自动把缺失 task_id 补为 classification

- Severity: P0
- Status: VERIFIED
- Executed semantics: `batch.setdefault('task_id', 'classification')`。
- Scientific impact: 非分类或错误 batch 被改写为分类任务。
- Minimal correction: 普通 classification adapter 显式写 task_id，Task 缺键时失败；或单任务合同不需要该键。
- Acceptance test: 缺 task identity 的多任务路径失败。

**Minimal PR order**

1. `fix(task): remove implicit classification task_id`。
2. `fix(runtime): remove Multitask config mutation from shared spine`。

---

<a id="r09-user-experience-cli-results-docs"></a>
## G02 · R09 — User Experience, CLI and Result Discoverability

**Verdict:** `REQUEST_CHANGES`

**Verified facts**

- `UX-01`：普通结果、run attestation 和 artifacts manifest 形成多套结果入口（VERIFIED，P1）。
- `CFG-04`：公共运行成功受 hash/attestation/evidence 后处理支配（VERIFIED，P1）。
- `TASK-03`：未知 metric 仅打印 warning 并跳过（VERIFIED，P0）。

### [G02-R09-F01] 普通结果、run attestation 和 artifacts manifest 形成多套结果入口

- Severity: P1
- Status: VERIFIED
- Executed semantics: CSVLogger/test_result/all_results、run_manifest、artifacts/manifest 和 evidence registry 并存。
- Scientific impact: 增加误用旧/错误文件的机会。
- Minimal correction: 普通 run 只保留 checkpoint、metrics、logs 和最小状态。
- Acceptance test: smoke 完成后关键结果只有一个清晰位置。

**Minimal PR order**

1. `fix(metrics): reject unsupported metric names`。
2. `cleanup(runtime): remove governance from experiment success`。

---

<a id="r10-data-factory-boundary-and-user-contract"></a>
## G02 · R10 — Data Factory Boundary and User Contract

**Verdict:** `REQUEST_CHANGES`

**Verified facts**

- `DATA-06`：Same_system_Sampler 静默跳过缺 metadata 的样本（VERIFIED，P0）。
- `DATA-04`：cache 可复用性只看 ID，不看 reader 语义（VERIFIED，P0）。
- `DATA-05`：None 或空 dataset 被警告后跳过（VERIFIED，P0）。

### [G02-R10-F01] Same_system_Sampler 静默跳过缺 metadata 的样本

- Severity: P0
- Status: VERIFIED
- Executed semantics: 构造 `indices_per_system` 时直接 `continue`。
- Scientific impact: 系统、类别或文件可能不参与训练。
- Minimal correction: 缺 metadata 或 system key 立即失败，列出 file_id 和缺失字段。
- Acceptance test: dataset sample count 等于 sampler represented sample count。

**Minimal PR order**

1. `fix(data): invalidate cache on reader-semantic changes`。
2. `fix(data): reject missing selected datasets`。

---

<a id="r11-model-factory-resolution-and-construction"></a>
## G02 · R11 — Model Factory Resolution and Construction

**Verdict:** `REQUEST_CHANGES`

**Verified facts**

- `MODEL-02`：Trainer 将所有非 cpu 设备请求转换为 auto（VERIFIED，P0）。
- `MODEL-01`：Task 根据 gpus 与 CUDA 可用性主动把网络移到 GPU（VERIFIED，P0）。
- `MODEL-04`：HSE 在 patch 过大时重复时间和通道（VERIFIED，P0）。

### [G02-R11-F01] Trainer 将所有非 cpu 设备请求转换为 auto

- Severity: P0
- Status: VERIFIED
- Executed semantics: `accelerator = 'cpu' if device == 'cpu' else 'auto'`。
- Scientific impact: 设备相关实验条件被改写。
- Minimal correction: 只允许 `cpu`、`cuda`、`auto` 三种显式语义；cuda 不可用时报错。
- Acceptance test: cuda unavailable + device=cuda 失败；device=auto 才可降级。

**Minimal PR order**

1. `fix(device): make Trainer the sole device authority`。
2. `fix(trainer): honor explicit device selection`。

---

<a id="r12-task-factory-semantics-and-objective-boundary"></a>
## G02 · R12 — Task Factory and Objective Boundary

**Verdict:** `REQUEST_CHANGES`

**Verified facts**

- `TASK-06`：未知 regularization 类型被警告后跳过（VERIFIED，P0）。
- `TASK-01`：Default_task 自动把缺失 task_id 补为 classification（VERIFIED，P0）。
- `TASK-05`：regularization 设备探测会消耗参数生成器的第一个参数（VERIFIED，P0）。

### [G02-R12-F01] 未知 regularization 类型被警告后跳过

- Severity: P0
- Status: VERIFIED
- Executed semantics: 打印警告并 `continue`，总 loss 不含该项。
- Scientific impact: 配置声明的优化目标消失。
- Minimal correction: 未知类型直接失败，支持值仅为当前真实实现。
- Acceptance test: 未知 key 在第一次训练前失败。

**Minimal PR order**

1. `fix(task): reject unsupported regularization`。
2. `fix(task): remove implicit classification task_id`。

---

<a id="r13-trainer-factory-device-checkpoint-callbacks"></a>
## G02 · R13 — Trainer Factory, Device and Callbacks

**Verdict:** `REQUEST_CHANGES`

**Verified facts**

- `TRAIN-04`：Pipeline 02 吞掉 trainer.test 异常并返回空 metrics 成功（VERIFIED，P0）。
- `MODEL-02`：Trainer 将所有非 cpu 设备请求转换为 auto（VERIFIED，P0）。
- `MODEL-01`：Task 根据 gpus 与 CUDA 可用性主动把网络移到 GPU（VERIFIED，P0）。

### [G02-R13-F01] Pipeline 02 吞掉 trainer.test 异常并返回空 metrics 成功

- Severity: P0
- Status: VERIFIED
- Executed semantics: `run_pretrain` 与 `run_adapt` 使用 `except Exception: pass`，返回 `metrics={}`。
- Scientific impact: 训练阶段可被错误报告为完成。
- Minimal correction: 传播原始 test 异常；空 test result 也显式失败。
- Acceptance test: test 抛异常/返回空 list 时 stage 失败。

**Minimal PR order**

1. `fix(device): make Trainer the sole device authority`。
2. `fix(pipeline02): fail when stage evaluation fails`。

---

<a id="r14-adversarial-meta-review-factory-arbitration"></a>
## G02 · R14 — Adversarial Meta-Review and Factory Arbitration

**Verdict:** `REQUEST_CHANGES`

**Verified facts**

- `TRAIN-06`：Pipeline 02 在运行前补齐多项 trainer 属性（VERIFIED，P0）。
- `TRAIN-04`：Pipeline 02 吞掉 trainer.test 异常并返回空 metrics 成功（VERIFIED，P0）。
- `PR-02`：开放 PR #148 治理范围重且基线过旧（VERIFIED，P1）。

### [G02-R14-F01] Pipeline 02 在运行前补齐多项 trainer 属性

- Severity: P0
- Status: VERIFIED
- Executed semantics: `_ensure_trainer_attributes()` 写入默认 monitor/save_dir/device/devices/log interval/num_epochs。
- Scientific impact: 不同 stage 的训练语义被自动修改。
- Minimal correction: 删除科学语义默认；仅允许纯输出路径等运行上下文在调用点传递。
- Acceptance test: 缺关键 stage trainer 字段立即失败。

**Minimal PR order**

1. `fix(pipeline02): stop repairing stage trainer config`。
2. `fix(pipeline02): fail when stage evaluation fails`。

---

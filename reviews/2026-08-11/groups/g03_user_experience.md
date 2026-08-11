# g03_user_experience — 14-Reviewer Dossier

- Repository: `PHMbench/PHM-Vibench`
- Baseline: `dev@7b604a06802b2053611430916d278ee807c6d772`
- Review branch: `reviews/2026-08-11`
- Review count: 14
- Source-code modifications: none

## G03 · R01 — Scientific Contract and Claim Alignment

**Verdict:** `REQUEST_CHANGES`

- `DATA-11`：不可满足的 target_domain_num 被自动缩减或退化为 train-only（VERIFIED，P0）。
- `PIPE-02`：当前所有 maintained demo 仍为 smoke_only，没有 baseline_valid（VERIFIED，P1）。
- `PIPE-01`：部分 compatibility/experimental Pipeline 默认可运行，无显式 opt-in（VERIFIED，P1）。

### [G03-R01-F01] 不可满足的 target_domain_num 被自动缩减或退化为 train-only

- Executed semantics: `min(requested, max_testable)` 可能把全部域放入训练并留下空测试集。
- Scientific impact: 请求的 DG/CDDG 问题被改写。
- Minimal correction: 在 ID selection 边界验证域数；不足时明确失败并报告可用域。
- Acceptance test: 不可满足的域请求立即失败。

**PR order:** require explicit system population → fail on impossible target domains.

---

## G03 · R02 — Configuration Authority and Runtime Dispatch

**Verdict:** `REQUEST_CHANGES`

- `CFG-04`：公共运行成功受 hash/attestation/evidence 后处理支配（VERIFIED，P1）。
- `CFG-02`：非法 override 值解析失败后被当作普通字符串继续执行（VERIFIED，P1）。
- `TRAIN-06`：Pipeline 02 在运行前补齐多项 trainer 属性（VERIFIED，P0）。

### [G03-R02-F01] 公共运行成功受治理后处理支配

- Executed semantics: CLI 在执行前创建 attestation，在执行后注册 evidence；后处理失败会覆盖科学运行状态。
- Scientific impact: 实验失败与记录失败被混为一谈。
- Minimal correction: 公共主路径只保留最小运行状态和用户结果；移除 artifact hash/evidence registry 对成功状态的控制。
- Acceptance test: train/checkpoint/test/metrics 成功即可形成科学成功。

**PR order:** require explicit trainer budget/device → stop Pipeline 02 config repair.

---

## G03 · R03 — Reader, Metadata and Raw-to-Tensor Semantics

**Verdict:** `REQUEST_CHANGES`

- `DATA-04`：cache 可复用性只看 ID，不看 reader 语义（VERIFIED，P0）。
- `DATA-02`：metadata reader 猜测 delimiter 与 encoding（VERIFIED，P0）。
- `MODEL-04`：HSE 在 patch 过大时重复时间和通道（VERIFIED，P0）。

### [G03-R03-F01] cache 可复用性只看 ID

- Executed semantics: published cache 主要检查 expected ID keys 是否存在。
- Scientific impact: 旧通道、旧 dtype 或旧读法可能被当成本次输入。
- Minimal correction: 直接记录并比较 reader name、selected columns、delimiter、dtype；不一致则重建。
- Acceptance test: 修改 reader-relevant config 后旧 cache 不被复用。

**PR order:** explicit metadata format → semantic cache invalidation.

---

## G03 · R04 — Split, Sampling, Transform and Leakage

**Verdict:** `REQUEST_CHANGES`

- `DATA-11`：不可满足的 target_domain_num 被自动缩减或退化为 train-only（VERIFIED，P0）。
- `DATA-08`：未知 task type 会退化为 train=test=all IDs（VERIFIED，P0）。
- `PIPE-02`：当前 maintained demo 全部 smoke_only（VERIFIED，P1）。

### [G03-R04-F01] impossible domain request 被静默改写

- Executed semantics: 代码自动缩减 test_count。
- Scientific impact: 配置协议与实际 split 不一致。
- Minimal correction: 不可满足的 domain request 立即失败。
- Acceptance test: target 域不足时返回可操作错误。

**PR order:** reject unknown task types → fail on impossible target domains.

---

## G03 · R05 — Model, Device, Shape and Determinism

**Verdict:** `REQUEST_CHANGES`

- `MODEL-04`：HSE 在 patch 过大时重复时间和通道（VERIFIED，P0）。
- `MODEL-03`：HSE 在 eval 模式仍随机选择 patch（VERIFIED，P0）。
- `TRAIN-04`：Pipeline 02 evaluation failure 仍可成功（VERIFIED，P0）。

### [G03-R05-F01] HSE 重复时间和通道

- Executed semantics: `repeat()` 扩展 L/C。
- Scientific impact: 输入信号与传感器结构被改变。
- Minimal correction: patch 大于输入时直接失败。
- Acceptance test: incompatible patch config 在首次 forward 失败。

**PR order:** deterministic HSE eval → reject oversized patches.

---

## G03 · R06 — Task, Loss, Metric and Estimator

**Verdict:** `REQUEST_CHANGES`

- `TASK-05`：regularization 消耗参数生成器第一个参数（VERIFIED，P0）。
- `TASK-03`：未知 metric 仅 warning 并跳过（VERIFIED，P0）。
- `MODEL-05`：HSE contrastive 在 val/test 仍随机增强（VERIFIED，P0）。

### [G03-R06-F01] 第一个参数不参与 regularization

- Executed semantics: `next(iter(params))` 先消费 generator，再遍历剩余参数。
- Scientific impact: 实际 objective 与配置不一致。
- Minimal correction: 先 materialize trainable parameter list，再从完整列表计算。
- Acceptance test: toy model 正则项等于手算结果。

**PR order:** single device authority → deterministic validation augmentation.

---

## G03 · R07 — Trainer, Checkpoint and Evaluation Lifecycle

**Verdict:** `REQUEST_CHANGES`

- `TRAIN-04`：Pipeline 02 吞 test 异常并返回空 metrics（VERIFIED，P0）。
- `UX-01`：多套结果入口并存（VERIFIED，P1）。
- `TRAIN-05`：资源清理不在 finally（VERIFIED，P1）。

### [G03-R07-F01] Pipeline 02 wrong-success

- Executed semantics: `except Exception: pass` 后返回 `metrics={}`。
- Scientific impact: evaluation failure + stage success。
- Minimal correction: 传播原始 test 异常；空 result 也失败。
- Acceptance test: test 异常/空 list 均导致 stage failure。

**PR order:** honor device → fail on stage evaluation error.

---

## G03 · R08 — Module Decoupling and Replaceability

**Verdict:** `REQUEST_CHANGES`

- `TASK-01`：Default_task 自动补 task_id=classification（VERIFIED，P0）。
- `ARCH-01`：shared runtime 特判 Multitask 并修改 Data/Model config（VERIFIED，P1）。
- `MODEL-02`：Trainer 把非 cpu 设备请求转为 auto（VERIFIED，P0）。

### [G03-R08-F01] Task 自动补 classification identity

- Executed semantics: `batch.setdefault('task_id','classification')`。
- Scientific impact: 错误 batch 被改写为另一任务。
- Minimal correction: task identity 由明确 adapter/config 提供；缺失则失败。
- Acceptance test: 多任务路径缺 task identity 时失败。

**PR order:** Trainer sole device authority → honor explicit device.

---

## G03 · R09 — User Experience, CLI and Result Discoverability

**Verdict:** `APPROVE_WITH_CONDITIONS`

- `UX-01`：普通结果、run attestation、artifacts manifest 形成多入口（VERIFIED，P1）。
- `CFG-04`：公共运行成功受 governance 后处理支配（VERIFIED，P1）。
- `TRAIN-03`：ManifestWriterCallback broad-exception best-effort（VERIFIED，P1）。

### [G03-R09-F01] 结果入口过多

- Executed semantics: CSVLogger、test_result、all_results、run_manifest、artifacts/manifest、evidence registry 并存。
- User impact: 用户难以判断哪个是权威结果。
- Minimal correction: 普通 run 只保留 checkpoint、metrics、logs 和最小状态。
- Acceptance test: smoke 后关键结果只有一个清晰位置。

**PR order:** remove governance from success → remove default artifact manifest callback.

---

## G03 · R10 — Data Factory Boundary and User Contract

**Verdict:** `REQUEST_CHANGES`

- `DATA-06`：Same_system_Sampler 静默跳过缺 metadata 样本（VERIFIED，P0）。
- `DATA-04`：cache 不看 reader 语义（VERIFIED，P0）。
- `DATA-07`：drop_last 可能让小 system 完全消失（VERIFIED，P0）。

### [G03-R10-F01] sampler 静默丢样本

- Executed semantics: 缺 file metadata/Dataset_id 时 `continue`。
- Scientific impact: selected population 与 trained population 不一致。
- Minimal correction: 缺字段立即失败并列出 file IDs。
- Acceptance test: dataset sample count 等于 sampler represented count。

**PR order:** semantic cache invalidation → selected sample coverage.

---

## G03 · R11 — Model Factory Resolution and Construction

**Verdict:** `REQUEST_CHANGES`

- `MODEL-02`：Trainer 将所有非 cpu 请求转为 auto（VERIFIED，P0）。
- `MODEL-01`：Task 根据 gpus/CUDA 主动 `.cuda()`（VERIFIED，P0）。
- `TASK-04`：类别数 max(Label)+1 未验证连续性（VERIFIED，P0）。

### [G03-R11-F01] device=cuda 可能被自动降级

- Executed semantics: non-cpu → `accelerator='auto'`。
- Scientific impact: 设备实验条件被改写。
- Minimal correction: cpu/cuda/auto 三种显式语义；cuda unavailable 立即失败。
- Acceptance test: 仅显式 auto 可自动选择。

**PR order:** Trainer sole device authority → honor explicit device.

---

## G03 · R12 — Task Factory and Objective Boundary

**Verdict:** `REQUEST_CHANGES`

- `TASK-06`：未知 regularization 被 warning 后跳过（VERIFIED，P0）。
- `TASK-05`：regularization 首参数缺失（VERIFIED，P0）。
- `TASK-02`：metric data_name 只取 batch 首个 file_id（VERIFIED，P0）。

### [G03-R12-F01] 未知 regularization 被跳过

- Executed semantics: warning + continue。
- Scientific impact: 配置 objective 消失。
- Minimal correction: 未知类型直接失败。
- Acceptance test: unsupported key 在训练前失败。

**PR order:** sole device authority → homogeneous metric batch validation.

---

## G03 · R13 — Trainer Factory, Device and Callbacks

**Verdict:** `REQUEST_CHANGES`

- `MODEL-02`：non-cpu → auto（VERIFIED，P0）。
- `MODEL-01`：Task 主动 `.cuda()`（VERIFIED，P0）。
- `TRAIN-04`：Pipeline 02 wrong-success（VERIFIED，P0）。

### [G03-R13-F01] device authority 冲突

- Executed semantics: Task 与 Trainer 都决定 device。
- Scientific impact: 配置设备不等于实际设备。
- Minimal correction: Trainer 为唯一 device authority；Task/Model 不迁移设备。
- Acceptance test: device=cpu 在 CUDA 机器仍保持 CPU。

**PR order:** sole device authority → explicit device semantics.

---

## G03 · R14 — Adversarial Meta-Review and Factory Arbitration

**Verdict:** `APPROVE_WITH_CONDITIONS`

- `UX-01`：多套结果入口（VERIFIED，P1）。
- `PR-02`：PR #148 治理重且基线过旧（VERIFIED，P1）。
- `GOV-01`：Pipeline 06 可因缺 stage ledger 覆盖科学阶段成功（VERIFIED，P1）。

### [G03-R14-F01] 用户结果与治理记录未分离

- Executed semantics: artifact/evidence/ledger 参与成功判定。
- Scientific impact: 记录失败覆盖科学成功。
- Minimal correction: 用户结果为 metrics/checkpoint/logs；治理记录退出关键路径。
- Acceptance test: Pipeline 科学阶段完成不依赖 ledger/hash。

**PR order:** remove governance from success → decouple Pipeline 06 from ledger.

---

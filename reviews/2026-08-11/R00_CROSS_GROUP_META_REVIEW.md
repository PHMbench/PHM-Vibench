# Cross-Group Meta-Review

- Repository: `PHMbench/PHM-Vibench`
- Reviewed authority: `dev@7b604a06802b2053611430916d278ee807c6d772`
- Review branch: `reviews/2026-08-11`
- Review population: **140 reviews = 10 groups × 14 reviewers**

# Verdict

```text
REQUEST_CHANGES
```

# Executive judgment

Across ten independent review lenses, the repository is no longer primarily blocked by missing Factory abstractions. Existing model and compatible-dataset extension seams are already sufficient for ordinary classification. The dominant blockers are requested-versus-executed semantic mismatches, device authority conflicts, stochastic evaluation, wrong-success in multi-stage evaluation, label/metric assumptions, and silent data/protocol fallback. The current public path is also burdened by hash/attestation/evidence machinery that does not establish scientific correctness. The shortest path is to close core semantic defects with Dummy/CSV focused tests, then validate one real dataset using a transparent baseline, and only then promote a `baseline_valid` configuration. CWRU can remain the final local-agent acceptance dataset; MFPT, SEU, or PU can exercise the same contracts earlier.

# Consensus finding index

| Finding | Severity | Mentions | Consensus issue | Minimal PR |
|---|---:|---:|---|---|
| `TRAIN-04` | P0 | 40/140 | Pipeline 02 吞掉 trainer.test 异常并返回空 metrics 成功 | `fix(pipeline02): fail when stage evaluation fails` |
| `MODEL-01` | P0 | 33/140 | Task 根据 gpus 与 CUDA 可用性主动把网络移到 GPU | `fix(device): make Trainer the sole device authority` |
| `TRAIN-06` | P0 | 32/140 | Pipeline 02 在运行前补齐多项 trainer 属性 | `fix(pipeline02): stop repairing stage trainer config` |
| `MODEL-02` | P0 | 30/140 | Trainer 将所有非 cpu 设备请求转换为 auto | `fix(trainer): honor explicit device selection` |
| `MODEL-04` | P0 | 28/140 | HSE 在 patch 过大时重复时间和通道 | `fix(hse): reject patches larger than input` |
| `DATA-04` | P0 | 23/140 | cache 可复用性只看 ID，不看 reader 语义 | `fix(data): invalidate cache on reader-semantic changes` |
| `PIPE-02` | P1 | 23/140 | 当前所有 maintained demo 仍为 smoke_only，没有 baseline_valid | `test(baseline): promote one scientifically closed experiment` |
| `DATA-11` | P0 | 22/140 | 不可满足的 target_domain_num 被自动缩减或退化为 train-only | `fix(protocol): fail on impossible target-domain requests` |
| `MODEL-03` | P0 | 20/140 | HSE 在 eval 模式仍随机选择 patch | `fix(hse): make evaluation patch selection deterministic` |
| `MODEL-05` | P0 | 20/140 | HSE contrastive 在 validation/test 中仍生成随机增强视图 | `fix(hse-task): make validation augmentation deterministic` |
| `TASK-02` | P0 | 20/140 | 指标 data_name 只取 batch 第一个 file_id | `fix(task): enforce dataset-homogeneous metric batches` |
| `TASK-05` | P0 | 19/140 | regularization 设备探测会消耗参数生成器的第一个参数 | `fix(loss): include every parameter in regularization` |
| `CFG-01` | P0 | 18/140 | 显式实验配置缺少 pipeline 时被静默改写为 Pipeline 01 | `fix(config): reject missing pipeline in explicit experiments` |
| `TASK-01` | P0 | 18/140 | Default_task 自动把缺失 task_id 补为 classification | `fix(task): remove implicit classification task_id` |
| `TASK-04` | P0 | 18/140 | 类别数由 max(Label)+1 推断但未验证标签连续性 | `fix(labels): validate contiguous classification ontology` |
| `TASK-06` | P0 | 13/140 | 未知 regularization 类型被警告后跳过 | `fix(loss): reject unsupported regularization` |
| `DATA-06` | P0 | 12/140 | Same_system_Sampler 静默跳过缺 metadata 的样本 | `fix(sampler): preserve selected sample coverage` |
| `DATA-08` | P0 | 12/140 | 未知 task type 会退化为 train=test=all IDs | `fix(protocol): reject unknown task types` |
| `ARCH-01` | P1 | 12/140 | 共享 classification runtime 特判 Multitask 并修改 Data/Model 配置 | `refactor(runtime): remove task-specific config mutation` |
| `CFG-04` | P1 | 12/140 | 公共运行成功受 hash/attestation/evidence 后处理支配 | `cleanup(runtime): remove governance from experiment success` |
| `TASK-07` | P1 | 12/140 | window-level global micro estimator 尚未被代码显式固定 | `test(metrics): freeze window-level micro estimator` |
| `TRAIN-05` | P1 | 12/140 | Pipeline 02 资源清理不在 finally，且 data 未关闭 | `fix(pipeline02): close resources through finally` |
| `UX-01` | P1 | 12/140 | 普通结果、run attestation 和 artifacts manifest 形成多套结果入口 | `refactor(outputs): simplify user-visible run results` |
| `DATA-02` | P0 | 11/140 | 通用 metadata reader 猜测 delimiter 与 encoding | `fix(data): make metadata format explicit` |
| `TRAIN-01` | P0 | 11/140 | Default Trainer 自动补 epochs、gpus 与 pruning | `fix(trainer): require explicit training budget and device` |
| `DATA-07` | P0 | 9/140 | 训练 drop_last=True 可能让小 system 完全消失 | `fix(sampler): prevent whole-system drop_last loss` |
| `CFG-02` | P1 | 9/140 | 非法 override 值解析失败后被当作普通字符串继续执行 | `fix(config): fail on malformed override values` |
| `GOV-01` | P1 | 8/140 | Pipeline 06 科学阶段完成后仍可能因缺 stage ledger 被判失败 | `cleanup(generative): decouple stage success from ledger` |
| `TASK-03` | P0 | 7/140 | 未知 metric 仅打印 warning 并跳过 | `fix(metrics): reject unsupported metric names` |
| `MODEL-07` | P1 | 6/140 | Model Factory 在构造过程中修改 args_model.num_classes | `refactor(model): make class-count input explicit` |
| `TRAIN-02` | P1 | 6/140 | Default Trainer 无条件追加 artifact manifest callback | `cleanup(trainer): remove default artifact manifest callback` |
| `DATA-10` | P0 | 5/140 | 非法或缺失监督标签被静默过滤 | `fix(data): reject invalid supervised labels` |
| `DATA-03` | P1 | 5/140 | reader 输出先自动扩维，dataset 又自动展平 | `fix(data): enforce explicit L-C reader shape` |
| `PIPE-01` | P1 | 5/140 | 部分 compatibility/experimental Pipeline 默认可运行，无显式 opt-in | `fix(pipelines): align public access with maturity` |
| `PR-01` | P1 | 5/140 | 开放 PR #147 与 no-ledger/no-hash 原则冲突且基线过旧 | `close/rebuild: population-aware CFM on latest dev` |
| `PR-02` | P1 | 14/140 | 开放 PR #148 治理范围重且基线过旧 | `defer/rebase: optional data backend after baseline` |

# Accepted critical path

1. `fix(config): reject missing pipeline and malformed overrides`.
2. `fix(protocol): reject unknown tasks, invalid labels and impossible target domains`.
3. `fix(device/hse): establish Trainer-only device authority, deterministic eval and shape fail-fast`.
4. `fix(task/metrics): include every objective parameter, reject unknown metrics, freeze estimator semantics`.
5. `fix(pipeline02): fail when stage evaluation fails and close resources` — parallel P1 if the first baseline uses Pipeline 01.
6. `test(baseline): promote one real ordinary-classification experiment to baseline_valid`.

# Factory responsibility decision

| Concern | Final owner |
|---|---|
| raw selection / reader / selected IDs / loaders | Data Factory |
| model identity / construction / explicit external weights | Model Factory |
| task identity / objective / metric lifecycle | Task Factory |
| device / checkpoint callbacks / logger lifecycle | Trainer Factory |
| orchestration / success gating / user result path | Runtime/Pipeline |

# Work-in-progress limit

```text
Critical scientific PR: 1
Independent correctness PR: 1
Total implementation PRs in flight: <= 2
```

# Do not build

```text
PluginSpec
ComponentSpec
ReaderPlugin
FactoryManager
UniversalContext
UniversalBatch
second registry
second schema system
hash/evidence/ledger replacement framework
```

# Real-data sequence

```text
Dummy + CSV fixtures
→ core semantics and 2×2 Data×Model diagnosis

MFPT
→ minimal single-channel portability

SEU
→ multi-channel and condition/domain protocol

PU
→ current + vibration channel contract

CWRU
→ final local-agent baseline_valid acceptance
```

# Promotion rule

`protocol_status: smoke_only → baseline_valid` is allowed only in the same PR that contains or references:

```text
canonical config
reader semantics test
split requirement test
deterministic evaluation test
exact metric estimator test
real end-to-end run
best-checkpoint evaluation
non-empty finite metrics
claim text matching evaluation unit
```

# Limitations

The review environment inspected GitHub content and PR metadata but did not execute external-data experiments. Runtime-only and dataset-only numerical claims remain explicitly unverified in the group dossiers.

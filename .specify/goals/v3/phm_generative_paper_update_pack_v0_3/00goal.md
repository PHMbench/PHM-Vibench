总控 Prompt：先给本地 Agent
你正在 PHMbench/PHM-Vibench 的 `Feature_factory-update` 分支上工作。

当前任务不是继续添加新生成模型，而是把现有 PHM-GenBench v0.3 的 train/sample/eval/paperpack/submission 证据链收敛到可审计、可复现、可 review 的论文级状态。

必须遵守：

1. 维护主路径只能是：
   `python main.py --config <yaml> [--override key=value ...]`

2. 不允许大重构。
3. 不允许一次完成多个互相独立的目标。
4. 不允许破坏 environment/data/model/task/trainer 五段式 config。
5. 不允许把 exploratory 方法包装成 benchmark-valid。
6. CFM / Rectified Flow / DDPM 是核心 baseline。
7. Score SDE / MeanFlow / Drifting / TFM / OT-NFM 保持 exploratory。
8. benchmark-valid 必须由以下证据共同证明：
   - sample synthetic_data_manifest.json
   - eval_evidence_manifest.json
   - stage_ledger.json
   - paperpack traceability
   - reviewer gate
9. 所有失败必须 fail fast，不能 silent fallback。
10. 每个 goal 完成后必须运行 validation commands。
11. 每个 goal 完成后必须用：
   `.specify/goals/v3/phm_generative_paper_update_pack_v0_3/14_reviewer.md`
   作为 review rubric，或在本地等价创建该 review rubric。
12. 如果 reviewer 输出 BLOCKING item，必须转化为新的小 `/goal`，不得直接进入真实六数据集 paper claim。

当前分支已具备：
- `main.py` 中已有 pipeline whitelist、YAML probe、Pydantic preflight、`--preflight-only`。:contentReference[oaicite:1]{index=1} :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}
- `Pipeline_06_generative.py` 已有 train/sample/eval 三模式。:contentReference[oaicite:4]{index=4}
- sample 阶段会写 samples payload 和 synthetic manifest。:contentReference[oaicite:5]{index=5}
- eval 阶段会写 `generative_eval_metrics.csv`。:contentReference[oaicite:6]{index=6}
- task registry 已注册 CFM、Rectified Flow、DDPM、Score SDE、MeanFlow、Drifting、TFM、OT-NFM。:contentReference[oaicite:7]{index=7}
- synthetic manifest 已有 evidence gating 和 benchmark-valid downgrade。:contentReference[oaicite:8]{index=8}
- paperpack 和 submission draft 脚本已存在，但需要 stage ledger / eval evidence / reviewer gate 串联。:contentReference[oaicite:9]{index=9} :contentReference[oaicite:10]{index=10}

请按下面 goal 顺序执行。每次只做一个 goal。
GOAL-V3-000：固化 Reviewer Gate
/goal

## Goal ID
GOAL-V3-000-REVIEWER-GATE

## Objective
把 PHM-GenBench v0.3 的 paper-readiness reviewer 固化进分支，并对当前分支状态执行一次 baseline review。

## Why
后续每个 runtime fix、metric fix、paperpack fix 都必须被统一 reviewer gate 检查。否则 agent 容易只看测试通过，不看论文证据链是否完整。

## Current facts
- 当前分支已有 `.specify/goals/v2/` 和 `specs/002-phm-genbench-frontier/` 过程产物。
- 当前分支已有 generative pipeline、manifest、metrics、paperpack 和 submission draft。
- paper-ready 的关键不是继续加模型，而是确认 evidence chain 是否闭环。

## Scope
允许新增或修改：
- `.specify/goals/v3/phm_generative_paper_update_pack_v0_3/14_reviewer.md`
- `specs/002-phm-genbench-frontier/reviews/v3/baseline-review.md`
- `specs/002-phm-genbench-frontier/reviews/v3/scorecard.csv`
- `specs/002-phm-genbench-frontier/reviews/v3/blocking_backlog.md`

## Out of scope
- 不修改 runtime 代码。
- 不修改模型、loss、task、pipeline。
- 不运行真实训练。
- 不把任何结果标记为 submission-ready。

## Required behavior
1. 如果 `14_reviewer.md` 已存在，直接使用它。
2. 如果不存在，在 `.specify/goals/v3/phm_generative_paper_update_pack_v0_3/14_reviewer.md` 创建 reviewer rubric。
3. Reviewer rubric 必须检查：
   - main config-first 主路径
   - train/sample/eval/paperpack stage traceability
   - synthetic manifest completeness
   - eval evidence manifest
   - metric naming
   - condition split evidence
   - leakage guard
   - benchmark-valid gating
   - paperpack source traceability
   - submission draft readiness
4. 对当前 repo 执行 baseline review。
5. 输出：
   - decision: PASS / PASS_WITH_WARNINGS / BLOCKED
   - readiness_score: 0-100
   - scorecard
   - blocking issues
   - non-blocking issues
   - metric gap matrix
   - evidence matrix
   - validation commands
   - Codex-ready backlog

## Deliverables
- `.specify/goals/v3/phm_generative_paper_update_pack_v0_3/14_reviewer.md`
- `specs/002-phm-genbench-frontier/reviews/v3/baseline-review.md`
- `specs/002-phm-genbench-frontier/reviews/v3/scorecard.csv`
- `specs/002-phm-genbench-frontier/reviews/v3/blocking_backlog.md`

## Acceptance criteria
- Review 文件必须明确当前是否能进入真实六数据集 run。
- Review 文件必须明确哪些问题是 BLOCKING。
- 所有 BLOCKING 问题都必须能映射到一个小 `/goal`。
- 不允许出现“后续再说”“需要人工判断”但不形成 backlog 的条目。

## Validation commands
python -m scripts.validate_docs
python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml --preflight-only

## Failure handling
如果 `scripts.validate_docs` 失败，不要修改 runtime；先把 docs failure 写入 review 的 blocking issues。

## Review checklist
- Reviewer rubric 是否能阻止不完整 evidence 被写成 benchmark-valid？
- 是否明确区分 exploratory 与 benchmark-valid？
- 是否输出 Codex-ready backlog？
GOAL-V3-001：Stage Ledger 串联 train/sample/eval/paperpack
/goal

## Goal ID
GOAL-V3-001-STAGE-LEDGER

## Objective
实现 `stage_ledger.json`，把 train/sample/eval/paperpack 的关键 artifact 串成可审计链路。

## Why
当前 train/sample/eval/paperpack 是分阶段运行。sample manifest 和 eval metrics 可能位于不同 run dir；paperpack 如果只扫描当前目录，可能找不到 sibling sample manifest。必须用 stage ledger 显式记录 artifact 路径。

## Current facts
- `Pipeline_06_generative.py` 已有 train/sample/eval 三个 mode。:contentReference[oaicite:11]{index=11}
- sample 阶段会写 `samples.pt` 和 `synthetic_data_manifest.json`。:contentReference[oaicite:12]{index=12}
- eval 阶段会写 `generative_eval_metrics.csv`。:contentReference[oaicite:13]{index=13}
- `paperpack_generative.py` 当前从 run_dir 搜索 manifests 和 metrics。:contentReference[oaicite:14]{index=14}

## Scope
允许新增：
- `src/task_factory/Components/generative/manifests/stage_ledger.py`
- `test/generative/test_stage_ledger.py`

允许修改：
- `src/Pipeline_06_generative.py`
- `scripts/paperpack_generative.py`
- `scripts/generative_benchmark_effect.py`
- 相关 README 或 docs

## Out of scope
- 不新增新模型。
- 不改 CFM/DDPM/RF loss 数学定义。
- 不运行真实六数据集训练。
- 不改 `main.py` 主入口语义。

## Required behavior
1. 每个 stage 写入或更新同一个 `stage_ledger.json`。
2. ledger 至少包含：
   - benchmark_id
   - dataset
   - method
   - seed
   - stage
   - config_path
   - output_dir
   - created_at
   - status
3. train stage 记录：
   - train_result_path
   - checkpoint_dir
   - best_checkpoint_path if available
   - normalization_params_path
4. sample stage 记录：
   - samples_path
   - synthetic_manifest_path
   - condition_counts
   - sampler_id
   - num_steps
5. eval stage 记录：
   - metrics_path
   - eval_evidence_manifest_path if available
   - eval_split
6. paperpack stage 记录：
   - paperpack_dir
   - tables paths
   - figure_sources paths
   - reproducibility_statement path
7. `paperpack_generative.py` 增加参数：
   - `--stage_ledger <path>`
8. 如果传入 `--stage_ledger`，paperpack 必须优先从 ledger 解析 sample manifest 和 eval metrics，不只依赖 rglob。
9. `scripts/generative_benchmark_effect.py --dry-run` 生成的 run plan 中应包含 stage ledger 预期路径或 ledger policy。

## Deliverables
- `src/task_factory/Components/generative/manifests/stage_ledger.py`
- `test/generative/test_stage_ledger.py`
- 更新后的 `src/Pipeline_06_generative.py`
- 更新后的 `scripts/paperpack_generative.py`
- 更新后的 dry-run plan 或 docs

## Acceptance criteria
- train/sample/eval 三个 stage 都能写 ledger。
- paperpack 能通过 `--stage_ledger` 找到 sibling sample manifest。
- ledger 缺失关键文件时，paperpack 必须报清晰错误或写 missing evidence，不允许 silent skip。
- test fixture 能证明：
  - sample manifest 在 sibling sample dir
  - eval metrics 在 eval dir
  - paperpack 仍能串联二者

## Validation commands
python -m pytest test/generative/test_stage_ledger.py
python -m pytest test/generative/test_paperpack_generative.py
python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml --preflight-only
python -m scripts.validate_docs

## Failure handling
如果无法定位 checkpoint 或 samples，不要写假的路径。ledger 字段应记录：
`status: missing`
`reason: <exact reason>`

## Review checklist
- stage ledger 是否覆盖 train/sample/eval/paperpack？
- paperpack 是否不再依赖脆弱的 `<experiment_name>` placeholder？
- ledger 是否能作为 submission evidence 的 source index？
GOAL-V3-002：Eval Evidence Manifest
/goal

## Goal ID
GOAL-V3-002-EVAL-EVIDENCE-MANIFEST

## Objective
在 eval 阶段生成 `eval_evidence_manifest.json`，把 metrics、sample manifest、reference split、label/domain evidence、missing metric reasons 绑定在一起。

## Why
sample manifest 只能证明样本如何生成，不能证明 eval 指标是否可计算、是否用错 split、是否有 label/domain 条件证据。benchmark-valid 必须由 sample manifest + eval evidence 共同判断。

## Current facts
- synthetic manifest 当前会检查 protocol/config/dependency/normalization/leakage/condition/metric status evidence。:contentReference[oaicite:15]{index=15}
- eval 阶段当前写 `generative_eval_metrics.csv`，但还没有独立 eval evidence sidecar。:contentReference[oaicite:16]{index=16}
- `evaluate_generated_windows()` 已经为不可计算指标写 status/reason。:contentReference[oaicite:17]{index=17}

## Scope
允许新增：
- `src/task_factory/Components/generative/manifests/eval_evidence_manifest.py`
- `test/generative/test_eval_evidence_manifest.py`
- `templates/eval_evidence_manifest.schema.json` 或 `docs/schemas/eval_evidence_manifest.schema.json`

允许修改：
- `src/Pipeline_06_generative.py`
- `scripts/paperpack_generative.py`
- `scripts/generative_benchmark_effect.py`

## Out of scope
- 不实现 full downstream classifier TSTR。
- 不修改核心 loss。
- 不把 exploratory run 升级为 benchmark-valid。

## Required behavior
1. eval 阶段在 `generative_eval_metrics.csv` 同目录写：
   `eval_evidence_manifest.json`
2. manifest 至少包含：
   - schema_version
   - generated_path
   - synthetic_manifest_path if available
   - metrics_path
   - eval_split
   - allow_test_reference_eval
   - real_shape
   - fake_shape
   - real_label_available
   - fake_label_available
   - real_domain_available
   - fake_domain_available
   - metric_status_summary
   - missing_metric_reasons
   - leakage_metric_keys
   - utility_metric_keys
   - quality_metric_keys
3. 如果 eval_split 是 test 或 target_test 且 `allow_test_reference_eval` 不是 true，必须 fail fast。
4. 如果 metrics 有 NaN/Inf，必须在 manifest 中记录原因。
5. paperpack 必须收集 eval evidence manifest，并输出到 appendix。
6. benchmark-effect aggregation 必须能读取 eval evidence manifest 的 missing reasons。

## Deliverables
- `eval_evidence_manifest.py`
- `eval_evidence_manifest.schema.json`
- `test/generative/test_eval_evidence_manifest.py`
- 更新 pipeline eval 阶段
- 更新 paperpack appendix 输出

## Acceptance criteria
- eval 后必有 `eval_evidence_manifest.json`。
- manifest 能说明每个 missing metric 的原因。
- test split 未授权时 eval 失败。
- paperpack 能列出 eval evidence manifest path。
- sample manifest 仍然只负责 generation provenance，不承担 eval evidence。

## Validation commands
python -m pytest test/generative/test_eval_evidence_manifest.py
python -m pytest test/generative/test_generative_metrics.py
python -m pytest test/generative/test_paperpack_generative.py
python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml --preflight-only

## Failure handling
如果 generated_path 不存在，直接 ValueError，不能写空 metrics。
如果 sample manifest 找不到，eval 可以继续 exploratory，但必须在 eval_evidence_manifest 中写：
`synthetic_manifest_status: missing`

## Review checklist
- eval evidence 是否能独立解释 metric validity？
- 是否防止 test split 被默认使用？
- 是否明确区分 quality / utility / leakage / efficiency？
GOAL-V3-004：Condition Split Evidence
/goal

## Goal ID
GOAL-V3-004-CONDITION-SPLIT-EVIDENCE

## Objective
为 `condition_sampling_policy=train_distribution` 增加 split evidence；没有 train split 证据时不得 benchmark-valid。

## Why
train_distribution 如果没有真实 train split 标记，可能从全部 metadata 中采样 condition，导致 val/test 条件泄漏。生成样本的 condition provenance 必须可审计。

## Current facts
- Pipeline 支持 condition policies：`first_metadata_repeated / grid / train_distribution / explicit`。:contentReference[oaicite:18]{index=18}
- `_metadata_condition_pairs(metadata, split="train")` 当前会尝试读取 split 字段，但如果 metadata 中没有 split 字段，可能无法证明 train-only。:contentReference[oaicite:19]{index=19}
- synthetic manifest 已记录 `condition_sampling_policy` 和 `condition_counts`。:contentReference[oaicite:20]{index=20}

## Scope
允许修改：
- `src/Pipeline_06_generative.py`
- `src/task_factory/Components/generative/manifests/synthetic_data_manifest.py`
- `test/generative/test_condition_sampling.py`

允许新增：
- `src/task_factory/Components/generative/manifests/condition_evidence.py`

## Out of scope
- 不改 data factory split 逻辑。
- 不重做 metadata schema。
- 不修改非 generative pipeline。

## Required behavior
1. `train_distribution` 必须返回 condition evidence：
   - requested_policy
   - effective_policy
   - split_requested
   - split_field_detected
   - split_verified
   - total_candidate_pairs
   - sampled_pairs_count
   - condition_counts
2. 如果 policy 是 `train_distribution`，但 metadata 中没有 split 字段：
   - sample 可以继续 exploratory
   - `split_verified=false`
   - manifest missing_evidence 中必须包含 `condition_split_evidence`
   - status 不得 benchmark-valid
3. `grid` 和 `explicit` 也要记录 condition evidence，但不强制 split_verified。
4. synthetic manifest evidence gate 增加：
   `condition_split_evidence`
5. 如果用户请求 benchmark-valid 但 condition split 未验证，必须 downgrade exploratory。

## Deliverables
- `condition_evidence.py`
- 更新 `Pipeline_06_generative.py`
- 更新 `synthetic_data_manifest.py`
- `test/generative/test_condition_sampling.py`

## Acceptance criteria
- train_distribution + metadata split=train → split_verified=true。
- train_distribution + no split field → split_verified=false 且 exploratory。
- grid policy 能记录 condition_counts。
- explicit policy 能记录每个 condition count。
- benchmark-valid 不可能在 condition_split_evidence 缺失时通过。

## Validation commands
python -m pytest test/generative/test_condition_sampling.py
python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml --preflight-only

## Failure handling
如果 condition policy 未知，必须 ValueError。
如果 explicit condition 为空，必须 ValueError。
如果 condition count 与 num_samples 不一致，必须 ValueError。

## Review checklist
- condition evidence 是否进入 manifest？
- train_distribution 是否真的证明只来自 train split？
- 没有 split evidence 时是否只能 exploratory？
GOAL-V3-005：严格化 Pipeline _to_ncl()
/goal

## Goal ID
GOAL-V3-005-PIPELINE-TO-NCL-STRICT

## Objective
修复 `Pipeline_06_generative._to_ncl()`：无法根据 expected channels 判断 `[N,C,L]` 或 `[N,L,C]` 时必须 fail fast。

## Why
生成模型评估和 normalization artifact 都依赖 `[N,C,L]` shape。如果 pipeline 层在 ambiguous shape 下直接返回原 tensor，会导致 metrics 和 manifest 使用错误 shape 而不报错。

## Current facts
- task 里的 `_to_ncl()` 在无法推断 channel 轴时会报错。:contentReference[oaicite:21]{index=21}
- pipeline 里的 `_to_ncl()` 当前如果两边都不等于 channels，会直接返回 `x.contiguous()`。:contentReference[oaicite:22]{index=22}

## Scope
允许修改：
- `src/Pipeline_06_generative.py`
- `test/generative/test_pipeline_to_ncl.py`

## Out of scope
- 不改 data reader。
- 不改 task `_to_ncl()`。
- 不改 model forward。
- 不修改 metrics 数学定义。

## Required behavior
`_to_ncl(x, channels)` 必须：
1. 接受 rank-3 tensor。
2. 如果 `x.shape[1] == channels`，返回 `[N,C,L]`。
3. 如果 `x.shape[2] == channels`，transpose 为 `[N,C,L]`。
4. 如果两者都不匹配，raise ValueError。
5. 错误信息必须包含：
   - actual shape
   - expected channels
   - accepted formats `[N,C,L]` and `[N,L,C]`

## Deliverables
- 更新 `Pipeline_06_generative.py`
- 新增 `test/generative/test_pipeline_to_ncl.py`

## Acceptance criteria
- `[N,C,L]` 通过。
- `[N,L,C]` 通过并 transpose。
- `[N,L,D]` 且 D/C 不匹配时报错。
- rank-2 / rank-4 tensor 报错。
- normalization artifact 和 eval 调用使用同一个严格函数。

## Validation commands
python -m pytest test/generative/test_pipeline_to_ncl.py
python -m pytest test/generative/test_eval_evidence_manifest.py
python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml --preflight-only

## Failure handling
不允许 fallback 到原 shape。
不允许只 print warning。

## Review checklist
- 是否完全消除 ambiguous shape？
- 错误信息是否足够定位数据问题？
- 是否没有影响 task 层已有 shape contract？
GOAL-V3-003：TSTR/TRTS 指标重命名为 Probe
/goal

## Goal ID
GOAL-V3-003-TSTR-PROBE-RENAME

## Objective
将当前 nearest-centroid 的 TSTR/TRTS 指标重命名为 probe 指标，避免论文中误解为 full downstream classifier TSTR。

## Why
当前 `tstr_metrics()` 使用 nearest-centroid probe，不是完整下游分类器训练。指标名如果仍叫 `tstr_accuracy`，论文表格容易误导审稿人。

## Current facts
- `tstr_metrics()` 当前实现为 nearest-centroid accuracy。:contentReference[oaicite:23]{index=23}
- 函数注释中说明 TSTR/TRTS 是 nearest-centroid probes。:contentReference[oaicite:24]{index=24}

## Scope
允许修改：
- `src/task_factory/Components/generative/metrics/tstr.py`
- `src/task_factory/task/generative/generative_eval.py`
- `scripts/paperpack_generative.py`
- `scripts/generative_benchmark_effect.py`
- `scripts/generative_submission_draft.py`
- `test/generative/test_generative_metrics.py`
- `test/generative/test_paperpack_generative.py`
- `test/generative/test_benchmark_effect.py`

## Out of scope
- 不实现 full downstream classifier TSTR。
- 不新增 sklearn classifier。
- 不新增训练 pipeline。
- 不改变 nearest-centroid probe 的计算方式。

## Required behavior
1. 新指标名：
   - `tstr_nearest_centroid_accuracy`
   - `trts_nearest_centroid_accuracy`
   - `tstr_probe_status_code`
2. 可选兼容 alias：
   - `tstr_accuracy`
   - `trts_accuracy`
   但如果保留 alias，必须输出：
   - `tstr_accuracy_status = deprecated_alias`
   - `tstr_accuracy_reason = use tstr_nearest_centroid_accuracy`
3. paperpack utility prefix 必须捕获：
   - `tstr_nearest_centroid_`
   - `trts_nearest_centroid_`
   - `utility_`
4. benchmark-effect utility prefix 必须同步更新。
5. submission draft 必须写：
   `nearest-centroid TSTR/TRTS probe`
   不得写 full downstream classifier TSTR。
6. missing metric reason 必须仍然保留。

## Deliverables
- 更新 metric 文件
- 更新 paperpack 脚本
- 更新 benchmark-effect 脚本
- 更新 submission draft 文案
- 更新相关测试

## Acceptance criteria
- metrics CSV 中出现新 probe 名称。
- paperpack table_utility 能收集新 probe 指标。
- benchmark-effect summary 能聚合新 probe 指标。
- submission draft 不再把 probe 写成 full TSTR。
- 兼容 alias 若存在，必须被明确标记 deprecated。
- 测试证明没有 paper table 混淆 probe 与 full classifier。

## Validation commands
python -m pytest test/generative/test_generative_metrics.py
python -m pytest test/generative/test_paperpack_generative.py
python -m pytest test/generative/test_benchmark_effect.py
python -m scripts.validate_docs

## Failure handling
如果旧测试依赖 `tstr_accuracy`，不要简单保留旧名蒙混通过；必须更新测试语义，明确 probe naming。

## Review checklist
- 表格中是否能看出这是 nearest-centroid probe？
- 是否没有任何地方声称 full downstream classifier TSTR？
- paperpack / benchmark-effect / submission draft 是否同步？
GOAL-V3-006：六数据集 Matrix Dry-run 与 Blocked Ledger 测试
/goal

## Goal ID
GOAL-V3-006-PAPER-MATRIX-DRYRUN-TESTS

## Objective
为六数据集 benchmark matrix 的 dry-run、missing metadata failure、stage coverage、blocked ledger、baseline method existence 补齐测试。

## Why
真实 GPU 六数据集 run 成本高，必须先证明 matrix planner 生成的命令、stage、ledger、blocked 状态都是可审计的。dry-run 只证明 pipeline readiness，不产生论文 claim。

## Current facts
- `six_dataset_benchmark_matrix.yaml` 已定义 6 个数据集、GPU 6/7、seeds [0,1]、CFM/RF/DDPM 三方法。:contentReference[oaicite:25]{index=25} :contentReference[oaicite:26]{index=26}
- `generative_benchmark_effect.py` 支持 dry-run 和 from-runs 聚合。:contentReference[oaicite:27]{index=27}
- run plan 中包含 train/sample/eval/paperpack stage。:contentReference[oaicite:28]{index=28}

## Scope
允许修改：
- `scripts/generative_benchmark_effect.py`
- `configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml` 仅限必要字段修正
- `test/generative/test_benchmark_effect.py`

允许新增：
- `test/fixtures/generative/benchmark_matrix_minimal.yaml`
- `test/fixtures/generative/benchmark_matrix_missing_metadata.yaml`
- `test/fixtures/generative/benchmark_matrix_bad_baseline.yaml`

## Out of scope
- 不运行真实训练。
- 不要求真实 metadata 存在，除非测试 strict failure。
- 不改生成模型。
- 不改 pipeline train/sample/eval 语义。

## Required behavior
1. `--dry-run --allow-missing-data` 必须输出：
   - run_plan.csv
   - run_status_ledger.csv
   - benchmark_effect_manifest.json
2. run_plan 必须覆盖：
   - train
   - sample
   - eval
   - paperpack
3. 每个 dataset/method/seed 组合必须有完整 stage 计划。
4. baseline_method 必须存在于 methods，否则 fail fast。
5. strict mode 下 metadata 缺失必须失败。
6. allow-missing-data 模式下 metadata 缺失不失败，但必须写 blocked ledger。
7. blocked ledger 必须记录：
   - dataset
   - method
   - seed
   - planned_stages
   - status
   - reason
8. run plan 不允许把 exploratory method 标成 benchmark-valid。
9. matrix manifest 必须记录：
   - configured dataset count
   - observed configured dataset count
   - min_datasets
   - missing_datasets
   - unexpected_datasets
   - input_gaps

## Deliverables
- 更新 `scripts/generative_benchmark_effect.py`
- 新增或更新 `test/generative/test_benchmark_effect.py`
- 新增 fixtures
- 若必要，更新 matrix README

## Acceptance criteria
- dry-run 可以在无真实数据环境下通过。
- strict missing metadata 测试必须失败。
- bad baseline 测试必须失败。
- run_plan stage coverage 完整。
- blocked ledger 可解释为什么不能真实 run。
- 不产生任何 submission-ready claim。

## Validation commands
python -m pytest test/generative/test_benchmark_effect.py
python -m scripts.generative_benchmark_effect --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml --dry-run --allow-missing-data
python -m scripts.validate_docs

## Failure handling
如果 matrix 文件字段缺失，必须输出具体缺失字段。
如果 baseline method 不存在，必须 ValueError。
如果 strict mode 找不到 metadata，必须非零退出。

## Review checklist
- dry-run 是否足以在无 GPU/无数据环境验证命令队列？
- blocked ledger 是否能阻止误报 paper readiness？
- stage 数量是否等于 datasets × methods × seeds × 4？
GOAL-V3-007：Reviewer-driven Closure
/goal

## Goal ID
GOAL-V3-007-REVIEWER-DRIVEN-CLOSURE

## Objective
在 V3-001..V3-006 完成后，重新运行 v0.3 reviewer gate，生成 closure scorecard、blocking backlog 和 paper-readiness decision。

## Why
前六个 goal 修的是 evidence-chain blocker。修完后必须统一复核，确认是否可以进入真实六数据集 GPU run。不能只凭测试通过就进入 paper claim。

## Dependencies
- GOAL-V3-000
- GOAL-V3-001
- GOAL-V3-002
- GOAL-V3-003
- GOAL-V3-004
- GOAL-V3-005
- GOAL-V3-006

## Scope
允许新增：
- `specs/002-phm-genbench-frontier/reviews/v3/closure-review.md`
- `specs/002-phm-genbench-frontier/reviews/v3/closure-scorecard.csv`
- `specs/002-phm-genbench-frontier/reviews/v3/closure-backlog.md`
- `specs/002-phm-genbench-frontier/reviews/v3/paper-readiness-decision.md`

允许修改：
- `.specify/goals/v3/phm_generative_paper_update_pack_v0_3/14_reviewer.md` 仅限补充 reviewer rubric，不允许降低门槛。

## Out of scope
- 不改 runtime。
- 不新增模型。
- 不运行真实 GPU 六数据集训练。
- 不把 draft 改成 submission-ready。

## Required behavior
1. 用 reviewer rubric 检查当前分支。
2. 必须输出：
   - decision
   - readiness_score
   - stage_ledger_status
   - eval_evidence_status
   - condition_split_status
   - metric_naming_status
   - paper_matrix_dryrun_status
   - benchmark_valid_gating_status
   - remaining_blockers
3. 如果仍有 BLOCKING：
   - 写入 `closure-backlog.md`
   - 每个 blocking item 必须有一个新小 goal 草案
4. 如果没有 BLOCKING：
   - 写明可以进入 `GOAL-V3-008-REAL-SIX-DATASET-RUN`
   - 但仍不能标记 submission-ready，直到真实 run evidence 完成。

## Deliverables
- closure review
- scorecard
- backlog
- readiness decision

## Acceptance criteria
- reviewer 不再报告以下 blocker：
  - missing stage ledger
  - missing eval evidence manifest
  - ambiguous TSTR naming
  - missing train_distribution split evidence
  - permissive pipeline `_to_ncl`
  - missing dry-run matrix tests
- 如果 reviewer 报告新 blocker，必须写出可执行 goal。
- 决策必须是：
  - BLOCKED
  - READY_FOR_REAL_RUN
  - READY_FOR_PAPER_DRAFT
  三者之一。

## Validation commands
python -m scripts.validate_docs
python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml --preflight-only
python -m scripts.generative_benchmark_effect --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml --dry-run --allow-missing-data

## Failure handling
如果 reviewer 文件不存在，返回 BLOCKED。
如果 review 输出缺 section，返回 BLOCKED。
如果 validation 命令失败，返回 BLOCKED。

## Review checklist
- closure review 是否足以作为进入真实 run 的准入门？
- 是否没有隐藏 blocker？
- 是否明确下一步是否能使用 GPU/data？
GOAL-V3-008：真实六数据集 Run 队列
/goal

## Goal ID
GOAL-V3-008-REAL-SIX-DATASET-RUN

## Objective
在 reviewer gate 判定 READY_FOR_REAL_RUN 后，执行六数据集真实 train/sample/eval/paperpack 队列。

## Why
代码 smoke/dry-run 只能证明 pipeline readiness，不能支撑论文 claim。论文必须有真实六数据集、真实 GPU、真实 metric、真实 manifest、真实 paperpack evidence。

## Dependencies
- GOAL-V3-007 must output READY_FOR_REAL_RUN

## Scope
允许执行：
- preflight
- dry-run
- real train/sample/eval/paperpack commands
- run status ledger update
- paperpack generation

允许写入：
- `results/paper/phm_generative/six_dataset_submission_v1/`
- `specs/002-phm-genbench-frontier/reviews/codex/<date>-real-run-log.md`
- `specs/002-phm-genbench-frontier/reviews/codex/<date>-run-status-ledger.csv`

## Out of scope
- 不修改模型实现。
- 不修改 loss。
- 不修改 paper draft claim。
- 不跳过失败 stage。
- 不手工伪造 checkpoint 或 metric。

## Required behavior
1. 先执行：
   - preflight
   - dry-run
   - data existence check
2. 真实 run 必须覆盖：
   - 6 datasets
   - 3 methods: CFM / Rectified Flow / DDPM
   - 2 seeds
   - 4 stages: train / sample / eval / paperpack
3. 每个 run 必须生成：
   - stage_ledger.json
   - synthetic_data_manifest.json
   - eval_evidence_manifest.json
   - generative_eval_metrics.csv
   - paperpack
4. 所有失败都必须写入 run status ledger。
5. 不允许把失败 run 从 summary 中删除。
6. 不允许用 test split 做 reference eval，除非 config 显式允许且 reviewer gate 认可。
7. 不允许 exploratory run 进入 benchmark-valid 主表。

## Deliverables
- run_plan.csv
- run_status_ledger.csv
- per-run stage_ledger.json
- per-run synthetic manifests
- per-run eval evidence manifests
- per-run paperpacks
- real-run log markdown

## Acceptance criteria
- 每个 dataset/method/seed 都有 status。
- 每个 failed run 有 reason。
- 每个 succeeded run 有完整 artifact paths。
- 至少生成 benchmark-effect 输入所需的 metrics 和 manifests。
- 没有 artifact path 是 `<experiment_name>` placeholder。

## Validation commands
python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml --preflight-only
python -m scripts.generative_benchmark_effect --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml --dry-run
python -m scripts.validate_docs

## Failure handling
如果 GPU 不可用，停止真实 run，写 BLOCKED。
如果数据路径不存在，停止真实 run，写 BLOCKED。
如果某一 run 失败，不删除结果，记录 failed 并继续或按 resource policy 停止。
如果 checkpoint 找不到，sample stage 必须失败并写 ledger。

## Review checklist
- 是否所有真实结果都有 manifest？
- 是否所有失败都被记录？
- 是否没有使用 dry-run 结果冒充真实结果？
GOAL-V3-009：Benchmark Effect 与 Submission Draft
/goal

## Goal ID
GOAL-V3-009-PAPER-EVIDENCE-PACKAGE

## Objective
基于真实六数据集 run outputs 生成 benchmark effect summary、manifest、submission draft、evidence package，并保持 submission readiness gate。

## Why
论文 draft 不能凭代码结构生成，必须由真实 run evidence 驱动。submission draft 只有在所有 configured dataset rows 都 benchmark-valid 且 quality + utility + leakage + efficiency evidence 完整时才能标记 SUBMISSION_READY。

## Dependencies
- GOAL-V3-008 must produce real run evidence

## Scope
允许执行或修改：
- `scripts/generative_benchmark_effect.py`
- `scripts/generative_submission_draft.py`
- `scripts/paperpack_generative.py`
- `specs/002-phm-genbench-frontier/paper/PAPER_DRAFT.md`
- `specs/002-phm-genbench-frontier/paper/evidence_gaps.md`
- `specs/002-phm-genbench-frontier/paper/submission_readiness.md`

允许新增：
- `results/paper/phm_generative/six_dataset_submission_v1/benchmark_effect_summary.csv`
- `results/paper/phm_generative/six_dataset_submission_v1/benchmark_effect_manifest.json`
- `results/paper/phm_generative/six_dataset_submission_v1/paper_evidence_package/`

## Out of scope
- 不手动补数值。
- 不删除失败 run。
- 不把 exploratory rows 写成 benchmark-valid。
- 不绕过 submission readiness gate。

## Required behavior
1. 从真实 run dirs 聚合：
   - quality
   - utility
   - leakage
   - efficiency
2. 输出：
   - benchmark_effect_summary.csv
   - benchmark_effect_manifest.json
3. manifest 必须包含：
   - configured datasets
   - observed datasets
   - missing datasets
   - unexpected datasets
   - min_datasets_met
   - input_gaps
   - benchmark-valid row count
   - exploratory row count
4. submission draft 必须：
   - 引用 summary 和 manifest
   - 不编造缺失结果
   - 不含 TODO/TBD/PLACEHOLDER
   - 若 evidence 不足，保持 NOT_SUBMISSION_READY
5. 生成 paper evidence package：
   - paper draft
   - evidence gaps
   - submission readiness
   - tables
   - figure sources
   - appendix run index
   - manifest completeness
   - missing metrics audit

## Deliverables
- benchmark effect summary
- benchmark effect manifest
- paper draft
- evidence gaps
- submission readiness
- paper evidence package

## Acceptance criteria
- 如果任一 configured dataset 缺 benchmark-valid quality + utility evidence，draft 必须 NOT_SUBMISSION_READY。
- 如果存在 exploratory rows，必须在 evidence gaps 中说明。
- 如果全部 evidence 完整，draft 才可 SUBMISSION_READY。
- 所有 numeric claim 必须能追溯到 metric_source_paths 和 manifest_paths。
- paper draft 不含 placeholder tokens。

## Validation commands
python -m scripts.generative_benchmark_effect --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml --from-runs results/paper/phm_generative/six_dataset_submission_v1/runs
python -m scripts.generative_submission_draft --summary results/paper/phm_generative/six_dataset_submission_v1/benchmark_effect_summary.csv --manifest results/paper/phm_generative/six_dataset_submission_v1/benchmark_effect_manifest.json --output specs/002-phm-genbench-frontier/paper/PAPER_DRAFT.md
python -m scripts.validate_docs

## Failure handling
如果 summary 不存在，draft generator 必须输出 NOT_SUBMISSION_READY。
如果 manifest 不存在，draft generator 必须输出 evidence gap。
如果 placeholder token 存在，draft generator 必须失败。

## Review checklist
- 是否所有结论都有 source path？
- 是否严格执行 submission readiness gate？
- 是否没有把 dry-run/smoke/exploratory 结果写成论文结果？
本地 Agent 执行顺序
GOAL-V3-000  固化 reviewer gate
GOAL-V3-001  stage ledger
GOAL-V3-002  eval evidence manifest
GOAL-V3-004  condition split evidence
GOAL-V3-005  strict _to_ncl
GOAL-V3-003  TSTR/TRTS probe rename
GOAL-V3-006  six-dataset matrix dry-run tests
GOAL-V3-007  reviewer-driven closure
GOAL-V3-008  real six-dataset run
GOAL-V3-009  paper evidence package
每轮执行后的强制 Review Gate 指令

每完成一个 goal，把下面这段给本地 agent 或 Claude Code：

请使用 `.specify/goals/v3/phm_generative_paper_update_pack_v0_3/14_reviewer.md` 作为唯一 review rubric，审查当前 repo。

必须输出：

1. Decision
   - PASS
   - PASS_WITH_WARNINGS
   - BLOCKED

2. Readiness score
   - 0-100

3. Scorecard
   - main config-first path
   - pipeline stage traceability
   - sample manifest
   - eval evidence
   - condition split evidence
   - metric naming
   - leakage guard
   - paperpack traceability
   - submission readiness gate
   - tests and validation commands

4. Blocking issues
   每个 blocking issue 必须包含：
   - issue
   - evidence file/path
   - risk
   - required fix
   - proposed next /goal

5. Non-blocking issues

6. Metric gap matrix

7. Evidence matrix

8. Validation commands actually run

9. Codex-ready backlog

规则：
- 只要存在 BLOCKING，就不得进入真实六数据集 paper run。
- 只要 sample manifest、eval evidence、stage ledger、condition split evidence 任一缺失，就不得 benchmark-valid。
- 只要 TSTR/TRTS 仍可能被误解为 full classifier TSTR，就不得进入 paper claim。
- 不得建议“继续加模型”作为解决 evidence-chain blocker 的方案。
最小验证命令组
python -m pytest test/generative/test_condition_sampling.py
python -m pytest test/generative/test_generative_metrics.py
python -m pytest test/generative/test_paperpack_generative.py
python -m pytest test/generative/test_benchmark_effect.py
python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml --preflight-only
python -m scripts.generative_benchmark_effect --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml --dry-run --allow-missing-data
python -m scripts.validate_docs
禁止事项
1. 禁止继续新增生成模型来掩盖 evidence blocker。
2. 禁止把 MeanFlow / Drifting / TFM / OT-NFM 写成 benchmark-valid。
3. 禁止把 nearest-centroid probe 写成 full TSTR。
4. 禁止 sample manifest 单独决定 benchmark-valid。
5. 禁止 eval 使用 test split 但没有显式 allow_test_reference_eval。
6. 禁止 paperpack 依赖模糊 rglob 而不使用 stage ledger。
7. 禁止 matrix dry-run 结果进入论文 claim。
8. 禁止在没有真实六数据集 evidence 时把 draft 标记 SUBMISSION_READY。
最终目标状态

本地 agent 执行完 V3-000 到 V3-007 后，仓库应达到：

代码可跑
证据链可审计
paper matrix 可 dry-run
reviewer 无 P0 blocker
可以进入真实六数据集 run

执行完 V3-008 到 V3-009 后，仓库应达到：

每个 run 有 stage ledger
每个 sample 有 synthetic manifest
每个 eval 有 eval evidence manifest
每个 paperpack 有表格、图源和 reproducibility statement
benchmark effect 有 summary 和 manifest
submission draft 自动保持 NOT_SUBMISSION_READY 或 SUBMISSION_READY
所有状态由 evidence gate 决定，而不是人工口头判断
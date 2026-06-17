请审查 PHM-Vibench `Feature_factory-update` 的 PHM-GenBench v0.3 PR。

只做架构和证据链审查，不直接大改代码。

重点检查：

1. 是否破坏 `python main.py --config <yaml>` 主路径。
2. 是否仍保留 `environment/data/model/task/trainer` 五段式配置。
3. `Pipeline_06_generative.py` 是否只负责 train/sample/eval orchestration。
4. train/sample/eval/paperpack 之间是否有可追踪 artifact ledger。
5. paperpack 是否能找到 sample 阶段的 `synthetic_data_manifest.json`。
6. `condition_sampling_policy=train_distribution` 是否有真实 train split evidence。
7. sample manifest 是否错误地在 eval 之前声明 benchmark-valid。
8. `metric_status_reason_recorded` 是否由 eval evidence sidecar 提供。
9. TSTR/TRTS 是否被误写成完整下游分类器结果。
10. MeanFlow/Drifting/Transition Flow/OT-NFM 是否仍是 exploratory placeholder。
11. `mamba1d_backbone` 是否被误当成真实 Mamba baseline。
12. six-dataset matrix 是否能 dry-run 并生成 blocked ledger。
13. paper draft 是否在 evidence incomplete 时保持 `NOT_SUBMISSION_READY`。
14. 是否一次修改过多，应该拆成多个 PR。

输出格式：

- Blocking issues
- Non-blocking issues
- Evidence from files
- Suggested patch boundaries
- Merge decision: merge / request changes / reject
- Paper status: docs-only / exploratory / benchmark-candidate / submission-ready

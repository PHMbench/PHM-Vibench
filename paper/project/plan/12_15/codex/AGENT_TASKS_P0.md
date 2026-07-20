# Paper 5 执行官（Agent）P0 任务包（按 Paper2 schema 统一）

## P0-S1：跑通统一基线入口（FuzzyLogic v2）
- `CUDA_VISIBLE_DEVICES=0 python main.py --config_dir configs/unified_baseline/config_FuzzyLogic_v2.yaml`

## P0-S2：在本 paper 下建立 `<RUN_DIR>` 并写入 schema 文件
- `<RUN_DIR>/run_meta.yaml`：paper_id=paper5；model_id=FuzzyLogic_v2
- `<RUN_DIR>/metrics.json`：至少 `split_metrics.test.accuracy`

## P0-S3：Schema 校验
- `python Paper/Explainable_FD_Toolkit/scripts/validate_schema.py --run_dir <RUN_DIR>`
- 验收：OK

## P0-S4：失败案例（先占位，P0阶段允许“零样本但结构齐全”）
- 输出 `failure_cases.md` 与 `case_*.json`（字段先按 schema 的 artifacts/notes 组织）
- 验收：结构齐全、可扩展到真实案例


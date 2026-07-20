# Paper 6 执行官（Agent）P0 任务包（按 Paper2 schema 统一）

> Paper6 的核心产出是“命题实验脚本+图表”。本 P0 先让命题2最小实验产出 L0 合规的 `run_meta.yaml`+`metrics.json`，便于纳入 master 表。

## P0-S1：命题2最小实验可跑通
- `python Paper/Neuralsymbolic_theory/simple_validation_demo.py`

## P0-S2：建立 `<RUN_DIR>` 并写入 schema 文件
- `<RUN_DIR>/run_meta.yaml`：paper_id=paper6；model_id=NeSy_Prop2_Minimal；dataset_id=Synthetic（或按实际）
- `<RUN_DIR>/metrics.json`：至少 `split_metrics.test.accuracy`（若非分类，用 0.0 + notes 解释，并在 P1 改为合适指标）

## P0-S3：Schema 校验
- `python Paper/Explainable_FD_Toolkit/scripts/validate_schema.py --run_dir <RUN_DIR>`
- 验收：OK


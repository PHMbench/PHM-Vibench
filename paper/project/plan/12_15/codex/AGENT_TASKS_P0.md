# Paper 4 执行官（Agent）P0 任务包（按 Paper2 schema 统一）

## P0-S1：至少跑通 3 experts（统一基线入口）
- `CUDA_VISIBLE_DEVICES=0 python main.py --config_dir configs/unified_baseline/config_MoE.yaml`

## P0-S2：跑通 5/8 experts（至少各一次）
- `CUDA_VISIBLE_DEVICES=0 python main.py --config_dir configs/unified_baseline/config_MoE_5experts.yaml`
- `CUDA_VISIBLE_DEVICES=0 python main.py --config_dir configs/unified_baseline/config_MoE_8experts.yaml`

## P0-S3：为每次运行建立 `<RUN_DIR>` 并写入 schema 文件
- `<RUN_DIR>/run_meta.yaml`（paper_id=paper4；model_id=MoE；dataset_id=THU_018_basic 或 CWRU/XJTU）
- `<RUN_DIR>/metrics.json`（至少 `split_metrics.test.accuracy`）

## P0-S4：Schema 校验
- `python Paper/Explainable_FD_Toolkit/scripts/validate_schema.py --run_dir <RUN_DIR>`
- 验收：OK


# 本次新增脚本/产物整理（UXFD merge）

目标：把“本次新增的脚本与产物（证据链闭环）”整理成可直接复用的清单。

## 1) 新增/关键脚本（主仓库内）

> 说明：脚本源码仍然放在主仓库规范位置（不复制到 paper 目录），这里仅做索引与用法记录。

- `scripts/collect_uxfd_runs.py`
  - 功能：扫描 `**/artifacts/manifest.json`，扁平化导出 `CSV`（一行一个 run）
  - 用法：
    - `python -m scripts.collect_uxfd_runs --input <runs_root> --out_dir <out_dir>`
  - 主要输出：
    - `<out_dir>/uxfd_runs.csv`
  - 备注：
    - 会自动把 `manifest.json` 里的 `metrics_inline` 展开为 `metric/<key>` 列

## 2) 新增/关键运行产物（每次 run 的证据链）

这些文件由默认流水线/回调 best-effort 写出：

- `<run_dir>/config_snapshot.yaml`
  - 含义：本次 run 的“最终解析配置快照”（含 base_configs + override 合并结果）
  - 入口：`src/Pipeline_01_default.py`

- `<run_dir>/artifacts/manifest.json`
  - 含义：单次 run 的证据链索引（SSOT），供 collect 脚本稳定消费
  - 入口：`src/trainer_factory/extensions/manifest.py`（fit/test 末尾写出；pipeline 在 test_result 写出后再补写一次）

- `<run_dir>/artifacts/data_metadata_snapshot.json`
  - 含义：从 test dataloader 抽样一个 batch 的 meta 快照（best-effort）
  - 入口：`src/Pipeline_01_default.py`

- `<run_dir>/artifacts/explain/eligibility.json`（可选）
  - 含义：当 `trainer.extensions.explain.enable=true` 时写出，说明 explain 是否具备运行条件（缺什么、建议是什么）
  - 入口：`src/Pipeline_01_default.py` + `src/explain_factory/eligibility.py`

## 3) 示例输出（本次 smoke run）

本次执行使用：
- `python main.py --config configs/demo/00_smoke/dummy_dg.yaml --override trainer.num_epochs=1`

把关键小文件拷贝到本目录，便于查阅：
- `paper/UXFD_paper/merge_uxfd/12_22/results/uxfd_runs_smoke.csv`
- `paper/UXFD_paper/merge_uxfd/12_22/results/manifest_smoke_test.json`
- `paper/UXFD_paper/merge_uxfd/12_22/results/config_snapshot_smoke.yaml`
- `paper/UXFD_paper/merge_uxfd/12_22/results/test_result_smoke.csv`


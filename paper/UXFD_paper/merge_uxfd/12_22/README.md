# 12_22（本次实现产物整理）

本目录用于把“本次 UXFD merge 的新增脚本/产物样例/失败问题”集中落盘，避免打扰主仓库 `docs/`。

## 快速导航

- 脚本与产物清单：`paper/UXFD_paper/merge_uxfd/12_22/scripts_and_outputs.md`
- 失败与问题分析：`paper/UXFD_paper/merge_uxfd/12_22/failures_report.md`
- Final plan 执行复盘 + TODO：`paper/UXFD_paper/merge_uxfd/12_22/status_review_and_todos.md`
- 上游项目缺口分析 + 集成计划：`paper/UXFD_paper/merge_uxfd/12_22/upstream_gap_analysis_and_plan.md`
- 证据链样例（小文件）：`paper/UXFD_paper/merge_uxfd/12_22/results/README.md`

## 复现命令（最小闭环）

```bash
python main.py --config configs/demo/00_smoke/dummy_dg.yaml --override trainer.num_epochs=1
python -m scripts.collect_uxfd_runs --input results/demo/dummy_dg_smoke --out_dir reports
```

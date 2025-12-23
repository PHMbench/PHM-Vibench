# 12_23（算子库补全计划）

本目录用于在开始大规模代码移植前，先把“WP0 submodule(min.yaml) 驱动验证 + WP1 算子库移植（含融合/路由）+ HookStore + 最小单测”写清楚，并让你确认后再动手实现。

## 核心文档

- 执行计划：`paper/UXFD_paper/merge_uxfd/12_23/ops_library_completion_plan.md`（算子库补全主计划）
- TODO 清单：`paper/UXFD_paper/merge_uxfd/12_23/TODO_BACKLOG.md`（从 12_22 迁移的未完成事项）

## 配套计划（plot/post-run）

- Plot 目录迁移计划：`paper/UXFD_paper/merge_uxfd/12_23/plot_factory_migration_plan.md`（plot/ → offline 工具 + 可选 src/plot_factory）
- Post-run 检查+绘图脚本：`scripts/uxfd_postrun.py`
- Post-run 配置示例：`paper/UXFD_paper/merge_uxfd/12_23/uxfd_postrun_config_example.yaml`

## 文件依赖关系（简图）

`paper/UXFD_paper/merge_uxfd/12_23/README.md`
├── `paper/UXFD_paper/merge_uxfd/12_23/ops_library_completion_plan.md` ←── `paper/UXFD_paper/merge_uxfd/12_23/TODO_BACKLOG.md`
└── `paper/UXFD_paper/merge_uxfd/12_23/plot_factory_migration_plan.md` → `scripts/uxfd_postrun.py` + `paper/UXFD_paper/merge_uxfd/12_23/uxfd_postrun_config_example.yaml`

说明（命名变更）：
- 组件区统一命名为 `src/model_factory/X_model/UXFD_component/`（后续会把现有 `src/model_factory/X_model/UXFD/` 作为兼容 shim 逐步淘汰）。

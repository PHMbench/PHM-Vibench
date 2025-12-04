# v0.1.0 Planning Snapshot

- Date: 2025-11-xx
- Base branch: `main` (commit: `<填上当前 main 的短 hash>`)

## Scope

- 版本目标：完成 v0.1.0 的第一版「可跑、可解释」发行。
- 配置系统与 demo 设计：见 `docs/v0.1.0/configupdate.md`。
- 发行流程与分支合并计划：见 `docs/v0.1.0/v0.1.0_update.md`。

## Existing branches

- unified_metric_learning_work: 统一度量学习相关实验分支（包含部分配置与实现尝试）。

## Decisions

- 仅从 `unified_metric_learning_work` 中挑选：
  - 明确需要进入 v0.1.0 的配置修正与关键 bugfix；
  - 与 v0.1.0 目标强相关、且已在本地验证过的实验配置。
- v0.1.0 版本阶段：
  - 不强制提供 shell 脚本入口，核心使用 `python main.py --config <yaml>`。
  - 先把 YAML 结构与 ConfigWrapper 行为整理稳定，再考虑后续版本的脚本与更复杂 pipeline。

## TODO / 状态记录（v0.1.0）

- [x] 根据 `docs/v0.1.0/configupdate.md` 设计，梳理并创建：
  - `configs/base/{data,model,task,trainer}/` 下的基础 yaml；
  - `configs/demo/` 下 6 类 demo 配置的目录与命名（已落地并在 `configs/config_registry.csv` 中登记）。
- [x] 在本地用最小规模的命令验证每一类 demo：
  - 使用 `python main.py --config <demo_yaml> --override trainer.num_epochs=1 --override data.num_workers=0` 对 6 个代表性 demo 做 sanity check，结果记录于 `docs/v0.1.0/done/demo_validation_report.md`。
- [x] 根据 `docs/v0.1.0/v0.1.0_update.md`，整理本次配置/工厂/流水线的更新摘要，作为 v0.1.0 的 Release Note。
- [ ] 在真实 Git 流程中，基于 `release/v0.1.0` 或等价分支，按计划从 `unified_metric_learning_work` 等分支挑选需要纳入 v0.1.0 的改动（cherry-pick / merge），并在最终合并到 `main` 前跑一遍 smoke test。

> 说明：本规划文件现在更偏向“状态记录 + 后续 Git 操作提醒”；  
> 代码与配置层面的实际变更，请参考 `docs/v0.1.0/v0.1.0_update.md` 与 `docs/v0.1.0/done/` 下的详细 plan/报告。

# Changelog

本目录用于维护“合并到 `main` 的大版本 Changelog”。

## Changelog 更新方法（约定）

目标：让 reviewer 一眼看清“这次合并带来了什么、为何需要、如何验证”。

- 位置：只维护本文件 `src/changelog/CHANGELOG.md`（每个大版本一个 changelog）。
- 结构：每次合并到 `main` 前，在最上方 `Unreleased` 下新增/更新条目，按以下小节组织：
  - `Added`：新增能力/入口/文档（面向用户或维护者的新增项）。
  - `Changed`：行为变化/默认值变化（可能影响复现或使用方式的变更）。
  - `Fixed`：bug 修复（包含“失败现象→原因→修复点”的最小描述）。
  - `Validation`：**必须**给出可复制的验证命令（尽量使用仓库内 demo + pytest）。
- 写法要求：
  - 每条尽量是“动词开头 + 影响范围 + 指向路径”，避免泛泛而谈。
  - `Validation` 中的命令要能在 CPU + dummy 数据上跑通（除非明确说明依赖外部数据/硬件）。
  - 如果涉及 submodule：默认只写“主仓库如何验证”；submodule 的提交/指针更新另开 PR 或在 PR 描述中说明。
  - 如果是 breaking change（不兼容）：在条目中明确标注，并在 `Validation` 增加回归命令。

## Unreleased (2026-01-31)

### Added
- 新增 UXFD→论文第 2–5 章支撑的人工验证清单：`paper/UXFD_paper/VALIDATE_UXFD_CHAP02_CHAP05.md`
- 新增合并到 `main` 的逐步测试计划：`paper/UXFD_paper/TEST_PLAN_MERGE_TO_MAIN.md`

### Changed
- `Default_trainer` 对可选 logger 依赖更健壮：未安装 `swanlab`/`wandb` 时不再导致 trainer import 失败或训练中断（会打印 warning 并跳过对应 logger）。

### Fixed
- 修复 `swanlab` 缺失时 `Default_trainer` 导入失败导致的 `trainer=None` 进而 `AttributeError: 'NoneType' object has no attribute 'fit'`。

### Validation
- `python -m scripts.validate_configs`
- `python -m scripts.validate_docs`
- `python -m pytest test/ -q`
- `python main.py --config configs/demo/uxfd/00_smoke_tspn_uxfd.yaml --override trainer.num_epochs=1`
- `python main.py --config configs/demo/uxfd/10_smoke_tspn_uxfd_sp2d.yaml --override trainer.num_epochs=1`
- `python main.py --config configs/demo/uxfd/20_smoke_tspn_uxfd_full.yaml --override trainer.num_epochs=1`

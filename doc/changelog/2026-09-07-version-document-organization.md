# 2026-09-07：版本文档目录整理

## 目标

将散落在仓库根目录的迁移指南和发布说明收敛到一个明确的版本文档入口，同时保留历史外部链接。

## 结果

完整内容现在位于：

```text
doc/migration/MIGRATION_v0.1_to_v0.2.md
doc/migration/MIGRATION_v0.2_to_v0.3.md
doc/release/RELEASE_NOTES_v0.2.0.md
doc/release/RELEASE_NOTES_v0.3.0.md
```

新增 [`doc/README.md`](../README.md) 作为索引，并更新 `docs/index.md` 与根目录 `CHANGELOG.md`。

原有四个根路径保留为短兼容页，只负责跳转，不再复制正文。这样可以同时满足：

- 仓库根目录保持简洁；
- 历史 PR、论文和外部链接继续可用；
- 当前文档只有一个正文 authority；
- 既有 release-readiness 和 package workflow 不因路径迁移失效。

## 内容校正

v0.2 → v0.3 迁移指南和 v0.3 发布说明同步删除了已经失效的仓库改名与 byte-hash 发布要求，改为当前实际的仓库名称、显式配置、installed-wheel 首跑、直接结果路径和科学语义边界。

v0.1 → v0.2 指南及 v0.2 发布说明属于历史版本记录，除补充当前索引链接外不改写其历史结论。

## 未改变

本次不修改：

```text
数据、split、模型、任务、Trainer、指标
版本号、tag、GitHub Release、包发布状态
benchmark 或 baseline_valid 状态
历史审计中记录的原始路径
```

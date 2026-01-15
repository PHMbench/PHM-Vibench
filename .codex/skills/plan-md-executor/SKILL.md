---
name: plan-md-executor
description: 读取当天的 `<MM_DD>/codex/plan/plan_<topic>.md`，将 Tasks/Dependencies/DoD/Gates 拆成可执行序列（含并行组、检查点、证据要求与回滚点），并落盘为 `<MM_DD>/codex/exec/exec_<topic>.md`。适用于“基于 plan 生成执行步骤/拆解并行/给我执行顺序与检查点”等请求。
---

# Plan MD Executor（当天落盘版）

## 目录协议（强制）
- 输入：`<MM_DD>/codex/plan/plan_<topic>.md`
- 输出：`<MM_DD>/codex/exec/exec_<topic>.md`
- 若目标文件已存在：先给差异摘要，等待用户确认后再覆盖/更新

## 执行前先给 Plan（强制）
在写文件前，先输出 3 行以内的计划，让用户确认：
1) 读取 plan 并抽取 Tasks/DoD/Dependencies/Gates
2) 生成执行序列（含并行组/检查点/证据/回滚点）
3) 写入 `exec_<topic>.md` 并给下一步建议

## 输出模板（建议）
```md
# Exec: <topic>

## Source
- Plan: <MM_DD>/codex/plan/plan_<topic>.md

## Assumptions
- ...

## Execution Sequence
- Phase 0 (preflight):
  - Step:
  - Evidence:
  - Rollback:
- Phase 1:
  - ...

## Checkpoints / Gates
- Gate:
  - Command:
  - Expected:
  - On fail:

## Evidence Log
- Files:
- Commands:
- Notes:

## Rollback
- ...

## Next
- ...
```

## 操作步骤
1) 要求用户给出（或确认）`<MM_DD>` 与 `<topic>`
2) 读取 `plan_<topic>.md`，抽取并规范化：
   - Tasks/DoD/Dependencies/Gates/Deliverables/Rollback
3) 拆解每个 Task 为“最小可执行子步骤”（建议 1-5 条）：
   - 每条写清：动作、输入/输出、完成判定（DoD）、最小证据
4) 建议执行顺序与并行组：
   - 先跑低成本门禁/快速失败项
   - 依赖项先行，互不依赖的任务分到并行组
5) 将 Gates 映射为检查点（在哪一步跑、失败怎么办、是否允许继续）
6) 为高风险步骤补回滚点：
   - 撤销变更/恢复文件/保留旧实现/feature flag（按任务选择最小策略）
7) 创建目录（如不存在）：`<MM_DD>/codex/exec/`
8) 写入 `exec_<topic>.md`
9) 返回下一步建议（按需）：
   - 实际执行：按 `exec_<topic>.md` 顺序执行，并把证据记录到 Evidence Log
   - 证据链：用 `artifact-manifest-writer` 生成 `manifest_<topic>.json`
   - 日更/待办：用 `daily-vibe-update` 更新 `daily.md` + `todo.md`

## 完成判定（DoD）
- `exec_<topic>.md` 已创建在约定路径
- 每个 Task 都有可执行步骤、最小证据；存在风险的步骤写明回滚点

---
name: daily-vibe-update
description: 生成当天日更 `daily.md` 与明日待办 `todo.md`，落盘到 `<MM_DD>/codex/daily/` 与 `<MM_DD>/codex/todo/`，并在 Evidence 中引用当天的 report/manifest/关键产物路径。适用于“写日更/生成daily+todo/总结今天Done+Blockers+Next+Evidence”等请求。
---

# Daily Vibe Update（当天落盘版）

## 目录协议（强制）
- 日更：`<MM_DD>/codex/daily/daily.md`（当天唯一）
- 待办：`<MM_DD>/codex/todo/todo.md`（当天唯一）
- 若目标文件已存在：先给差异摘要，等待用户确认后再覆盖/更新

## 执行前先给 Plan（强制）
在写文件前，先输出 3 行以内的计划，让用户确认：
1) 收集 Done/Blockers/Next/Evidence（含路径）
2) 写入 daily.md 与 todo.md
3) 返回链接与下一步建议

## `daily.md` 模板（必须块）
```md
# Daily <MM_DD>

## Summary
...

## Done
- ...

## Evidence
- <MM_DD>/codex/report/report_<topic>.md
- <MM_DD>/codex/artifact/manifest_<topic>.json

## Blockers
- ...

## Next
- See: <MM_DD>/codex/todo/todo.md
```

## `todo.md` 模板（建议）
```md
# Todo <MM_DD>

## Top
- [ ] ...
- [ ] ...

## Notes
- ...
```

## 操作步骤
1) 要求用户给出（或确认）`<MM_DD>`，并询问今天涉及的 `<topic>`（可多个，用于 Evidence 路径）
2) 让用户给素材（或从对话中提取）：
   - Summary：一句话总结
   - Done：每条必须可验证（对应产物/命令/结果）
   - Evidence：列出当天目录下的关键路径（manifest/report/重要文件）
   - Blockers：原因 + 需要什么
   - Next：明日 Top5/Top10
3) 创建目录（如不存在）：`<MM_DD>/codex/daily/`、`<MM_DD>/codex/todo/`
4) 写入 `daily.md` 与 `todo.md`
5) 在 `daily.md` 的 Next 区块引用 `todo.md` 路径；Evidence 中尽量引用 manifest/report 路径

## 完成判定（DoD）
- `daily.md` 与 `todo.md` 已创建在约定路径
- Done 与 Evidence 至少各 1 条；Evidence 路径位于当天目录

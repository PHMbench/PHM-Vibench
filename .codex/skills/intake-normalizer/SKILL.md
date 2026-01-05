---
name: intake-normalizer
description: 将用户给的口水输入/日志/想法清单/issue 等，归一成结构化 Intake Markdown，并落盘到当天目录 `<MM_DD>/codex/intake/intake_<topic>.md`。适用于“整理成intake/规范化输入/把材料写成intake_<topic>.md/把这堆信息结构化”等请求。
---

# Intake Normalizer（当天落盘版）

## 目录协议（强制）
- 当天目录：`<MM_DD>/`（例如 `12_30/`）
- `<topic>`：短、可复用、`hyphen-case`（例如 `oom-fix`）
- 输出文件：`<MM_DD>/codex/intake/intake_<topic>.md`
- 若目标文件已存在：先给差异摘要，等待用户确认后再覆盖/更新

## 执行前先给 Plan（强制）
在写文件前，先输出 3 行以内的计划，让用户确认：
1) 解析输入并提取字段
2) 确认 `<MM_DD>`/`<topic>` 与输出路径
3) 写入 intake 文件并返回下一步建议

## 输出模板（字段固定顺序，空也要保留）
```md
# Intake: <topic>

- Goal:
- Scope:
- Tasks:
  - T1:
  - T2:
- Priority:
- Due:
- Owner:
- Background:
- Details:
- Acceptance / DoD:
- Notes:
- Evidence:
```

## 操作步骤
1) 要求用户给出（或确认）`<MM_DD>` 与 `<topic>`；缺失时你提出 1-3 个候选供选择
2) 让用户粘贴原始材料（或用当前对话中已给材料）
3) 按模板字段提取信息：
   - Tasks 必须改写为动词开头的可执行项（必要时拆分）
   - Acceptance/DoD 写成可验证标准（能看日志/文件/结果）
   - Evidence 写“最小证据”（具体路径/命令/日志关键字）
4) 创建目录（如不存在）：`<MM_DD>/codex/intake/`
5) 写入 `intake_<topic>.md`
6) 在文件末尾追加：`Next: plan-md-writer`（若用户同意进入下一步）

## 完成判定（DoD）
- `intake_<topic>.md` 已创建在约定路径
- 字段齐全；`Tasks` 至少 1 条；`Goal` 与 `Acceptance/DoD` 不为空（或明确写 “TBD”）

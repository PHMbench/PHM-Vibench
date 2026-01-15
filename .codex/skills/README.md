# PHM-Vibench Codex Skills

本目录是仓库内置的 Codex Skills（每个 skill 一个子目录，核心文件为 `SKILL.md`）。

## 目录结构约定
- `.<repo>/.codex/skills/<skill-name>/SKILL.md`：skill 主文件（YAML 头 + Markdown 指令）
- 可选：`scripts/`（可复用脚本）、`references/`（规范/模板/参考）、`assets/`（静态资源）
- 产物默认落盘到当天目录：`<MM_DD>/codex/{intake,plan,exec,artifact,daily,todo}/`

## 内置 Skills（维护集合）
- `workflow-to-skill`：把手工 SOP 抽取成可复用 Skill 规格与 `SKILL.md` 草案（元 skill）。
- `vibe-batch-orchestrator`：批量分诊→Sprint 计划→（可选）落盘 intake/plan/exec（编排元 skill）。
- `intake-normalizer`：将口水输入归一为 `intake_<topic>.md`（当天落盘）。
- `plan-md-writer`：基于 intake 生成 `plan_<topic>.md`（含 DoD/Gates/Deliverables/Rollback）。
- `plan-md-executor`：基于 plan 拆成 `exec_<topic>.md`（并行组/检查点/证据/回滚点）。
- `artifact-manifest-writer`：生成结构化 `manifest_<topic>.json`（变更/命令/验证/产物/回滚）。
- `daily-vibe-update`：生成当天 `daily.md` + `todo.md`（并在 Evidence 中引用关键产物）。
- `tmux-ai-cli`：通过 `ai` 命令管理 tmux-ai-cli 多 AI 会话（默认先只读自检，破坏性操作二次确认）。

## 推荐串联（最小闭环）
`intake-normalizer` → `plan-md-writer` → `plan-md-executor` → `artifact-manifest-writer` → `daily-vibe-update`

## Reference（非维护集合）
`reference/awesome-claude-skills/` 为参考用的第三方 skill 集合（用于借鉴结构/模板；不保证与本仓库协议一致）。

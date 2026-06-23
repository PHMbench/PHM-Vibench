# Session Handoff: Slice 2 Tasks And Taskstoissues Blocker

**Date:** 2026-05-10
**Project:** `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench_fix`

## Current State

**Task:** Execute Slice 2 from `.specify/goals/phm-vibench-full-phm-experiment-platform.md`
**Phase:** `speckit-tasks` complete; blocked at `speckit-taskstoissues`
**Active feature:** `specs/002-phm-task-experiment-matrix`
**Branch:** `002-phm-task-experiment-matrix`

## What We Did

Generated Slice 2 tasks and attempted the `speckit-taskstoissues` prechecks. The
repository remote is a GitHub URL, but both available GitHub authentication paths
are expired or invalid. A local issue draft was written for future authenticated
creation.

## Decisions Made

- `speckit-taskstoissues` is blocked instead of creating issues blindly. The skill
  requires creating issues only in the repository that matches the Git remote.
- Do not proceed to Slice 2 `speckit-analyze` or `speckit-implement` while this
  step is blocked.
- Optional git auto-commit hooks were not executed.

## Files Changed

- `specs/002-phm-task-experiment-matrix/tasks.md`
- `specs/002-phm-task-experiment-matrix/github-issues-draft.md`
- `.claude/handoffs/2026-05-10-phm-vibench-slice2-tasks.md`
- `.claude/handoffs/2026-05-10-phm-vibench-slice2-tasks-blocked-taskstoissues.md`

## Validation

Commands run:

- `.specify/scripts/bash/check-prerequisites.sh --json --require-tasks --include-tasks`
  - Result: `AVAILABLE_DOCS` includes `tasks.md`.
- `git config --get remote.origin.url`
  - Result: `git@github.com:PHMbench/PHM-Vibench.git`.
- `gh auth status`
  - Result: failed; default GitHub token for `liq22` is invalid.
- `mcp__codex_apps__github._search_installed_repositories_v2`
  - Result: failed; GitHub connector returned `token_expired`.

Task validation already recorded in
`.claude/handoffs/2026-05-10-phm-vibench-slice2-tasks.md`:

- 36 task lines.
- No malformed unchecked task lines or unresolved task placeholders.

## Open Questions

- [ ] Re-authenticate GitHub access, then decide whether to create grouped issues
  from `github-issues-draft.md` or one issue per task from `tasks.md`.

## Blockers / Issues

- `speckit-taskstoissues` is blocked. Safe completion needs one of:
  - valid `gh` authentication so existing issues can be checked before creation; or
  - valid GitHub connector authentication plus an issue search/list path; or
  - explicit user approval to waive issue creation for this slice.

## Next Steps

1. Resolve GitHub issue sync authorization/idempotency or obtain explicit waiver.
2. Use `specs/002-phm-task-experiment-matrix/github-issues-draft.md` as the draft
   body source if grouped issue creation is approved.
3. Only after issue sync succeeds or is explicitly waived, run `speckit-analyze`.

## Files to Review on Resume

- `specs/002-phm-task-experiment-matrix/tasks.md`
- `specs/002-phm-task-experiment-matrix/github-issues-draft.md`
- `.specify/goals/phm-vibench-full-phm-experiment-platform.md`

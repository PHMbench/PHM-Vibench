# Session Handoff: Slice 1 Tasks And Taskstoissues Blocker

**Date:** 2026-05-10
**Project:** `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench_fix`

## Current State

**Task:** Execute Slice 1 from `.specify/goals/phm-vibench-full-phm-experiment-platform.md`
**Phase:** `speckit-tasks` complete; blocked at `speckit-taskstoissues`
**Progress:** Slice 1 `tasks.md` generated and validated; GitHub issue sync not executed;
local issue-sync draft added for future authenticated creation

## What We Did

Generated `specs/001-core-runtime-config-contract/tasks.md` with 31 executable tasks,
organized by setup, foundational work, three user stories, and final validation.
Validated that Spec Kit sees `tasks.md` and that task lines follow the required
checkbox/ID format.

## Decisions Made

- **Taskstoissues blocked instead of blind issue creation** - the repository remote is
  GitHub, but `gh auth status` reports an invalid token and the GitHub connector also
  reports an expired token. Creating issues without authentication and duplicate
  detection is unsafe.
- **Do not proceed to analyze or implement** - the controlling goal says to stop at a
  blocked Speckit step and not continue to later steps for that slice.
- **No optional auto-commit** - optional git commit hooks were not executed.

## Code Changes

**Files added or modified:**

- `specs/001-core-runtime-config-contract/tasks.md`
- `specs/001-core-runtime-config-contract/github-issues-draft.md`
- `.claude/handoffs/2026-05-10-phm-vibench-slice1-tasks-blocked-taskstoissues.md`

## Validation

Commands run:

- `.specify/scripts/bash/setup-tasks.sh --json`
  - Result: resolved active feature, available docs, and tasks template.
- `.specify/scripts/bash/check-prerequisites.sh --json --require-tasks --include-tasks`
  - Result: `AVAILABLE_DOCS` includes `tasks.md`.
- `rg -n "^- \\[ \\] T[0-9]{3}" specs/001-core-runtime-config-contract/tasks.md | wc -l`
  - Result: 31 tasks.
- `rg --pcre2 -n "^- \\[ \\](?! T[0-9]{3})|TXXX|\\[FEATURE|\\[Title\\]|\\[name\\]|\\[endpoint\\]|\\[Entity\\]|NEEDS CLARIFICATION|ACTION REQUIRED" specs/001-core-runtime-config-contract/tasks.md`
  - Result: no invalid task placeholders or malformed unchecked task lines.
- `git config --get remote.origin.url`
  - Result: `git@github.com:PHMbench/PHM-Vibench.git`.
- `gh auth status`
  - Result: failed; default GitHub token is invalid.
- `mcp__codex_apps__github._list_repositories`
  - Result: failed; GitHub connector returned `token_expired`.
- Resume check on 2026-05-10:
  - `gh auth status`
    - Result: still failed; default GitHub token for `liq22` is invalid.
  - `mcp__codex_apps__github._search_installed_repositories_v2`
    - Result: still failed; GitHub connector returned `token_expired`.
  - `.specify/scripts/bash/check-prerequisites.sh --json --require-tasks --include-tasks`
    - Result: active feature remains `specs/001-core-runtime-config-contract`, and
      `AVAILABLE_DOCS` still includes `tasks.md`.

## Open Questions

- [ ] Re-authenticate GitHub access, then decide whether to create one issue per
  phase/story from `github-issues-draft.md` or one issue per task from `tasks.md`.

## Blockers / Issues

- `speckit-taskstoissues` is blocked. Safe completion needs one of:
  - valid `gh` authentication so existing issues can be checked before creation; or
  - valid GitHub connector authentication plus an issue list/search path; or
  - explicit user approval to create issues without duplicate detection after
    authentication is restored.

## Next Steps

1. [ ] Resolve GitHub issue sync authorization/idempotency.
2. [ ] Use `specs/001-core-runtime-config-contract/github-issues-draft.md` as the
   draft body source if grouped issue creation is approved.
3. [ ] Run `speckit-taskstoissues` for Slice 1 tasks.
4. [ ] Only after issue sync succeeds or is explicitly waived, run `speckit-analyze`.

## Files to Review on Resume

- `specs/001-core-runtime-config-contract/tasks.md`
- `specs/001-core-runtime-config-contract/github-issues-draft.md`
- `.specify/goals/phm-vibench-full-phm-experiment-platform.md`
- `.specify/extensions/git/commands/speckit.git.commit.md`

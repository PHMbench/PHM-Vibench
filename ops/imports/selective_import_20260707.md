# Selective Import Record - 2026-07-07

## Intent

Import only the source materials needed by the repo-slim branch while preserving the
full source refs through local bundles and archive tags.

## Baseline

- Target branch: `cleanup/repo-slim-2026-07-05`
- Target baseline commit: `462adac5fe110007bf8406dfd88ba446d797e9ff`
- Integration branch: `integrate/selective-import-20260707`
- Backup tag: `backup/pre-selective-import-20260707`

## Bundles

Created under `/tmp/phm-vibench-import-20260707/`:

- `target-pre-selective-import.bundle`
- `src-fix.bundle`
- `src-state-flow.bundle`
- `src-state-flow-submodule-paper_state_flow.bundle`

## Archive Tags

- `archive/src-fix-goal-repo-slim-v020-plan-20260706`
- `archive/src-fix-004-uxfd-paper-alignment-20260707`
- `archive/src-fix-lq-merge-uxfd-20260707`
- `archive/src-fix-uxfd-merge-251223-20260707`
- `archive/src-state-flow-paper_state_flow-20260707`
- `archive/src-state-flow-vk-base-20260707`
- `archive/src-state-flow-vk-e530-agent-20260707`

## Imported Scope

- Added `paper/paper_state_flow` as a submodule gitlink at
  `eca41359c8c1151e629c3b1ae3cb102a50808d2d`, matching the source
  `paper_state_flow` superproject.
- Added paper-local guidance files from `src_state_flow/paper_state_flow`:
  `paper/AGENTS.md` and `paper/CLAUDE.md`.
- Imported the v020 work-pack documents from
  `src_fix/origin/goal/repo-slim-v020-plan-20260706` under `ops/v020_work/`.

## Explicitly Not Imported

- No full source branch merge.
- No UXFD code path import from `004-uxfd-paper-alignment`.
- No `.claude/`, `.codex/`, `.agents/`, `.tmp/`, `.pytest_cache/`,
  `.specify/`, or `specs/` content from either source.
- No large data, results, checkpoints, PDFs, zip files, or `data/Reference/`
  content from either source.

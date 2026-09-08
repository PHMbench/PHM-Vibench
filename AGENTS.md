# Working on PHMFactory

Read [CORE.md](CORE.md) for project invariants and [CONTRIBUTING.md](CONTRIBUTING.md)
for change and branch rules. [README.md](README.md) is the user entrypoint. Read only the
module READMEs relevant to the current task, not every guide in the repository.

## Establish the task and working state

Start with read-only checks from the repository root:

```bash
git status --short
git branch --show-current
git rev-parse HEAD
```

Do not automatically switch branches, pull, stash, reset, clean, or format the repository.
Preserve unrelated user changes. If edits overlap the requested work, stop before
modifying them and identify the conflict; an unrelated dirty file alone is not a blocker.
When remote changes are needed, inspect the relevant open PRs and actual branch state.
Do not treat a HEAD, PR number, or next-goal statement in an old report as current.

A review-only task does not authorize edits, commits, merges, data downloads, or cleanup.
For an implementation task, state one short plan, then make the smallest useful change.
Use authorization already given for this task, but do not extend it to unrelated branches,
data, tags, package publication, or machine settings.

## Find the existing owner

- Public configuration: `phmfactory/config.py`; use `analyze_config()`, not a new loader.
- Legacy direct-Python configuration: `src/configs/README.md`; do not expand that path.
- Data, models, tasks and training: the corresponding `src/*_factory/README.md`.
- Runtime and results: `src/runtime/` and `phmfactory/runtime/`.
- Tests: [docs/testing.md](docs/testing.md) and the existing tests for the changed owner.
- Completed upgrades: [doc/changelog/](doc/changelog/), not a fixed queue in this file.

CORE states what must hold; code and tests establish current behavior. Report a conflict
as a defect rather than weakening the invariant or pretending it is already fixed.
Catalogue entries and successful imports do not prove an exact experiment is supported.

## Make a bounded change

Prefer deletion or reuse over a new abstraction. Preserve data population, split, model,
objective, device request, checkpoint selection and estimator unless the task changes them.
Do not hide failures by changing algorithms, skipping samples, filling metrics with zero,
choosing another checkpoint, or automatically switching hardware or data sources.
Do not add hash/checksum/digest auditing, receipts, ledgers, goal registries, or duplicate
configuration/runtime/result managers. Ordinary Git references are not such a system.
Use Python for real computation, experiments, file generation or focused tests, not a
fallback layer that repairs invalid inputs. Comments explain reasons; docs use real
interfaces and separate runnable examples from illustrative fragments.

## Validate and report

For documentation, run `python -m scripts.validate_docs` and check changed links/examples.
For runtime changes, run the relevant focused tests and affected maintained user path.
Generate Atlas/support documents only when their source registry or generator changes.
Do not download or rerun THU or another real dataset just to validate a documentation edit.
Reuse supplied validation within its stated scope; do not invent unseen numerical results.

Report changed files, commands and PASS / FAIL / NOT RUN with reasons. Distinguish local
checks, GitHub Actions, software smoke and scientific validation. A green audit can still
report blocked publication. Never claim a PR merged before verifying its merge result.
Record actual user-facing changes in `doc/changelog/`; do not add a parallel status system.

## Keep instructions shared

This is the only maintained AI instruction body. Root `CLAUDE.md` imports it with
`@AGENTS.md`. Keep module guidance in neutral READMEs. Do not commit personal Agent
workspaces, credentials, tool permissions, local overrides, or conversation logs.

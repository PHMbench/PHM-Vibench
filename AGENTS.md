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

## Repository ownership and one-way dependency

Resolve the actual remote, role and pinned dependencies; a package or directory name
alone does not identify a checkout. This repository is upstream infrastructure.
Maintained public models, operators, tasks, data interfaces, standard benchmarks,
reproducibility seeds, generic analysis and legitimate citations belong here when they
have a real paper-independent purpose. Existing ownership is not changed by a keyword.

Manuscript-specific methods, objectives, task bindings, sweeps, seed lists, ablations,
control stimuli, publication figures/tables and research results belong downstream,
regardless of their folder or a generic-looking name. Use existing public functions,
registries or plugin hooks; do not copy the runtime or invent an abstraction to relocate
one study. New abstractions must satisfy CORE's current-consumer requirement.

Dependency direction is paper -> public upstream API. Installation, default execution,
packaging, standard tests and CI must not require a downstream checkout/package or
hardcode discovery/imports/default paths for a named paper. Caller-supplied callbacks
through existing public hooks are permitted when upstream ships and tests without that
paper plugin and has no paper-specific branch or default.

During a paper task, upstream source trees are read-only: submodules, editable installs,
symlink targets, shared checkouts and installed package sources. Do not patch,
monkey-patch, vendor, shadow imports, advance pins or write outputs/caches into them.
Resolve the real write destination and preserve pre-existing changes. A general defect
requires a separately authorized upstream task/PR, the smallest public-contract fix,
relevant tests and a separately authorized dependency update before downstream use.
Continue independent paper work rather than silently switching implementations.

Use caller-supplied data/output paths and existing explicit local configuration. Do not
commit private paths or machine-specific defaults in shared code/configuration. Local
provenance stays in its existing designated record. Do not move/delete legacy material
or change filesystem permissions merely to satisfy this policy; migrations require
separate scope. These instructions and scans are not filesystem isolation.

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

### Pre-commit and pre-PR acceptance

- [ ] Review actual staged and complete PR diffs, including names, contents and modes;
      no new manuscript-only payload or unapproved ownership migration.
- [ ] Inspect imports, dynamic loading, defaults, packaging, tests and CI for reverse
      dependencies; verify the affected standalone path when those contracts change.
- [ ] Resolve every scan candidate as a violation or justified benign reference. No
      blanket docs exemption; zero matches is not proof of architectural independence.
- [ ] Preserve populations, physical grouping, label access, preprocessing, budgets,
      seeds, selection, metric denominators and failure handling. A correctness fix
      affecting these has separate scope and an explicit reproducibility impact.
- [ ] Run existing applicable checks, preserve required CI, and record exact results.
      Missing required validation blocks merge; a Draft may disclose that blocker.
- [ ] Commit only task-owned files; do not repair unrelated dirty dependencies.
      Git status is useful context, not proof that ignored/untracked contents are unchanged.

Run in Bash after verifying the actual, current PR target in `BASE_REF` (normally
`origin/dev`). Do not silently fetch, select another comparison, or change branches.

```bash
set -euo pipefail
: "${BASE_REF:?Set BASE_REF to the verified PR target}"
base_oid=$(git rev-parse --verify "${BASE_REF}^{commit}")
git diff --cached --check
git diff --cached --name-status
git diff --cached --summary
git diff --check "${base_oid}...HEAD"
git diff --name-status "${base_oid}...HEAD"
git diff --summary "${base_oid}...HEAD"

scan_added_lines() (
    set -euo pipefail
    patch=$(git diff --no-ext-diff --no-textconv --no-color \
        --unified=0 "$@" --) || exit $?
    added=$(printf '%s\n' "$patch" | awk '
        /^diff --git / { file=$0; hunk=""; next }
        /^@@ / { hunk=$0; next }
        hunk != "" && /^\+/ { print file " | " hunk " | " substr($0, 2) }
    ')
    rc=0
    printf '%s\n' "$added" | rg --no-config --color never -n -i \
        -e '(/home/|/Users/|[A-Za-z]:[\\/])' \
        -e '\b(ablation|ours|camera[_-]?ready)\b' \
        -e '\b(table|fig)[_-][0-9]+\b' \
        -e '\bP[0-9]{2,3}\b' \
        -e 'AI4Engineering-L/P[0-9]|liq22/(P01-PHMskills|P02_agent_langraph)' \
        || rc=$?
    case "$rc" in
        0) printf '%s\n' 'REVIEW_REQUIRED: classify every candidate; not PASS.' ;;
        1) printf '%s\n' 'NO_CANDIDATES: semantic/dependency review still required.' ;;
        *) printf 'SCAN_ERROR: exit=%s\n' "$rc" >&2; exit "$rc" ;;
    esac
)
scan_added_lines --cached
scan_added_lines "${base_oid}...HEAD"
```

The scan is advisory, not a secret scanner or import proof. Policy examples themselves
may match. Inspect deletion, rename, binary, symlink and gitlink changes separately.
Do not suppress tool errors or claim that unmatched terms establish neutrality.
Reuse the existing boundary checks when their scope is affected:

```bash
python tools/repo/check_agent_boundaries.py
python tools/repo/check_case_collisions.py
python -m pytest -q test/test_agent_boundary_guard.py test/test_case_collision_guard.py
```

These guards check tracked paths and shared instruction boundaries, not all runtime
imports. Select other existing tests by [docs/testing.md](docs/testing.md); no new
validation framework or full benchmark rerun is required for wording changes.

## Keep instructions shared

This is the only maintained AI instruction body. Root `CLAUDE.md` imports it with
`@AGENTS.md`. Keep module guidance in neutral READMEs. Do not commit personal Agent
workspaces, credentials, tool permissions, local overrides, or conversation logs.

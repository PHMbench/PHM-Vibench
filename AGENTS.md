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

Paper ownership is a hard boundary in `CORE.md` section 3. A request for a
single paper does not authorize storing its method, protocol, configs, scripts,
results or manuscript assets here. Edit the paper repository and reuse the
existing explicit `task.module` interface. Do not add `src/research`, a paper
namespace, an import shim or a whitelist exception. Shared promotion needs
actual independent consumers and a study-neutral contract.

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

## Upstream admission and one-way dependency

Paper repositories depend on this framework, never the reverse. Runtime, package
metadata, build/tests and default configs must not require a particular paper checkout,
import its modules, discover sibling directories, or inject its path into `sys.path`.
An explicitly caller-supplied config, output path or supported extension module is an
input to an existing public contract, not permission to hardcode that caller upstream.
The framework must install and run its generic checks without downstream repositories.

Do not add manuscript-only seed lists, sweeps, ablation matrices, table/figure runners,
case-study selection, publication layouts/results, claim-specific prompts, thresholds,
model branches or metric overrides. Moving or renaming them does not change ownership.
Do not encode paper IDs, venues or private author/data paths into dispatch or defaults.
Resolve local resources from explicit caller configuration; never invent dataset folders,
metadata, labels or readers to satisfy an experiment. An editable installation is not
permission for a paper Agent to patch this checkout or its installed package in place.

Standard reference methods, reusable operators/readers/metrics, generic analysis, public
examples and deterministic software fixtures are legitimate framework content. Judge
purpose and dependencies, not a word blacklist: literature citations, licenses, method
names, generic tables and documented historical evidence are not paper coupling. A new
abstraction still needs the real maintained consumers required by CORE; renaming a paper
protocol or duplicating a toy example does not establish generality. A public dataset
adapter needs an independent input/output contract, not support for invented datasets.

### Required checks before committing and updating a PR

- [ ] Inspect both the staged diff and the complete branch diff against the actual PR
  base, including renamed/binary files and dependency declarations. No new paper-owned
  experiment, publication artifact or unresolved private-path leak enters upstream.
- [ ] Inspect changed imports, dynamic module loading, config defaults and packaging:
  no hardcoded reverse dependency on a downstream paper. Caller-provided public extension
  bindings remain explicit; no automatic paper discovery or fallback is introduced.
- [ ] Explain the shared contract and actual reuse. For changed executable interfaces,
  reuse or minimally extend a paper-independent test/example at the existing owner and
  run the affected standalone path without paper packages. Do not add tests for prose.
- [ ] Preserve standard splits, labels, budgets, missing-score rules and estimators.
  A deliberate benchmark protocol change requires its own authorization and comparison,
  not a hidden adjustment to favor the current paper. Run affected regressions and the
  existing required CI; do not launch all datasets or model calls for each commit.
- [ ] Resolve every actual boundary violation before publishing the patch. Record
  PASS / FAIL / NOT RUN and limitations in the existing PR, not a new audit ledger.
  Required unexecuted checks prevent acceptance/merge; a clearly marked Draft may retain
  a reviewed documentation patch when execution is unavailable. Never label it passing.

Use Bash in the existing checkout. Verify `BASE` against the current PR; do not fetch,
change branches or assume a missing/stale ref proves an empty diff:

```bash
set -euo pipefail
BASE=origin/dev
git rev-parse --verify "$BASE^{commit}"
git diff --cached --check
git diff --cached --name-status
git diff --check "$BASE"...HEAD
git diff --name-status "$BASE"...HEAD

# Triage added lines only; full diff/ownership review remains mandatory.
review_added() {
  local patch rc
  patch="$(git diff --no-ext-diff --no-textconv --no-color --unified=0 "$@")" || return "$?"
  if printf '%s\n' "$patch" | sed -n '/^+++ /d;s/^+//p' | rg -n -i -- '(ablation|camera[_-]?ready|table[_-]?[0-9]+|fig(ure)?[_-]?[0-9]+|\bours\b|/home/|/Users/|[A-Za-z]:[\\/])'; then
    printf '%s\n' 'REVIEW REQUIRED: classify each hit; this is not an automatic failure.'
  else
    rc=$?
    if [[ "$rc" -ne 1 ]]; then return "$rc"; fi
  fi
}
review_added --cached
review_added "$BASE"...HEAD

# Existing committed-tree ownership check and documentation check.
python tools/repo/check_paper_migration_status.py --mode release --base-ref "$BASE"
python -m scripts.validate_docs
```

Zero search matches do not prove architectural independence. Review hits in context;
require zero unresolved violations, not zero occurrences of these words. The policy
and its regex will themselves match. Do not use blanket documentation exclusions or
`|| true` to hide execution errors. The committed-tree checker does not replace staged
review. Reuse a valid result for unchanged code/config/environment; do not repeatedly
run unrelated suites. Missing tools, refs or host access remain explicit NOT RUN.

Historical paper material is not permission to add more. Preserve it and its immutable
references; this admission policy does not authorize broad migration, deletion, data
replacement, protocol changes, gitlink upgrades or merging another paper's PR.

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

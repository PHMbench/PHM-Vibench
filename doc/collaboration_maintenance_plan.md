# Collaboration & Maintenance Plan

A practical, step‑by‑step guide to keep this multi‑pipeline project healthy when several contributors work across branches and machines.

## 0) One‑Time Setup
- Repository hygiene
  - Protect `main` and `develop` branches (required PR, required reviews, required checks).
  - Enable branch deletion on merge; disallow force‑push on protected branches.
  - Define CODEOWNERS (see step 5) for critical paths: `src/*_factory/**`, `src/configs/**`, `scripts/**`, `configs/**`.
- Automation
  - Configure CI to run: `pytest -q test/` + lint (flake8) + type checks (mypy on `src/`).
  - Cache datasets disabled by default in CI; synthetic/unit tests only.
- Local override mechanism (cross‑device)
  - Use `configs/local/local.yaml` or `--local_config` to set per‑machine `data.data_dir` (no env vars required).
  - Pipelines 01/02/03/ID already support auto‑merge with `configs/local/local.yaml`.

## 1) Branching Model (Git)
- Long‑lived branches
  - `main`: Only tagged releases and hotfix merges. Protected.
  - `develop`: Integration branch for next release. Protected.
- Short‑lived branches
  - `feature/<slug>` for new features
  - `fix/<slug>` for bug fixes
  - `docs/<slug>` for docs only
  - `refactor/<slug>` for structural changes without behavior changes
  - `chore/<slug>` for CI/tooling/non‑code assets
- Release support
  - `release/x.y.z` branch cut from `develop` for final hardening
  - `hotfix/x.y.(z+1)` cut from `main` for urgent fixes

## 2) Commit & PR Conventions
- Commits (imperative, focused, small)
  - Style: `type(scope): subject` — e.g., `feat(prompt): add hybrid library`, `fix(trainer): guard logger when missing`.
  - Types: feat, fix, docs, refactor, perf, test, chore.
- Pull Requests
  - Small scope (≤ 300 LOC diff if possible), linked to an issue.
  - Provide: problem statement, impact summary, reproduction/validation commands, updated docs path(s).
  - Attach artifact paths under `save/...` or screenshots for UI; confirm data‑source compliance with `SECURITY.md`.
  - Draft until CI is green; request review when ready.

## 3) Review Policy
- Minimum 1 approval from CODEOWNER for touching critical paths; otherwise 1 reviewer.
- Blockers to merge
  - CI red
  - Missing tests for fix/feature
  - Docs not updated when public APIs/configs changed
  - New configuration keys without defaults or documentation
- Suggested checklist (reviewer)
  - Does it break existing configs? Any hardcoded paths?
  - Factory registry updated when adding new components?
  - Backward compatibility: YAML keys lowercase, hyphen‑separated where applicable.
  - Performance: obvious O(N^2) hazards on large tensors?

## 4) Testing Strategy
- Fast tier (PR gating)
  - `pytest -q test/` (CPU only, marks: `slow`, `gpu` skipped)
  - Lint: `flake8 src/ test/` (100‑char lines), Format: `black --check`, Import order: `isort --check` (advisory)
  - Static: `mypy src/` (advisory if codebase not fully typed)
- Extended tier (nightly or pre‑release)
  - `pytest -q -m 'not gpu' --cov=src --cov-report=term`
  - Selected `docs/hse-implementation/*` integration scripts in dry‑run mode
  - Synthetic HSE pipeline: `python scripts/hse_synthetic_demo.py`
- Dataset‑heavy validation (manual/triggered only)
  - Unified metric 1‑epoch sanity: `python script/unified_metric/pipeline/quick_validate.py --mode full_validation --config script/unified_metric/configs/unified_experiments_1epoch.yaml`

## 5) Ownership & Areas
- CODEOWNERS (suggested — create `.github/CODEOWNERS`)
  - `src/model_factory/**` @ml-core
  - `src/data_factory/**` @data-core
  - `src/task_factory/**` @task-core
  - `src/trainer_factory/**` @train-core
  - `src/configs/**` `configs/**` @config-core
  - `script/unified_metric/**` @unified-core
  - `doc/**` `docs/**` @docs-core

## 6) Versioning & Releases
- Semantic Versioning (SemVer): MAJOR.MINOR.PATCH
  - MAJOR: incompatible changes (e.g., breaking YAML keys)
  - MINOR: new features (backward compatible)
  - PATCH: bug fixes
- Release flow
  1. Cut `release/x.y.z` from `develop`.
  2. Freeze: bug fixes only; update docs & CHANGELOG.
  3. Tag on `main`: `vX.Y.Z`; upload wheel if packaging; attach release notes.
  4. Merge `main -> develop` to keep lines in sync.
- Hotfix
  - `hotfix/x.y.(z+1)` from `main`, PR back to `main`, then back‑merge to `develop`.

## 7) Configuration Governance
- Single‑source of truth is YAML in `configs/**`.
- Local per‑machine overrides only in `configs/local/local.yaml` (or pass `--local_config`).
- For new keys
  - Provide defaults and document in `src/configs/README.md`.
  - Ensure keys are lowercase with hyphenated values to match `configs/demo/` samples.
  - Add validation (if non‑optional) and examples.

## 8) Data, Secrets, and Paths
- Data paths
  - Never hardcode absolute paths in code; use YAML + local override.
  - Raw input under `data/`, results under `save/`, visuals in `pic/`, docs in `docs/`.
- Secrets
  - Keep tokens (wandb, swanlab) out of code; use environment or CI secrets store.
- Reproducibility
  - For every major PR, include quick commands to reproduce the main figure/metric.

## 9) Backport & Sync Policy
- Backport (patches) strategy
  - If a bug exists on both `main` and `develop`, fix on the oldest affected branch, then cherry‑pick forward.
- Cherry‑picking
  - Use `-x` flag to keep origin reference and cross‑link issues.
- Divergence handling
  - Prefer `develop` as integration target for features; minimize long‑running branches.

## 10) Documentation Workflow
- Co‑locate user docs in `docs/`, dev docs in `doc/`.
- When adding/updating features
  - Update a “What’s New” snippet in `docs/CHANGELOG.md` or `CHANGELOG.md`.
  - Add quickstart or example config under `configs/experiments/` if applicable.
- Keep `AGENTS.md` in sync for coding agents.

## 11) Issue Management & Labels
- Labels (suggested): `bug`, `feature`, `refactor`, `docs`, `performance`, `help wanted`, `good first issue`, `blocked`, `needs discussion`, `breaking change`.
- Triage process
  - Weekly triage: assign owners, set milestone, ensure reproduction steps exist.
  - Use project board for `Backlog → Ready → In Progress → Review → Done`.

## 12) Daily/Weekly/Release Checklists
- Daily
  - Rebase your branch on latest `develop`; keep diffs small.
  - Run fast tier tests locally before pushing.
- Weekly
  - Triage issues; close stale PRs; review flaky tests.
  - Sync docs and configs status with recent merges.
- Release
  - Verify CI matrix passes on `release/x.y.z`.
  - Ensure local override docs are correct; provide 1‑epoch unified metric smoke result.
  - Update `CHANGELOG.md` and tag.

## 13) Contributor Quickstart (Step‑By‑Step)
1. Clone repo; create and activate venv.
2. `pip install -r requirements.txt` (add `dev/test_history/requirements-test.txt` if you develop tests).
3. Create `configs/local/local.yaml` with your `data.data_dir`.
4. Create a feature branch: `git checkout -b feature/prompt-soft-hard-selector`.
5. Implement changes; add tests under `test/` if needed; run `pytest -q`.
6. Run a quick pipeline smoke (CPU): `python script/unified_metric/pipeline/quick_validate.py --mode health_check --config script/unified_metric/configs/unified_experiments_1epoch.yaml`.
7. Push, open PR; fill template with problem, validation, impact, and sample commands.
8. Address reviews; squash when merging if commits are noisy.

## 14) Deprecation & Breaking Changes
- Announce deprecations in release notes; keep old keys for one MINOR version if feasible.
- Provide migration snippets for YAML and CLI (before/after examples).

## 15) Long‑term Roadmap (Sample)
- Q1: Stabilize HSE prompt library/selector; expand tests; publish ablations.
- Q2: Full CI matrix (Linux + CUDA runners), unified metric end‑to‑end nightly.
- Q3: Data reader registry audit; add dataset validation script; doc hardening.
- Q4: Packaging (pip installable core) and example notebooks.

---

Maintainers can adapt this plan incrementally—start with branching/CI/PR reviews, then expand CODEOWNERS, labels, and release cadence once the team grows.

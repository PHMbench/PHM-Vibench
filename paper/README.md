# paper/

Research materials, paper experiments, and paper-facing workspaces live here.
This directory is intentionally separate from the maintained core onboarding
path. Core validation must not depend on paper-only scripts, raw paper results,
or private submodules.

## Read First

- Submodule operations: `paper/README_SUBMODULE.md`
- UXFD suite index: `paper/UXFD_paper/README.md`
- Repo-wide navigation and avoid-read rules: `docs/REPO_INDEX.md`
- Core run contract: `AGENTS.md` and `CLAUDE.md`

## Directory Index

| Path | Type | Role | Entry document |
|---|---|---|---|
| `paper/2025-10_foundation_model_0_metric/` | git submodule | HSE/HSE-Prompt metric and foundation-model paper work | `paper/2025-10_foundation_model_0_metric/README.md` |
| `paper/LQ_vibench_fix/` | git submodule | LQ fix history, UXFD merge notes, review artifacts, and reports | `paper/LQ_vibench_fix/merge_uxfd/README.md` |
| `paper/UXFD_paper/` | parent index plus submodules | UXFD paper family and 7 paper repositories | `paper/UXFD_paper/README.md` |

## Local Or Ignored Workspaces

| Path | Status | Notes |
|---|---|---|
| `paper/UXFD_paper/thu_liqi_phd_thesis/` | ignored local repo | Independent thesis workspace, not tracked by this parent repo and not one of the 7 UXFD submodules. |
| `paper/UXFD_paper/results/` | parent artifact area | Shared UXFD result artifacts; inspect only when a task names a result or figure. |

## Working Rules

- Read the paper index files before opening paper submodule internals.
- Do not recursively read all of `paper/`; it contains submodules, drafts,
  results, generated figures, and local agent artifacts.
- Paper-specific configs, scripts, and outputs stay inside the relevant
  submodule unless they are reusable core PHM-Vibench code.
- If a submodule file changes, commit inside that submodule first. The parent
  repo should only record the updated gitlink when that pointer change is
  intentional.
- Keep `python main.py --config <yaml> [--override key=value ...]` as the
  maintained core entrypoint. Older paper README commands may still mention
  historical `Paper/...`, `--config_dir`, or `--config_path` forms; treat those
  as paper-local history until updated inside the submodule.

## Current Review Findings

- `paper/README_SUBMODULE.md` is the parent-level source for submodule handling.
- The UXFD family has 7 tracked paper submodules under `paper/UXFD_paper/`.
- The UXFD submodules provide `VIBENCH.md` and `configs/vibench/min.yaml` as the
  parent-facing mapping and minimal config contract.
- Several paper READMEs still contain historical paths, absolute paths, or old
  CLI flags. Do not use them as core-repo contracts without checking the current
  root docs and `VIBENCH.md`.

# Documentation Audit and Information Architecture Plan

Baseline: `main` at `9554eee5cae2dd2d9aa178008cb7056096d0e44d`  
Audit date: 2026-07-14

This audit covers the maintained user, developer, contributor, release, policy,
configuration, data, application, factory, template, research, historical, and
agent-facing documentation surfaces in PHM-Vibench. It establishes the intended
single sources of truth before any broad cleanup or deletion.

## Audit rules

1. Runtime code, maintained configs, registries, tests, and generated outputs are
   treated as stronger evidence than prose.
2. A registry entry means discoverable, not release-supported.
3. `configs/config_registry.csv` is the configuration inventory source; the
   generated `docs/CONFIG_ATLAS.md` is its human-readable view.
4. Historical and research records are not current user documentation.
5. Files are not deleted merely because they are old or verbose. Deletion requires
   an authoritative replacement and a reference check.
6. Bilingual pages should share the same structure and support boundary. English
   is the canonical maintenance source unless a page explicitly says otherwise.

## Executive findings

### P1: policy templates contain false or unusable information

- `SECURITY.md` advertises support for `1.0.x`, although the repository is alpha
  and is preparing a bounded `v0.2.0` surface.
- `SECURITY.md` contains the unresolved placeholder
  `[INSERT SECURITY CONTACT EMAIL ADDRESS HERE]` and promises response times that
  are not backed by a published support process.
- `citation.cff` is empty and uses the non-canonical lowercase filename.
- `contributing.md` embeds a Code of Conduct with `[INSERT CONTACT METHOD]` and
  collides with `CONTRIBUTING.md` on case-insensitive filesystems.

### P1: issue templates are not usable as written

- English and Chinese bug templates contain `<YOUR_REPO_URL>` placeholders.
- Both bug templates show the invalid command
  `python src/main.py --config-name=...` instead of the maintained
  `python main.py --config <yaml> [--override key=value ...]` entrypoint.
- Feature templates refer to the project as `PHMbench` rather than
  `PHM-Vibench` and do not request compatibility or maintenance-cost analysis.

### P1: contribution guides conflict with the current architecture

- `CONTRIBUTING.md` and `CONTRIBUTING_CN.md` still declare Python 3.8+ while the
  maintained workflow and CI use Python 3.10.
- They instruct contributors to register components primarily through
  `__init__.py`, while current task/model/config traceability relies on registry
  CSV files and dynamic import paths.
- They recommend adding examples directly to `configs/demo/`; unverified work
  should begin under `configs/experiments/` and be promoted only with smoke
  evidence.
- Their example test filenames do not describe the maintained test surface.
- Factory-specific contribution guides contain additional stale paths and
  patterns: raw data inside the repository, a missing `data/contribute.md`,
  `if __name__ == '__main__'` as a model-test strategy, and obsolete task/trainer
  registration instructions.

### P2: user onboarding is accurate but duplicated

The current root READMEs are materially more accurate than the historical docs,
but installation, quickstart, configuration, test, and data-path instructions
are repeated across:

- `README.md` / `README_CN.md`;
- `configs/README.md`;
- `docs/developer_guide.md`;
- `CONTRIBUTING.md` / `CONTRIBUTING_CN.md`;
- `docs/testing.md`;
- factory and demo READMEs.

This makes command drift likely. The root README should retain only the shortest
successful path and link to canonical installation, quickstart, testing, and
configuration pages.

### P2: the documentation index is missing

`docs/README.md` is a short directory note rather than a reader-oriented
navigation page. The repository lacks a clear progressive path from project
positioning to installation, first run, concepts, advanced configuration,
development, contribution, troubleshooting, and history.

### P2: current and historical material are mixed

The following trees contain useful evidence, but they are not current user
instructions:

- `docs/v0.1.0/**`;
- `docs/past/**`;
- `src/configs/plan/**`;
- `dev/agent_log/**` and other `dev/**` research records;
- `.claude/specs/**`, `.claude/bugs/**`, and `.codex/**` workflow assets;
- paper-specific notes under `paper/**`.

They should be excluded from the main navigation and described as archived,
research, or tooling material. They should not be mass-moved because published
links, paper references, or provenance may depend on their current paths.

### P2: validation scans only a subset of Markdown

`scripts/validate_docs.py` currently checks local links only in files named
`README.md`, `CLAUDE.md`, `AGENTS.md`, `GEMINI.md`, or `API_REFERENCE.md`.
Broken relative links in `CONTRIBUTING.md`, `SECURITY.md`, ordinary `docs/*.md`,
issue templates, and release notes can therefore pass the documentation gate.

## Document inventory and disposition

The table uses these states:

- **current**: part of the maintained documentation surface;
- **generated**: derived from a machine-readable source and not hand-edited;
- **historical**: preserved evidence, not current instructions;
- **research**: experimental material without release-support implications;
- **tooling**: contributor or agent workflow material, not user documentation;
- **invalid**: empty, contradictory, placeholder-based, or factually wrong.

| Path or path group | Audience / purpose | Current state | Decision | Authoritative source |
|---|---|---|---|---|
| `README.md` | New users; project positioning and shortest path | current, partly duplicated | shorten and link | `README.md` for positioning only |
| `README_CN.md` | Chinese new users | current, partly duplicated | synchronize structure with English | `README.md` + translated wording |
| `docs/README.md` | Documentation directory entry | current but insufficient | replace with redirect to `docs/index.md` | `docs/index.md` |
| `docs/index.md` | Missing central navigation | missing | add | `docs/index.md` |
| `docs/installation.md` | Missing environment/install SSOT | missing | add | `docs/installation.md` |
| `docs/quickstart.md` | Missing first successful-run SSOT | missing | add | `docs/quickstart.md` |
| `docs/troubleshooting.md` | Missing common-failure SSOT | missing | add | `docs/troubleshooting.md` |
| `docs/testing.md` | Test commands and evidence levels | too sparse | rewrite | `docs/testing.md` |
| `docs/developer_guide.md` | Architecture and extension boundaries | current, command duplication | retain and reduce duplication | `docs/developer_guide.md` |
| `configs/README.md` | Config composition and tooling | current, title says v0.1.0 | rewrite config-only scope | `configs/README.md` |
| `configs/config_registry.csv` | Config inventory | current | retain | itself |
| `docs/CONFIG_ATLAS.md` | Generated config reference | generated | retain; never hand-edit | `configs/config_registry.csv` |
| `docs/config_registry_schema.md` | Registry column contract | current | retain and link from config guide | itself |
| `configs/base/**/README.md` | Base-block details | current | retain; link upward | local README + `configs/README.md` |
| `configs/demo/**/README.md` | Per-demo details | current | retain; avoid repeating install/testing | local README + registry row |
| `configs/experiments/README.md` | Local/unverified experiment boundary | current | retain | itself |
| `configs/reference/README.md` | Reference-config boundary | current | retain; mark unverified | itself |
| `data/README.md` | Data location and licensing boundary | current | retain | itself |
| `docs/custom_dataset.md` | Dataset integration tutorial | current but conflicts with data policy | rewrite | root contribution guide + data factory guide |
| `src/data_factory/contributing.md` | Data/reader implementation checklist | stale | rewrite as factory-specific addendum | `CONTRIBUTING.md` + local registry/runtime |
| `src/model_factory/contributing.md` | Model implementation checklist | stale | rewrite | `CONTRIBUTING.md` + model registry |
| `src/task_factory/contributing.md` | Task implementation checklist | stale | rewrite | `CONTRIBUTING.md` + task registry |
| `src/trainer_factory/contributing.md` | Trainer implementation checklist | stale | rewrite | `CONTRIBUTING.md` + trainer factory |
| `src/*/README.md` | Factory/component usage | mixed but generally current | retain; remove duplicated global setup | local implementation |
| lowercase `src/**/readme.md` | Redirects or stale case-colliding docs | mixed | inspect individually; canonicalize cautiously | sibling `README.md` |
| `CONTRIBUTING.md` | Contribution policy and workflow | stale and overlong | rewrite | `CONTRIBUTING.md` |
| `CONTRIBUTING_CN.md` | Chinese contribution guide | stale | rewrite after English | English guide + translation |
| `contributing.md` | Legacy short guide + embedded conduct policy | invalid duplicate | split valid policy, then delete | `CONTRIBUTING.md`; future `CODE_OF_CONDUCT.md` |
| `.github/ISSUE_TEMPLATE/**` | Structured bug/feature intake | stale placeholders and commands | rewrite and deduplicate | `CONTRIBUTING.md` |
| `.github/PULL_REQUEST_TEMPLATE.md` | Missing PR evidence checklist | missing | add | `CONTRIBUTING.md` |
| `SECURITY.md` | Vulnerability reporting and support boundary | invalid template | rewrite without invented SLA | `SECURITY.md` |
| `citation.cff` | Citation metadata | invalid: empty/lowercase | replace with `CITATION.cff` | `CITATION.cff` |
| `CHANGELOG.md` | Release change history | current | retain | itself |
| `RELEASE_NOTES_v0.2.0.md` | Release-candidate narrative | current, evidence-bounded | retain | itself |
| `MIGRATION_v0.1_to_v0.2.md` | Migration path | current | retain | itself |
| `SUPPORTED_COMPONENTS.md` | Release-supported component boundary | current | retain | itself + runtime evidence |
| `SUPPORTED_COMBINATIONS.md` | Release-supported combinations | current | retain | config registry + smoke evidence |
| `KNOWN_LIMITATIONS.md` | Explicit support/evidence limits | current | retain | itself |
| `LICENSE` | Repository software license | current | retain | itself |
| `AGENTS.md` / `AGENTS_CN.md` | Maintainer runbook | tooling/current | retain; link from development docs, not user path | `AGENTS.md` |
| `CLAUDE.md` / `CLAUDE_CN.md` | Change-strategy constraints | tooling/current | retain; do not use as primary contributor guide | `CLAUDE.md` |
| `GEMINI.md`, `Codex_agent.md` | Agent entrypoints | one current, one empty | retain `GEMINI.md`; delete empty file after reference check | canonical maintainer docs |
| `docs/app_usage.md` | Streamlit user workflow | current, overlaps app README | keep user-focused; remove architecture duplication | `docs/app_usage.md` |
| `apps/streamlit/README.md` | Streamlit architecture/testing | current | keep maintainer-focused | itself |
| `app/README.md` | Legacy app boundary | historical/compatibility | retain with explicit legacy banner | maintained Streamlit docs |
| `examples/README.md` | Example-directory boundary | current but minimal | retain; link quickstart | `docs/quickstart.md` |
| `scripts/README.md` | Script catalog | current | retain | actual scripts |
| `docs/PIPELINE_06_GENERATIVE_MIGRATION.md` | Future migration contract | current design, not support claim | retain outside user onboarding | itself |
| `docs/branch_governance_20260709.md` | Branch-decision evidence | historical governance | retain; link only from archive/governance section | itself |
| `docs/REPOSITORY_OPTIMIZATION_SOP.md` | Maintainer optimization process | current maintainer policy | retain | itself |
| `docs/HPC.md` and specialized experiment guides | Advanced/research operation | needs evidence review | retain but exclude from beginner path | local guide + validated commands |
| `docs/v0.1.0/**`, `docs/past/**` | Prior release/history | historical | preserve in place; add archive index/banners incrementally | immutable historical record |
| `src/configs/plan/**` | Old config-refactor plans | historical | preserve, exclude from current nav | historical record |
| `dev/**` documentation | Development experiments and logs | research/historical | preserve selectively; not public support evidence | local provenance |
| `paper/**` documentation | Paper/research workflows | research | preserve; validate separately from core docs | paper-specific provenance |
| `.claude/**`, `.codex/**` Markdown | Agent templates/specs/bugs/skills | tooling/research | exclude from user docs; remove only proven personal/obsolete files | tool-specific entrypoints |
| source-code block comments used as user docs | API/implementation notes | mixed | keep implementation contracts near code; move user procedures to docs | code for API contract, docs for workflow |

## Target information architecture

The target is intentionally small. Empty category directories are not required.

```text
README.md
README_CN.md
CONTRIBUTING.md
CONTRIBUTING_CN.md
SECURITY.md
CODE_OF_CONDUCT.md          # only after reporting contact is resolved
CITATION.cff
CHANGELOG.md
LICENSE

docs/
├── index.md
├── installation.md
├── quickstart.md
├── troubleshooting.md
├── testing.md
├── developer_guide.md
├── app_usage.md
├── custom_dataset.md
├── configuration/          # add only when multiple maintained pages justify it
├── development/            # add only when pages are moved for a clear reason
└── archive/
    └── README.md            # index only; do not mass-move historical files
```

Existing local READMEs under `configs/`, `data/`, `apps/`, `scripts/`, and
`src/*_factory/` remain close to the code they explain.

## Single-source-of-truth map

| Information | Canonical source | Treatment elsewhere |
|---|---|---|
| Project purpose and bounded capability | `README.md` | translated summary in `README_CN.md`; other pages link |
| Installation and environment | `docs/installation.md` | README keeps minimal commands only |
| First successful run and expected outputs | `docs/quickstart.md` | README and examples link |
| Configuration composition and tools | `configs/README.md` | quickstart links; module docs do not repeat precedence |
| Config inventory | `configs/config_registry.csv` | `docs/CONFIG_ATLAS.md` generated from it |
| Release-supported components | `SUPPORTED_COMPONENTS.md` | README summarizes without copying full tables |
| Release-supported combinations | `SUPPORTED_COMBINATIONS.md` | registry rows and release notes reference it |
| Data location/licensing policy | `data/README.md` | dataset tutorials link |
| Test commands and evidence levels | `docs/testing.md` | README/CONTRIBUTING use short links |
| Architecture and extension boundaries | `docs/developer_guide.md` | factory guides add only local details |
| Contribution process | `CONTRIBUTING.md` | README links; Chinese translation mirrors structure |
| Security reporting | `SECURITY.md` | contribution docs link; no duplicate contact policy |
| Citation metadata | `CITATION.cff` | README explains exact commit/tag citation |
| Streamlit user workflow | `docs/app_usage.md` | app README links for usage |
| Streamlit architecture/testing | `apps/streamlit/README.md` | user page avoids implementation detail |
| Release changes | `CHANGELOG.md` | release notes provide version-specific narrative |
| Historical evidence | existing paths + `docs/archive/README.md` | excluded from current navigation |

## Planned implementation PRs

### PR D1 — audit and navigation

- add this audit;
- add `docs/index.md`;
- turn `docs/README.md` into a stable pointer;
- no deletion or runtime change.

### PR D2 — user onboarding

- add canonical installation, quickstart, troubleshooting, and testing pages;
- shorten English/Chinese root READMEs to positioning + minimal success path;
- narrow `configs/README.md` to configuration-specific content;
- clarify Streamlit user/maintainer page boundaries.

### PR D3 — contribution and repository policy

- rewrite English/Chinese contribution guides;
- replace placeholder issue templates and add a PR template;
- rewrite `SECURITY.md` without invented versions, contacts, or SLAs;
- create valid `CITATION.cff` and remove the empty lowercase file;
- resolve `contributing.md` only after preserving any valid conduct policy.

### PR D4 — local factory docs and validation

- rewrite factory-specific contribution addenda;
- fix custom-dataset guidance;
- expand documentation validation to all maintained Markdown and templates;
- add focused tests for link scope and archive exclusions;
- remove empty/case-colliding files only after reference checks.

### PR D5 — historical and research boundary

- add `docs/archive/README.md`;
- add concise archive/research banners where needed;
- do not mass-move or delete paper, release, migration, or experiment evidence;
- produce a separate deletion list for maintainer approval.

## Validation requirements

Each implementation PR must run the narrow applicable subset of:

```bash
python -m scripts.validate_docs
python -m scripts.validate_configs
python -m scripts.gen_config_atlas
git diff --exit-code docs/CONFIG_ATLAS.md
git diff --check
```

User-onboarding changes must additionally verify:

```bash
python main.py --help
python main.py --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

Commands not executed in the available environment must be marked
`NOT_EXECUTED`; they must not be reported as passing.

## Decisions requiring maintainer confirmation

1. The private contact method for security and conduct reports.
2. Whether `CODE_OF_CONDUCT.md` should be adopted now or only after a private
   contact channel is published.
3. Whether old `docs/v0.1.0/**`, `docs/past/**`, and `dev/**` files have external
   citations that prevent future moves.
4. Whether bilingual parity is mandatory for every deep technical page or only
   for top-level onboarding/contribution pages.
5. Whether a stable publication/DOI exists; none is assumed in the citation
   metadata plan.

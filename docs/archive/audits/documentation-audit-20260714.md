# Documentation Audit — 2026-07-14

Status: **historical audit snapshot**

Baseline: `main` at `9554eee5cae2dd2d9aa178008cb7056096d0e44d`

This audit covers the root documentation surface, `docs/`, contribution and
security files, configuration/data/application guides, repository templates, and
representative module-level documentation discoverable in the tracked repository.
Historical research and release subtrees were classified rather than rewritten
line by line. The audit does not treat a document's existence as proof that its
claims are implemented.

## Executive findings

1. The root README had become the de facto source for installation, quickstart,
   configuration, testing, architecture, support boundaries, and contribution.
   That made it accurate but too broad and encouraged downstream duplication.
2. There was no maintained `docs/index.md`, installation page, or quickstart page.
3. English and Chinese contribution guides still described Python 3.8, broad
   component support, obsolete test filenames, and generic CI behavior that no
   longer matched the repository.
4. The lowercase `contributing.md` mixed a legacy redirect with a copied Code of
   Conduct containing `[INSERT CONTACT METHOD]`.
5. `SECURITY.md` was an uncustomized template: it claimed `1.0.x` support and
   contained an email placeholder.
6. English and Chinese issue templates contained `<YOUR_REPO_URL>` placeholders
   and an invalid historical command (`python src/main.py --config-name=...`).
7. `docs/app_usage.md` duplicated the maintained `apps/streamlit/README.md`.
8. `configs/README.md` was still titled `v0.1.0` and duplicated first-run guidance.
9. `docs/testing.md` did not describe focused tests, generated-doc checks, smoke
   evidence, CI evidence, or environment limitations.
10. `docs/HPC.md` is a long YCRC-specific note with generic `src/train.py`
    commands, cluster-specific assertions, and externally changing operational
    details. It is not a maintained PHM-Vibench user guide.
11. Historical material under `docs/v0.1.0/`, `docs/past/`, `configs/v0.0.9/`, and
    `dev/` was not clearly separated by a central navigation policy.
12. Several lowercase README files are intentional compatibility redirects and
    should not be removed solely for stylistic consistency.

## Document disposition matrix

| Path or group | Audience | Finding | Action |
| --- | --- | --- | --- |
| `README.md` | users | Accurate but overextended; stale statement that active CI is still needed | Rewrite as concise project/front-door page |
| `README_CN.md` | users | Mirrors README structure and same stale CI statement | Rewrite in parallel with English README |
| `docs/index.md` | all | Missing | Add canonical navigation and source-of-truth map |
| `docs/installation.md` | users | Missing | Add authoritative installation page |
| `docs/quickstart.md` | users | Missing | Add authoritative offline smoke walkthrough |
| `configs/README.md` | users/developers | Useful config authority, but version title and first-run duplication are stale | Retain; update title and link to quickstart |
| `data/README.md` | users/contributors | Recently rewritten; clearly separates dummy data, local data, and references | Retain as data authority |
| `docs/developer_guide.md` | developers | Current, focused, and factory-aligned | Retain; link from documentation index |
| `docs/testing.md` | developers | Too small to serve as test authority | Rewrite as test/evidence authority |
| `apps/streamlit/README.md` | users/developers | Detailed maintained UI authority | Retain |
| `docs/app_usage.md` | users | Duplicates Streamlit guide | Replace with compatibility redirect |
| `docs/custom_dataset.md` | users/contributors | Useful short bridge; overlaps factory guide | Retain for now; link to data and contribution authorities |
| `CONTRIBUTING.md` | contributors | Outdated environment, tests, support assumptions, and component instructions | Rewrite in contributor-governance PR |
| `CONTRIBUTING_CN.md` | contributors | Same problems as English guide | Rewrite together with English guide |
| `contributing.md` | contributors | Duplicate plus embedded Code of Conduct placeholder | Replace with compatibility redirect |
| `CODE_OF_CONDUCT.md` | community | Missing | Add customized conduct policy |
| `SECURITY.md` | security reporters | Incorrect supported versions and placeholder contact | Rewrite without unverifiable response-time promises |
| `CITATION.cff` | researchers | Missing | Add minimal software citation metadata after maintainer identity check |
| `.github/ISSUE_TEMPLATE/*` | users/contributors | Duplicate bilingual templates with placeholders and invalid commands | Rewrite with real repository links and evidence fields |
| `.github/PULL_REQUEST_TEMPLATE.md` | contributors | Missing | Add scope, evidence, compatibility, docs, and rollback checklist |
| `CHANGELOG.md` | users/release managers | Current release-candidate record | Retain |
| `RELEASE_NOTES_v0.2.0.md` | users/release managers | Release-specific authority | Retain; do not duplicate in general docs |
| `MIGRATION_v0.1_to_v0.2.md` | users | Migration authority | Retain |
| `SUPPORTED_COMPONENTS.md` | users/developers | Support boundary authority | Retain |
| `SUPPORTED_COMBINATIONS.md` | users/developers | Maintained combination authority | Retain |
| `KNOWN_LIMITATIONS.md` | users/developers | Limitation authority | Retain |
| `docs/CONFIG_ATLAS.md` | users/developers | Generated view of registry | Retain; never hand-edit |
| `docs/branch_governance_20260709.md` | maintainers | Important governance decision record | Retain |
| `docs/REPOSITORY_OPTIMIZATION_SOP.md` | maintainers | Stable change-management SOP | Retain |
| `docs/PIPELINE_06_GENERATIVE_MIGRATION.md` | developers/researchers | Explicit future migration contract | Retain, clearly outside release support |
| `docs/HPC.md` | specific HPC users | Unverified, cluster-specific, and not aligned to current entrypoint | Archive or replace with a warning page in a separate PR |
| `docs/v0.1.0/**` | maintainers/researchers | Historical release and planning evidence | Preserve; classify as historical |
| `docs/past/**` | maintainers/researchers | Historical guides that may contain obsolete commands | Preserve; classify as historical |
| `src/**/README.md` | developers | Local component documentation; mixed depth but useful near code | Retain; prefer links to central policies |
| lowercase redirect READMEs | compatibility | Short redirects already prevent case/link breakage | Retain unless a reference audit proves deletion safe |
| `AGENTS.md`, `CLAUDE.md`, `GEMINI.md` | maintainers/agents | Engineering runbook and constraints, not user documentation | Retain; link only from developer navigation |

## Single-source-of-truth decisions

| Information | Authority |
| --- | --- |
| Project positioning and shortest entry | `README.md` |
| Installation | `docs/installation.md` |
| First successful run | `docs/quickstart.md` |
| Documentation navigation | `docs/index.md` |
| Configuration behavior | `configs/README.md` |
| Config inventory | `configs/config_registry.csv` |
| Rendered config inventory | generated `docs/CONFIG_ATLAS.md` |
| Data layout and external-data boundary | `data/README.md` |
| Supported release components | `SUPPORTED_COMPONENTS.md` |
| Supported release combinations | `SUPPORTED_COMBINATIONS.md` |
| Known constraints | `KNOWN_LIMITATIONS.md` |
| Tests and evidence terminology | `docs/testing.md` |
| General contribution process | `CONTRIBUTING.md` |
| Factory-specific extension steps | each `src/*_factory/contributing.md` |
| Streamlit behavior | `apps/streamlit/README.md` |
| Security reporting | `SECURITY.md` |
| Release history | `CHANGELOG.md` plus versioned release notes |

## Planned PR sequence

### PR A — user documentation architecture

- add documentation index, installation guide, quickstart, archive policy, and
  this audit snapshot;
- shorten English and Chinese root READMEs;
- make testing guidance authoritative;
- turn duplicate Streamlit usage content into a redirect;
- remove the stale version label from the config guide.

### PR B — contribution and repository templates

- rewrite bilingual contribution guides;
- separate the lowercase compatibility page from the Code of Conduct;
- replace the security template;
- add a pull-request template;
- repair bilingual bug and feature templates;
- add minimal citation metadata after maintainer review.

### PR C — historical and specialist cleanup

- audit references to `docs/HPC.md` and either archive it with a warning or remove
  it if no provenance value remains;
- add explicit archive landing pages for existing historical subtrees without
  mass-renaming paths;
- review large code comments or AI workflow documents that act as unofficial
  user documentation.

## Validation requirements

Each documentation PR must run:

```bash
python -m scripts.validate_docs
python -m scripts.validate_configs
python -m scripts.gen_config_atlas
git diff --exit-code docs/CONFIG_ATLAS.md
git diff --check
```

A quickstart or command change additionally requires the repository-shipped smoke
command in an environment with the maintained runtime dependencies. Missing local
runtime dependencies must be reported as `NOT_EXECUTED` or `FAILED`, never as a
pass.

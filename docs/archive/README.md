# Historical and Research Documentation

This page defines the boundary between current PHM-Vibench documentation and
preserved evidence. It is an index, not a new copy of historical content.

For maintained instructions, start at [`docs/index.md`](../index.md).

## Why historical material remains

Older plans, migration notes, failed experiments, paper workflows, and agent
records may provide:

- release and compatibility evidence;
- architectural decision context;
- experiment provenance;
- references used by papers, issues, or external links;
- explanations for code or configuration that still exists;
- recovery information for superseded branches and workflows.

Age, verbosity, or an obsolete command is not enough reason to delete evidence.
At the same time, preserved material must not be presented as current usage or
release support.

## Preserved locations

| Location | Classification | Use |
|---|---|---|
| `docs/v0.1.0/` | release history | v0.1 planning, migration, validation, and change records |
| `docs/past/` | historical user/developer docs | prior guides retained for provenance or old links |
| `src/configs/plan/` | historical design | old configuration-refactor plans |
| `configs/v0.0.9/` | compatibility/history | prior configuration snapshots, not current demos |
| `dev/` | research/development evidence | experiments, logs, notebooks, scripts, and prior validation material |
| `paper/` | paper-specific research | manuscripts, experiment protocols, and publication evidence |
| `.claude/` and `.codex/` | tooling/research | agent skills, specs, prompts, and workflow records |
| branch-governance and migration contracts in `docs/` | maintained design evidence | current decisions about future or historical migrations |

These locations are excluded from the beginner navigation and may be excluded
from current documentation link gates. Exclusion does not validate their
commands or claims.

## How to read archived material

Before using a historical command or config:

1. identify the commit/release it describes;
2. compare paths and parameters with current code;
3. inspect the current config with `scripts.config_inspect`;
4. check current registries and support documents;
5. run the narrowest applicable test or smoke path;
6. label any result as historical, reproduced, modified, or not reproduced.

Do not copy old claims or procedures into current docs without new evidence.

## Promote useful content back to current docs

When a historical page contains still-valid information:

- move only the verified concept or procedure into its current single source of
  truth;
- link to the historical record for provenance instead of copying the whole page;
- update commands, paths, terminology, and support boundaries;
- add tests or runtime evidence when the content describes behavior;
- avoid rewriting the historical source unless it contains a serious legal,
  security, or privacy problem.

## Delete or move historical material only after review

A deletion or move requires:

- exact file inventory;
- repository reference search;
- Git history review;
- external publication/link assessment when known;
- data/code/license assessment;
- replacement or reason no replacement is needed;
- recovery commit, tag, or bundle for significant material;
- focused PR with no unrelated cleanup.

Do not mass-delete `docs/past/`, `docs/v0.1.0/`, `dev/`, `paper/`, `.claude/`, or
`.codex/` in the name of repository neatness.

## Current decisions

The active documentation inventory and planned dispositions are recorded in
[`docs/DOCUMENTATION_AUDIT.md`](../DOCUMENTATION_AUDIT.md). Release-supported
behavior is defined separately by:

- [`SUPPORTED_COMPONENTS.md`](../../SUPPORTED_COMPONENTS.md)
- [`SUPPORTED_COMBINATIONS.md`](../../SUPPORTED_COMBINATIONS.md)
- [`KNOWN_LIMITATIONS.md`](../../KNOWN_LIMITATIONS.md)

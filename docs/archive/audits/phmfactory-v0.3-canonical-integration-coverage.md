# PHMFactory v0.3 canonical integration audit

## Current disposition

```text
repository:             PHMbench/PHM-Vibench
main:                   a331769d4005018bc833534ecf4efeb5e8a5a78d
canonical PR:           #127
integration branch:     agent/v030-canonical-integration-r2
review state:           ready for human review
source PR cleanup:      complete; closed unmerged as superseded
main modified:          no
repository renamed:     no
version finalized:      no
release/tag published:  no
```

PR #127 is the only active PHMFactory v0.3 integration PR. This audit records ancestry,
authority, protected boundaries, and merge rules; it is not a release announcement.

## Canonical ancestry

The integration branch physically preserves the accepted implementation history:

```text
main
  -> #84 repository contract
  -> #93 baseline inventory
  -> #94 public package / CLI
  -> #95 Pipeline rename
  -> #96 config resolver
  -> #97 CWRU bundle
  -> #98 dependency ownership
  -> #99 UI consolidation
  -> #103 workspace cleanup
  -> #105 case guard
  -> #108 historical docs
  -> #120 readiness base
  -> #110 Agent boundary
  -> #113 reader documentation
  -> #114 branding and v0.2 RC provenance
  -> #119 backend policy and final readiness authority
  -> #122 paper migration tracker
  -> integration reconciliation and review commits
```

The exact heads of PRs #108, #110, #113, #114, #119, #120, and #122 are physical
ancestors of PR #127. PR #117 and PR #121 are intentionally not ancestors:

- PR #117 contributed only its non-conflicting baseline/frozen-gitlink audit evidence;
  its personal-account backend candidate was superseded by PR #119.
- PR #121 contributed only the exact `MIGRATION_v0.2_to_v0.3.md`; its stale ancestry
  and duplicate release-document variants were excluded.

## Single-source authority map

| Subject | Authority in PR #127 |
| --- | --- |
| Repository and protected-runtime contract | PR #84 documents |
| Runtime/reader fingerprints | PR #93 audit artifacts |
| Public package, CLI, config, Pipeline names | implementation inherited from PRs #94–#96 |
| CWRU bundle interface | PR #97 implementation and `docs/CWRU_DEMO_V0_3.md` |
| Dependency ownership | PR #98 implementation and `docs/DEPENDENCY_BOUNDARIES_V0_3.md` |
| Maintained UI | `apps/streamlit/` from PR #99 plus bounded lifecycle correction |
| Historical and Agent boundaries | PRs #103, #105, #108, #110, and #113 |
| v0.2 release-candidate provenance | `docs/releases/v0.2.0-rc-provenance.yaml` |
| Current submodule/backend policy | `.github/phmfactory-v0.3-submodules.allowlist.yml` from PR #119 |
| Release-readiness implementation | checker/workflow/readiness page from PR #119 on PR #120 ancestry |
| Paper gitlink migration state | PR #122 machine-readable tracker |

Release documentation is deliberately split by role:

| Document | Role |
| --- | --- |
| `CHANGELOG.md` | concise release delta |
| `RELEASE_NOTES_v0.3.0.md` | user-facing overview and limitations |
| `MIGRATION_v0.2_to_v0.3.md` | detailed upgrade procedure |
| `docs/PHMFACTORY_V0_3_RELEASE_READINESS.md` | exact machine-gated blockers |

Do not duplicate the full migration map or blocker rationale across those files.

## Source PR consolidation

The following PHMFactory v0.3 source PRs were closed **without merge** after their
accepted content and ancestry were preserved in PR #127:

```text
#84 #85 #93 #94 #95 #96 #97 #98 #99
#103 #105 #108 #110 #113 #114
#117 #118 #119 #120 #121 #122
```

Special cases:

- #85 is superseded by the broader #110 Agent cleanup.
- #117 is a historical audit input; #119 owns current machine policy.
- #118 was validation-only and never mergeable by design.
- #121 had stale ancestry; only its migration guide was retained.
- #122 remains the paper-tracker source; closing it does not remove any gitlink.

The older feature/research PRs #42, #79, #80, #81, and #83 were also closed without
merge as `post-v0.3/rebuild-required`. Their branches and evidence remain available for
bounded reconstruction after the v0.3 topology is established.

Temporary integration PRs #123–#126 targeted only the integration branch. Their closed
state records ancestry construction, not a merge to `main`.

## Protected boundary

The consolidation and document cleanup do not mechanically modify:

```text
src/data_factory/reader/*.py
src/data_factory/data_factory.py
src/data_factory/dataset_task/**
src/data_factory/samplers/**
src/model_factory/**
src/task_factory/**
src/trainer_factory/**
Pipeline algorithms
```

No backend gitlink is added. No paper gitlink is removed. No CWRU revision or hash is
invented. The package remains `0.3.0.dev0`.

The only reader-area changes are documentation-only:

```text
src/data_factory/reader/README.md
removed src/data_factory/reader/CLAUDE.md
```

All six renamed Pipeline files retain the canonical PR #95 Git blob identities.

## Bounded Windows lifecycle correction

After release-document deduplication, a Windows Streamlit test exposed a process/monitor
handoff race: `get_run()` could read a stale `running` manifest and overwrite a terminal
`failed` result as `orphaned`.

The bounded fix:

- keeps completed managed processes registered until terminal state is persisted;
- reads and reconciles the run manifest under the same lock as the monitor thread;
- changes no experiment command, Pipeline, configuration, output schema, or protected
  runtime behavior.

The temporary one-shot patch workflow removed itself from the final tree.

## Validation contract

PR #127 must keep these ten workflows green on its current head:

```text
Core quality gates
Repository layout
PHMFactory public package
PHMFactory dependency ownership
PHMFactory CWRU bundle contract
Streamlit quality gates (Ubuntu and Windows)
Pipeline 06 CFM quality gates
Submodule policy
Paper migration status
PHMFactory release readiness
```

The online CWRU provider job is intentionally not PR evidence and remains a release
blocker. A green readiness workflow means the expected blocker set was found; it does
not mean strict release mode passed.

## Remaining release blockers

```text
2 x CWRU_HASH_MISSING
2 x CWRU_REVISION_FLOATING
1 x LEGACY_SUBMODULES_REMAIN
1 x PHM_DATA_FACTORY_BACKEND_PENDING
1 x REPOSITORY_RENAME_PENDING
1 x VERSION_NOT_FINAL
```

All eight paper tracker entries remain `safe_to_remove: false` until their independent
migration conditions are satisfied.

## Merge contract

If the repository owner later authorizes PR #127 to enter `main`, it must use a
**merge commit**.

```text
required: merge commit
forbidden: squash merge
forbidden: rebase merge
```

Squash or rebase would discard the exact source-head ancestry that TASK-170 exists to
preserve. Until separate merge authorization is given:

```text
PR #127 remains open and unmerged
main remains unchanged
repository remains PHMbench/PHM-Vibench
version remains 0.3.0.dev0
no v0.3.0 tag or release exists
```

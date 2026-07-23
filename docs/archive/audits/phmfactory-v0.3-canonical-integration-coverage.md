# PHMFactory v0.3 canonical integration coverage

## Status

```text
TASK:                 TASK-170 canonical ancestry consolidation
repository:           PHMbench/PHM-Vibench
integration branch:   agent/v030-canonical-integration-r2
frozen main:          a331769d4005018bc833534ecf4efeb5e8a5a78d
physical integration base:
                      19748c93fb4716a5314cdaf38317b39a84da0cd7
                      PR #120 exact head
head before this audit:
                      9ee30a516559b65c157d3eac6738ae3d2fe6c650
main modified:        no
source PR merged to main:
                      no
repository renamed:  no
version finalized:   no
tag or release:       no
```

This document records the Git ancestry and authority decisions used to consolidate the
previously divergent PHMFactory v0.3 Draft PRs. It is an integration audit, not a release
announcement and not approval to merge the integration branch.

## Canonical ancestry

The integration branch was created from the exact reviewed head of PR #120:

```text
#105 case-collision guard
  -> #108 historical documentation migration
    -> #120 release-readiness base
      -> canonical integration commits
```

The following reviewed source heads are physical ancestors of the integration branch:

| Source | Head SHA | Integration treatment |
| --- | --- | --- |
| PR #120 | `19748c93fb4716a5314cdaf38317b39a84da0cd7` | exact physical base |
| PR #110 | `7b6ccc119d8128b18b43682cdfd2e91c3868f278` | merged after exact 78-path scope inspection |
| PR #113 | `3d098efc11943b1f5ef477ccdf93a10ef27d6911` | merged after exact three-path scope inspection |
| PR #114 | `b1afdab6b0db54fce3f551e505d3b184498ee41f` | merged with explicit documentation/readiness conflict resolution |
| PR #119 | `bc43654b32c4b939ca5a9f4037c3b03a30edcf69` | merged as current backend-policy and readiness authority |
| PR #122 | `7c2e8b7e654239c396081ed011a378078764a8a2` | merged as an exact four-file paper-tracker increment |

Because PR #120 physically contains PR #108, the historical-document migration head
`1c079079ee0776b2a5b6e60772c85b130c931e30` is also inherited.

## Integration commits

| Step | Commit | Result |
| --- | --- | --- |
| I0 | branch at `19748c93...` | exact PR #120 base, ahead 0 / behind 0 |
| I1 | `a287397ea6310b5ee189c45dce6d4ac4d5d064fe` | merge PR #110 Agent boundary |
| I2 | `cfa4ad070582fef075bb7818fdfc926c3b89ce67` | merge PR #113 reader documentation |
| I3 | `097a39a1e30ae11b4a8ca697b98cd56efd4327b6` | add reconciled PR #117 audit evidence |
| I4 | `6ec5deb0bf34eb512fd0aa43e358e565a00c049b` | merge PR #114 branding and v0.2 RC provenance with conflict resolution |
| I5 | `51dcef593b536b67acfdc720c7624d3f93418280` | merge PR #119 organization backend governance and final checker authority |
| I6 | `779c89ee07916392e91985502ceb3cf8ad1e4bd5` | merge PR #122 paper migration tracker |
| I7 | `9ee30a516559b65c157d3eac6738ae3d2fe6c650` | rebuild PR #121's unique migration guide on canonical ancestry |

Temporary staging PRs #123 through #126 were used to obtain or inspect GitHub-generated
merge trees against the integration branch. They were never targeted at `main`; their
closed/merged status means only that their source head became reachable from the
integration branch.

## Authority decisions

### Repository and reader contracts

PR #84 remains the governing repository contract. PR #93 remains the immutable
protected-runtime and reader baseline. No integration decision weakens either contract.

### Historical documentation

PR #108 is authoritative for removal of:

```text
docs/past/
docs/v0.1.0/
```

`configs/v0.0.9/` remains because the protected compatibility loader still references
it. Documentation conflict resolution must not restore the removed public trees or
claim that they remain active directories.

### Agent and reader documentation

PR #110 is authoritative for the root/hidden Agent boundary. PR #113 is authoritative
for the function-based reader documentation. Integration changes no reader Python file,
reader signature, signal shape, dtype, channel order, numerical transform, or cache
behavior.

### Submodule inventory: PR #117 versus PR #119

PR #117 supplies bounded inventory evidence:

- ten frozen-baseline submodules;
- two personal gitlinks removed after complete archive verification;
- eight paper/research gitlinks retained frozen;
- reviewed source-tree evidence for `phm-data-factory`.

Its earlier personal-account backend candidate is not retained as current policy. The
reconciled audit is:

```text
docs/archive/audits/phmfactory-v0.3-submodule-inventory.md
```

PR #119 is the sole current machine-policy authority:

```text
.github/phmfactory-v0.3-submodules.allowlist.yml
```

The final target is `https://github.com/PHMbench/phm-data-factory.git`, status
`blocked_pending_org_transfer`, with no pinned organization commit and no current
`.gitmodules` integration.

### Release readiness: PR #120 versus PR #114/#119

PR #120 supplies the correct physical ancestry and the initial three-file readiness
gate. PR #114 adds the approved v0.2 release-candidate provenance decision. PR #119
supplies the final expanded checker and workflow, including backend and legacy-submodule
blockers.

The integration tree contains one checker authority:

```text
tools/repo/check_release_readiness.py
.github/workflows/phmfactory-release-readiness.yml
docs/PHMFACTORY_V0_3_RELEASE_READINESS.md
```

Those paths use PR #119 content on top of PR #120's physical ancestry.

### Paper migration: PR #122

PR #122 is authoritative for the four-file content-level migration tracker. It changes
no `.gitmodules` section and no gitlink. All eight entries remain unsafe to remove until
the target workflow/review/retention conditions are recorded.

### Release documentation: PR #121

PR #121 was based on stale readiness ancestry and failed the repository-layout guard.
Its `CHANGELOG.md` and `RELEASE_NOTES_v0.3.0.md` content is superseded by the more
complete PR #119 versions. Its unique valid deliverable was rebuilt as:

```text
MIGRATION_v0.2_to_v0.3.md
```

The stale PR #121 head is intentionally not an integration ancestor. No case guard was
weakened.

## Supersession and do-not-merge map

| PR | Disposition |
| --- | --- |
| #82 | implementation evidence only; personal URL and protected-runtime changes make it non-mergeable for v0.3 |
| #85 | superseded by the broader, validated PR #110 Agent-boundary cleanup |
| #100 | superseded by #103 |
| #104 | superseded by #105 |
| #106/#109 | superseded by #120 |
| #107 | superseded by #108 |
| #117 | audit evidence ported; old allowlist representation superseded by #119 |
| #118 | validation-only; must not be merged |
| #121 | stale ancestry; unique migration guide rebuilt, other content superseded |

Source PR cleanup is a separate repository-maintenance action. This integration audit
does not close those PRs automatically.

## Protected-boundary result

The consolidation introduces no new algorithmic or data-semantic change. In particular:

```text
reader Python files changed by TASK-170: 0
src/data_factory/data_factory.py changed by TASK-170: no
model/task/trainer implementation changes by TASK-170: no
Pipeline algorithm changes by TASK-170: no
.gitmodules changes by TASK-170: no
gitlink changes by TASK-170: no
CWRU revisions or hashes fabricated: no
package version changed to 0.3.0: no
```

The reader-area changes inherited from PR #113 are documentation-only:

```text
src/data_factory/reader/README.md
removed src/data_factory/reader/CLAUDE.md
```

## Expected pre-release blockers

Strict release mode is expected to remain blocked by exactly these genuine conditions:

```text
2 x CWRU_HASH_MISSING
2 x CWRU_REVISION_FLOATING
1 x LEGACY_SUBMODULES_REMAIN
1 x PHM_DATA_FACTORY_BACKEND_PENDING
1 x REPOSITORY_RENAME_PENDING
1 x VERSION_NOT_FINAL
```

A successful readiness workflow at this stage means the audit ran and found the expected
blocker set. It does not mean release mode passed.

## Required integration validation

Before this branch can be considered a reviewed release-candidate input, GitHub Actions
must validate the combined tree rather than the former independent source trees:

```text
Repository layout / case-insensitive paths
Agent boundary
Core documentation and configuration contracts
Generated CONFIG_ATLAS parity
Whitespace
Public package, wheel, sdist, and entrypoints
Dependency ownership
CWRU offline bundle contract
Streamlit Ubuntu and Windows
Offline Dummy smoke
Pipeline 06 shell and CFM contracts
UXFD focused contract
Submodule policy
Paper migration policy
Release-readiness audit and expected strict failure
```

Any integration failure must be corrected on this branch without weakening an existing
guard or claiming a release blocker is complete.

## Merge and release boundary

This audit authorizes neither merge nor publication. Until a separate review accepts the
combined tree:

```text
main remains unchanged
all source work remains unmerged to main
repository remains PHMbench/PHM-Vibench
version remains 0.3.0.dev0
no v0.3.0 tag or release is created
```

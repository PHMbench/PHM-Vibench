# PHMFactory v0.3 canonical integration coverage

## Status

```text
TASK:                    TASK-170 canonical ancestry consolidation
repository:              PHMbench/PHM-Vibench
integration branch:      agent/v030-canonical-integration-r2
frozen main:             a331769d4005018bc833534ecf4efeb5e8a5a78d
physical integration base:
                          19748c93fb4716a5314cdaf38317b39a84da0cd7
                          PR #120 exact head
full-tree validated head:
                          5a5944208c57f634a91a5294ee370382319e8de8
independent AI review:   accepted after audit-metadata correction
GitHub human approval:   not recorded
main modified:           no
repository renamed:      no
version finalized:       no
tag or release:          no
```

This document records the physical Git ancestry, source authority, content-only ports,
combined-tree validation, and review boundary for the PHMFactory v0.3 integration Draft
PR. It is not a release announcement and does not authorize a merge to `main`.

## Terminology

The source heads below are **validated source heads**. This means their declared scope,
Git ancestry, exact-file identity, and relevant CI evidence were checked. It does not
mean GitHub contains a human `APPROVED` review submission for every source PR.

At the time of this audit, PR #127 has no GitHub human approval. GitHub Copilot did not
perform a code review because the pull request changes more than its 300-file limit.
The independent review recorded here therefore uses layered ancestry, blob-identity,
protected-boundary, net-diff, and workflow evidence rather than a single monolithic UI
review.

## Canonical ancestry

The integration branch was created from the exact head of PR #120:

```text
#105 case-collision guard
  -> #108 historical documentation migration
    -> #120 release-readiness base
      -> canonical integration commits
```

The following source heads are physical ancestors of the integration branch. Connector
comparison returned `behind_by: 0` and a merge base equal to each source head.

| Source | Head SHA | Integration treatment |
| --- | --- | --- |
| PR #108 | `1c079079ee0776b2a5b6e60772c85b130c931e30` | inherited through PR #120 |
| PR #110 | `7b6ccc119d8128b18b43682cdfd2e91c3868f278` | physical ancestor after bounded Agent-boundary merge |
| PR #113 | `3d098efc11943b1f5ef477ccdf93a10ef27d6911` | physical ancestor after exact three-path reader-document merge |
| PR #114 | `b1afdab6b0db54fce3f551e505d3b184498ee41f` | physical ancestor with explicit documentation/readiness conflict resolution |
| PR #119 | `bc43654b32c4b939ca5a9f4037c3b03a30edcf69` | physical ancestor and final backend/readiness content authority |
| PR #120 | `19748c93fb4716a5314cdaf38317b39a84da0cd7` | exact physical base |
| PR #122 | `7c2e8b7e654239c396081ed011a378078764a8a2` | physical ancestor as exact four-file paper tracker |

PR #117 and PR #121 are deliberately not ancestors:

- PR #117 remains on the earlier personal-backend policy representation. Its valid
  inventory evidence was reconciled into a new audit while PR #119 remains the current
  machine-policy authority.
- PR #121 has stale readiness ancestry. Only its unique migration-guide file was ported
  exactly; its stale ancestry and superseded release documents were excluded.

## Integration commits

| Step | Commit | Result |
| --- | --- | --- |
| I0 | branch at `19748c93...` | exact PR #120 base |
| I1 | `a287397ea6310b5ee189c45dce6d4ac4d5d064fe` | merge PR #110 Agent boundary |
| I2 | `cfa4ad070582fef075bb7818fdfc926c3b89ce67` | merge PR #113 reader documentation |
| I3 | `097a39a1e30ae11b4a8ca697b98cd56efd4327b6` | add reconciled PR #117 audit evidence |
| I4 | `6ec5deb0bf34eb512fd0aa43e358e565a00c049b` | merge PR #114 branding and v0.2 RC provenance |
| I5 | `51dcef593b536b67acfdc720c7624d3f93418280` | merge PR #119 backend governance and final readiness authority |
| I6 | `779c89ee07916392e91985502ceb3cf8ad1e4bd5` | merge PR #122 paper migration tracker |
| I7 | `9ee30a516559b65c157d3eac6738ae3d2fe6c650` | port PR #121 migration guide without stale ancestry |
| I8 | `5a5944208c57f634a91a5294ee370382319e8de8` | record canonical integration coverage |

Temporary staging PRs #123 through #126 were targeted at the integration branch, never
at `main`. Their closed/merged state records reachability from the integration branch;
it is not a release or `main` merge.

## Authority decisions

### Repository and protected runtime

PR #84 remains the governing repository contract. PR #93 remains the immutable reader
and protected-runtime baseline. The consolidation does not weaken either contract.

### Historical documentation

PR #108 is authoritative for removing:

```text
docs/past/
docs/v0.1.0/
```

`configs/v0.0.9/` remains because protected compatibility code still references it.
Conflict resolution does not restore the removed trees or represent them as current
user guidance.

### Agent and reader documentation

PR #110 is authoritative for the root and hidden Agent boundary. PR #113 is authoritative
for the function-based reader documentation. No reader Python implementation is changed
by TASK-170.

### Submodule inventory: PR #117 versus PR #119

PR #117 supplies bounded inventory evidence:

- ten frozen-baseline submodules;
- two personal gitlinks removed after complete archive verification;
- eight paper/research gitlinks retained frozen;
- reviewed personal-source tree evidence for `phm-data-factory`.

Its personal-account backend candidate is not retained as current policy. The reconciled
history is stored in:

```text
docs/archive/audits/phmfactory-v0.3-submodule-inventory.md
```

PR #119 is the sole current machine-policy authority:

```text
.github/phmfactory-v0.3-submodules.allowlist.yml
```

The final target is `https://github.com/PHMbench/phm-data-factory.git`, status
`blocked_pending_org_transfer`, with no approved organization commit and no backend
`.gitmodules` entry.

### Release readiness: PR #120 versus PR #114/#119

PR #120 supplies the correct physical ancestry and the initial readiness gate. PR #114
supplies the v0.2 release-candidate provenance decision. PR #119 supplies the final
expanded checker, workflow, and readiness document, including backend and legacy-paper
blockers.

The following files in the integration tree have the exact Git blob identity of PR #119:

```text
.github/phmfactory-v0.3-submodules.allowlist.yml
tools/repo/check_release_readiness.py
.github/workflows/phmfactory-release-readiness.yml
docs/PHMFACTORY_V0_3_RELEASE_READINESS.md
CHANGELOG.md
RELEASE_NOTES_v0.3.0.md
```

### Paper migration: PR #122

PR #122 is authoritative for the four-file content-level paper migration tracker. All
four files retain the exact Git blob identity of PR #122. No `.gitmodules` section or
gitlink is deleted, and all eight entries remain `safe_to_remove: false`.

### Release documentation: PR #121

PR #121 failed the case-insensitive repository-layout guard because it was based on stale
ancestry. Its `CHANGELOG.md` and `RELEASE_NOTES_v0.3.0.md` are superseded by PR #119.
Its unique valid deliverable was copied exactly as:

```text
MIGRATION_v0.2_to_v0.3.md
```

The PR #121 head remains outside the canonical ancestry. No repository guard was weakened.

## Protected-boundary result

The independent review found no TASK-170 change to the protected runtime beyond the
already accepted source-PR scopes:

```text
reader Python files changed by TASK-170: 0
src/data_factory/data_factory.py changed by TASK-170: no
model/task/trainer implementation changes by TASK-170: no
Pipeline algorithm changes by TASK-170: no
backend .gitmodules entry added: no
paper gitlink removed: no
CWRU revisions or hashes fabricated: no
package version changed to 0.3.0: no
```

The reader-area changes inherited from PR #113 are documentation-only:

```text
src/data_factory/reader/README.md
removed src/data_factory/reader/CLAUDE.md
```

All six renamed Pipeline files have the same Git blob identities as the canonical PR #95
head, so later integration commits did not rewrite their algorithms.

## Combined-tree validation

The full integration tree at
`5a5944208c57f634a91a5294ee370382319e8de8` completed ten pull-request workflows with
terminal `success`:

| Workflow | Run |
| --- | ---: |
| Core quality gates | `30004188592` |
| Repository layout | `30004188617` |
| PHMFactory public package | `30004188597` |
| PHMFactory dependency ownership | `30004188555` |
| PHMFactory CWRU bundle contract | `30004188613` |
| Streamlit quality gates | `30004188650` |
| Pipeline 06 CFM quality gates | `30004188583` |
| Submodule policy | `30004188547` |
| Paper migration status | `30004188509` |
| PHMFactory release readiness | `30004188528` |

The jobs covered documentation/config validation, Atlas parity, whitespace, offline Dummy
smoke, public package and entrypoint tests, wheel/sdist build, wheel inspection, clean
installation, dependency ownership, offline CWRU validation, Streamlit on Ubuntu and
Windows, Pipeline 06/CFM, UXFD, portable paths, case-insensitive paths, Agent boundaries,
submodule policy, paper policy, and expected strict release blocking.

The manual online CWRU provider job was skipped by design on the pull request. It remains
a release blocker and is not online parity evidence.

## Expected pre-release blockers

Strict release mode remains blocked by exactly eight genuine findings:

```text
2 x CWRU_HASH_MISSING
2 x CWRU_REVISION_FLOATING
1 x LEGACY_SUBMODULES_REMAIN
1 x PHM_DATA_FACTORY_BACKEND_PENDING
1 x REPOSITORY_RENAME_PENDING
1 x VERSION_NOT_FINAL
```

A successful readiness workflow at this stage means the checker found the expected
pre-release blocker set. It does not mean release mode passed.

## Independent-review disposition

The independent AI review examined:

- physical ancestry and merge bases;
- exact source-authority blob identity;
- the two content-only port boundaries;
- protected runtime and Pipeline identity;
- the 366-file net diff by source layer;
- job-level combined-tree CI evidence;
- remaining release blockers and non-actions.

The code and governance tree is accepted for the next decision point after correcting
two audit-only issues:

1. the machine snapshot previously said combined-tree validation was pending after all
   ten workflows had completed;
2. the audit used “reviewed source heads” although GitHub contains no human review
   submission for those source PRs.

Both are corrected here and in the machine-readable snapshot. The source heads are now
accurately described as validated, and GitHub human approval remains explicitly pending.

## Merge contract

If a later authorization permits PR #127 to enter `main`, it must use a **merge commit**.

```text
required: merge commit
forbidden: squash merge
forbidden: rebase merge
```

The exact source-head ancestry is the principal deliverable of TASK-170. Squash or rebase
would discard that ancestry in `main` and invalidate the consolidation evidence. PR #127
must remain Draft until the merge sequence and source-PR supersession plan are separately
authorized.

## Supersession and do-not-merge map

| PR | Disposition |
| --- | --- |
| #82 | implementation evidence only; not mergeable for v0.3 |
| #85 | superseded by PR #110 |
| #100 | superseded by PR #103 |
| #104 | superseded by PR #105 |
| #106/#109 | superseded by PR #120 |
| #107 | superseded by PR #108 |
| #117 | audit evidence ported; old allowlist representation superseded by PR #119 |
| #118 | validation-only; must not be merged |
| #121 | stale ancestry; migration guide ported and other content superseded |

Source-PR cleanup is a separate repository-maintenance action. This audit does not close
or merge those PRs.

## Merge and release boundary

This review authorizes neither merge nor publication:

```text
main remains unchanged
repository remains PHMbench/PHM-Vibench
version remains 0.3.0.dev0
paper gitlinks remain 8/8 present
backend organization transfer remains pending
CWRU immutable revisions and hashes remain pending
no v0.3.0 tag or release is created
```

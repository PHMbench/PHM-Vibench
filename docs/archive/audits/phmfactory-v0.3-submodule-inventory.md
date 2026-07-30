# PHMFactory v0.3 submodule inventory and governance state

## Purpose

This audit preserves the bounded inventory evidence developed in PR #117 while applying
the later organization-ownership decision from PR #119. It separates four states that
must not be conflated:

```text
removed personal workspaces
frozen paper/research gitlinks
reviewed personal-source backend evidence
organization-owned backend target
```

A repository name, an existing destination, or a historical gitlink is not by itself
sufficient evidence that a submodule belongs in the public PHMFactory runtime.

## Frozen baseline

The immutable repository baseline contained ten configured submodules:

```text
source commit: a331769d4005018bc833534ecf4efeb5e8a5a78d
configured paths: 10
gitlinks:        10
```

Classification at that baseline:

```text
personal workspaces: 2
paper workspaces:    8
framework backends:  0
```

The original `.gitmodules`, URLs, branch metadata, and all ten gitlink commits are
preserved in the approved personal-fork archive. That archive has no installation,
runtime, test, build, or release role in PHMFactory.

## Personal workspaces already removed

The following gitlinks were removed only after their complete external commit trees were
reconstructed and verified in the personal fork:

| Former upstream path | Fixed external commit | Preserved blobs | Status |
| --- | --- | ---: | --- |
| `data/Rotor_simulation` | `d46d089c5a086965dda5555734692114bc347437` | 41 | removed by canonical PR #103 |
| `paper/LQ_vibench_fix` | `1a15710fd532fad73c552704f48349576d843ee0` | 152 | removed by canonical PR #103 |

Neither path is permitted to return through a later rebase, merge, or `.gitmodules`
conflict resolution.

## Current frozen paper gitlinks

The current v0.3 migration tree retains eight paper/research gitlinks:

| Path | Fixed gitlink | State |
| --- | --- | --- |
| `paper/2025-10_foundation_model_0_metric` | `2dd7dabe10c11a18e7a1d865ddcf70ba95f26ac7` | frozen |
| `paper/UXFD_paper/1D-2D_fusion_explainable` | `b385b07e82d6323a291d90e55a5ef4aff9336c0b` | frozen |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `379244dc8410eea0580714bc216c71074acba3a9` | frozen |
| `paper/UXFD_paper/LLM_Explainable_FD_Toolkit` | `08eb944dd9acdbbd6c69cf39f050854a936e9b78` | frozen |
| `paper/UXFD_paper/MOE_explainable` | `0da06ae366eeb66d852c144013af6925446615cb` | frozen |
| `paper/UXFD_paper/Paper_fuzzy_XFD` | `1bedd533bd52e7ac2592d3a6f7aeffaf25a1014f` | frozen |
| `paper/UXFD_paper/Neuralsymbolic_theory` | `ad7dc2e2851f59d941e402e68fa3d3409b39c583` | frozen |
| `paper/UXFD_paper/TII_operator_attention` | `20f47bac5c02763e1f6b856c90ce32861025c003` | frozen |

`frozen` means:

- the gitlink is retained for provenance and loss prevention;
- PHMFactory does not import it, package it, or require it for tests or release;
- it is not an approved framework dependency;
- deletion requires exact source-commit evidence and complete destination coverage;
- destination workflow and human-review status must be recorded;
- a similarly named organization repository is not sufficient evidence;
- deletion remains forbidden until the paper tracker records `safe_to_remove: true`.

## Reviewed backend source evidence

The source tree reviewed during v0.3 governance remains:

```text
repository: https://github.com/liq22/P01-phm-data-factory.git
commit:     5580fafec2ea5615f6d3276d95e1e5a948cc0f13
package:    phm-data-factory 0.2.0
license:    Apache-2.0
```

This personal-account repository is implementation and provenance evidence only. It is
not the permitted final `.gitmodules` authority.

## Organization-owned backend target

PHMFactory v0.3 permits exactly one optional framework-submodule exception target:

```text
path:   packages/phm-data-factory
URL:    https://github.com/PHMbench/phm-data-factory.git
status: blocked_pending_org_transfer
```

Current state:

```text
organization repository present: no
approved organization commit:    none
present in current .gitmodules:   no
ownership migration issue:       liq22/P01-phm-data-factory#5
```

The old consumer PR #82 must not be merged directly. It is based on the pre-cleanup
repository topology, points to a personal repository, modifies protected Data Factory
lifecycle behavior, mutates `sys.path`, and combines several ownership and runtime
concerns in one change.

Before activation, all of the following must hold:

1. the backend is public under the approved organization-owned HTTPS URL;
2. the reviewed source commit remains reachable or replacement-tree parity is proven;
3. the Apache-2.0 license remains explicit;
4. the exact organization commit is approved and pinned;
5. `.gitmodules` adds only `packages/phm-data-factory` and tracks no branch;
6. the backend remains optional and absent from the default import path;
7. uninitialized-submodule core wheel, CLI, Dummy smoke, and CWRU quickstart pass;
8. no `sys.path` mutation or silent backend fallback is introduced;
9. protected runtime fingerprints remain unchanged;
10. no paper, personal fork, or Agent runtime dependency is introduced.

## Machine-readable policy authority

The canonical machine-readable authority is:

```text
.github/phmfactory-v0.3-submodules.allowlist.yml
```

Its current-policy content is owned by PR #119 and supersedes the earlier candidate
representation in PR #117. The policy remains deny-by-default and records:

- the single organization-owned backend target;
- the two verified personal-gitlink removals;
- the eight legacy paper entries that must migrate before release;
- strict structural and release conditions.

The paper-specific content-level state is separately recorded by:

```text
docs/archive/audits/phmfactory-v0.3-paper-submodule-migration-status.yaml
docs/archive/audits/phmfactory-v0.3-paper-submodule-migration-status.md
```

## Protected boundary

This inventory does not modify or authorize changes to:

```text
.gitmodules
any gitlink
src/data_factory/data_factory.py
src/data_factory/reader/**
src/data_factory/dataset_task/**
src/data_factory/samplers/**
src/model_factory/**
src/task_factory/**
src/trainer_factory/**
src/Pipeline_*.py
```

## Source authority and supersession

```text
PR #117: bounded baseline/current-state audit evidence
PR #119: canonical backend ownership and machine-policy authority
PR #122: canonical paper content-migration tracker
```

The integration tree retains the non-conflicting audit evidence from #117 without
restoring its personal-account backend candidate as current policy.

## Rollback

A normal revert removes this audit document. It does not alter `.gitmodules`, any
gitlink, backend source, paper source, or Git history.

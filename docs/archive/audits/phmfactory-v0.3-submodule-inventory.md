# PHMFactory v0.3 submodule inventory and governance state

## Purpose

This audit separates three distinct states that must not be conflated:

```text
removed personal workspaces
frozen paper/research gitlinks
candidate framework backend
```

A repository name, existing GitHub destination, or historical gitlink is not by itself
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
preserved in the approved personal-fork archive.

## Personal workspaces already removed

The following gitlinks were removed only after their full external commit trees were
reconstructed and verified in the personal fork:

| Former upstream path | Fixed external commit | Preserved blobs | Status |
| --- | --- | ---: | --- |
| `data/Rotor_simulation` | `d46d089c5a086965dda5555734692114bc347437` | 41 | removed by canonical PR #103 |
| `paper/LQ_vibench_fix` | `1a15710fd532fad73c552704f48349576d843ee0` | 152 | removed by canonical PR #103 |

Neither path is permitted to return through a later rebase or `.gitmodules` conflict
resolution.

## Current frozen paper gitlinks

The current migration chain retains eight paper/research gitlinks:

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

- the gitlink is retained to avoid data or research loss;
- PHMFactory does not import it, package it, or require it for tests or release;
- it is not an approved framework dependency;
- deletion requires destination mapping plus content-level or hash-level coverage;
- existence of an `AI4Engineering-L` repository with a similar name is not enough.

## Candidate `phm-data-factory` backend

PR #82 proposes the only framework-submodule exception candidate:

```text
path:       packages/phm-data-factory
repository: https://github.com/liq22/P01-phm-data-factory.git
commit:     5580fafec2ea5615f6d3276d95e1e5a948cc0f13
license:    Apache-2.0
optional:   true
```

Current state:

```text
provider PR: liq22/P01-phm-data-factory#4 — Draft, unmerged
consumer PR: PHMbench/PHM-Vibench#82 — Draft, based on the old main baseline
present in current .gitmodules: no
```

Therefore the candidate is **not yet approved or active**. It must not be inserted by
blindly merging or rebasing the old `.gitmodules` patch, because that could restore the
removed personal entries or conflict with the eight frozen paper entries.

Before activation, all of the following must hold:

1. the provider commit is explicitly accepted or its provider review is merged;
2. the consumer integration is rebased onto the clean v0.3 chain;
3. `.gitmodules` adds only `packages/phm-data-factory` and changes no frozen entry;
4. the URL is public HTTPS and the gitlink is the reviewed full commit;
5. the provider remains optional;
6. uninitialized-submodule core installation, CLI, Dummy smoke, and CWRU quickstart pass;
7. focused backend and configuration tests pass;
8. no paper, personal fork, or Agent runtime dependency is introduced.

## Policy authority

The machine-readable state is:

```text
.github/phmfactory-v0.3-submodules.allowlist.yml
```

Its policy is deny-by-default. It records:

- the two verified removals;
- the eight frozen paper entries;
- the single blocked backend candidate;
- release conditions for any transition to active status.

## Scope of this PR

This audit and policy update do not modify:

```text
.gitmodules
any gitlink
src/data_factory/**
provider code
reader code
Pipeline code
configuration behavior
```

## Rollback

A normal revert restores the previous policy document. It does not alter submodule
content or Git history.

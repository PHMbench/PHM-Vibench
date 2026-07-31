# PHMFactory v0.3 P05 Gitlink Removal

## Scope

This bounded change removes only the migrated P05 parent-side gitlink:

```text
paper/UXFD_paper/Paper_fuzzy_XFD
```

It does not modify the accepted P05 destination repository, scientific claims, runtime code, configuration semantics, package version, repository identity, tags, or releases.

## Source authority

```yaml
parent_repository: PHMbench/PHM-Vibench
source_path: paper/UXFD_paper/Paper_fuzzy_XFD
source_gitlink_commit: 1bedd533bd52e7ac2592d3a6f7aeffaf25a1014f
```

## Accepted destination evidence

```yaml
target_repository: AI4Engineering-L/P05-Neuro-Fuzzy-Safe-XFD
target_pr: 2
target_validated_head: bad91311b343638d3f1a929691899dffc024ace7
target_merge_commit: c2b6238f422fb2b4d6229146db2c016b2051f69e
source_blob_count: 53
uncovered_count: 0
coverage_manifest_sha256: 5094bff21ad30dd7d5bffa519edf6cbcc3781544199bda5e06e419825ab13337
source_archive_sha256: be0efbcf96c7cceb00a00283c03db42af6c3495a37c2212737658dfeb1de0e55
```

The canonical migration tracker records `coverage_status: complete`, `target_review_status: target_merged`, and `safe_to_remove: true` for P05.

## Tree change

```text
.gitmodules: remove the P05 section
mode-160000 entry: remove the exact P05 gitlink
other paper gitlinks: unchanged
Foundation gitlink: unchanged
```

## Validation contract

The pull request must prove:

```text
P05 path is absent from .gitmodules
P05 mode-160000 entry is absent from the Git tree
remaining P06-P07 and Foundation gitlinks retain exact commits
submodule policy passes in policy mode
release mode remains blocked
core package and offline smoke remain unchanged
```

## Rollback

Before merge, reset or delete the topic branch. After merge, revert the bounded removal commit to restore the exact `.gitmodules` section and gitlink commit. The original P05 content remains independently preserved in the accepted destination repository, its provenance archive, and immutable Git history.

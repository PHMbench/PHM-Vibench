# PHMFactory v0.3 P07 Gitlink Removal

## Scope

This bounded change removes only the migrated P07 parent-side gitlink:

```text
paper/UXFD_paper/TII_operator_attention
```

It does not modify the accepted P07 destination repository, scientific claims, runtime code, configuration semantics, package version, repository identity, tags, or releases.

## Source authority

```yaml
parent_repository: PHMbench/PHM-Vibench
source_path: paper/UXFD_paper/TII_operator_attention
source_gitlink_commit: 20f47bac5c02763e1f6b856c90ce32861025c003
```

## Accepted destination evidence

```yaml
target_repository: AI4Engineering-L/P07-XOAN-Operator-Attention
target_pr: 2
target_validated_head: 74d54dd3bc22a6651748bc54024d8dfb1818e6f5
target_merge_commit: 0df3f1d57a05cef1faa5d7bca826dccea28f74f4
source_blob_count: 84
uncovered_count: 0
coverage_manifest_sha256: 029263c26aea26261f0ce207d327003de102f5dd04edce1e9f5be5d4c59d6f22
source_archive_sha256: da07d7f28ee678c0087ca001bfe1c76acdf042c0dc5a80b079963b2f9cf4af9c
```

The canonical migration tracker records `coverage_status: complete`, `target_review_status: target_merged`, and `safe_to_remove: true` for P07.

## Tree change

```text
.gitmodules: remove the P07 section
mode-160000 entry: remove the exact P07 gitlink
Foundation gitlink: unchanged
```

## Validation contract

The pull request must prove:

```text
P07 path is absent from .gitmodules
P07 mode-160000 entry is absent from the Git tree
Foundation is the only remaining legacy gitlink and retains commit 2dd7dabe10c11a18e7a1d865ddcf70ba95f26ac7
submodule policy passes in policy mode
release mode remains blocked
core package and offline smoke remain unchanged
```

## Rollback

Before merge, reset or delete the topic branch. After merge, revert the bounded removal commit to restore the exact `.gitmodules` section and gitlink commit. The original P07 content remains independently preserved in the accepted destination repository, its provenance archive, and immutable Git history.

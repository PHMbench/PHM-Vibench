# PHMFactory v0.3 P06 Gitlink Removal

## Scope

This bounded change removes only the migrated P06 parent-side gitlink:

```text
paper/UXFD_paper/Neuralsymbolic_theory
```

It does not modify the accepted P06 destination repository, scientific claims, runtime code, configuration semantics, package version, repository identity, tags, or releases.

## Source authority

```yaml
parent_repository: PHMbench/PHM-Vibench
source_path: paper/UXFD_paper/Neuralsymbolic_theory
source_gitlink_commit: ad7dc2e2851f59d941e402e68fa3d3409b39c583
```

## Accepted destination evidence

```yaml
target_repository: AI4Engineering-L/P06-Verifiable-Neural-Symbolic-XFD
target_pr: 2
target_validated_head: 3f62686809f595783322c39a1a2e99616e11bd18
target_merge_commit: 53c2ccc37ae8eae4fe07872db2225e9fff8b09bf
source_blob_count: 68
uncovered_count: 0
coverage_manifest_sha256: dd1839fb1c5f577485dfc13dd22899c09084e5508270d1f2491457c15cb3f732
source_archive_sha256: a9ac2213f311d5f17f0f76a5269ba5025d43e6e493a38449c3595e11162da702
```

The canonical migration tracker records `coverage_status: complete`, `target_review_status: target_merged`, and `safe_to_remove: true` for P06.

## Tree change

```text
.gitmodules: remove the P06 section
mode-160000 entry: remove the exact P06 gitlink
other paper gitlinks: unchanged
Foundation gitlink: unchanged
```

## Validation contract

The pull request must prove:

```text
P06 path is absent from .gitmodules
P06 mode-160000 entry is absent from the Git tree
remaining P07 and Foundation gitlinks retain exact commits
submodule policy passes in policy mode
release mode remains blocked
core package and offline smoke remain unchanged
```

## Rollback

Before merge, reset or delete the topic branch. After merge, revert the bounded removal commit to restore the exact `.gitmodules` section and gitlink commit. The original P06 content remains independently preserved in the accepted destination repository, its provenance archive, and immutable Git history.

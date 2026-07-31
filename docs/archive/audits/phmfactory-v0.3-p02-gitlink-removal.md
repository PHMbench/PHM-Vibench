# PHMFactory v0.3 P02 Gitlink Removal

## Scope

This bounded change removes only the migrated P02 parent-side gitlink:

```text
paper/UXFD_paper/Explainable_FD_Toolkit
```

It does not modify the accepted P02 destination repository, scientific claims, runtime code, configuration semantics, package version, repository identity, tags, or releases.

## Source authority

```yaml
parent_repository: PHMbench/PHM-Vibench
source_path: paper/UXFD_paper/Explainable_FD_Toolkit
source_gitlink_commit: 379244dc8410eea0580714bc216c71074acba3a9
```

## Accepted destination evidence

```yaml
target_repository: AI4Engineering-L/P02-XFD-Benchmark-Toolkit
target_pr: 2
target_validated_head: 6d73f731cd2308616b5fdac4335069878477c827
target_merge_commit: 062a4bd8a900317f14cd50d4a6227f454d2c5145
source_blob_count: 163
uncovered_count: 0
coverage_manifest_sha256: 05770099dfd98477b1eaca0a5a1f01db18595b763ef1b529f8f289c1ce61f3c0
source_archive_sha256: f6adc63df49697b1f50f6e5f99325c968c97acc8ded60efb3c4fbbb183a1a633
```

The canonical migration tracker records `coverage_status: complete`, `target_review_status: target_merged`, and `safe_to_remove: true` for P02.

## Tree change

```text
.gitmodules: remove the P02 section
mode-160000 entry: remove the exact P02 gitlink
other paper gitlinks: unchanged
Foundation gitlink: unchanged
```

## Validation contract

The pull request must prove:

```text
P02 path is absent from .gitmodules
P02 mode-160000 entry is absent from the Git tree
remaining six P03-P07 and Foundation gitlinks retain their exact commits
submodule policy passes in policy mode
paper migration tracker remains structurally valid
release mode remains blocked
core package and offline smoke remain unchanged
```

## Rollback

Before merge, reset or delete the topic branch. After merge, revert the bounded removal commit to restore the exact `.gitmodules` section and gitlink commit. The original P02 content remains independently preserved in the accepted destination repository, its provenance archive, and immutable Git history.

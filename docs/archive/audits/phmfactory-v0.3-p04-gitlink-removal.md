# PHMFactory v0.3 P04 Gitlink Removal

## Scope

This bounded change removes only the migrated P04 parent-side gitlink:

```text
paper/UXFD_paper/MOE_explainable
```

It does not modify the accepted P04 destination repository, scientific claims, runtime code, configuration semantics, package version, repository identity, tags, or releases.

## Source authority

```yaml
parent_repository: PHMbench/PHM-Vibench
source_path: paper/UXFD_paper/MOE_explainable
source_gitlink_commit: 0da06ae366eeb66d852c144013af6925446615cb
```

## Accepted destination evidence

```yaml
target_repository: AI4Engineering-L/P04-Physics-Informed-MoE-XFD
target_pr: 2
target_validated_head: bca10311e4d25145b5a7d18e491c1af639b7eca8
target_merge_commit: b983481ce02645c1e9044b98ad2528168c07ad28
source_blob_count: 48
uncovered_count: 0
coverage_manifest_sha256: 95cbebba433e8965344cce955b5d28230485a82afbb8cafa8946fb0c8d0e329a
source_archive_sha256: 8d143549375271a795462c37b0b3b103148315754e97d52683e1506193e10339
```

The canonical migration tracker records `coverage_status: complete`, `target_review_status: target_merged`, and `safe_to_remove: true` for P04.

## Tree change

```text
.gitmodules: remove the P04 section
mode-160000 entry: remove the exact P04 gitlink
other paper gitlinks: unchanged
Foundation gitlink: unchanged
```

## Validation contract

The pull request must prove:

```text
P04 path is absent from .gitmodules
P04 mode-160000 entry is absent from the Git tree
remaining P05-P07 and Foundation gitlinks retain exact commits
submodule policy passes in policy mode
release mode remains blocked
core package and offline smoke remain unchanged
```

## Rollback

Before merge, reset or delete the topic branch. After merge, revert the bounded removal commit to restore the exact `.gitmodules` section and gitlink commit. The original P04 content remains independently preserved in the accepted destination repository, its provenance archive, and immutable Git history.

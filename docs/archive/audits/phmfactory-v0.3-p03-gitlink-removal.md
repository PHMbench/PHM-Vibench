# PHMFactory v0.3 P03 Gitlink Removal

## Scope

This bounded change removes only the migrated P03 parent-side gitlink:

```text
paper/UXFD_paper/LLM_Explainable_FD_Toolkit
```

It does not modify the accepted P03 destination repository, scientific claims, runtime code, configuration semantics, package version, repository identity, tags, or releases.

## Source authority

```yaml
parent_repository: PHMbench/PHM-Vibench
source_path: paper/UXFD_paper/LLM_Explainable_FD_Toolkit
source_gitlink_commit: 08eb944dd9acdbbd6c69cf39f050854a936e9b78
```

## Accepted destination evidence

```yaml
target_repository: AI4Engineering-L/P03-Evidence-Grounded-LLM-XFD
target_pr: 2
target_validated_head: b14bd9dea0f2838c5681b2e34edaf58c91aaa6aa
target_merge_commit: 6a95f27c35b1acd905d60c6848b74f1f1337cbce
source_blob_count: 90
uncovered_count: 0
coverage_manifest_sha256: 9c34faa184756cb7a2db5d85ba3bbabac4a7116f21afbd82f3c95230641e3883
source_archive_sha256: 6db015b3505435cef73632f9e6e15cbf3fdf4c67d9fb47c38cc0d7ebdb6e366c
```

The canonical migration tracker records `coverage_status: complete`, `target_review_status: target_merged`, and `safe_to_remove: true` for P03.

## Tree change

```text
.gitmodules: remove the P03 section
mode-160000 entry: remove the exact P03 gitlink
other paper gitlinks: unchanged
Foundation gitlink: unchanged
```

## Validation contract

The pull request must prove:

```text
P03 path is absent from .gitmodules
P03 mode-160000 entry is absent from the Git tree
remaining P04-P07 and Foundation gitlinks retain their exact commits
submodule policy passes in policy mode
release mode remains blocked
core package and offline smoke remain unchanged
```

## Rollback

Before merge, reset or delete the topic branch. After merge, revert the bounded removal commit to restore the exact `.gitmodules` section and gitlink commit. The original P03 content remains independently preserved in the accepted destination repository, its provenance archive, and immutable Git history.

# PHMFactory v0.3 P01 Gitlink Removal

## Scope

This bounded change removes only the migrated P01 parent-side gitlink:

```text
paper/UXFD_paper/1D-2D_fusion_explainable
```

It does not modify the accepted P01 destination repository, scientific claims, runtime code, configuration semantics, package version, repository identity, tags, or releases.

## Source authority

```yaml
parent_repository: PHMbench/PHM-Vibench
source_path: paper/UXFD_paper/1D-2D_fusion_explainable
source_gitlink_commit: b385b07e82d6323a291d90e55a5ef4aff9336c0b
```

## Accepted destination evidence

```yaml
target_repository: AI4Engineering-L/P01-UXFD-Multimodal-Alignment
target_pr: 2
target_validated_head: a62db89f18783c9078d107f90e838900dd095a42
target_merge_commit: 964446803d58bf21472a3d7933b752d0f47aa8ef
source_blob_count: 89
uncovered_count: 0
coverage_manifest_sha256: 82cf569c2a900a9a5f0931a179cc2a5965ba48cf1f1101f407748e9721614780
source_archive_sha256: c498b48144524174e7054efc1451c52c72c51c0bc5603a0c35f0b358f5bfcb51
```

The canonical migration tracker records `coverage_status: complete`, `target_review_status: target_merged`, and `safe_to_remove: true` for P01.

## Tree change

```text
.gitmodules: remove the P01 section
mode-160000 entry: remove the exact P01 gitlink
other paper gitlinks: unchanged
Foundation gitlink: unchanged
```

## Validation contract

The pull request must prove:

```text
P01 path is absent from .gitmodules
P01 mode-160000 entry is absent from the Git tree
remaining seven legacy paper/Foundation gitlinks retain their exact commits
submodule policy passes in policy mode
paper migration tracker passes in policy mode
release mode remains blocked
core package and offline smoke remain unchanged
```

## Rollback

Before merge, reset or delete the topic branch. After merge, revert the bounded removal commit to restore the exact `.gitmodules` section and gitlink commit. The original P01 content remains independently preserved in the accepted destination repository, its provenance archive, and immutable Git history.

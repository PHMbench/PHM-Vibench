# PHMFactory v0.3 Foundation Gitlink Removal

## Scope

This bounded change removes the final legacy gitlink:

```text
paper/2025-10_foundation_model_0_metric
```

Because no configured submodules remain, `.gitmodules` is removed as an empty authority file.

## Source authority

```yaml
source_repository: liq22/PHM-Vibench-Paper-2025-Metric
source_commit: 2dd7dabe10c11a18e7a1d865ddcf70ba95f26ac7
source_paths: 257
unique_git_blob_oids: 243
```

## Accepted evidence

```yaml
partition_program_merge: 8ff9e6a776230111888b0ecb719f26fdaf4283f0
partition_manifest_sha256: d0ee37b6d1d3d6704042c59c1d16dcca0a9f70bf454b6c14c901db8e4acedf5c
p08_target_merge: 476246e0a8fe6b513e1a33638cca6dd999ae1d06
p09_target_merge: 9374427e09491cbab281c4167db5bf3539d7ac92
unassigned_paths: 0
cross_authority_overlap: 0
paper_migration_pending: 0
tracker_safe_to_remove: true
```

## Tree change

```text
.gitmodules: removed
paper/2025-10_foundation_model_0_metric: mode-160000 entry removed
all regular runtime/package/config files: unchanged
```

## Validation contract

The pull request must prove:

```text
zero mode-160000 entries
no .gitmodules file
paper migration release mode passes with zero pending papers
submodule policy passes in policy mode
release readiness no longer reports LEGACY_SUBMODULES_REMAIN
repository layout and Core quality gates pass
```

Remaining release blockers such as CWRU immutable revisions/hashes, backend decision, repository rename and final version remain independent and are not changed here.

## Rollback

A normal revert restores the exact `.gitmodules` content and Foundation gitlink commit. The original content remains preserved in the fixed source commit, the accepted partition manifest and the merged P08/P09 provenance imports.

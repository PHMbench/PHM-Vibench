# PHMFactory v0.2 provenance resolution

## Decision

The repository does not contain a final `v0.2.0` Git tag. The maintained changelog
entry is explicitly titled:

```text
v0.2.0 Release Candidate - 2026-07-11
```

PHMFactory does not retroactively create or reinterpret a final v0.2 release tag.
Instead, the v0.3 migration records one immutable pre-migration runtime baseline:

```text
repository: PHMbench/PHM-Vibench
baseline commit: a331769d4005018bc833534ecf4efeb5e8a5a78d
baseline role: pre-v0.3 runtime and repository snapshot
v0.2 status: release candidate, not a tagged final release
```

## Why this is sufficient provenance

The baseline commit is the source used for the v0.3 reader, runtime, submodule,
personal-path, and repository-boundary inventories. Protected runtime files were
fingerprinted against this commit before cleanup and migration changes began.

The record distinguishes three facts that must not be conflated:

1. a v0.2.0 release-candidate changelog existed;
2. no final `v0.2.0` Git tag was published;
3. `a331769d4005018bc833534ecf4efeb5e8a5a78d` is the immutable pre-v0.3 migration
   baseline, not a fabricated final release tag.

## Migration interpretation

Documentation may describe the transition as:

```text
PHM-Vibench v0.2 release-candidate baseline
                    ->
PHMFactory v0.3.0
```

It must not claim that `a331769...` was a tagged v0.2.0 final release.

## Release-readiness marker

```text
provenance_status: resolved_without_final_tag
baseline_sha: a331769d4005018bc833534ecf4efeb5e8a5a78d
```

The v0.3 release-readiness checker accepts this explicit record as the truthful
alternative to a visible `v0.2*` tag.

## Rollback

Removing this record reopens the `V020_PROVENANCE_UNRESOLVED` release blocker. A final
v0.2 tag must not be created after the fact merely to satisfy automation.

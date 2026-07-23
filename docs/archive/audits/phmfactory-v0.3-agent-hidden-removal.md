# PHMFactory v0.3 Hidden Agent Workspace Removal

## Scope

This audit records the v0.3.0 removal of the repository-specific Agent tooling
workspaces:

```text
.claude/
.codex/
```

## Source and preservation

- Immutable source commit: `a331769d4005018bc833534ecf4efeb5e8a5a78d`
- Source-controlled files: `65`
- Preservation method: `git archive` from the immutable source commit
- Verification method: per-file Git blob SHA comparison
- Preservation result: passed before the public deletion branch was created

The preserved copy is maintained outside the public PHMFactory upstream. It is
not a runtime, build, test, data, or release dependency of this repository.

## Guarded removal

The deletion automation required all of the following before committing:

1. exactly 65 tracked files under the approved paths;
2. no path outside `.claude/` or `.codex/`;
3. successful removal of only those paths;
4. removal of the temporary cleanup workflow from the final diff.

Deletion commit: `8d4836a9cde13abe7b27088fd41a4528d0d8c16b`

## Compatibility boundary

This removal does not modify readers, factories, Pipeline implementations,
configuration semantics, tests, datasets, or the maintained CLI. Agent tooling
may continue in downstream personal workspaces but cannot be required by the
public PHMFactory runtime.

## Rollback

A normal Git revert restores the removed paths. The independently preserved
copy retains the original path and blob identity for every removed file.

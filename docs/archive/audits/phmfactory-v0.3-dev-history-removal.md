# PHMFactory v0.3 Dev and Historical Workspace Removal

## Scope

This batch removes two non-runtime workspaces from the public framework:

```text
.archive/
dev/
```

## Immutable source

```text
repository: PHMbench/PHM-Vibench
commit:     a331769d4005018bc833534ecf4efeb5e8a5a78d
```

## Private-fork preservation

The approved personal fork stores exact reconstructed trees under:

```text
upstream-archive/phmfactory-v0.3.0/dev-history/.archive/
upstream-archive/phmfactory-v0.3.0/dev-history/dev/
```

| Source path | Preserved blob entries | Nested gitlinks | Blob verification |
| --- | ---: | ---: | --- |
| `.archive/` | 1 | 0 | PASS |
| `dev/` | 114 | 0 | PASS |

Every regular file and symlink was reconstructed from the immutable Git object
and rehashed with the Git blob algorithm.

The stacked public branch contained 113 `dev/` files at deletion time because
`dev/test_history/AGENTS.md` had already been independently archived and removed
in PR #88. Thus the private archive preserves the complete 114-file frozen
baseline while this PR removes the 113 files still present in its base branch.

## Guarded deletion

```text
deletion commit: 473b2b805fc506f103db2fd439a6d65f4080fb45
scope check:      PASS
```

The same-repository workflow verified exact path counts, rejected consumers in
runtime/config/test/script/CI surfaces, removed only `.archive/` and `dev/`, and
removed its own temporary workflow.

## Boundary

The removed trees are historical reports, scratch scripts, local development
notes, test history, or paper-oriented experiments. They are not imported by the
maintained framework runtime.

This batch does not change:

- `src/data_factory/reader/*.py`;
- any factory, model, task, trainer, or Pipeline implementation;
- maintained configurations;
- tests;
- the remaining paper submodules;
- the optional `phm-data-factory` integration proposal.

## Recovery

Recovery is possible from:

1. the immutable source commit;
2. the private-fork object-verified archive;
3. a normal revert of the deletion commit.

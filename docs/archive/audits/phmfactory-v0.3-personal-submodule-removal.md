# PHMFactory v0.3 Personal Submodule Removal

## Scope

This audit records removal of two personal-workspace gitlinks from the public
PHMFactory repository:

```text
data/Rotor_simulation
paper/LQ_vibench_fix
```

No reader, factory, Pipeline, configuration, test, or runtime implementation was
changed.

## Immutable source state

| Former upstream path | External repository | Gitlink commit |
| --- | --- | --- |
| `data/Rotor_simulation` | `liq22/Rotor_simulation` | `d46d089c5a086965dda5555734692114bc347437` |
| `paper/LQ_vibench_fix` | `liq22/LQ_vibench_fix` | `1a15710fd532fad73c552704f48349576d843ee0` |

The original `.gitmodules` and all ten baseline gitlink commits were preserved
before any public deletion.

## Content preservation

The two external commit trees were reconstructed outside the public upstream
directly from Git blob objects.

| Former path | Preserved blob entries | Nested gitlinks | Blob verification |
| --- | ---: | ---: | --- |
| `data/Rotor_simulation` | 41 | 0 | PASS |
| `paper/LQ_vibench_fix` | 152 | 0 | PASS |

For every regular file and symbolic link, the preserved bytes were rehashed with
the Git blob algorithm and compared with the source tree entry. All 193 blob
entries matched.

The preserved copy is not a runtime, build, test, data, or release dependency of
PHMFactory.

## Consumer review

Before removal:

- `data/Rotor_simulation` appeared only in `.gitmodules` and a historical
  `.archive/README.md` description;
- `paper/LQ_vibench_fix` appeared only in `.gitmodules` and `src/README.md`;
- no Python runtime, maintained config, test, script, or CI consumer referenced
  either path.

The two documentation references were neutralized before the gitlinks were
removed.

## Guarded deletion

The deletion automation required:

1. each path to be a mode `160000` gitlink;
2. each gitlink SHA to match the archived immutable commit;
3. both `.gitmodules` definitions to be absent;
4. runtime, config, test, script, and CI consumers to be absent;
5. the staged deletion to contain only the two gitlinks and the temporary
   cleanup workflow.

All conditions passed.

Deletion commit:

```text
cbe5a451222db244a3b7fcc708fcb5840445980d
```

The temporary cleanup workflow removed itself and is not part of the final
repository state.

## Remaining submodule boundary

This removal does not authorize deletion of the eight paper/research
submodules. Each requires an exact destination-repository and commit mapping.

The proposed `packages/phm-data-factory` backend is also outside this batch. It
remains the only allowed optional-backend candidate and must be reviewed and
rebased independently.

## Rollback

A normal Git revert restores the two gitlinks and their `.gitmodules` entries.
The independently preserved file trees and source manifests remain available
for recovery and audit.

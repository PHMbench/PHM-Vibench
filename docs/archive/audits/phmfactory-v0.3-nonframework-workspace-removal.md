# PHMFactory v0.3 Non-framework Workspace Removal Audit

## Scope

This audit covers the bounded PR-10 removal set:

```text
.archive/
dev/
results/README.md
metrics_reports/README.md
data/Rotor_simulation          Git submodule
paper/LQ_vibench_fix           Git submodule
```

The following are explicitly not removed in this PR:

```text
reports/
plot/
eight paper/research submodules
packages/phm-data-factory      proposed optional backend
```

`plot/` contains reusable public tooling and an owned requirements file. `reports/`
requires document-level review. The remaining paper gitlinks require content-level
paper-repository mapping or a complete personal archive before deletion.

## Regular workspace preservation

The exact frozen trees are stored in the user's personal fork:

```text
repository: liq22/PHM-Vibench
branch:     archive/phmfactory-v0.3.0-removals
```

| Source path | Frozen source commit | Preserved blobs | Verification |
| --- | --- | ---: | --- |
| `.archive/` | `a331769d4005018bc833534ecf4efeb5e8a5a78d` | 1 | per-file Git blob PASS |
| `dev/` | `a331769d4005018bc833534ecf4efeb5e8a5a78d` | 114 | per-file Git blob PASS |

Private archive locations:

```text
upstream-archive/phmfactory-v0.3.0/dev-history/.archive/
upstream-archive/phmfactory-v0.3.0/dev-history/dev/
```

## Output-placeholder preservation

The two tracked output-directory placeholders were copied byte-for-byte into:

```text
upstream-archive/phmfactory-v0.3.0/runtime-output-placeholders/
```

| Source path | Source/destination Git blob | Verification |
| --- | --- | --- |
| `results/README.md` | `8aee63ebb3e992f241314349d8195ad59332c09f` | exact blob match |
| `metrics_reports/README.md` | `4b9e74841619784841247960d897f4d1231b94b0` | exact blob match |

These files are placeholders, not source-of-truth configuration or experiment
results. Runtime output directories remain ignored after their tracked README files
are removed.

## Personal submodule preservation

Complete fixed-commit trees were reconstructed in the personal fork before the
public gitlinks are removed:

| Former public path | External repository and commit | Preserved blobs | Verification |
| --- | --- | ---: | --- |
| `data/Rotor_simulation` | `liq22/Rotor_simulation@d46d089c5a086965dda5555734692114bc347437` | 41 | per-file Git blob PASS |
| `paper/LQ_vibench_fix` | `liq22/LQ_vibench_fix@1a15710fd532fad73c552704f48349576d843ee0` | 152 | per-file Git blob PASS |

Private archive locations:

```text
upstream-archive/phmfactory-v0.3.0/personal-submodules/data/Rotor_simulation/
upstream-archive/phmfactory-v0.3.0/personal-submodules/paper/LQ_vibench_fix/
```

Neither archived workspace becomes a PHMFactory runtime, build, test, data, or
release dependency.

## Guarded public change

The deletion workflow must:

1. verify the exact regular-file counts and known source blobs;
2. verify both gitlinks are mode `160000` and equal the archived commits;
3. remove only the two approved `.gitmodules` sections;
4. remove only the approved regular paths and gitlinks;
5. retain all eight paper/research gitlinks;
6. retain `reports/` and `plot/`;
7. remove its temporary workflow and inventory workflow from the final tree;
8. run documentation, configuration, package, dependency, CWRU, Streamlit, Dummy,
   Pipeline 06, and UXFD gates.

## Rollback

A normal revert restores the regular files, gitlinks, and `.gitmodules` sections.
Exact content remains independently recoverable from Git history and the personal
fork archive.

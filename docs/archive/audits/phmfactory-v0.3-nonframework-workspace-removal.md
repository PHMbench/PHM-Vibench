# PHMFactory v0.3 Non-framework Workspace Removal Audit

## Completed scope

PR-10 removes only the following content from the public framework branch:

```text
.archive/
dev/
results/README.md
metrics_reports/README.md
data/Rotor_simulation          Git submodule
paper/LQ_vibench_fix           Git submodule
```

The atomic deletion commit is:

```text
f52a4b75b6617a03c4eaf259352590ab5b523997
```

The following remain intentionally unchanged:

```text
reports/
plot/
eight paper/research submodules
packages/phm-data-factory      proposed optional backend
```

`plot/` contains reusable public tooling and an owned requirements file. `reports/`
still requires document-level review. The eight paper gitlinks require exact target
repository and content-level verification before removal.

## Regular workspace preservation

The frozen public trees are stored in the approved personal fork:

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

The tracked output-directory placeholders were copied byte-for-byte into:

```text
upstream-archive/phmfactory-v0.3.0/runtime-output-placeholders/
```

| Source path | Source/destination Git blob | Verification |
| --- | --- | --- |
| `results/README.md` | `8aee63ebb3e992f241314349d8195ad59332c09f` | exact blob match |
| `metrics_reports/README.md` | `4b9e74841619784841247960d897f4d1231b94b0` | exact blob match |

These files were placeholders, not source-of-truth configurations or experiment
results. Runtime output paths remain local and ignored after their tracked README
files are removed.

## Personal submodule preservation

Complete fixed-commit trees were reconstructed in the personal fork before the
public gitlinks were removed:

| Former public path | External repository and commit | Preserved blobs | Verification |
| --- | --- | ---: | --- |
| `data/Rotor_simulation` | `liq22/Rotor_simulation@d46d089c5a086965dda5555734692114bc347437` | 41 | per-file Git blob PASS |
| `paper/LQ_vibench_fix` | `liq22/LQ_vibench_fix@1a15710fd532fad73c552704f48349576d843ee0` | 152 | per-file Git blob PASS |

Private archive locations:

```text
upstream-archive/phmfactory-v0.3.0/personal-submodules/data/Rotor_simulation/
upstream-archive/phmfactory-v0.3.0/personal-submodules/paper/LQ_vibench_fix/
```

The public PHMFactory repository has no runtime, build, test, data, or release
dependency on these archives.

## Atomic Git-tree verification

The deletion was constructed against parent:

```text
cbf929e26240b2b73d78a1461fde862bea388720
```

and applied as one fast-forward Git-tree commit. The resulting tree:

1. removes the approved regular paths and two personal gitlinks;
2. removes only their two `.gitmodules` sections;
3. retains all eight paper/research gitlinks;
4. retains `reports/` and `plot/`;
5. removes temporary inventory/deletion workflows from the final tree;
6. removes the stale `paper/LQ_vibench_fix` reference from `src/README.md`;
7. leaves protected reader, factory, Pipeline, config, CWRU, dependency, and UI code unchanged.

## Remaining release gate

Repository-native documentation, configuration, package, dependency, CWRU,
Streamlit, Dummy, Pipeline 06, and UXFD checks must pass on the final PR head before
this Draft PR is eligible for review or merge.

## Rollback

A normal revert of the atomic deletion commit restores the regular files, gitlinks,
and `.gitmodules` sections. Exact content is also independently recoverable from
Git history and the personal-fork archives above.

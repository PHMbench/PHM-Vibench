# PHMFactory v0.3 Submodule Inventory

## Purpose

This audit freezes the current submodule state before PHMFactory removes
personal and paper workspaces from the public framework repository. It does not
remove or modify a gitlink.

## Immutable baseline

```text
Repository: PHMbench/PHM-Vibench
Commit:     a331769d4005018bc833534ecf4efeb5e8a5a78d
```

The baseline `.gitmodules` file and every gitlink were preserved outside the
public upstream before this inventory was written.

Verification result:

```text
configured submodule paths: 10
gitlink paths:              10
path/gitlink parity:        PASS
```

The preservation record contains the original `.gitmodules`, all URLs, the one
configured branch, and the immutable gitlink commit for every path.

## Current main-branch submodules

| Path | URL | Gitlink commit | v0.3 disposition |
| --- | --- | --- | --- |
| `data/Rotor_simulation` | `git@github.com:liq22/Rotor_simulation.git` | `d46d089c5a086965dda5555734692114bc347437` | Move the simulation workspace to the personal fork; remove the public gitlink after consumer and destination verification. |
| `paper/2025-10_foundation_model_0_metric` | `git@github.com:liq22/PHM-Vibench-Paper-2025-Metric.git` | `2dd7dabe10c11a18e7a1d865ddcf70ba95f26ac7` | Maintain in its paper repository; remove from the public framework after paper destination review. |
| `paper/LQ_vibench_fix` | `git@github.com:liq22/LQ_vibench_fix.git` | `1a15710fd532fad73c552704f48349576d843ee0` | Move to the personal fork; remove from the public framework after destination verification. |
| `paper/UXFD_paper/1D-2D_fusion_explainable` | `https://github.com/liq22/1D-2D_fusion_explainable.git` | `b385b07e82d6323a291d90e55a5ef4aff9336c0b` | Maintain in the corresponding paper repository; remove after destination mapping and commit verification. |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `https://github.com/liq22/Explainable_FD_Toolkit.git` | `379244dc8410eea0580714bc216c71074acba3a9` | Maintain in the corresponding paper repository; remove after destination mapping and commit verification. |
| `paper/UXFD_paper/LLM_Explainable_FD_Toolkit` | `https://github.com/liq22/LLM_Explainable_FD_Toolkit.git` | `08eb944dd9acdbbd6c69cf39f050854a936e9b78` | Maintain in the corresponding paper repository; remove after destination mapping and commit verification. |
| `paper/UXFD_paper/MOE_explainable` | `https://github.com/liq22/MOE_explainable.git` | `0da06ae366eeb66d852c144013af6925446615cb` | Maintain in the corresponding paper repository; remove after destination mapping and commit verification. |
| `paper/UXFD_paper/Neuralsymbolic_theory` | `https://github.com/liq22/Neuralsymbolic_theory.git` | `ad7dc2e2851f59d941e402e68fa3d3409b39c583` | Maintain in the corresponding paper repository; remove after destination mapping and commit verification. |
| `paper/UXFD_paper/Paper_fuzzy_XFD` | `https://github.com/liq22/Paper_fuzzy_XFD.git` | `1bedd533bd52e7ac2592d3a6f7aeffaf25a1014f` | Maintain in the corresponding paper repository; remove after destination mapping and commit verification. |
| `paper/UXFD_paper/TII_operator_attention` | `git@github.com:liq22/TII_operator_attention.git` | `20f47bac5c02763e1f6b856c90ce32861025c003` | Maintain in the corresponding paper repository; remove after destination mapping and commit verification. |

`paper/LQ_vibench_fix` additionally configures `branch = lqfix_25-12`. The
branch field is provenance only; the recorded gitlink commit remains the actual
checked-out revision.

## Classification

The ten current entries fall into three non-framework categories:

```text
personal simulation workspace: 1
personal fix workspace:        1
paper/research workspaces:      8
```

None is approved as a PHMFactory core or optional backend dependency.

Four entries use personal SSH URLs. The remaining six use public HTTPS URLs but
still point to personal paper repositories. URL accessibility does not change
the ownership boundary: paper code remains downstream of PHMFactory.

## Approved backend exception candidate

Open integration work proposes one separately governed optional backend:

```text
Path:       packages/phm-data-factory
Repository: https://github.com/liq22/P01-phm-data-factory.git
Commit:     5580fafec2ea5615f6d3276d95e1e5a948cc0f13
License:    Apache-2.0
Status:     experimental candidate; not present in the frozen main baseline
```

This is the only submodule candidate allowed by the v0.3 repository contract.
It may be retained only when all of the following remain true:

1. the URL is public HTTPS;
2. the gitlink is pinned to a reviewed full commit SHA;
3. the repository has a compatible explicit license;
4. the backend is optional and does not block the core wheel, CLI, Dummy smoke,
   or CWRU quickstart when uninitialized;
5. PHMFactory owns and tests the adapter boundary;
6. no Agent, paper, or private-fork functionality is pulled into the public
   runtime;
7. `.gitmodules` contains no other entry after the legacy removals.

The integration PR must be rebased after the legacy-submodule cleanup so that
its `.gitmodules` change adds only the approved backend entry and cannot
reintroduce historical submodules.

## Required migration evidence

Before removing any current gitlink, record:

```text
source path
source repository URL
source gitlink commit
current consumers in code/config/tests/docs/CI
confirmed destination repository
confirmed destination commit containing required content
reviewer
verification date
safe_to_remove status
```

A path is not safe to remove merely because its external repository still
exists. References and reproduction instructions must also be redirected.

## Removal sequencing

Recommended batches:

1. personal workspaces:
   - `data/Rotor_simulation`
   - `paper/LQ_vibench_fix`
2. foundation-model paper workspace;
3. UXFD paper workspaces, grouped only after exact destination mapping;
4. normalize `.gitmodules` to the approved optional backend entry, or remove
   `.gitmodules` entirely until that backend integration is ready.

Each deletion batch must remain separate from runtime, reader, Pipeline,
configuration, or package refactoring.

## Public dependency boundary

After cleanup:

```text
paper repositories ──> PHMFactory
personal fork      ──> PHMFactory

PHMFactory ─X─> paper repositories
PHMFactory ─X─> personal fork
```

The optional data backend is an explicitly reviewed infrastructure dependency,
not a paper or personal-workspace exception.

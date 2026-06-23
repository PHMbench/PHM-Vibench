# Paper Submodule Guide

This file is the parent-repo guide for paper submodules. Paper repositories keep
their own code, configs, drafts, results, and README files. The parent
PHM-Vibench repo tracks only the submodule commit pointers.

## Registered Paper Submodules

| Path | Remote | Notes |
|---|---|---|
| `paper/2025-10_foundation_model_0_metric` | `git@github.com:liq22/PHM-Vibench-Paper-2025-Metric.git` | HSE/HSE-Prompt metric and foundation-model paper. |
| `paper/LQ_vibench_fix` | `git@github.com:liq22/LQ_vibench_fix.git` | LQ fix history and UXFD merge work. |
| `paper/UXFD_paper/1D-2D_fusion_explainable` | `https://github.com/liq22/1D-2D_fusion_explainable.git` | UXFD paper submodule. |
| `paper/UXFD_paper/Explainable_FD_Toolkit` | `https://github.com/liq22/Explainable_FD_Toolkit.git` | UXFD paper submodule. |
| `paper/UXFD_paper/LLM_Explainable_FD_Toolkit` | `https://github.com/liq22/LLM_Explainable_FD_Toolkit.git` | UXFD paper submodule. |
| `paper/UXFD_paper/MOE_explainable` | `https://github.com/liq22/MOE_explainable.git` | UXFD paper submodule. |
| `paper/UXFD_paper/Neuralsymbolic_theory` | `https://github.com/liq22/Neuralsymbolic_theory.git` | UXFD paper submodule. |
| `paper/UXFD_paper/Paper_fuzzy_XFD` | `https://github.com/liq22/Paper_fuzzy_XFD.git` | UXFD paper submodule. |
| `paper/UXFD_paper/TII_operator_attention` | `git@github.com:liq22/TII_operator_attention.git` | UXFD paper submodule. |

The source of truth for this list is `.gitmodules`.

## Common Operations

Clone with submodules:

```bash
git clone --recurse-submodules git@github.com:liq22/Vbench.git
```

Initialize after a normal clone:

```bash
git submodule update --init --recursive
```

Check submodule pointers and dirty state:

```bash
git submodule status --recursive
git status --short
```

Update a submodule to the commit recorded by the parent repo:

```bash
git submodule update --recursive paper/UXFD_paper/Explainable_FD_Toolkit
```

Update a submodule to a newer upstream commit:

```bash
cd paper/UXFD_paper/Explainable_FD_Toolkit
git fetch
git checkout <branch>
git pull --ff-only
cd -
git status --short
git add paper/UXFD_paper/Explainable_FD_Toolkit
git commit -m "Update Explainable_FD_Toolkit submodule"
```

Make content changes inside a submodule:

```bash
cd paper/UXFD_paper/Explainable_FD_Toolkit
# edit files inside the submodule
git status --short
git add <changed-files>
git commit -m "Describe paper change"
git push
cd -
git add paper/UXFD_paper/Explainable_FD_Toolkit
git commit -m "Record Explainable_FD_Toolkit submodule update"
```

## Parent-Repo Rules

- Do not copy paper-specific configs or scripts into the parent repo unless they
  are reusable core PHM-Vibench functionality.
- Do not edit submodule files when the intended change is only a parent index
  update.
- Do not commit parent gitlink changes accidentally. A changed submodule pointer
  should correspond to a real commit inside that submodule.
- Keep parent docs focused on navigation and boundaries. Detailed paper
  roadmaps, experiments, and evidence stay inside each paper repository.

## Related Indexes

- Parent paper index: `paper/README.md`
- UXFD family index: `paper/UXFD_paper/README.md`
- UXFD setup notes: `paper/UXFD_paper/README_SUBMODULE.md`

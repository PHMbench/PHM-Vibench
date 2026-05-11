# UXFD Paper Submodules

This directory hosts the 7 UXFD paper repositories as git submodules. It also
contains parent-level UXFD indexes and shared result artifacts.

## 总控索引

- OpenClaw 索引页：`/home/user/.openclaw/workspace/research_OS/01_projects/UXFD_suite.md`
- 系列统一依赖与符号映射：`paper/UXFD_paper/UXFD_Family_Tree.md`
- 子模块操作说明：`paper/UXFD_paper/README_SUBMODULE.md`
- 父仓 paper 总入口：`paper/README.md`
- 当前阶段：`repo_understanding`

## 系列级规则

- 每个子项目继续维护自己的 README、CORE、paper_blueprint 和 paper 资产。
- 跨项目依赖、统一符号、统一协议由 `UXFD_Family_Tree.md` 和 `research_OS` 共同维护。
- 当新增或重构超过 3 个子项目的系列工作时，第一步先更新统一依赖与符号映射表。
- Paper-specific configs and artifacts live inside each submodule.
- The main PHM-Vibench repo only keeps reusable common code under `src/`.
- Mapping docs (`VIBENCH.md`) live in each submodule. Do not add paper mapping
  docs to the main `docs/` directory.
- Each UXFD paper submodule is expected to provide:
  - `configs/vibench/min.yaml`: 5-block config runnable through the maintained
    root contract, `python main.py --config ...`.
  - `VIBENCH.md`: parent-facing mapping and reproduction document.

If you update `configs/vibench/min.yaml`, `VIBENCH.md`, or a paper README,
commit inside the submodule repo first. The parent repo only records the updated
gitlink pointer when that pointer change is intentional.

## UXFD Submodule Index

| Path | Layer | README | VIBENCH | Min config | Notes |
|---|---|---|---|---|---|
| `paper/UXFD_paper/Explainable_FD_Toolkit` | Infrastructure | yes | yes | yes | Common explainability API, metrics, and visualization toolkit. |
| `paper/UXFD_paper/1D-2D_fusion_explainable` | Method | yes | yes | yes | 1D time-series plus 2D spectrum fusion paper. |
| `paper/UXFD_paper/MOE_explainable` | Method | yes | yes | yes | Physics-constrained MoE and path-level interpretability. |
| `paper/UXFD_paper/Paper_fuzzy_XFD` | Method | yes | yes | yes | Fuzzy rule and fuzzy-inference XFD paper. |
| `paper/UXFD_paper/TII_operator_attention` | Theory | yes | yes | yes | Operator-attention theory; `CORE.md` is not present at the top level. |
| `paper/UXFD_paper/LLM_Explainable_FD_Toolkit` | Application | yes | yes | yes | LLM natural-language explanation layer. |
| `paper/UXFD_paper/Neuralsymbolic_theory` | Cross-layer theory | yes | yes | yes | Neural-symbolic theory and concept unification. |

## Other Directories

| Path | Status | Notes |
|---|---|---|
| `paper/UXFD_paper/results/` | parent artifact area | Shared UXFD results and figures. Read only named artifacts. |
| `paper/UXFD_paper/thu_liqi_phd_thesis/` | ignored local repo | Independent thesis workspace. It is not tracked by this repo and not one of the 7 UXFD submodules. |

## Current Cleanup Findings

- The 7 UXFD submodules have `VIBENCH.md` and `configs/vibench/min.yaml`.
- Most submodule READMEs still contain historical `Paper/...` paths, old
  `--config_dir` or `--config_path` commands, absolute paths, or historical
  roadmap sections.
- Treat `VIBENCH.md` and `configs/vibench/min.yaml` as the parent-facing
  reproduction contract until each submodule README is cleaned inside its own
  repository.
- Do not recursively read all UXFD submodules. Start with this file, then open
  the target submodule README, `VIBENCH.md`, and only the files named by the task.

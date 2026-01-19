# UXFD Paper Submodules

This directory hosts the 7 UXFD paper repositories as **git submodules**.

Rules:
- Paper-specific configs and artifacts live **inside each submodule**.
- The main PHM‑Vibench repo only keeps reusable common code under `src/`.
- Mapping docs (`VIBENCH.md`) live in each submodule (do not add paper mapping docs to the main `docs/`).
- Each paper submodule is expected to provide:
  - `configs/vibench/min.yaml` (5-block config; runnable via `python main.py --config ...`)
  - `VIBENCH.md` (the only supported mapping / reproduction doc for that paper)

Note:
- If you update `configs/vibench/min.yaml` or `VIBENCH.md`, commit inside the submodule repo; the parent repo only
  updates the gitlink pointer.

Submodules (paths):
- `paper/UXFD_paper/1D-2D_fusion_explainable`
- `paper/UXFD_paper/Explainable_FD_Toolkit`
- `paper/UXFD_paper/LLM_Explainable_FD_Toolkit`
- `paper/UXFD_paper/MOE_explainable`
- `paper/UXFD_paper/Paper_fuzzy_XFD`
- `paper/UXFD_paper/Neuralsymbolic_theory`
- `paper/UXFD_paper/TII_operator_attention`

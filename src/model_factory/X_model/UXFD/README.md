# UXFD (Common Modules)

This package hosts **common, reusable UXFD building blocks** that are shared across the 7 UXFD paper submodules.

Scope:
- Put reusable operators/modules here (signal processing 1D/2D, fusion, fuzzy logic, operator attention, TSPN wrappers).
- Do **not** put paper-specific experiment configs here (those live in each `paper/UXFD_paper/<paper_id>/` submodule).

Notes:
- The current maintained runnable TSPN implementation lives in `src/model_factory/X_model/TSPN.py`.
- This package provides an organized landing area; some modules may initially re-export from legacy files for compatibility.


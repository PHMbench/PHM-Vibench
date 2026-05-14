# UXFD SOTA Aggregate Templates

- Template root: `paper/UXFD_paper/results/sota_aggregate_templates`
- Templates: `7`
- Status: templates only; not accepted SOTA evidence.
- Fill one `sota_aggregate.yaml` per paper only after accepted run coverage exists.
- Activation preflight: `python -m scripts.uxfd_artifact_gate paper/UXFD_paper/results/accepted_runs --require-queue-coverage` must pass before creating `paper/UXFD_paper/results/sota_aggregates`.
- Do not commit template-derived `sota_aggregate.yaml` files while `accepted_runs` has zero accepted records or incomplete queue coverage.
- Required statistics: per-seed values, finite mean/std/95% CI, and finite effect size or paired test p-value in [0, 1].
- Required run refs: every proposed, baseline, and TOP entry lists existing relative `run_meta.yaml` paths under accepted_runs.

| Queue | Paper | Minimum Seeds | Baselines | TOP Bindings | Template |
|---|---|---:|---:|---:|---|
| `Q1` | `TII_operator_attention` | 3 | 7 | 1 | `paper/UXFD_paper/results/sota_aggregate_templates/TII_operator_attention/sota_aggregate.template.yaml` |
| `Q2` | `1D-2D_fusion_explainable` | 3 | 6 | 1 | `paper/UXFD_paper/results/sota_aggregate_templates/1D-2D_fusion_explainable/sota_aggregate.template.yaml` |
| `Q3` | `Explainable_FD_Toolkit` | 3 | 6 | 1 | `paper/UXFD_paper/results/sota_aggregate_templates/Explainable_FD_Toolkit/sota_aggregate.template.yaml` |
| `Q4` | `MOE_explainable` | 3 | 6 | 1 | `paper/UXFD_paper/results/sota_aggregate_templates/MOE_explainable/sota_aggregate.template.yaml` |
| `Q5` | `Paper_fuzzy_XFD` | 3 | 7 | 1 | `paper/UXFD_paper/results/sota_aggregate_templates/Paper_fuzzy_XFD/sota_aggregate.template.yaml` |
| `Q6` | `Neuralsymbolic_theory` | 3 | 6 | 1 | `paper/UXFD_paper/results/sota_aggregate_templates/Neuralsymbolic_theory/sota_aggregate.template.yaml` |
| `Q7` | `LLM_Explainable_FD_Toolkit` | 3 | 7 | 1 | `paper/UXFD_paper/results/sota_aggregate_templates/LLM_Explainable_FD_Toolkit/sota_aggregate.template.yaml` |

# UXFD Accepted Run Artifact Templates

- Template root: `paper/UXFD_paper/results/accepted_run_templates`
- Templates: `104`
- Status: templates only; not accepted evidence.
- Accepted metrics rule: `metrics.json` or `metrics.csv` must include at least one numeric metric; status-only payloads are rejected.
- Source-tree rule: accepted runs must set `source_tree_status: clean`.
- Run-control rule: `seed` must be a non-negative integer and `batch_size` must be a positive integer.
- Provenance rule: `git_sha_or_submodule_sha` must be a concrete SHA record without dirty, modified, unknown, or uncommitted markers.

| Queue | Paper | Phase | Entry | GPU | Template |
|---|---|---|---|---:|---|
| `Q1` | `TII_operator_attention` | `proposed` | `P00` | `0` | `paper/UXFD_paper/results/accepted_run_templates/TII_operator_attention/proposed/P00__gpu0/run_meta.template.yaml` |
| `Q1` | `TII_operator_attention` | `baselines` | `B01` | `1` | `paper/UXFD_paper/results/accepted_run_templates/TII_operator_attention/baselines/B01__gpu1/run_meta.template.yaml` |
| `Q1` | `TII_operator_attention` | `baselines` | `B02` | `0` | `paper/UXFD_paper/results/accepted_run_templates/TII_operator_attention/baselines/B02__gpu0/run_meta.template.yaml` |
| `Q1` | `TII_operator_attention` | `baselines` | `B03` | `1` | `paper/UXFD_paper/results/accepted_run_templates/TII_operator_attention/baselines/B03__gpu1/run_meta.template.yaml` |
| `Q1` | `TII_operator_attention` | `baselines` | `B04` | `0` | `paper/UXFD_paper/results/accepted_run_templates/TII_operator_attention/baselines/B04__gpu0/run_meta.template.yaml` |
| `Q1` | `TII_operator_attention` | `baselines` | `B05` | `1` | `paper/UXFD_paper/results/accepted_run_templates/TII_operator_attention/baselines/B05__gpu1/run_meta.template.yaml` |
| `Q1` | `TII_operator_attention` | `baselines` | `B06` | `0` | `paper/UXFD_paper/results/accepted_run_templates/TII_operator_attention/baselines/B06__gpu0/run_meta.template.yaml` |
| `Q1` | `TII_operator_attention` | `baselines` | `B07` | `1` | `paper/UXFD_paper/results/accepted_run_templates/TII_operator_attention/baselines/B07__gpu1/run_meta.template.yaml` |
| `Q1` | `TII_operator_attention` | `ablations` | `A01` | `0` | `paper/UXFD_paper/results/accepted_run_templates/TII_operator_attention/ablations/A01__gpu0/run_meta.template.yaml` |
| `Q1` | `TII_operator_attention` | `ablations` | `A02` | `1` | `paper/UXFD_paper/results/accepted_run_templates/TII_operator_attention/ablations/A02__gpu1/run_meta.template.yaml` |
| `Q1` | `TII_operator_attention` | `ablations` | `A03` | `0` | `paper/UXFD_paper/results/accepted_run_templates/TII_operator_attention/ablations/A03__gpu0/run_meta.template.yaml` |
| `Q1` | `TII_operator_attention` | `ablations` | `A04` | `1` | `paper/UXFD_paper/results/accepted_run_templates/TII_operator_attention/ablations/A04__gpu1/run_meta.template.yaml` |
| `Q1` | `TII_operator_attention` | `ablations` | `A05` | `0` | `paper/UXFD_paper/results/accepted_run_templates/TII_operator_attention/ablations/A05__gpu0/run_meta.template.yaml` |
| `Q1` | `TII_operator_attention` | `ablations` | `A06` | `1` | `paper/UXFD_paper/results/accepted_run_templates/TII_operator_attention/ablations/A06__gpu1/run_meta.template.yaml` |
| `Q2` | `1D-2D_fusion_explainable` | `proposed` | `P00` | `0` | `paper/UXFD_paper/results/accepted_run_templates/1D-2D_fusion_explainable/proposed/P00__gpu0/run_meta.template.yaml` |
| `Q2` | `1D-2D_fusion_explainable` | `baselines` | `B01` | `1` | `paper/UXFD_paper/results/accepted_run_templates/1D-2D_fusion_explainable/baselines/B01__gpu1/run_meta.template.yaml` |
| `Q2` | `1D-2D_fusion_explainable` | `baselines` | `B02` | `0` | `paper/UXFD_paper/results/accepted_run_templates/1D-2D_fusion_explainable/baselines/B02__gpu0/run_meta.template.yaml` |
| `Q2` | `1D-2D_fusion_explainable` | `baselines` | `B03` | `1` | `paper/UXFD_paper/results/accepted_run_templates/1D-2D_fusion_explainable/baselines/B03__gpu1/run_meta.template.yaml` |
| `Q2` | `1D-2D_fusion_explainable` | `baselines` | `B04` | `0` | `paper/UXFD_paper/results/accepted_run_templates/1D-2D_fusion_explainable/baselines/B04__gpu0/run_meta.template.yaml` |
| `Q2` | `1D-2D_fusion_explainable` | `baselines` | `B05` | `1` | `paper/UXFD_paper/results/accepted_run_templates/1D-2D_fusion_explainable/baselines/B05__gpu1/run_meta.template.yaml` |
| `Q2` | `1D-2D_fusion_explainable` | `baselines` | `B06` | `0` | `paper/UXFD_paper/results/accepted_run_templates/1D-2D_fusion_explainable/baselines/B06__gpu0/run_meta.template.yaml` |
| `Q2` | `1D-2D_fusion_explainable` | `ablations` | `A01` | `1` | `paper/UXFD_paper/results/accepted_run_templates/1D-2D_fusion_explainable/ablations/A01__gpu1/run_meta.template.yaml` |
| `Q2` | `1D-2D_fusion_explainable` | `ablations` | `A02` | `0` | `paper/UXFD_paper/results/accepted_run_templates/1D-2D_fusion_explainable/ablations/A02__gpu0/run_meta.template.yaml` |
| `Q2` | `1D-2D_fusion_explainable` | `ablations` | `A03` | `1` | `paper/UXFD_paper/results/accepted_run_templates/1D-2D_fusion_explainable/ablations/A03__gpu1/run_meta.template.yaml` |
| `Q2` | `1D-2D_fusion_explainable` | `ablations` | `A04` | `0` | `paper/UXFD_paper/results/accepted_run_templates/1D-2D_fusion_explainable/ablations/A04__gpu0/run_meta.template.yaml` |
| `Q2` | `1D-2D_fusion_explainable` | `ablations` | `A05` | `1` | `paper/UXFD_paper/results/accepted_run_templates/1D-2D_fusion_explainable/ablations/A05__gpu1/run_meta.template.yaml` |
| `Q2` | `1D-2D_fusion_explainable` | `ablations` | `A06` | `0` | `paper/UXFD_paper/results/accepted_run_templates/1D-2D_fusion_explainable/ablations/A06__gpu0/run_meta.template.yaml` |
| `Q2` | `1D-2D_fusion_explainable` | `ablations` | `A07` | `1` | `paper/UXFD_paper/results/accepted_run_templates/1D-2D_fusion_explainable/ablations/A07__gpu1/run_meta.template.yaml` |
| `Q3` | `Explainable_FD_Toolkit` | `proposed` | `P00` | `0` | `paper/UXFD_paper/results/accepted_run_templates/Explainable_FD_Toolkit/proposed/P00__gpu0/run_meta.template.yaml` |
| `Q3` | `Explainable_FD_Toolkit` | `baselines` | `B01` | `1` | `paper/UXFD_paper/results/accepted_run_templates/Explainable_FD_Toolkit/baselines/B01__gpu1/run_meta.template.yaml` |
| `Q3` | `Explainable_FD_Toolkit` | `baselines` | `B02` | `0` | `paper/UXFD_paper/results/accepted_run_templates/Explainable_FD_Toolkit/baselines/B02__gpu0/run_meta.template.yaml` |
| `Q3` | `Explainable_FD_Toolkit` | `baselines` | `B03` | `1` | `paper/UXFD_paper/results/accepted_run_templates/Explainable_FD_Toolkit/baselines/B03__gpu1/run_meta.template.yaml` |
| `Q3` | `Explainable_FD_Toolkit` | `baselines` | `B04` | `0` | `paper/UXFD_paper/results/accepted_run_templates/Explainable_FD_Toolkit/baselines/B04__gpu0/run_meta.template.yaml` |
| `Q3` | `Explainable_FD_Toolkit` | `baselines` | `B05` | `1` | `paper/UXFD_paper/results/accepted_run_templates/Explainable_FD_Toolkit/baselines/B05__gpu1/run_meta.template.yaml` |
| `Q3` | `Explainable_FD_Toolkit` | `baselines` | `B06` | `0` | `paper/UXFD_paper/results/accepted_run_templates/Explainable_FD_Toolkit/baselines/B06__gpu0/run_meta.template.yaml` |
| `Q3` | `Explainable_FD_Toolkit` | `ablations` | `A01` | `1` | `paper/UXFD_paper/results/accepted_run_templates/Explainable_FD_Toolkit/ablations/A01__gpu1/run_meta.template.yaml` |
| `Q3` | `Explainable_FD_Toolkit` | `ablations` | `A02` | `0` | `paper/UXFD_paper/results/accepted_run_templates/Explainable_FD_Toolkit/ablations/A02__gpu0/run_meta.template.yaml` |
| `Q3` | `Explainable_FD_Toolkit` | `ablations` | `A03` | `1` | `paper/UXFD_paper/results/accepted_run_templates/Explainable_FD_Toolkit/ablations/A03__gpu1/run_meta.template.yaml` |
| `Q3` | `Explainable_FD_Toolkit` | `ablations` | `A04` | `0` | `paper/UXFD_paper/results/accepted_run_templates/Explainable_FD_Toolkit/ablations/A04__gpu0/run_meta.template.yaml` |
| `Q3` | `Explainable_FD_Toolkit` | `ablations` | `A05` | `1` | `paper/UXFD_paper/results/accepted_run_templates/Explainable_FD_Toolkit/ablations/A05__gpu1/run_meta.template.yaml` |
| `Q3` | `Explainable_FD_Toolkit` | `ablations` | `A06` | `0` | `paper/UXFD_paper/results/accepted_run_templates/Explainable_FD_Toolkit/ablations/A06__gpu0/run_meta.template.yaml` |
| `Q4` | `MOE_explainable` | `proposed` | `P00` | `1` | `paper/UXFD_paper/results/accepted_run_templates/MOE_explainable/proposed/P00__gpu1/run_meta.template.yaml` |
| `Q4` | `MOE_explainable` | `baselines` | `B01` | `0` | `paper/UXFD_paper/results/accepted_run_templates/MOE_explainable/baselines/B01__gpu0/run_meta.template.yaml` |
| `Q4` | `MOE_explainable` | `baselines` | `B02` | `1` | `paper/UXFD_paper/results/accepted_run_templates/MOE_explainable/baselines/B02__gpu1/run_meta.template.yaml` |
| `Q4` | `MOE_explainable` | `baselines` | `B03` | `0` | `paper/UXFD_paper/results/accepted_run_templates/MOE_explainable/baselines/B03__gpu0/run_meta.template.yaml` |
| `Q4` | `MOE_explainable` | `baselines` | `B04` | `1` | `paper/UXFD_paper/results/accepted_run_templates/MOE_explainable/baselines/B04__gpu1/run_meta.template.yaml` |
| `Q4` | `MOE_explainable` | `baselines` | `B05` | `0` | `paper/UXFD_paper/results/accepted_run_templates/MOE_explainable/baselines/B05__gpu0/run_meta.template.yaml` |
| `Q4` | `MOE_explainable` | `baselines` | `B06` | `1` | `paper/UXFD_paper/results/accepted_run_templates/MOE_explainable/baselines/B06__gpu1/run_meta.template.yaml` |
| `Q4` | `MOE_explainable` | `ablations` | `A01` | `0` | `paper/UXFD_paper/results/accepted_run_templates/MOE_explainable/ablations/A01__gpu0/run_meta.template.yaml` |
| `Q4` | `MOE_explainable` | `ablations` | `A02` | `1` | `paper/UXFD_paper/results/accepted_run_templates/MOE_explainable/ablations/A02__gpu1/run_meta.template.yaml` |
| `Q4` | `MOE_explainable` | `ablations` | `A03` | `0` | `paper/UXFD_paper/results/accepted_run_templates/MOE_explainable/ablations/A03__gpu0/run_meta.template.yaml` |
| `Q4` | `MOE_explainable` | `ablations` | `A04` | `1` | `paper/UXFD_paper/results/accepted_run_templates/MOE_explainable/ablations/A04__gpu1/run_meta.template.yaml` |
| `Q4` | `MOE_explainable` | `ablations` | `A05` | `0` | `paper/UXFD_paper/results/accepted_run_templates/MOE_explainable/ablations/A05__gpu0/run_meta.template.yaml` |
| `Q4` | `MOE_explainable` | `ablations` | `A06` | `1` | `paper/UXFD_paper/results/accepted_run_templates/MOE_explainable/ablations/A06__gpu1/run_meta.template.yaml` |
| `Q5` | `Paper_fuzzy_XFD` | `proposed` | `P00` | `0` | `paper/UXFD_paper/results/accepted_run_templates/Paper_fuzzy_XFD/proposed/P00__gpu0/run_meta.template.yaml` |
| `Q5` | `Paper_fuzzy_XFD` | `baselines` | `B01` | `1` | `paper/UXFD_paper/results/accepted_run_templates/Paper_fuzzy_XFD/baselines/B01__gpu1/run_meta.template.yaml` |
| `Q5` | `Paper_fuzzy_XFD` | `baselines` | `B02` | `0` | `paper/UXFD_paper/results/accepted_run_templates/Paper_fuzzy_XFD/baselines/B02__gpu0/run_meta.template.yaml` |
| `Q5` | `Paper_fuzzy_XFD` | `baselines` | `B03` | `1` | `paper/UXFD_paper/results/accepted_run_templates/Paper_fuzzy_XFD/baselines/B03__gpu1/run_meta.template.yaml` |
| `Q5` | `Paper_fuzzy_XFD` | `baselines` | `B04` | `0` | `paper/UXFD_paper/results/accepted_run_templates/Paper_fuzzy_XFD/baselines/B04__gpu0/run_meta.template.yaml` |
| `Q5` | `Paper_fuzzy_XFD` | `baselines` | `B05` | `1` | `paper/UXFD_paper/results/accepted_run_templates/Paper_fuzzy_XFD/baselines/B05__gpu1/run_meta.template.yaml` |
| `Q5` | `Paper_fuzzy_XFD` | `baselines` | `B06` | `0` | `paper/UXFD_paper/results/accepted_run_templates/Paper_fuzzy_XFD/baselines/B06__gpu0/run_meta.template.yaml` |
| `Q5` | `Paper_fuzzy_XFD` | `baselines` | `B07` | `1` | `paper/UXFD_paper/results/accepted_run_templates/Paper_fuzzy_XFD/baselines/B07__gpu1/run_meta.template.yaml` |
| `Q5` | `Paper_fuzzy_XFD` | `ablations` | `A01` | `0` | `paper/UXFD_paper/results/accepted_run_templates/Paper_fuzzy_XFD/ablations/A01__gpu0/run_meta.template.yaml` |
| `Q5` | `Paper_fuzzy_XFD` | `ablations` | `A02` | `1` | `paper/UXFD_paper/results/accepted_run_templates/Paper_fuzzy_XFD/ablations/A02__gpu1/run_meta.template.yaml` |
| `Q5` | `Paper_fuzzy_XFD` | `ablations` | `A03` | `0` | `paper/UXFD_paper/results/accepted_run_templates/Paper_fuzzy_XFD/ablations/A03__gpu0/run_meta.template.yaml` |
| `Q5` | `Paper_fuzzy_XFD` | `ablations` | `A04` | `1` | `paper/UXFD_paper/results/accepted_run_templates/Paper_fuzzy_XFD/ablations/A04__gpu1/run_meta.template.yaml` |
| `Q5` | `Paper_fuzzy_XFD` | `ablations` | `A05` | `0` | `paper/UXFD_paper/results/accepted_run_templates/Paper_fuzzy_XFD/ablations/A05__gpu0/run_meta.template.yaml` |
| `Q5` | `Paper_fuzzy_XFD` | `ablations` | `A06` | `1` | `paper/UXFD_paper/results/accepted_run_templates/Paper_fuzzy_XFD/ablations/A06__gpu1/run_meta.template.yaml` |
| `Q6` | `Neuralsymbolic_theory` | `proposed` | `P00` | `0` | `paper/UXFD_paper/results/accepted_run_templates/Neuralsymbolic_theory/proposed/P00__gpu0/run_meta.template.yaml` |
| `Q6` | `Neuralsymbolic_theory` | `baselines` | `B01` | `1` | `paper/UXFD_paper/results/accepted_run_templates/Neuralsymbolic_theory/baselines/B01__gpu1/run_meta.template.yaml` |
| `Q6` | `Neuralsymbolic_theory` | `baselines` | `B02` | `0` | `paper/UXFD_paper/results/accepted_run_templates/Neuralsymbolic_theory/baselines/B02__gpu0/run_meta.template.yaml` |
| `Q6` | `Neuralsymbolic_theory` | `baselines` | `B03` | `1` | `paper/UXFD_paper/results/accepted_run_templates/Neuralsymbolic_theory/baselines/B03__gpu1/run_meta.template.yaml` |
| `Q6` | `Neuralsymbolic_theory` | `baselines` | `B04` | `0` | `paper/UXFD_paper/results/accepted_run_templates/Neuralsymbolic_theory/baselines/B04__gpu0/run_meta.template.yaml` |
| `Q6` | `Neuralsymbolic_theory` | `baselines` | `B05` | `1` | `paper/UXFD_paper/results/accepted_run_templates/Neuralsymbolic_theory/baselines/B05__gpu1/run_meta.template.yaml` |
| `Q6` | `Neuralsymbolic_theory` | `baselines` | `B06` | `0` | `paper/UXFD_paper/results/accepted_run_templates/Neuralsymbolic_theory/baselines/B06__gpu0/run_meta.template.yaml` |
| `Q6` | `Neuralsymbolic_theory` | `ablations` | `A01` | `1` | `paper/UXFD_paper/results/accepted_run_templates/Neuralsymbolic_theory/ablations/A01__gpu1/run_meta.template.yaml` |
| `Q6` | `Neuralsymbolic_theory` | `ablations` | `A02` | `0` | `paper/UXFD_paper/results/accepted_run_templates/Neuralsymbolic_theory/ablations/A02__gpu0/run_meta.template.yaml` |
| `Q6` | `Neuralsymbolic_theory` | `ablations` | `A03` | `1` | `paper/UXFD_paper/results/accepted_run_templates/Neuralsymbolic_theory/ablations/A03__gpu1/run_meta.template.yaml` |
| `Q6` | `Neuralsymbolic_theory` | `ablations` | `A04` | `0` | `paper/UXFD_paper/results/accepted_run_templates/Neuralsymbolic_theory/ablations/A04__gpu0/run_meta.template.yaml` |
| `Q6` | `Neuralsymbolic_theory` | `ablations` | `A05` | `1` | `paper/UXFD_paper/results/accepted_run_templates/Neuralsymbolic_theory/ablations/A05__gpu1/run_meta.template.yaml` |
| `Q6` | `Neuralsymbolic_theory` | `ablations` | `A06` | `0` | `paper/UXFD_paper/results/accepted_run_templates/Neuralsymbolic_theory/ablations/A06__gpu0/run_meta.template.yaml` |
| `Q6` | `Neuralsymbolic_theory` | `ablations` | `A07` | `1` | `paper/UXFD_paper/results/accepted_run_templates/Neuralsymbolic_theory/ablations/A07__gpu1/run_meta.template.yaml` |
| `Q7` | `LLM_Explainable_FD_Toolkit` | `proposed` | `P00` | `0` | `paper/UXFD_paper/results/accepted_run_templates/LLM_Explainable_FD_Toolkit/proposed/P00__gpu0/run_meta.template.yaml` |
| `Q7` | `LLM_Explainable_FD_Toolkit` | `baselines` | `B01` | `1` | `paper/UXFD_paper/results/accepted_run_templates/LLM_Explainable_FD_Toolkit/baselines/B01__gpu1/run_meta.template.yaml` |
| `Q7` | `LLM_Explainable_FD_Toolkit` | `baselines` | `B02` | `0` | `paper/UXFD_paper/results/accepted_run_templates/LLM_Explainable_FD_Toolkit/baselines/B02__gpu0/run_meta.template.yaml` |
| `Q7` | `LLM_Explainable_FD_Toolkit` | `baselines` | `B03` | `1` | `paper/UXFD_paper/results/accepted_run_templates/LLM_Explainable_FD_Toolkit/baselines/B03__gpu1/run_meta.template.yaml` |
| `Q7` | `LLM_Explainable_FD_Toolkit` | `baselines` | `B04` | `0` | `paper/UXFD_paper/results/accepted_run_templates/LLM_Explainable_FD_Toolkit/baselines/B04__gpu0/run_meta.template.yaml` |
| `Q7` | `LLM_Explainable_FD_Toolkit` | `baselines` | `B05` | `1` | `paper/UXFD_paper/results/accepted_run_templates/LLM_Explainable_FD_Toolkit/baselines/B05__gpu1/run_meta.template.yaml` |
| `Q7` | `LLM_Explainable_FD_Toolkit` | `baselines` | `B06` | `0` | `paper/UXFD_paper/results/accepted_run_templates/LLM_Explainable_FD_Toolkit/baselines/B06__gpu0/run_meta.template.yaml` |
| `Q7` | `LLM_Explainable_FD_Toolkit` | `baselines` | `B07` | `1` | `paper/UXFD_paper/results/accepted_run_templates/LLM_Explainable_FD_Toolkit/baselines/B07__gpu1/run_meta.template.yaml` |
| `Q7` | `LLM_Explainable_FD_Toolkit` | `ablations` | `A01` | `0` | `paper/UXFD_paper/results/accepted_run_templates/LLM_Explainable_FD_Toolkit/ablations/A01__gpu0/run_meta.template.yaml` |
| `Q7` | `LLM_Explainable_FD_Toolkit` | `ablations` | `A02` | `1` | `paper/UXFD_paper/results/accepted_run_templates/LLM_Explainable_FD_Toolkit/ablations/A02__gpu1/run_meta.template.yaml` |
| `Q7` | `LLM_Explainable_FD_Toolkit` | `ablations` | `A03` | `0` | `paper/UXFD_paper/results/accepted_run_templates/LLM_Explainable_FD_Toolkit/ablations/A03__gpu0/run_meta.template.yaml` |
| `Q7` | `LLM_Explainable_FD_Toolkit` | `ablations` | `A04` | `1` | `paper/UXFD_paper/results/accepted_run_templates/LLM_Explainable_FD_Toolkit/ablations/A04__gpu1/run_meta.template.yaml` |
| `Q7` | `LLM_Explainable_FD_Toolkit` | `ablations` | `A05` | `0` | `paper/UXFD_paper/results/accepted_run_templates/LLM_Explainable_FD_Toolkit/ablations/A05__gpu0/run_meta.template.yaml` |
| `Q7` | `LLM_Explainable_FD_Toolkit` | `ablations` | `A06` | `1` | `paper/UXFD_paper/results/accepted_run_templates/LLM_Explainable_FD_Toolkit/ablations/A06__gpu1/run_meta.template.yaml` |
| `Q7` | `LLM_Explainable_FD_Toolkit` | `ablations` | `A07` | `0` | `paper/UXFD_paper/results/accepted_run_templates/LLM_Explainable_FD_Toolkit/ablations/A07__gpu0/run_meta.template.yaml` |
| `TOP-Q1-GTM` | `TII_operator_attention` | `top_representatives` | `B04,B05,A04` | `0,1` | `paper/UXFD_paper/results/accepted_run_templates/TII_operator_attention/top_representatives/B04_B05_A04__gpu0_1/run_meta.template.yaml` |
| `TOP-Q2-GTM` | `1D-2D_fusion_explainable` | `top_representatives` | `B04,B05,A06` | `0,1` | `paper/UXFD_paper/results/accepted_run_templates/1D-2D_fusion_explainable/top_representatives/B04_B05_A06__gpu0_1/run_meta.template.yaml` |
| `TOP-Q3-TIMESEG` | `Explainable_FD_Toolkit` | `top_representatives` | `P00,A02,A03,A06` | `0,1` | `paper/UXFD_paper/results/accepted_run_templates/Explainable_FD_Toolkit/top_representatives/P00_A02_A03_A06__gpu0_1/run_meta.template.yaml` |
| `TOP-Q4-TSPULSE` | `MOE_explainable` | `top_representatives` | `B06,A04,A06` | `0,1` | `paper/UXFD_paper/results/accepted_run_templates/MOE_explainable/top_representatives/B06_A04_A06__gpu0_1/run_meta.template.yaml` |
| `TOP-Q5-TIMESLIVER` | `Paper_fuzzy_XFD` | `top_representatives` | `B07,A01,A04,A05,A06` | `0,1` | `paper/UXFD_paper/results/accepted_run_templates/Paper_fuzzy_XFD/top_representatives/B07_A01_A04_A05_A06__gpu0_1/run_meta.template.yaml` |
| `TOP-Q6-TIMESLIVER` | `Neuralsymbolic_theory` | `top_representatives` | `A01,A05,A06,A07` | `0,1` | `paper/UXFD_paper/results/accepted_run_templates/Neuralsymbolic_theory/top_representatives/A01_A05_A06_A07__gpu0_1/run_meta.template.yaml` |
| `TOP-Q7-TIMESEG` | `LLM_Explainable_FD_Toolkit` | `top_representatives` | `B02,A05,A07` | `0,1` | `paper/UXFD_paper/results/accepted_run_templates/LLM_Explainable_FD_Toolkit/top_representatives/B02_A05_A07__gpu0_1/run_meta.template.yaml` |

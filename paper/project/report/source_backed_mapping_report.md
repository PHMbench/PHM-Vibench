# Paper 06 Source-Backed Cross-Method Mapping

- Source-backed: `true`
- Accepted evidence: `false`
- Papers checked: `6`

| Paper | Mapping role | Source-backed | Matched required terms | Evidence paths |
|---|---|---:|---|---|
| 1D-2D_fusion_explainable | signal-to-representation fusion and alignment evidence | `true` | alignment, fusion, signal_processing_2d | ../1D-2D_fusion_explainable/VIBENCH.md<br>../1D-2D_fusion_explainable/submission_prep/baseline_ablation_matrix.yaml<br>../1D-2D_fusion_explainable/configs/vibench/min.yaml<br>../1D-2D_fusion_explainable/code/models/fusion_aligned.py<br>../1D-2D_fusion_explainable/code/alignment/physical_alignment.py |
| MOE_explainable | expert-routing and physics-constrained mixture evidence | `true` | expert, load_balance, router, sparsity | ../MOE_explainable/VIBENCH.md<br>../MOE_explainable/submission_prep/baseline_ablation_matrix.yaml<br>../MOE_explainable/configs/vibench/min.yaml<br>../MOE_explainable/code/moe_model.py<br>../MOE_explainable/code/router/statistical_router.py |
| Paper_fuzzy_XFD | fuzzy rule, membership, and decision-path evidence | `true` | fuzzy, membership, rule | ../Paper_fuzzy_XFD/VIBENCH.md<br>../Paper_fuzzy_XFD/submission_prep/baseline_ablation_matrix.yaml<br>../Paper_fuzzy_XFD/configs/vibench/min.yaml<br>../Paper_fuzzy_XFD/code/fuzzy_system/rule_base.py<br>../Paper_fuzzy_XFD/code/fuzzy_system/membership_functions.py |
| Explainable_FD_Toolkit | explanation schema, metric, manifest, and toolkit evidence | `true` | explain, manifest, schema | ../Explainable_FD_Toolkit/VIBENCH.md<br>../Explainable_FD_Toolkit/submission_prep/baseline_ablation_matrix.yaml<br>../Explainable_FD_Toolkit/configs/vibench/min.yaml<br>../Explainable_FD_Toolkit/scripts/run_toolkit_ablations.py<br>../Explainable_FD_Toolkit/scripts/validate_schema.py |
| LLM_Explainable_FD_Toolkit | LLM evidence-chain and unsupported-claim control evidence | `true` | evidence, llm, unsupported | ../LLM_Explainable_FD_Toolkit/VIBENCH.md<br>../LLM_Explainable_FD_Toolkit/submission_prep/baseline_ablation_matrix.yaml<br>../LLM_Explainable_FD_Toolkit/configs/vibench/min.yaml<br>../LLM_Explainable_FD_Toolkit/code/llm_explainable_toolkit/core/intermediate_representation.py<br>../LLM_Explainable_FD_Toolkit/experiments/scripts/run_llm_evidence_smoke.py |
| TII_operator_attention | operator-attention and signal-operator evidence | `true` | FFT, attention, operator | ../TII_operator_attention/VIBENCH.md<br>../TII_operator_attention/submission_prep/baseline_ablation_matrix.yaml<br>../TII_operator_attention/configs/vibench/min.yaml<br>../TII_operator_attention/code/synthetic_signals/operator_validation.py<br>../TII_operator_attention/code/synthetic_verification.py |

## Layer Support

### 1D-2D_fusion_explainable
- `signal_layer`: 1D, 2D, frequency, spectral
- `neural_layer`: features_1d, features_2d, fusion_layers
- `constraint_layer`: alignment_loss, geometric, physical, semantic
- `evidence_layer`: VIBENCH, accepted_evidence_status, baseline_ablation_matrix

### MOE_explainable
- `signal_layer`: envelope, frequency, harmonic, low_pass
- `neural_layer`: expert_outputs, moe, routing_weights
- `constraint_layer`: diversity, load_balance, orthogonal, sparsity
- `evidence_layer`: baseline_ablation_matrix, expert activation, route

### Paper_fuzzy_XFD
- `signal_layer`: diagnosis, fault, feature
- `neural_layer`: NSN, TSPN_UXFD, decision_configs
- `constraint_layer`: membership, predicate, rule
- `evidence_layer`: active_rules, baseline_ablation_matrix, rule-level

### Explainable_FD_Toolkit
- `signal_layer`: dataset, fault
- `neural_layer`: NSN, baseline, model
- `constraint_layer`: metric, schema, snapshot
- `evidence_layer`: accepted_evidence, artifact, manifest

### LLM_Explainable_FD_Toolkit
- `signal_layer`: diagnosis, fault, time
- `neural_layer`: intermediate, model
- `constraint_layer`: checker, hallucination, unsupported
- `evidence_layer`: evidence, metrics, run_meta

### TII_operator_attention
- `signal_layer`: FFT, Hilbert, operator, signal
- `neural_layer`: NSN, attention, operator_attention
- `constraint_layer`: identity, subset, temperature
- `evidence_layer`: accepted_evidence, baseline_ablation_matrix, validation

## Limitations

- This report is source-introspection evidence only.
- It does not prove model performance, mapping impact, TOP-method reproduction, GPU feasibility, or SOTA.
- Accepted train/eval evidence still requires same-protocol logs, metrics, run_meta.yaml, and local GPU metadata.

# Paper 07 Reviewer Traceability Matrix

Status date: 2026-05-14

This matrix is a rejection-recovery control document. It is not accepted experiment evidence
and it does not make the paper submission-ready.

## Purpose

Paper 07 was treated as the highest-risk rejection-recovery paper because prior
review risks included weak industrial performance, unclear innovation,
insufficient recent baselines, shallow ablations, and weak theory-experiment
linkage. This matrix maps each concern to the exact evidence gate that must be
closed before the revised manuscript may make a stronger claim.

## Traceability Matrix

| Reviewer concern | Current safe response | Required accepted evidence | Manuscript action | Current status |
|---|---|---|---|---|
| Weak industrial performance | Do not claim SOTA or industrial superiority from dummy, synthetic, or CPU-fallback runs. | Multi-seed P00 vs B01-B07 industrial table with identical datasets, splits, seeds, metrics, runtime, GPU model, confidence intervals, and accepted run refs. | Replace all final result tables only after parent artifact and SOTA gates pass. | blocked |
| Theory-experiment mismatch | Treat synthetic operator-selection results as mechanism sanity checks only. | P00 run package linking selected operators, attention temperature, sparsity weight, OAS, OSS, OCS, sample IDs, and failure cases to accepted industrial predictions. | Add a claim-to-evidence paragraph after the method and before final experiments. | blocked |
| Unclear innovation | Define DSOA v2 as operator-space attention, not generic feature attention. | Matched-budget comparison against feature/self-attention and transformer attention, plus operator-subset ablations A01-A06. | Add a novelty table contrasting operator dictionary, sparse selection, physics consistency, and explanation metrics. | blocked |
| Insufficient recent/SOTA baselines | Keep TOP methods representative-only unless exact external code/config is integrated. | Accepted same-protocol representative or exact evidence for the 2024-2026 TOP quota, including GTM proxy rows B04, B05, and A04. | Add recent-work scope labels: exact-run, representative-run, resource-blocked, or literature-only. | blocked |
| Shallow ablations | Use command-bound A01-A06 only as pending run plans until accepted artifacts exist. | Same-seed A01-A06 industrial ablation table with GPU metadata and matched metrics against P00. | Move ablation claims from "proved" to "planned evidence" until accepted artifacts exist. | blocked |
| Explainability not convincing | Keep attention visualizations qualitative unless linked to accepted metrics. | OAS, OSS, OCS, faithfulness or stability metrics, sample-level explanations, and failure-mode records tied to P00 artifacts. | Add one positive case, one failure case, and one operator-metric table only after accepted evidence exists. | blocked |
| Resource feasibility | Do not rely on nonlocal GPUs or unstated compute. | Parent Q0 preflight showing local RTX 4090 devices 0 and 1, plus per-run CUDA device, precision, batch size, and runtime metadata. | State the two-GPU execution budget and mark oversized exact TOP reproductions as resource-blocked. | blocked |

## Innovation Claim Boundaries

The revised manuscript may describe the intended innovation as:

- Dynamic Sparse Operator Attention v2 for operator-space selection.
- Physics-consistency regularization over signal-processing operators.
- Operator-level explanation metrics OAS, OSS, and OCS.
- Reviewer-facing traceability from each claim to accepted artifacts.

The manuscript must not claim:

- SOTA diagnosis accuracy before accepted industrial aggregate evidence exists.
- Exact reproduction of TOP methods when only local proxies were run.
- Theory validation from synthetic results alone.
- Submission readiness while the parent objective audit is not achieved.

## Evidence Closure Rules

Each row can move from `blocked` to `evidence-ready` only when all of the
following are true:

1. The accepted artifacts live under
   `paper/UXFD_paper/results/accepted_runs/TII_operator_attention/`.
2. The parent artifact gate accepts the run metadata and queue coverage.
3. The parent SOTA gate accepts matched-seed aggregate evidence where relevant.
4. The row cites concrete `run_meta.yaml` paths, metrics, logs, configs, and
   Paper07 submodule SHA.
5. The row does not rely on dummy, synthetic-only, template, CPU-fallback, or
   pending evidence.

## Next Update Point

Update this matrix only after Q0 GPU preflight and Paper07 accepted artifacts
exist. Until then, it is a strict reviewer planning aid, not evidence.

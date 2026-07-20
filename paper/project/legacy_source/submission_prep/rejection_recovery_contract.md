# Paper 07 Rejection-Recovery Evidence Contract

Status date: 2026-05-11

This contract converts the prior rejection-risk analysis into verifiable
evidence gates. It is not accepted experiment evidence and it does not make the
paper submission-ready.

## Claim Policy

The revised Paper 07 package must not use SOTA, submission-ready, or industrial
superiority wording until all gates below pass with accepted artifacts.

- Dummy-data PHM-Vibench runs may support command wiring only.
- Synthetic operator-selection runs may support mechanism sanity checks only.
- TeX compilation may support package normalization only.
- SOTA wording requires same-protocol industrial evidence where P00 beats every
  declared baseline, ablation comparator, and runnable TOP representative.
- Exact TOP-reproduction wording is blocked unless the external method is run
  from its own accepted code/config under the same protocol. Local proxies must
  be labeled representative only.

## Inputs That Must Stay Aligned

| Input | Required role |
|---|---|
| `submission_prep/baseline_ablation_matrix.yaml` | Machine-readable P00, B01-B07, A01-A06, and TOP status source. |
| `submission_prep/baseline_ablation_matrix.md` | Human-readable command-binding and next-evidence summary. |
| `submission_prep/ieee_trans_readiness.md` | Strict reviewer gate and current status note. |
| `submission_prep/reviewer_traceability_matrix.md` | Prior concern to evidence-gate map for rejection recovery. |
| `revision/review_response_plan.md` | Prior rejection-risk analysis and reviewer-response strategy. |
| Parent `paper/UXFD_paper/goal/07_tii_operator_attention.md` | Paper-level target, TOP quota, compute budget, and SOTA rules. |
| Parent `paper/UXFD_paper/goal/09_gpu_execution_queue.yaml` | 2x4090 scheduler, accepted metadata contract, and Q1 execution order. |

If any of these sources disagree, the most conservative status wins and the
paper remains not submission-ready.

## Accepted Artifact Package

Accepted evidence for this paper must be stored under the parent-repo artifact
root:

`paper/UXFD_paper/results/accepted_runs/TII_operator_attention/`

Each accepted run directory must contain:

- `run_meta.yaml`
- `metrics.json` or `metrics.csv`
- stdout/stderr or trainer log
- config snapshot or config path
- git SHA and Paper07 submodule SHA
- dataset split and preprocessing signature
- `CUDA_VISIBLE_DEVICES`, GPU count, and GPU model
- seed, batch size, precision, runtime, and command
- OOM or failure reason when a run is blocked or failed

The parent artifact gate must reject placeholders, TODO metadata, CPU fallback
runs, missing logs, missing metrics, missing local RTX 4090 metadata, and
partial queue coverage.

## Minimum Rejection-Recovery Gates

| Prior concern | Required accepted evidence | Blocks |
|---|---|---|
| Weak industrial performance | Multi-seed P00 vs B01-B07 table with mean, std, confidence interval, split, runtime, and GPU metadata. | SOTA claim, industrial value claim, final results table. |
| Theory-experiment mismatch | Operator-selection evidence linking selected operators, attention temperature, sparsity weight, OAS, OSS, OCS, and sample-level failure cases to P00 runs. | Theory contribution wording and explanation claim. |
| Missing recent/SOTA comparisons | TOP representative map covering TimeMixer, SARAD, CATCH, DADA, PGRFNet, GTM, CSLSTM, and TSPulse as exact-run, representative-run, or resource-blocked with logs. | Related-work completeness and SOTA comparison. |
| Shallow ablations | A01-A06 same-protocol ablation table with the same seeds, split, metrics, and GPU metadata as P00. | Mechanism necessity claim. |
| Unclear innovation | Evidence that operator-space attention differs from standard feature/self-attention and transformer attention under matched backbone or matched budget. | Innovation and novelty claim. |

## Q1 Execution Order

1. Q0 preflight: `nvidia-smi -L` must show local RTX 4090 devices `0` and `1`,
   and PyTorch must report CUDA available with device count `2`.
2. Promote the Paper07 accepted-run templates for P00, B01-B07, A01-A06, and
   the TOP representative rows into concrete run directories.
3. Run all Paper07 rows with `CUDA_VISIBLE_DEVICES=0` or
   `CUDA_VISIBLE_DEVICES=1`; at most two single-GPU jobs may run concurrently.
4. Run the parent artifact gate against
   `paper/UXFD_paper/results/accepted_runs`.
5. Update `submission_prep/baseline_ablation_matrix.yaml` only from accepted
   artifact summaries, not from dummy or synthetic logs.
6. Update manuscript tables and reviewer-response text only after the evidence
   tables exist.
7. Run the recent-work gate, submission gate, objective audit, and TeX compile
   gate before any submission-ready claim.

## Required Tables Before Manuscript Update

| Table | Required rows | Required fields |
|---|---|---|
| Baseline comparison | P00 and B01-B07 | dataset, split, seed count, metric mean/std/CI, runtime, GPU model, artifact path. |
| Ablation comparison | A01-A06 | same protocol fields as the baseline table plus ablated component. |
| TOP representative map | 2024-2026 TOP quota rows | exact/representative/resource-blocked status, command or blocker, log path, artifact path. |
| Operator explanation | P00 plus operator ablations | OAS, OSS, OCS, operator subset, temperature, sparsity weight, sample IDs, failure modes. |
| Reviewer traceability | Prior rejection concerns | safe response, required accepted evidence, manuscript action, current status. |

## Stop Rules

- Stop before experiments if Q0 GPU preflight fails.
- Stop SOTA wording if any declared baseline lacks same-protocol evidence.
- Stop exact TOP wording if a TOP method is represented only by a local proxy.
- Stop manuscript table updates if a row is backed only by dummy, synthetic, or
  CPU-fallback evidence.
- Stop submission-ready status if the parent objective audit reports
  `achieved=false`.

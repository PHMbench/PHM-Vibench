# Paper 03 LLM Evidence Package Contract

Status date: 2026-05-14

This contract defines the minimum accepted evidence package for the LLM
Explainable FD Toolkit paper. It is not accepted experiment evidence and it does
not make the paper submission-ready.

## Package Root

Accepted LLM evidence must be promoted under the parent artifact root:

`paper/UXFD_paper/results/accepted_runs/LLM_Explainable_FD_Toolkit/`

Paper-local smoke outputs under `/tmp`, `pipeline_test_results/`, `sessions/`,
or `results/llm_evidence/` are not accepted evidence unless they are promoted
through the parent artifact gate with complete metadata.

## Required Files Per Condition

Each accepted condition directory must contain:

- `run_meta.yaml`
- `metrics.json`
- `prompt_set.json`
- `responses.jsonl`
- `unsupported_claims.json`
- `latency.json`
- `config_snapshot.yaml` or a resolved config path
- stdout/stderr or execution log

The `run_meta.yaml` file must include:

- `paper_id: LLM_Explainable_FD_Toolkit`
- `queue_id` and matrix row id
- `evidence_level: accepted_same_protocol`
- `accepted_evidence: true`
- local RTX 4090 GPU id and model
- seed, prompt-set id, dataset split, batch size or prompt batch size
- precision or quantization mode
- runtime in positive `HH:MM:SS`
- clean parent SHA and clean Paper03 submodule SHA
- preprocessing signature with `sha256:<64 lowercase hex>`
- OOM or failure reason when a condition cannot run

## Required Metrics

`metrics.json` must contain numeric values for:

- diagnostic accuracy or task success rate
- unsupported-claim rate
- hallucination-check pass rate
- evidence-grounding score
- latency p50 and p95
- failure rate
- explanation length or token count

Status-only metrics are rejected. Smoke metrics with
`accepted_evidence=false`, dummy-only data, or CPU-fallback metadata cannot
support manuscript claims.

## Accepted Conditions

The accepted main protocol must cover the same prompt set and seed policy for:

| ID | Condition | Required comparison role |
|---|---|---|
| P00 | full LLM explanation toolkit | proposed method |
| B01 | structured report without LLM dialogue | no-LLM baseline |
| B02 | template-only explanation | template baseline |
| A05 | hallucination checker disabled | anti-hallucination ablation |
| A06 | domain context removed | retrieval/context ablation |
| A07 | short/medium/long explanation sweep | latency and length ablation |
| TOP-Q7-TIMESEG | TimeSeg representative proxy | 2026 TOP representative binding |

## Claim Rules

- No anti-hallucination claim is allowed without unsupported-claim and
  hallucination-check metrics.
- No latency claim is allowed without p50, p95, and failure-rate metrics under
  the same prompt set.
- No human-centered decision-support claim is allowed from template demos alone.
- No exact TOP reproduction claim is allowed when the TOP method is represented
  only by a local proxy.
- No SOTA claim is allowed until the parent SOTA gate accepts matched-seed
  aggregate evidence.

## Next Update Point

Update this contract only after Q0 GPU preflight passes and accepted LLM
evidence packages exist under the parent artifact root.

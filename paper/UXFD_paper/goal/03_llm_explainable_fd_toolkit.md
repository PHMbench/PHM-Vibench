# Paper Goal: LLM Explainable FD Toolkit

## Target

- Submodule: `paper/UXFD_paper/LLM_Explainable_FD_Toolkit`
- Default journal: IEEE Transactions on Industrial Informatics
- Alternate journal: IEEE Transactions on Human-Machine Systems
- Contribution: evidence-chain structured-to-text explanations and dialogue support for fault diagnosis decisions.

## Canonical Package

- Manuscript entrypoint: blocked; no final manuscript entrypoint discovered in the prior Slice 4 audit.
- Reproduction contract: `VIBENCH.md` and `configs/vibench/min.yaml`
- Current compile blocker: missing canonical final TeX package.

## Required Evidence

- Structured explanation input contract from `Explainable_FD_Toolkit`.
- Anti-hallucination evidence: generated text must cite structured evidence fields.
- Task study or proxy evaluation for decision time, decision correctness, quality score, and failure rate.
- End-to-end demo latency distribution including P95.

## Baseline Suite

- Template-only structured report without LLM generation.
- Generic LLM prompt without evidence-field grounding.
- Retrieval-augmented LLM prompt using the same knowledge base.
- `Explainable_FD_Toolkit` structured output without dialogue layer.
- Rule-based natural-language explanation from fuzzy rules.
- SHAP/LIME text summary generated from post-hoc feature importance.
- Human-written report subset as an upper-reference comparator when available.

## Ablation Suite

- Remove evidence-field grounding.
- Remove retrieval/domain knowledge context.
- Remove dialogue state tracking.
- Remove hallucination checker.
- Compare one-shot explanation versus multi-turn diagnostic dialogue.
- Compare latency and failure rate across short, medium, and long explanation templates.

## SOTA Optimization Gate

- The LLM paper may claim SOTA decision-support performance only if it beats all declared baselines on task accuracy, time-to-decision, evidence consistency, hallucination rate, and latency under the same task protocol.
- If no human/user study is available, the paper must label the result as proxy evaluation and cannot claim human-centered SOTA.

## TOP Recent-Work Quota

- RWTOP2024-TIMELLM: `representative-runnable` LLM/time-series adaptation baseline for evidence-chain generation.
- RWTOP2024-MOMENT: `representative-runnable` foundation-model representation baseline for structured evidence inputs.
- RWTOP2025-TIMEMOE: `representative-runnable` sparse foundation-model comparator for scalable time-series reasoning.
- RWTOP2025-CBAE: `literature-only` concept-bottleneck generation comparator until FD concept supervision exists.
- RWTOP2026-TIMESEG: `representative-runnable` segment-evidence source for grounded explanation reports.
- RWTOP2026-GTM: `representative-runnable` frequency-attention evidence encoder comparator.
- RWTOP2026-CALTSFM: `literature-only` calibration protocol until local confidence artifacts are generated.

## Compute Budget

- Available devices: local RTX 4090 GPUs `0,1` only; commands must bind `CUDA_VISIBLE_DEVICES=0` or `CUDA_VISIBLE_DEVICES=1` for local model/proxy runs.
- Default execution: evidence-chain generation and proxy evaluation should use one GPU or CPU; at most two GPU-backed jobs may run concurrently.
- Runtime tier: TOP LLM/time-series methods that require model sizes beyond 2x4090 are `resource-blocked` for exact reproduction and must be represented by a local proxy.
- Required metadata: device ID, GPU model, seed, batch size or prompt batch size, precision/quantization, runtime, and any OOM/failure reason must be captured with accepted artifacts.

## Strict-Reviewer Risks

- LLM prose without evidence fields will be rejected as ungrounded.
- User-study claims require a defined task protocol or must be scoped as proxy evaluation.
- Privacy and safety boundaries for uploaded/diagnostic data must be explicit.

## Acceptance Gates

- Canonical final manuscript entrypoint exists and compiles.
- Every natural-language explanation claim maps to structured evidence.
- Anti-hallucination and latency tables are generated from accepted artifacts.
- At least six baselines and the grounding/dialogue ablations above are present in accepted artifacts.
- Any SOTA decision-support claim satisfies the SOTA optimization gate.
- Submodule-local commit records the accepted paper package milestone.

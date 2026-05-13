# Status Report: Paper 03 - LLM Explainable FD Toolkit

Status reports are generated control-plane summaries, not accepted experiment evidence.

- Generated: `2026-05-14`
- Goal file: `paper/UXFD_paper/goal/03_llm_explainable_fd_toolkit.md`

## Current Verdict

- Submission ready: `False`
- Baselines declared: `7`
- Ablations declared: `7`
- Strict blockers: `8`
- Accepted artifact coverage: `0/16`
- Dirty submodule entries: `0`
- TOP recent-work methods in matrix: `7`
- Has 2026 TOP method: `True`
- TOP binding: `TOP-Q7-TIMESEG` -> `RWTOP2026-TIMESEG`
- TOP evidence ready: `False`
- TOP binding status: `pending_gpu_and_artifacts`

## Strict Blockers

- The manuscript/ieee_tii/main.tex entrypoint is a conservative compile checkpoint; it is not final evidence-bearing text.
- Only smoke run_meta.yaml/metrics.json are emitted; no accepted results/llm_evidence/**/{run_meta.yaml,metrics.json} package exists for the main protocol.
- No accepted six-condition LLM baseline table with matching prompts, seeds, metrics, latency, and unsupported-claim rate exists.
- Standalone and package-based template demos pass only as smoke checks; they are not accepted LLM evidence packages.
- Only smoke hallucination-checker, context-removal, and latency-sweep runners exist; no accepted main-protocol ablation artifacts exist yet.
- No accepted TOP representative command/log/artifact mapping yet.
- No GPU model/runtime metadata from local GPUs 0,1 yet.
- No SOTA or human-centered decision-support claim is allowed from this matrix alone.

## Next Gate

Do not mark this paper submission-ready until same-protocol accepted baseline, ablation, TOP representative, GPU metadata, and SOTA evidence are present under the artifact gate.

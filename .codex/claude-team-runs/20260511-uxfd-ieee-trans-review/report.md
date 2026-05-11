# UXFD Codex XHigh Subagent Integrated Report

Date: 2026-05-11

## Verdict

The seven-paper goal package has strong control-plane coverage, but it is not
IEEE Transactions submission-ready. Six local Codex xhigh read-only subagents
audited the package after the external Claude Team launch path was
policy-blocked.

## Scope Covered

- Paper02 and Paper07 manuscript/evidence readiness.
- Paper01 and Paper03 Toolkit/LLM evidence packages.
- Paper04 and Paper05 MoE/Fuzzy matrices and ablations.
- Paper06 proposition, mapping, and TeX readiness.
- Cross-paper TOP recent-work policy and representative bindings.
- Cross-paper execution gates, objective audit, and artifact gate.

## Cross-Paper Findings

- All named goal files and Spec Kit artifacts exist.
- All seven paper-local matrices exist and each has at least six baselines and
  six ablations.
- Paper-local command blockers have been converted to command-bound or
  non-accepted smoke surfaces, but the smoke artifacts are not accepted
  reviewer evidence.
- `paper/UXFD_paper/results/accepted_runs` is still absent, so there are no
  accepted `run_meta.yaml` records.
- `nvidia-smi -L` and PyTorch CUDA visibility fail in this session; accepted
  GPU evidence cannot be generated.
- TOP recent-work policy is coherent, but all seven TOP representative
  bindings are still `pending_gpu_and_artifacts`.
- Paper03 TOP representative binding was corrected from literature-only
  `RWTOP2026-CALTSFM` to representative-runnable `RWTOP2026-TIMESEG`.

## Paper Findings

| Paper | Current State | Primary Remaining Blocker |
|---|---|---|
| Paper01 Toolkit | Schema-shaped outputs and command-bound matrix exist; ablation runner is smoke-only. | No accepted six-baseline/ablation/TOP/GPU evidence; TeX still has placeholder/engine issues. |
| Paper02 1D-2D Fusion | Six baselines and seven ablations are command-bound; FFT/legacy surfaces have non-accepted smoke runners. | Canonical TeX still depends on `NatureMi.cls`; true Fusion1D2D real-data ablations are absent. |
| Paper03 LLM Toolkit | IEEE compile checkpoint exists; package and LLM smoke runners emit `accepted_evidence=false`. | No accepted `results/llm_evidence/**/{run_meta.yaml,metrics.json}` package. |
| Paper04 MoE | Six baselines and MoE ablation smoke surfaces are bound. | No accepted CWRU/XJTU or industrial multi-seed baseline/ablation artifacts or route metadata. |
| Paper05 Fuzzy-XFD | Seven baselines and six fuzzy ablations are command-bound. | Rule metrics, safety cases, and hard-threshold/safety/no-rule-output ablations remain non-accepted. |
| Paper06 Neural-Symbolic | Proposition hooks, mapping hook, and mapping-ablation smoke runner exist. | P2 remains failed/inconsistent; source-backed mapping and real-data proposition evidence are missing. |
| Paper07 Operator Attention | Canonical IEEE TeX wrapper exists and prior compile was recorded. | Synthetic evidence does not support industrial/SOTA claims; accepted industrial matrix is missing. |

## Recommended Non-GPU Next Actions

1. Fix manuscript truthfulness before GPU execution: Paper02 `NatureMi.cls`,
   Paper01 placeholder/Unicode compile issue, Paper06 placeholder/missing figure,
   and Paper07 performance overclaims.
2. Materialize local smoke artifact directories for Paper03 LLM evidence under
   `results/llm_evidence/demo_smoke/**` with `accepted_evidence=false`.
3. Add missing goal-suite baseline bindings where matrices are currently only
   count-complete, especially Paper06 `ISFM.M_01_ISFM` and MoE/Fuzzy declared
   baseline variants.
4. Strengthen citation hygiene beyond accepted-pool string checks before final
   manuscript submission.
5. Only after GPU Q0 passes, run queue entries with strict artifact metadata
   and keep SOTA wording blocked until same-protocol evidence beats baselines.

# T042 Submission Readiness Contract

Status date: 2026-05-11

This file is the submodule-local contract for moving T042 forward. It defines
the canonical manuscript package, the accepted LLM evidence protocol, and the
current blockers. It does not make this paper submission-ready.

## Current Verdict

Blocked.

- A conservative IEEE TeX entrypoint now exists, but it is a compile checkpoint
  rather than final evidence-bearing text.
- The existing draft contains numerical user-study and consistency claims that
  are not backed by accepted artifacts in this submodule.
- The LLM evidence protocol is now defined below. A command-bound matrix exists
  in `submission_prep/baseline_ablation_matrix.yaml`, but accepted baseline,
  ablation, SOTA, latency, and anti-hallucination artifacts are still missing.
- The standalone and package-based template LLM demos now run. The package demo
  emits smoke `run_meta.yaml` and `metrics.json`, but they remain non-accepted
  until main-protocol, same-prompt, GPU-recorded evidence packages exist.

## Canonical Manuscript Package

Target journal: IEEE Transactions on Industrial Informatics.

Alternate journal: IEEE Transactions on Human-Machine Systems.

Current canonical draft source:

- Body draft: `manuscript/drafts/paper.md`
- References draft: `manuscript/drafts/references.bib`
- Figure/table inventory: `manuscript/drafts/figures_and_tables.md`
- Conservative IEEE checkpoint: `manuscript/ieee_tii/main.tex`
- IEEE checkpoint bibliography: `manuscript/ieee_tii/references.bib`
- Existing generated assets:
  - `manuscript/figures/figure_5_quality_radar.pdf`
  - `manuscript/figures/figure_5_quality_radar.png`
  - `manuscript/tables/table_4_quality_metrics.tex`

Final IEEE package to complete before submission can be accepted:

- Entrypoint: `manuscript/ieee_tii/main.tex`
- Bibliography: `manuscript/ieee_tii/references.bib`
- Figures: `manuscript/ieee_tii/figures/`
- Tables: `manuscript/ieee_tii/tables/`

Compile command for the current IEEE checkpoint:

```bash
cd manuscript/ieee_tii
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Current manuscript blocker:

- `manuscript/ieee_tii/main.tex` is intentionally conservative and does not yet
  contain final claim-to-evidence content from accepted LLM artifacts.
- The final figure/table package remains blocked until accepted metrics,
  latency, hallucination, and TOP representative artifacts exist.

Claim rule:

- `manuscript/drafts/paper.md` is not accepted as final evidence-bearing text.
- Any numerical claim in the draft must map to an accepted run directory under
  `results/llm_evidence/` before it can be copied into final TeX.

## LLM Evidence Protocol

Accepted run directories must be submodule-local:

```text
results/llm_evidence/<protocol_id>/<condition_id>/seed_<seed>/
|-- run_meta.yaml
|-- metrics.json
|-- inputs/
|-- outputs/
|-- logs/
|   |-- stdout.log
|   `-- stderr.log
`-- artifacts/
```

Each accepted `run_meta.yaml` must record:

- `paper_id: LLM_Explainable_FD_Toolkit`
- `protocol_id`
- `condition_id`
- exact command
- working directory
- git commit for this submodule
- parent repository commit when available
- input artifact paths
- output artifact paths
- `CUDA_VISIBLE_DEVICES`
- GPU model
- GPU count
- seed
- batch size or prompt batch size
- precision or quantization
- start/end time
- runtime seconds
- OOM or failure reason, if any

Each accepted `metrics.json` must record the metric definitions and values for:

- decision accuracy or proxy task accuracy
- time-to-decision
- explanation quality score
- evidence consistency
- hallucination or unsupported-claim rate
- end-to-end latency p50 and p95
- failure rate
- sample count
- seed

No table, figure, or SOTA wording is accepted if either `run_meta.yaml` or
`metrics.json` is missing.

## Required Commands

Run from the parent repository root:

```bash
cd "$(git rev-parse --show-toplevel)"
```

Minimal PHM-Vibench smoke command, writing inside this submodule:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n LQ_signal python main.py --config paper/project/configs/vibench/min.yaml --override trainer.device=cuda --override model.device=cuda --override trainer.gpus=1 --override trainer.num_epochs=1 --override environment.seed=0 --override environment.output_dir=paper/project/results/vibench_min/gpu0_seed0
```

Local template LLM demo command, writing inside this submodule:

```bash
cd paper/project
CUDA_VISIBLE_DEVICES=0 conda run -n LQ_signal python experiments/scripts/run_minimal_llm_demo.py --mode pipeline --save --output results/llm_evidence/demo_smoke/template_llm/seed_0/artifacts
```

These commands are smoke commands only. The package demo writes `run_meta.yaml`
and `metrics.json` when `--save` is used, but those files are marked
`accepted_evidence: false` until the same prompt set, same protocol, GPU
metadata, and reviewer baseline/ablation gates are satisfied.

Non-accepted hallucination/context/latency smoke runner:

```bash
cd paper/project
CUDA_VISIBLE_DEVICES=0 conda run -n LQ_signal python experiments/scripts/run_llm_evidence_smoke.py --condition all --output results/llm_evidence/demo_smoke --seed 0
```

This runner writes `run_meta.yaml`, `metrics.json`, prompt sets, and responses
for grounded, no-checker, no-domain-context, and latency conditions. It is a
smoke runner only and marks all outputs `accepted_evidence: false`.

Current command-bound checkpoint:

- `submission_prep/baseline_ablation_matrix.yaml` records the PHM-Vibench
  proposed smoke, no-agent smoke, six diagnostic baseline smokes, standalone
  template LLM pipeline/single-case demos, and package-level smoke gates.
- `python experiments/scripts/run_minimal_llm_demo_standalone.py --mode pipeline`
  and `--mode single --case 0` pass in `LQ_signal`.
- `python experiments/scripts/run_minimal_llm_demo.py --mode pipeline --save ...`
  passes and writes demo artifacts plus smoke `run_meta.yaml` and `metrics.json`
  under `/tmp/uxfd_paper03_template_llm_artifacts`.
- `python experiments/scripts/run_llm_evidence_smoke.py --condition all ...`
  passes and writes smoke hallucination/context/latency ablation artifacts.
- `python -m pytest -q code/tests/test_basic_functionality.py` passes with
  `14 passed`.
- This validates only executable surfaces and blockers; it is not accepted LLM
  evidence for manuscript claims.

## Baseline Evidence Gate

At least six same-protocol baselines are required before performance claims are
accepted.

| ID | Baseline condition | Accepted evidence required | Current status |
|---|---|---|---|
| B1 | Template-only structured report without LLM generation | `results/llm_evidence/main_protocol/template_only/seed_*/` | blocked: no accepted run |
| B2 | Generic LLM prompt without evidence-field grounding | `results/llm_evidence/main_protocol/generic_llm_no_grounding/seed_*/` | blocked: no accepted run |
| B3 | Retrieval-augmented LLM prompt using the same knowledge base | `results/llm_evidence/main_protocol/rag_same_kb/seed_*/` | blocked: no accepted run |
| B4 | `Explainable_FD_Toolkit` structured output without dialogue layer | `results/llm_evidence/main_protocol/structured_no_dialogue/seed_*/` | blocked: upstream accepted structured inputs missing |
| B5 | Rule-based natural-language explanation from fuzzy rules | `results/llm_evidence/main_protocol/rule_based_fuzzy_text/seed_*/` | blocked: no accepted run |
| B6 | SHAP/LIME text summary from post-hoc feature importance | `results/llm_evidence/main_protocol/shap_lime_text/seed_*/` | blocked: no accepted run |
| B7 | Human-written report subset upper-reference comparator | `results/llm_evidence/main_protocol/human_reference/seed_*/` | optional, blocked until data exists |

The same dataset split, preprocessing, seeds, task prompts, metric definitions,
and report format must be used for every accepted baseline.

## TOP Recent-Work Gate

The accepted local recent-work pool for this paper is:

- `RWTOP2024-TIMELLM`: representative-runnable only under local LLM/proxy evidence.
- `RWTOP2024-MOMENT`: representative-runnable foundation-style structured input proxy.
- `RWTOP2025-TIMEMOE`: representative-runnable sparse/foundation comparator.
- `RWTOP2025-CBAE`: literature-only until fault-diagnosis concept supervision exists.

At least one evidence-grounded LLM or local proxy run is required before this
paper satisfies the runnable TOP baseline gate. `RWTOP2025-CBAE` must not be
counted as a reproduced baseline.

## Ablation Evidence Gate

The ablation suite must use the same protocol as the baseline suite:

- remove evidence-field grounding
- remove retrieval/domain knowledge context
- remove dialogue state tracking
- remove hallucination checker
- compare one-shot explanation vs. multi-turn diagnostic dialogue
- compare short, medium, and long explanation templates for latency and failure
  rate

Accepted ablation artifacts must live under:

```text
results/llm_evidence/main_protocol/ablation_<name>/seed_<seed>/
```

## SOTA Gate

SOTA decision-support wording is blocked until the proposed method beats every
declared baseline under the same protocol on:

- task accuracy or proxy task accuracy
- time-to-decision
- evidence consistency
- hallucination or unsupported-claim rate
- latency p95
- failure rate

If no human or user-task study exists, all results must be labeled as proxy
evaluation and the manuscript must not claim human-centered SOTA.

## Missing Artifacts To Unblock Submission Readiness

1. Expand `manuscript/ieee_tii/main.tex` into final evidence-bearing IEEE text
   after accepted artifacts exist.
2. Replace unverified numerical claims in `manuscript/drafts/paper.md` with
   claims backed by accepted `results/llm_evidence/` artifacts.
3. Produce accepted structured explanation inputs from `Explainable_FD_Toolkit`
   and record their paths in `results/llm_evidence/main_protocol/*/run_meta.yaml`.
4. Run at least six baseline conditions with matching data, prompts, seeds, and
   metrics.
5. Run the ablation suite above.
6. Produce anti-hallucination and latency tables from accepted artifacts,
   including latency p95 and failure rate.
7. Produce at least one evidence-grounded TOP representative run for the LLM
   recent-work gate.
8. Record GPU metadata for every accepted run using only local RTX 4090 GPUs
   `0` or `1`.

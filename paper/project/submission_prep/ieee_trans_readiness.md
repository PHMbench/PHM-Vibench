# Paper 03 LLM Explainable FD Toolkit IEEE Transactions Readiness

Date: 2026-05-11

This checkpoint adds a command-bound baseline and ablation blocker matrix for
the LLM Explainable FD Toolkit paper. It does not make the paper
submission-ready.

## Current Evidence

- Matrix: `submission_prep/baseline_ablation_matrix.yaml`
- Accepted evidence package contract:
  `submission_prep/llm_evidence_package_contract.md`
- Base config: `configs/vibench/min.yaml`
- Conservative IEEE TeX entrypoint: `manuscript/ieee_tii/main.tex`
- Bibliography: `manuscript/ieee_tii/references.bib`
- Existing gate: `SUBMISSION_READINESS.md`
- Evidence level: PHM-Vibench dummy smokes plus standalone and package-based
  template LLM demos plus a conservative IEEE compile checkpoint
- Compute policy: local RTX 4090 GPUs `0,1`; runnable PHM-Vibench commands bind
  `CUDA_VISIBLE_DEVICES=0` or `CUDA_VISIBLE_DEVICES=1`

## Dummy-Smoke Summary

The PHM-Vibench proposed smoke, the no-agent smoke, and six model baselines
completed in `LQ_signal` on dummy data with CPU fallback because the current
environment reported `GPU available: False` and `Can't initialize NVML`.

The standalone template demo completed in both pipeline and single-case modes.
The package-based template demo also runs after the local template/knowledge
fallback fix and can save demo artifacts plus smoke-level `run_meta.yaml` and
`metrics.json` under `/tmp`. Those files are marked `accepted_evidence: false`.
They do not create the required accepted main-protocol
`results/llm_evidence/**/{run_meta.yaml,metrics.json}` package.

The non-accepted LLM evidence smoke runner also exists:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/scripts/run_llm_evidence_smoke.py --condition all --output /tmp/uxfd_paper03_llm_evidence_smoke --seed 0
```

It writes per-condition `run_meta.yaml`, `metrics.json`, prompt sets, and
responses for grounded, no-checker, no-domain-context, and latency conditions.
These artifacts remain smoke-level only and are not accepted reviewer evidence.

## Current Package Gate

The actual package path now passes its local smoke gates:

- `python experiments/scripts/run_minimal_llm_demo.py --mode pipeline --save ...`
  passes and writes demo artifacts plus smoke `run_meta.yaml` and `metrics.json`
  under `/tmp/uxfd_paper03_template_llm_artifacts`.
- `python experiments/scripts/run_llm_evidence_smoke.py --condition all ...`
  passes and writes non-accepted smoke artifacts for hallucination checking,
  context removal, and latency p50/p95 proxies.
- `python -m pytest -q code/tests/test_basic_functionality.py` passes with
  `14 passed`.

This supports package importability and smoke-level workflow exploration only;
it still cannot support package-level LLM toolkit claims without accepted
main-protocol `run_meta.yaml`, `metrics.json`, latency, and unsupported-claim
artifacts.

## Current Manuscript Gate

`manuscript/ieee_tii/main.tex` now exists as a conservative IEEE Transactions
checkpoint with local references in `manuscript/ieee_tii/references.bib`. It is
intentionally evidence-conservative and does not copy unsupported numerical
claims from the Markdown draft.

The compile gate to run from `manuscript/ieee_tii/` is:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

This clears the missing-entrypoint blocker only. The paper remains blocked
until accepted LLM evidence packages, same-protocol tables, TOP representatives,
latency/hallucination metrics, and GPU metadata exist.

## Remaining Gaps

- Evidence-bearing final IEEE manuscript content mapped to accepted artifacts.
- Accepted `results/llm_evidence/**/{run_meta.yaml,metrics.json}` packages.
- Parent-gated LLM evidence package with prompt sets, responses,
  unsupported-claim metrics, latency metrics, and clean SHA/GPU metadata.
- Six-condition same-prompt LLM baseline table.
- Accepted hallucination/unsupported-claim, latency p50/p95, failure-rate, and
  time-to-decision metrics under the main protocol.
- Accepted retrieval/domain-context, hallucination-checker, dialogue-state, and
  template-length ablations.
- TOP representative artifacts for TimeLLM, MOMENT, Time-MoE, or a local
  faithful proxy under the 2x4090 budget.
- Complete strict local GPU metadata from devices `0,1`.
- SOTA or human-centered decision-support gate.

## Allowed Manuscript Wording

The manuscript may state that a standalone template LLM demonstration and a
package-based template LLM smoke surface are runnable. It must not claim
accepted package evidence, anti-hallucination performance, latency superiority,
human-task benefit, TOP-method reproduction, GPU feasibility, or SOTA from this
checkpoint.

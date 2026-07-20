# T044 Submission Readiness Evidence And Blockers

Date: 2026-05-11

Scope: Fuzzy-XFD only. This document records what can be claimed now and what
is still blocked before IEEE Transactions submission. It intentionally does not
promote old numeric claims from draft text unless a same-protocol artifact is
present in this submodule.

## Accepted Current-State Evidence

The following artifacts exist and may support only method-structure or
rule-visualization claims:

| Claim type | Accepted artifact | Allowed wording |
|---|---|---|
| Fuzzy rules are implemented locally | `code/fuzzy_system/rule_base.py` | The submodule contains fuzzy rule and rule-base code. |
| Fuzzy inference is implemented locally | `code/fuzzy_system/inference_engine.py` | The submodule contains fuzzy inference and explanation-path mechanics. |
| Membership functions can be visualized | `FuzzyLogic_explainable/results/fuzzy_membership_functions.pdf` | Membership-function visualization exists. |
| Rule activations can be visualized | `FuzzyLogic_explainable/results/fuzzy_rule_heatmap.pdf` | Rule-heatmap visualization exists. |
| Inference process can be visualized | `FuzzyLogic_explainable/results/fuzzy_inference_process.pdf` | Fuzzy-inference process visualization exists. |
| Canonical entrypoint is rewritten | `manuscript/final_tex/main.tex` | The entrypoint is a compilable readiness snapshot, not a final IEEE paper. |
| Minimal PHM-Vibench smoke path is declared | `VIBENCH.md`, `configs/vibench/min.yaml` | A smoke configuration exists for entrypoint validation. |
| Baseline and ablation commands are bound | `submission_prep/baseline_ablation_matrix.yaml` | The comparison surface has runnable dummy-smoke commands, not accepted paper results. |

Not accepted as final evidence:

- `manuscript/paper.md` performance, parameter-count, safety, and SOTA claims.
- `doc/safety_critical_case_studies.md` narrative examples without sample IDs,
  rule activations, membership values, and decision paths.
- Ignored files under `results/` unless copied into an accepted evidence package
  with metadata, logs, configs, and exact command provenance.
- `results/cwru_fuzzy.log`, which records a failed run caused by missing
  `pytorch_lightning`.

## Smoke Command

Run from the parent PHM-Vibench repo root:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config paper/UXFD_paper/Paper_fuzzy_XFD/configs/vibench/min.yaml \
  --override trainer.num_epochs=1
```

Expected smoke output location:

```text
paper/UXFD_paper/Paper_fuzzy_XFD/results/uxfd/pilot/
```

This smoke run is an instantiation check only. It does not satisfy the
baseline, ablation, rule-metric, safety-case, recent-work, or SOTA gates.

Observed validation on 2026-05-11:

- Default `python` failed preflight with
  `ModuleNotFoundError("No module named 'pytorch_lightning'")`.
- `conda run -n LQ_signal` completed the smoke run with the same config, but
  PyTorch Lightning reported `GPU available: False` and `Can't initialize NVML`.
  Therefore the command validates config/model instantiation only; it is not
  accepted GPU-feasibility evidence.

## Required Evidence Package Layout

Every accepted run must record:

- `command.txt`
- `run_meta.yaml` with device IDs, GPU model, GPU count, seed, batch size,
  precision, runtime, dataset, split, metric definitions, and OOM/failure reason
  if any
- `config_resolved.yaml`
- `stdout.log`
- `metrics.json`
- `artifacts/predictions.npz`
- rule artifacts when applicable: `rule_activations.json`,
  `membership_values.npz`, `decision_paths.json`

Required root:

```text
paper/UXFD_paper/Paper_fuzzy_XFD/results/evidence/t044/
```

## Baseline Gate

Status: command-bound, but accepted evidence remains blocked.

Current command-bound matrix:

```text
submission_prep/baseline_ablation_matrix.yaml
```

The matrix records dummy-smoke passes for six PHM-Vibench model baselines plus a
paper-local classical fuzzy script. These runs use dummy data and CPU fallback in
the current environment, so they do not replace the required same-protocol
CWRU/XJTU or industrial artifacts below.

Minimum accepted baseline suite:

| Baseline | Missing config | Missing artifacts |
|---|---|---|
| `ISFM.M_01_ISFM` | `configs/vibench/baselines/{cwru,xjtu}/isfm_m01.yaml` | `results/evidence/t044/baselines/{cwru,xjtu}/isfm_m01/seed_{42,123,456}/` |
| `NSN/TSPN_UXFD` without fuzzy rules | `configs/vibench/baselines/{cwru,xjtu}/tspn_no_fuzzy.yaml` | `results/evidence/t044/baselines/{cwru,xjtu}/tspn_no_fuzzy/seed_{42,123,456}/` |
| `CNN.ResNet1D` | `configs/vibench/baselines/{cwru,xjtu}/resnet1d.yaml` | `results/evidence/t044/baselines/{cwru,xjtu}/resnet1d/seed_{42,123,456}/` |
| `X_model.Sincnet` | `configs/vibench/baselines/{cwru,xjtu}/sincnet.yaml` | `results/evidence/t044/baselines/{cwru,xjtu}/sincnet/seed_{42,123,456}/` |
| `X_model.TFN` | `configs/vibench/baselines/{cwru,xjtu}/tfn.yaml` | `results/evidence/t044/baselines/{cwru,xjtu}/tfn/seed_{42,123,456}/` |
| `X_model.WKN` | `configs/vibench/baselines/{cwru,xjtu}/wkn.yaml` | `results/evidence/t044/baselines/{cwru,xjtu}/wkn/seed_{42,123,456}/` |
| Classical fuzzy/rule baseline | `configs/vibench/baselines/{cwru,xjtu}/classical_fuzzy.yaml` | `results/evidence/t044/baselines/{cwru,xjtu}/classical_fuzzy/seed_{42,123,456}/` |

Command pattern after each missing config exists:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config paper/UXFD_paper/Paper_fuzzy_XFD/configs/vibench/baselines/cwru/isfm_m01.yaml \
  --override environment.seed=42 \
  --override environment.output_dir=paper/UXFD_paper/Paper_fuzzy_XFD/results/evidence/t044/baselines/cwru/isfm_m01/seed_42
```

Use GPU `0` or `1` for one run at a time, with at most two concurrent
single-GPU jobs. A two-GPU command must use `CUDA_VISIBLE_DEVICES=0,1` and record
why both GPUs are necessary.

## Rule-Level Evidence Gate

Status: blocked.

Required missing artifacts:

```text
results/evidence/t044/rule_metrics/{cwru,xjtu}/seed_{42,123,456}/faithfulness.json
results/evidence/t044/rule_metrics/{cwru,xjtu}/seed_{42,123,456}/stability.json
results/evidence/t044/rule_metrics/{cwru,xjtu}/seed_{42,123,456}/sparsity.json
results/evidence/t044/rule_metrics/{cwru,xjtu}/seed_{42,123,456}/efficiency.json
```

Required missing evaluator:

```text
scripts/evaluate_rule_metrics.py
```

Expected command after the evaluator exists:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_rule_metrics.py \
  --run-dir paper/UXFD_paper/Paper_fuzzy_XFD/results/evidence/t044/proposed/cwru/seed_42 \
  --output paper/UXFD_paper/Paper_fuzzy_XFD/results/evidence/t044/rule_metrics/cwru/seed_42
```

## Ablation Gate

Status: command-bound for six supported fuzzy sensitivity/removal checks, but
accepted evidence remains blocked.

Current command-bound ablations:

- remove fuzzy decision head
- uncalibrated fuzzy residual scale
- weak fuzzy residual scale
- low rule-count fuzzy head
- single membership function
- narrow fuzzy feature bottleneck

These are dummy-smoke validation commands only. Hard-threshold inference,
safety-fallback removal, and no-rule-output ablations remain implementation
blockers because no PHM-Vibench config switch or evaluator exists for them yet.

Required ablations:

| Ablation | Missing config | Missing artifacts |
|---|---|---|
| Remove fuzzy rule layer | `configs/vibench/ablations/{cwru,xjtu}/no_fuzzy_rule_layer.yaml` | `results/evidence/t044/ablations/{cwru,xjtu}/no_fuzzy_rule_layer/seed_{42,123,456}/` |
| Remove membership calibration | `configs/vibench/ablations/{cwru,xjtu}/no_membership_calibration.yaml` | `results/evidence/t044/ablations/{cwru,xjtu}/no_membership_calibration/seed_{42,123,456}/` |
| Replace fuzzy inference with hard thresholds | `configs/vibench/ablations/{cwru,xjtu}/hard_threshold_inference.yaml` | `results/evidence/t044/ablations/{cwru,xjtu}/hard_threshold_inference/seed_{42,123,456}/` |
| Vary number of rules and membership functions | `configs/vibench/ablations/{cwru,xjtu}/rule_membership_sweep.yaml` | `results/evidence/t044/ablations/{cwru,xjtu}/rule_membership_sweep/seed_{42,123,456}/` |
| Remove safety fallback path | `configs/vibench/ablations/{cwru,xjtu}/no_safety_fallback.yaml` | `results/evidence/t044/ablations/{cwru,xjtu}/no_safety_fallback/seed_{42,123,456}/` |
| Remove rule-level explanation output | `configs/vibench/ablations/{cwru,xjtu}/no_rule_explanation_output.yaml` | `results/evidence/t044/ablations/{cwru,xjtu}/no_rule_explanation_output/seed_{42,123,456}/` |

Command pattern after each missing config exists:

```bash
CUDA_VISIBLE_DEVICES=1 python main.py \
  --config paper/UXFD_paper/Paper_fuzzy_XFD/configs/vibench/ablations/cwru/no_fuzzy_rule_layer.yaml \
  --override environment.seed=42 \
  --override environment.output_dir=paper/UXFD_paper/Paper_fuzzy_XFD/results/evidence/t044/ablations/cwru/no_fuzzy_rule_layer/seed_42
```

## Safety-Case Gate

Status: blocked.

Required missing artifacts:

```text
results/evidence/t044/safety_cases/case_001.md
results/evidence/t044/safety_cases/case_002.md
results/evidence/t044/safety_cases/case_003.md
results/evidence/t044/safety_cases/membership_values_case_*.npz
results/evidence/t044/safety_cases/decision_paths_case_*.json
```

Required missing collector:

```text
scripts/collect_safety_cases.py
```

Expected command after the collector exists:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/collect_safety_cases.py \
  --run-dir paper/UXFD_paper/Paper_fuzzy_XFD/results/evidence/t044/proposed/cwru/seed_42 \
  --output paper/UXFD_paper/Paper_fuzzy_XFD/results/evidence/t044/safety_cases
```

Each case must include true label, predicted label, sample ID, dataset, split,
triggered rules, membership values, final decision path, and why the case is
safety-critical.

## TOP Recent-Work Gate

Status: blocked.

Accepted TOP-source mapping from the goal package:

| ID | Local status for this paper | Required local artifact |
|---|---|---|
| `RWTOP2024-TIMEXPP` | representative-runnable, but no local artifact yet | `results/evidence/t044/top_recent/rwtop2024_timexpp_proxy/{cwru,xjtu}/seed_{42,123,456}/` |
| `RWTOP2025-CFCBM` | literature-only until FD concept labels are defined | concept-label protocol and local proxy config |
| `RWTOP2025-CBAE` | literature-only until concept supervision is adapted | concept-supervision protocol and local proxy config |
| `RWTOP2025-IFCBM` | literature-only until task mapping is defined | FD concept/task mapping and local proxy config |

Representative TimeX++ command pattern after config exists:

```bash
CUDA_VISIBLE_DEVICES=1 python main.py \
  --config paper/UXFD_paper/Paper_fuzzy_XFD/configs/vibench/top_recent/cwru/rwtop2024_timexpp_proxy.yaml \
  --override environment.seed=42 \
  --override environment.output_dir=paper/UXFD_paper/Paper_fuzzy_XFD/results/evidence/t044/top_recent/rwtop2024_timexpp_proxy/cwru/seed_42
```

## SOTA Gate

Status: blocked.

No SOTA wording is allowed until all of the following are true:

- The proposed Fuzzy-XFD run has accepted CWRU and XJTU 3-seed artifacts.
- At least six declared baselines have accepted same-protocol artifacts.
- The required ablations have accepted same-protocol artifacts.
- At least one TOP recent-work exact or representative run has accepted local
  artifacts under the `CUDA_VISIBLE_DEVICES=0` / `1` / `0,1` budget.
- The aggregate table proves the proposed method beats every accepted baseline
  on the claim being made. If accuracy drops while rule auditability improves,
  the manuscript must state that tradeoff instead of claiming accuracy SOTA.

## Submodule Commit Gate

Status: not ready yet.

Before the parent repo updates the `Paper_fuzzy_XFD` gitlink, the submodule must
commit the accepted content changes locally. At this checkpoint, source artifacts
needed for submission claims are still missing, so the submodule is not ready for
an accepted paper-package commit.

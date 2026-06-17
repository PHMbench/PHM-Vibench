# PHM-GenBench v0.3 Paper-Readiness Review

Date: 2026-06-10

Review source:
`.specify/goals/v3/phm_generative_paper_update_pack_v0_3/14_reviewer.md`

Branch target: `Feature_factory-update`

## 1. Executive decision

```xml
<REVIEW_DECISION>BLOCKING</REVIEW_DECISION>
<PAPER_READINESS_SCORE>76</PAPER_READINESS_SCORE>
<PAPER_READINESS_LEVEL>WORKSHOP_LEVEL</PAPER_READINESS_LEVEL>
<BRANCH>Feature_factory-update</BRANCH>
```

Rationale: the v0.3 evidence-chain blockers for stage ledger, eval evidence
sidecar, strict condition split evidence, strict pipeline shape conversion,
TSTR/TRTS probe naming, six-dataset dry-run planning, a first classifier utility
protocol skeleton, sampling-rate-aware PHM spectral metrics, and a conservative
benchmark-effect promotion gate are now addressed in code and focused tests.
The repository is still blocked for
benchmark-paper claims because no complete six-dataset train/sample/eval/
paperpack run set is available and no row can yet be promoted to benchmark-valid
paper evidence.

## 2. Scorecard

| Axis | Score 0-5 | Evidence paths | Evidence summary | Missing pieces | Blocking? |
| --- | ---: | --- | --- | --- | --- |
| A CLI/preflight | 4 | `main.py`, `configs/demo/10_generative/dummy_generative_cfm.yaml` | Generative preflight passes and exits before trainer execution. | Full malformed-config matrix not rerun in this review. | No |
| B config completeness | 4 | `configs/demo/10_generative/`, `configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml` | Six datasets, three baseline methods, two seeds, and dry-run matrix exist. | Real dataset metadata availability and complete run evidence. | Yes |
| C pipeline train/sample/eval | 4 | `src/Pipeline_06_generative.py`, `/tmp/phm_genbench_v3_smoke/stage_ledger.json` | CPU smoke train/sample/eval/paperpack loop writes a shared stage ledger, sample manifest, eval metrics, eval evidence sidecar, and paperpack dir. | Full train/sample/eval execution over real six-dataset runs. | Yes |
| D model_factory/generative_model | 3 | `src/model_factory/generative_model/`, `src/model_factory/model_registry.csv` | MLP1D, UNet1D, DiT1D, and SSM-style modules are factory native. | Paper-grade model-card/test coverage for every promoted backbone. | No |
| E loss correctness | 4 | `src/task_factory/Components/generative/losses/` | CFM target is `x1-z`; DDPM target is epsilon; Score SDE is score/DSM skeleton. | Score SDE remains research-only rather than paper baseline. | No |
| F task/training completeness | 3 | `src/task_factory/task/generative/`, `/tmp/phm_genbench_v3_smoke/train/` | Lightning train smoke produced a checkpoint and `train_result_0.csv`. | One-epoch checkpoint evidence for each paper method/dataset. | Yes |
| G sampler completeness | 4 | `src/task_factory/Components/generative/samplers/` | Euler ODE and DDPM samplers record finite-shape guarded outputs and NFE/steps through pipeline metadata. | Real sampling artifacts from trained checkpoints. | Yes |
| H normalization/domain evidence | 4 | `src/data_factory/data_utils.py`, `src/task_factory/Components/generative/manifests/synthetic_data_manifest.py` | Train-only normalization artifacts and domain-map hashes are manifest evidence gates. | Real-run normalization artifacts for six datasets. | Yes |
| I temporal metrics | 3 | `src/task_factory/Components/generative/metrics/temporal.py` | Smoke/exploratory temporal bundle exists with status/reason annotations. | Confidence intervals and PHM event/peak interval statistics. | No |
| J spectral metrics | 4 | `src/task_factory/Components/generative/metrics/spectral.py` | FFT/log-spectrum metrics plus sampling-rate-aware PHM band, envelope, fault peak, harmonic, and coherence metrics exist with status/reason behavior. | Real-run PHM spectral evidence and dataset-specific fault-frequency metadata coverage. | No |
| K distribution metrics | 3 | `src/task_factory/Components/generative/metrics/distribution.py` | Mean/variance/distribution distances exist. | Stronger feature-space paper metric such as PHM feature-FID. | No |
| L diversity metrics | 3 | `src/task_factory/Components/generative/metrics/diversity.py` | Diversity metrics return status/reason when sample count or labels are insufficient. | Larger-sample confidence intervals and per-condition reporting. | No |
| M leakage metrics | 3 | `src/task_factory/Components/generative/metrics/leakage.py` | Nearest-neighbor and duplicate-rate evidence exists. | Threshold sensitivity and train-vs-test leakage ratio. | No |
| N utility metrics | 4 | `src/task_factory/Components/generative/metrics/tstr.py` | Nearest-centroid TSTR/TRTS probe names are explicit; deterministic linear classifier TSTR/TRTS and real+synth gain metrics are emitted with status/reason fields. | Real-run utility evidence and stronger classifier protocol review. | Yes |
| O missing-metric status/reason | 4 | `src/task_factory/task/generative/generative_eval.py`, `scripts/paperpack_generative.py` | Metric status/reason propagates into paperpack missing-metric audit. | Policy file for primary metric exclusion is not wired as a runtime gate. | No |
| P paperpack | 4 | `scripts/paperpack_generative.py`, `/tmp/phm_genbench_v3_smoke/.../paperpack/` | Paperpack accepts `--stage_ledger`, includes sibling sample manifests, and writes `paperpack_dir` back to the stage ledger. | Real paperpack outputs for every run group. | Yes |
| Q benchmark-effect aggregation | 4 | `scripts/generative_benchmark_effect.py` | 144-row six-dataset dry-run plan is produced with ledger-aware commands; aggregation resolves sibling sample manifests through eval evidence sidecars or stage ledgers and only promotes rows when eval evidence is eligible and paperpack is traceable. | Aggregation from real six-dataset run directories. | Yes |
| R submission-draft gating | 4 | `scripts/generative_submission_draft.py` | Draft generator is conservative and refuses submission-ready status without evidence. | Real benchmark-effect summary/manifest. | Yes |
| S tests/validation coverage | 4 | `test/generative/`, `scripts.validate_docs` | 64 targeted v0.3 tests passed; preflight, dry-run, and docs validation passed. | Full repository suite not rerun in this review turn. | No |
| T branch goal/handoff hygiene | 4 | `.specify/goals/v3/`, `specs/002-phm-genbench-frontier/reviews/v3/` | v0.3 pack is present; canonical baseline and closure reviewer-gate artifacts are written under `reviews/v3/`. | v3 pack is currently untracked. | No |

## 3. Blocking issues

### B1. No complete six-dataset benchmark-valid run evidence

- Issue: `configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml` can
  produce the expected dry-run plan, but no complete real run set is available
  under `results/paper/phm_generative/six_dataset_submission_v1/runs`.
- Why it blocks: paper claims require train/sample/eval/paperpack artifacts with
  source paths, metric rows, manifests, and benchmark-valid status for each
  reported dataset/method/seed row.
- Affected files: `configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml`,
  `scripts/generative_benchmark_effect.py`, `scripts/generative_submission_draft.py`.
- Required fix: run the staged six-dataset queue after CUDA/data availability is
  confirmed, then aggregate from real runs.
- Validation command:

```bash
python -m scripts.generative_benchmark_effect \
  --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml \
  --from-runs results/paper/phm_generative/six_dataset_submission_v1/runs \
  --output-dir results/paper/phm_generative/six_dataset_submission_v1/effect
```

### B2. Exploratory one-step methods are not faithful paper baselines

- Issue: MeanFlow, Drifting Flow, Transition Flow Matching, and OT-NFM are
  registered but remain exploratory velocity-contract placeholders.
- Why it blocks: including them in main paper comparison tables would overclaim
  method fidelity.
- Affected files: `src/task_factory/task/generative/_experimental_one_step.py`,
  `src/task_factory/task_registry.csv`, `configs/base/task/generative_*.yaml`.
- Required fix: keep these methods out of benchmark-valid main tables until each
  receives method-specific loss/sampler evidence and review.
- Validation command:

```bash
python -m pytest test/generative/test_one_step_experimental.py
```

## 4. Metric gap matrix

| Metric family | Current implementation | Paper-level requirement | Gap | Blocking? | Recommended goal |
| --- | --- | --- | --- | --- | --- |
| Temporal | Basic time-domain distances and status/reason | Per-domain/fault aggregation, CIs, PHM event stats | Aggregation exists in paperpack, but CIs/event stats missing | No | GOAL-V3-METRIC-001 |
| Spectral | FFT/log spectrum plus PHM band, envelope, fault peak, harmonic ratio, and coherence metrics | Dataset-specific sampling-rate/fault-frequency evidence across real runs | Real-run metadata coverage missing | No | GOAL-V3-RUN-001 |
| Distribution | Mean/variance/distribution distances | Feature-space PHM similarity | No feature-FID/discriminative score | No | GOAL-V3-METRIC-003 |
| Diversity | PRDC/intra-class style smoke metrics | Per-condition diversity with CIs | Larger-sample statistical reporting missing | No | GOAL-V3-METRIC-004 |
| Leakage | Nearest neighbor and duplicate rate | Threshold sensitivity and train/test ratio | Threshold sweep missing | No | GOAL-V3-METRIC-005 |
| Utility | Nearest-centroid TSTR/TRTS plus deterministic linear classifier TSTR/TRTS and real+synth gain | Stronger downstream classifier protocol over real benchmark splits | Real-run utility evidence missing | Yes | GOAL-V3-RUN-001 |
| Efficiency | parameter count, NFE, time, samples/sec, memory | Consistent real-run hardware metadata | Real GPU evidence missing | Yes | GOAL-V3-RUN-001 |
| Status/reason | Metric statuses propagate into eval/paperpack | Primary metric policy gate | Runtime gate not wired to policy YAML | No | GOAL-V3-EVIDENCE-001 |

## 5. Paper evidence matrix

| Paper artifact | Required evidence | Exists? | Source path | Missing action | Benchmark-valid gate |
| --- | --- | --- | --- | --- | --- |
| Table 1 dataset/domain summary | Six configured datasets and protocol rows | Partial | `configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml` | Add observed real-run coverage | configured and observed dataset count >= 6 |
| Table 2 model-family summary | Method/backbone/loss cards | Partial | `src/model_factory/generative_model/README.md` | Add promoted model cards for final methods | registry + config + smoke forward |
| Table 3 fidelity metrics | Quality metrics per dataset/method/seed | Missing real evidence | `scripts/paperpack_generative.py`, `src/task_factory/Components/generative/metrics/spectral.py` | Run eval/paperpack on real outputs | benchmark-valid metric rows |
| Table 4 utility metrics | Utility metrics with source paths | Partial | `src/task_factory/Components/generative/metrics/tstr.py` | Produce real utility rows from benchmark runs | benchmark-valid utility evidence |
| Table 5 leakage/validity | Manifest leakage and validity gates | Partial | `synthetic_data_manifest.py` | Produce real manifests | leakage pass and manifest evidence |
| Table 6 efficiency | NFE/time/memory/parameter count | Partial | `src/Pipeline_06_generative.py` | Produce real sample/eval artifacts | hardware + source path |
| Table 7 ablation | Multi-seed/backbone/condition ablations | Partial | `scripts/paperpack_generative.py` | Run ablations | all contributing rows valid |
| Figure 1 pipeline architecture | Source-linked pipeline description | Partial | `src/Pipeline_06_generative.py` | Add final figure source | paperpack traceability |
| Figure 2 model/loss taxonomy | Model/loss registry evidence | Partial | registry CSVs and READMEs | Separate exploratory methods | no placeholder baselines |
| Figure 3 signal gallery | Generated samples and conditions | Missing real evidence | `samples.pt` via sample stage | Run sample stage | sample manifest present |
| Figure 4 temporal/spectral overlays | Figure-source CSVs | Scaffold | `paperpack/figure_sources/` | Produce from real runs | source paths present |
| Figure 5 distribution/diversity plots | Figure-source CSVs | Scaffold | `paperpack/figure_sources/` | Produce from real runs | source paths present |
| Figure 6 utility plots | Utility figure-source rows | Scaffold | `paperpack/figure_sources/` | Add full utility protocol | utility evidence exists |
| Figure 7 leakage plots | Leakage rows and manifest checks | Scaffold | `paperpack/tables/table_leakage.csv` | Produce from real runs | leakage pass |
| Figure 8 efficiency/quality trade-off | Efficiency + quality rows | Scaffold | `paperpack/figure_sources/metric_barplot.csv` | Produce from real runs | quality and efficiency source paths |
| Reproducibility statement | Config/protocol/hash/run paths | Scaffold | `scripts/paperpack_generative.py` | Generate per run | manifest completeness |
| Missing metric appendix | Missing status/reason audit | Scaffold | `paperpack/appendix/missing_metrics.csv` | Produce from real runs | no silent missing primary metric |
| Benchmark effect manifest | Aggregation manifest | Scaffold | `scripts/generative_benchmark_effect.py` | Aggregate from real runs | observed configured dataset count |
| Submission readiness report | Conservative draft sidecars | Scaffold | `scripts/generative_submission_draft.py` | Feed real summary/manifest | no exploratory rows |

## 6. Branch-native validation commands

Commands run during the v0.3 review pass:

```bash
python -m pytest \
  test/generative/test_condition_sampling.py \
  test/generative/test_manifest_validity.py \
  test/generative/test_stage_ledger.py \
  test/generative/test_generative_metrics.py \
  test/generative/test_utility_protocols.py \
  test/generative/test_paperpack_generative.py \
  test/generative/test_benchmark_effect.py
```

Status: passed, 64 tests after adding classifier utility, PHM spectral coverage,
stage-ledger-aware benchmark-effect aggregation, conservative promotion gating
for eval evidence plus paperpack traceability, and dry-run status-ledger/manifest
coverage.

```bash
python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml --preflight-only
```

Status: passed with `[OK] preflight passed`.

```bash
python -m scripts.generative_benchmark_effect \
  --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml \
  --dry-run --allow-missing-data \
  --output-dir /tmp/phm_genbench_v3_dryrun
```

Status: passed. The generated plan has 144 rows covering six datasets, three
methods, two seeds, and train/sample/eval/paperpack stages. Runtime commands
carry `task.generative.stage_ledger_path`; paperpack commands carry
`--stage_ledger`. The dry-run output now includes `run_plan.csv`,
`run_status_ledger.csv`, and `benchmark_effect_manifest.json`.

```bash
python -m scripts.validate_docs
```

Status: passed, 121 files scanned.

Additional commands and status:

```bash
python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml \
  --override environment.iterations=1 \
  --override trainer.num_epochs=1 \
  --override trainer.gpus=0 \
  --override trainer.device=cpu
```

Status: run with `/tmp/phm_genbench_v3_smoke/train` output and
`task.generative.stage_ledger_path=/tmp/phm_genbench_v3_smoke/stage_ledger.json`.
The smoke run produced a checkpoint, normalization artifacts, `train_result_0.csv`,
and a train-stage ledger entry.

```bash
PHM_TRUSTED_CHECKPOINT_ROOTS=/tmp/phm_genbench_v3_smoke \
python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml \
  --override environment.output_dir=/tmp/phm_genbench_v3_smoke/sample \
  --override task.generative.stage_ledger_path=/tmp/phm_genbench_v3_smoke/stage_ledger.json \
  --override task.generative.mode=sample \
  --override task.generative.checkpoint_path=<train-checkpoint> \
  --override task.generative.condition_sampling_policy=grid \
  --override 'task.generative.condition_grid.fault_label=[0,1]' \
  --override 'task.generative.condition_grid.domain_id=[0,1]' \
  --override task.generative.condition_grid.samples_per_condition=1
```

Status: run. The sample stage wrote `samples.pt`,
`synthetic_data_manifest.json`, condition counts for four fault/domain pairs,
and a sample-stage ledger entry.

```bash
PHM_TRUSTED_CHECKPOINT_ROOTS=/tmp/phm_genbench_v3_smoke \
python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml \
  --override environment.output_dir=/tmp/phm_genbench_v3_smoke/eval \
  --override task.generative.stage_ledger_path=/tmp/phm_genbench_v3_smoke/stage_ledger.json \
  --override task.generative.mode=eval \
  --override task.generative.generated_path=<path-to-synthetic-samples.pt> \
  --override task.generative.eval_split=train \
  --override task.generative.sampling_rate_hz=12000 \
  --override task.generative.shaft_rpm=1797
```

Status: run. The eval stage wrote `generative_eval_metrics.csv`,
`eval_evidence_manifest.json`, PHM spectral metrics with `ok` status for the
provided sampling/rpm metadata, and explicit classifier utility missing reasons
for insufficient fake label class support.

```bash
python -m scripts.paperpack_generative \
  --run_dir <eval-run-dir> \
  --stage_ledger /tmp/phm_genbench_v3_smoke/stage_ledger.json
```

Status: run. Paperpack wrote tables, figure sources, appendices, included the
sibling sample manifest in `manifest_completeness.csv`, and wrote
`paperpack.paperpack_dir` back to the stage ledger.

```bash
python -m scripts.generative_submission_draft \
  --summary <benchmark_effect_summary.csv> \
  --manifest <benchmark_effect_manifest.json> \
  --output <paper.md> \
  --require-submission-ready
```

Status: not_run. Reason: complete real-run benchmark-effect summary/manifest
do not exist yet.

## 7. Codex-ready `/goal` backlog

```md
/goal

## Goal ID
GOAL-V3-RUN-001

## Objective
Execute the six-dataset train/sample/eval/paperpack queue and aggregate real
benchmark evidence.

## Why
The repository can plan 144 jobs, but paper claims require real metrics,
manifests, stage ledgers, paperpacks, and benchmark-effect aggregation.

## Current facts
- Dry-run writes the expected 144-row plan.
- Runtime commands include `task.generative.stage_ledger_path`.
- Paperpack commands include `--stage_ledger`.

## Scope
Allowed: staged execution, run ledger updates, aggregation, submission draft
generation.

## Out of scope
Do not add new methods or claim submission readiness with exploratory rows.

## Required behavior
Run train, sample, eval, and paperpack stages for configured datasets/methods/seeds.
Aggregate from real runs and preserve metric/manifest source paths.

## Acceptance criteria
`benchmark_effect_manifest.json` reports observed configured dataset count >= 6,
and every claim-table row links metric and manifest source paths.

## Validation commands
python -m scripts.generative_benchmark_effect --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml --from-runs results/paper/phm_generative/six_dataset_submission_v1/runs --output-dir results/paper/phm_generative/six_dataset_submission_v1/effect
```

## 8. Final recommendation

Current readiness: workshop-level internal benchmark scaffold, not a full
conference or high-level journal submission.

Top actions to reach conference-candidate readiness:

1. Execute and aggregate the real six-dataset queue with complete stage ledgers.
2. Produce real classifier utility evidence from benchmark-valid runs.
3. Confirm real-run sampling-rate and fault-frequency metadata coverage.
4. Keep exploratory one-step methods out of benchmark-valid main tables.
5. Regenerate paperpack and submission draft only from benchmark-valid real-run
   evidence.

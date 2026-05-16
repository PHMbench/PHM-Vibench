# UXFD Goal Follow-up Tasks

Date: 2026-05-16

Status: execution backlog. This file is a task plan, not accepted experiment
evidence, not SOTA evidence, and not IEEE Transactions submission readiness.

Current gate baseline:

```text
objective_audit: Achieved=False, Met=86, Not met=14, Blocked=1
experiment_launch_gate: Ready=False
submission_gate: Ready=False, Blocking findings=20
accepted_runs: records=0
sota_aggregates: missing
owner_review: pending_records=6
gpu_preflight: CUDA/RTX 4090 unavailable in this session
```

## Execution Policy

- Do not call `update_goal` until the final objective audit, experiment launch
  gate, artifact gate, SOTA gate, and submission gate all pass.
- Do not create accepted evidence from templates, smoke runs, demo runs, dummy
  data, pending records, or historical autoresearch notes.
- Do not claim SOTA from a single accepted run or from a paper-local matrix.
- Do not commit or discard dirty submodule files without owner-review decisions.
- Every important update must be committed in the relevant repository or
  submodule with a focused commit.

## Critical Path

| ID | Task | Depends on | Done when | Verification |
|---|---|---|---|---|
| `T00` | Freeze current task baseline | none | This task file is committed under `.specify/goals/v2/tasks/` | `git log -1 --oneline` |
| `T01` | Resolve parent goal-status dirty files | `T00` | The 10 dirty `paper/UXFD_paper/goal/status/*.md` files are reviewed and either committed or intentionally left outside the UXFD goal-control checkpoint | `python -m scripts.uxfd_objective_audit --format markdown` no longer reports `dirty_parent_goal_control_paths=10` unless intentionally documented |
| `T02` | Produce real owner-review decisions | `T00` | `submodule_owner_review_decisions.json` exists, keeps `OR-01..OR-06`, uses status `owner_review_decisions`, real reviewer names, ISO dates, and no pending decisions | `python -m scripts.uxfd_owner_review_gate --format markdown` |
| `T03` | Clean or commit dirty paper submodules | `T02` | Dirty files in `Explainable_FD_Toolkit`, `1D-2D_fusion_explainable`, and `MOE_explainable` are resolved according to owner decisions | `python -m scripts.uxfd_submodule_dirty_triage --format markdown` reports clean or no parent-blocking dirty entries |
| `T04` | Restore local GPU visibility | `T00` | GPUs `0,1` are visible as RTX 4090 devices and CUDA is available to torch | `python -m scripts.uxfd_gpu_queue --format markdown --live-preflight --require-preflight` |
| `T05` | Pass experiment launch gate | `T02,T03,T04` | Owner-review, static queue, and live GPU preflight are all ready | `python -m scripts.uxfd_experiment_launch_gate --format markdown` exits 0 without `--allow-not-ready` |
| `T06` | Execute Q0/Q1 queue and create accepted run artifacts | `T05` | Accepted run artifacts exist for Paper07 proposed, baselines, ablations, and TOP representative bindings | `python -m scripts.uxfd_artifact_gate paper/UXFD_paper/results/accepted_runs --require-queue-coverage` |
| `T07` | Execute remaining Q2-Q7 queue artifacts | `T05,T06` | Accepted run artifacts exist for all seven papers, all required baselines, all ablations, and all TOP representatives | Same artifact gate with full queue coverage |
| `T08` | Build SOTA aggregates from accepted run refs | `T07` | Seven `sota_aggregate.yaml` files exist and reference only accepted same-protocol run_meta files | `python -m scripts.uxfd_sota_gate --format markdown` |
| `T09` | Update paper matrices to submission-ready only from accepted evidence | `T07,T08` | Each paper-local `baseline_ablation_matrix.yaml` has blockers resolved and `submission_ready=True` only where supported | `python -m scripts.uxfd_submission_gate --format markdown` |
| `T10` | Final objective completion audit | `T09` | Objective audit shows achieved and no missing prompt-to-artifact requirements | `python -m scripts.uxfd_objective_audit --format markdown` |

## Owner Review Tasks

| ID | Owner item | Required decision | Done when |
|---|---|---|---|
| `OR-01` | `Explainable_FD_Toolkit/EXPERIMENT_DESIGN.md` | `rewrite_then_commit` or `discard_from_submodule` | Current-root, parent-gated plan is committed or the draft is explicitly left out |
| `OR-02` | `Explainable_FD_Toolkit/manuscript/AUTORESEARCH_EVIDENCE.md` | `discard_from_submodule` or `rewrite_then_commit` | Historical accepted/readiness wording is removed, rewritten as non-evidence history, or discarded |
| `OR-03` | `1D-2D_fusion_explainable/EXPERIMENT_DESIGN.md` | `rewrite_then_commit` or `discard_from_submodule` | Deprecated `--config_dir` flow is replaced with maintained `python main.py --config ...` flow or left out |
| `OR-04` | `1D-2D_fusion_explainable/manuscript/AUTORESEARCH_EVIDENCE.md` | `discard_from_submodule` or `rewrite_then_commit` | Historical accepted/readiness wording is removed, rewritten as non-evidence history, or discarded |
| `OR-05` | `MOE_explainable/EXPERIMENT_DESIGN.md` | `rewrite_then_commit` or `discard_from_submodule` | Deprecated config dispatch and nonlocal GPU references are rewritten or left out |
| `OR-06` | `MOE_explainable/manuscript/AUTORESEARCH_EVIDENCE.md` | `discard_from_submodule` or `rewrite_then_commit` | Historical evidence wording, stale root references, and nonlocal GPU references are removed or rewritten |

Required owner-review checks:

```bash
python -m scripts.uxfd_owner_review_gate --format markdown
python -m scripts.uxfd_submodule_dirty_triage --format markdown
```

## GPU and Accepted Artifact Tasks

Every accepted run artifact must include:

- `run_meta.yaml` with paper id, queue id, entry id, seed, GPU identity,
  runtime, precision, source tree status, evidence level
  `accepted_same_protocol`, and clean SHA provenance.
- `metrics.json` or `metrics.csv` with at least one finite numeric metric.
- Preprocessing signature as `sha256:<64 lowercase hex>`.
- No TODO, template, smoke, demo, dummy, pending, NaN, infinite, or dirty
  provenance markers.
- A path under `paper/UXFD_paper/results/accepted_runs`.

Verification:

```bash
python -m scripts.uxfd_gpu_queue --format markdown --live-preflight --require-preflight
python -m scripts.uxfd_experiment_launch_gate --format markdown
python -m scripts.uxfd_artifact_gate paper/UXFD_paper/results/accepted_runs --require-queue-coverage
```

## Per-paper Evidence Tasks

| ID | Paper | Required accepted evidence |
|---|---|---|
| `P07-A` | `TII_operator_attention` | Industrial same-protocol multi-seed proposed model, 7 baselines, 6 ablations, TOP representative, GPU metadata, rejection-recovery traceability |
| `P02-A` | `1D-2D_fusion_explainable` | CWRU/XJTU same-protocol proposed model, 6 baselines, 7 branch ablations, TOP representative, GPU metadata |
| `P01-A` | `Explainable_FD_Toolkit` | Toolkit schema/report evidence, proposed method, 6 baselines, 6 ablations, TOP representative, complete local 2x4090 metadata |
| `P04-A` | `MOE_explainable` | MoE proposed method, 6 baselines, 6 ablations, route entropy, expert activation, expert-count surfaces, TOP representative, GPU metadata |
| `P05-A` | `Paper_fuzzy_XFD` | Fuzzy proposed method, 7 baselines, 6 ablations, rule metrics, safety-case package, TOP representative, GPU metadata |
| `P06-A` | `Neuralsymbolic_theory` | Neuralsymbolic proposed method, 6 baselines, 7 ablations, proposition validation, real-data robustness support for final P2, TOP representative, GPU metadata |
| `P03-A` | `LLM_Explainable_FD_Toolkit` | LLM evidence packages with `run_meta.yaml` and `metrics.json`, 7 baselines, 7 ablations, hallucination checks, context-removal, latency sweep, TOP representative, GPU metadata |

Each paper must pass these local conditions before final submission:

- At least 6 accepted baseline artifacts.
- At least 6 accepted ablation artifacts.
- TOP 2024-2026 representative command/log/artifact mapping accepted.
- Local GPU `0,1` runtime metadata accepted.
- Matrix blockers removed only after accepted evidence exists.
- SOTA wording either supported by `uxfd_sota_gate` or explicitly downgraded
  to bounded contribution wording.

## SOTA Aggregate Tasks

| ID | Task | Done when |
|---|---|---|
| `SOTA-01` | Create accepted-run refs for proposed methods | Every proposed method has matched seed `run_meta.yaml` refs |
| `SOTA-02` | Create accepted-run refs for all baseline comparators | Every required baseline id has matched seed refs |
| `SOTA-03` | Create accepted-run refs for all TOP representative bindings | All seven TOP bindings have accepted same-protocol refs |
| `SOTA-04` | Compute aggregate statistics | Each aggregate has finite mean, std, 95% CI, effect size or paired-test p-value in `[0, 1]` |
| `SOTA-05` | Declare exact or representative scope | Exact SOTA is used only if the proposed method beats every accepted comparator under the metric direction |
| `SOTA-06` | Run gate | `python -m scripts.uxfd_sota_gate --format markdown` exits 0 |

## Manuscript and Submission Tasks

| ID | Task | Done when |
|---|---|---|
| `M-01` | Replace smoke/demo wording | Manuscripts no longer cite smoke, dummy, template, pending, or historical autoresearch artifacts as evidence |
| `M-02` | Update claims from gates | Claims match accepted artifacts and SOTA aggregate scope |
| `M-03` | Verify low-tier source hygiene | Active manuscripts and bib files contain no blocked low-tier sources such as Scientific Reports, MDPI, or IEEE TIM as target support |
| `M-04` | Refresh recent-work README/citations | 2024-2026 TOP journal/conference methods are represented and linked to accepted comparison artifacts |
| `M-05` | Final submission gate | `python -m scripts.uxfd_submission_gate --format markdown` exits 0 |

## Final Completion Checklist

The UXFD goal can be considered complete only when all commands below pass
without override flags:

```bash
python -m scripts.uxfd_owner_review_gate --format markdown
python -m scripts.uxfd_submodule_dirty_triage --format markdown
python -m scripts.uxfd_gpu_queue --format markdown --live-preflight --require-preflight
python -m scripts.uxfd_experiment_launch_gate --format markdown
python -m scripts.uxfd_artifact_gate paper/UXFD_paper/results/accepted_runs --require-queue-coverage
python -m scripts.uxfd_sota_gate --format markdown
python -m scripts.uxfd_submission_gate --format markdown
python -m scripts.uxfd_objective_audit --format markdown
```

Expected final state:

- `objective_audit`: `Achieved=True`.
- `experiment_launch_gate`: `Ready=True`.
- `submission_gate`: `Ready=True`.
- `accepted_runs`: full queue coverage for 104 queue rows.
- `sota_aggregates`: 7 accepted paper records.
- All seven paper matrices: `submission_ready=True`.
- No dirty UXFD goal-control paths.
- Paper submodule working trees clean or intentionally documented outside the
  parent handoff.

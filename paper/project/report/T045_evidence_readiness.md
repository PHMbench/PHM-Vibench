# T045 Evidence Readiness: Proposition, Baseline, Ablation, And SOTA Gates

Date: 2026-05-11

Scope: bind the Neural-Symbolic Theory paper's formal propositions to runnable
local validation artifacts, and define the minimum evidence gates before any
baseline, ablation, TOP-source, or SOTA claim is accepted.

## Assumptions

- All paths are relative to `paper/UXFD_paper/Neuralsymbolic_theory/` unless
  explicitly marked as parent-repo commands.
- Existing proposition demos are validation hooks, not final paper evidence.
- Final accepted evidence must use local RTX 4090 GPUs `0,1` only, with one GPU
  per experiment by default.
- SOTA wording is blocked until the same CWRU/XJTU protocol beats every declared
  baseline on the stated metric axis.

## Canonical Entry Points

| Purpose | Path or command | Current status |
|---|---|---|
| Manuscript entrypoint | `manuscript/final_tex/main.tex` | evidence-bound IEEEtran checkpoint; no placeholder title/body or missing example figure reference; two `pdflatex` passes succeeded to `/tmp/uxfd_paper06_tex/main.pdf` |
| VIBENCH contract | `VIBENCH.md` | present; T045 gates added here |
| Minimal PHM-VIBench config | `configs/vibench/min.yaml` | present; CPU smoke config for `NSN` with logic decision slot |
| Submission-prep matrix | `submission_prep/baseline_ablation_matrix.yaml` | present; six baseline dummy smokes, proposition hooks, scripted/source-backed mapping hooks, and blockers recorded |
| Submission-prep readiness note | `submission_prep/ieee_trans_readiness.md` | present; explains allowed wording and remaining IEEE blockers |
| Proposition demo hook | `python simple_validation_demo.py` | runnable hook; synthetic/preliminary only; current P2 outcome is a blocker if unsupported |
| Proposition 2 tracked artifact hook | `python experiments/proposition2_simple.py` | runnable hook; writes tracked `experiments/results/proposition2_12_14/` artifacts |
| Cross-method mapping hook | `python code/validate_mapping.py` plus `python scripts/build_source_backed_mapping.py` | runnable scripted hook plus source-introspection report; not accepted train/eval impact evidence |

## Proposition-To-Artifact Binding

| Proposition | Formal claim to validate | Runnable command | Artifact path | Acceptance status | Missing final evidence |
|---|---|---|---|---|---|
| P1 symbolic constraints improve reliability/trustworthiness | constrained variant improves reliability, consistency, or trustworthiness over unconstrained neural variant without hiding accuracy loss | `python simple_validation_demo.py` | `results/theory_validation/validation_summary.json`, `results/theory_validation/proposition_1_demo.png` | runnable demo only | CWRU+XJTU configs, 5 seeds, unconstrained vs constrained logs, reliability/consistency metrics, CI table |
| P2 physical homomorphism improves robustness | physics-consistent variant has lower degradation slope under noise or shift than free model | `python experiments/proposition2_simple.py` | `experiments/results/proposition2_12_14/simple_results.json`, `experiments/results/proposition2_12_14/simple_validation.png`; `submission_prep/p2_evidence_contract.md` | preliminary tracked synthetic artifact plus explicit boundary/failure contract; unsupported demo outcomes must be reported as boundary cases | CWRU+XJTU robustness runs, noise/shift protocol, slope estimates, boundary/failure cases |
| P3 interpretability-performance Pareto boundary exists | declared model family contains nondominated points under accuracy/F1 and interpretability metrics | `python simple_validation_demo.py` | `results/theory_validation/proposition_3_demo.png`, `results/theory_validation/validation_summary.json` | runnable demo only | same-protocol 6+ baseline table with interpretability metrics and Pareto-front calculation |
| Cross-method mapping | 1D-2D, MoE, Fuzzy, Toolkit, LLM, and Operator Attention mechanisms map to the four-layer neural-symbolic framework | `python code/validate_mapping.py` and `python scripts/build_source_backed_mapping.py` | `report/mapping_validation_report.json`, `manuscript/figures/mapping_validation.png`, `report/source_backed_mapping_report.json`, `report/source_backed_mapping_report.md` | runnable scripted hook plus source-backed sibling-submodule report only | explicit failed mappings and same-protocol mapping-impact artifacts |

## Six-Plus Baseline Evidence Gate

Each accepted baseline row must include: config, exact command, seed list,
dataset IDs, split protocol, preprocessing, metrics, log path, artifact path,
GPU metadata, runtime, and whether the method is exact, representative, or
blocked.

Minimum datasets: `RM_001_CWRU` and `RM_002_XJTU`.

Minimum seeds: `0,1,2,3,4`.

Metrics: accuracy, macro-F1, reliability, consistency, faithfulness, robustness
slope, and run status.

| Gate ID | Method | Role | TOP-source status | Required command pattern | Acceptance rule |
|---|---|---|---|---|---|
| B0 | proposed constrained `NSN`/`TSPN_UXFD` | main method | local method | parent repo: `CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/Neuralsymbolic_theory/configs/vibench/min.yaml ...` | full neural-symbolic constraint stack enabled and logged |
| B1 | `ISFM.M_01_ISFM` | strong diagnostic baseline | local representative | parent repo config under `configs/vibench/` required | same split/seeds/metrics as B0 |
| B2 | `X_model.NSN` or `TSPN_UXFD` without neural-symbolic constraints | paper-specific unconstrained baseline | local representative | disable logic/constraint slot in a dedicated config | isolates contribution of symbolic constraints |
| B3 | `CNN.ResNet1D` or `X_model.Resnet` | strong CNN diagnostic baseline | local representative | dedicated config required | same preprocessing and input window as B0 |
| B4 | `Transformer.PatchTST` or `Transformer.ConvTransformer` | competitive recent architecture baseline | representative for foundation/time-series family | dedicated config required | same protocol; no extra pretraining unless declared for all comparable methods |
| B5 | `X_model.Sincnet` | signal-processing baseline | local representative | dedicated config required | same protocol and metrics |
| B6 | `X_model.TFN` | frequency/time-frequency baseline | representative for frequency-aware TOP methods | dedicated config required | same protocol and metrics |
| B7 | TimeX++ representative | TOP recent explanation baseline | `RWTOP2024-TIMEXPP`, representative-runnable | toolkit/explanation proxy command required | may count only as representative, not exact reproduction |
| B8 | SARAD representative | TOP recent diagnosis/association baseline | `RWTOP2024-SARAD`, representative-runnable | association/channel proxy command required | may count only as representative, not exact reproduction |

Blocked as performance baselines until local concepts/protocols exist:
`RWTOP2025-CFCBM` and `RWTOP2025-IFCBM`. They may support related-work
positioning only.

Current command-bound checkpoint:

- `submission_prep/baseline_ablation_matrix.yaml` records P00 plus six
  PHM-Vibench baseline dummy smokes: no-symbolic NSN/TSPN_UXFD, ResNet,
  SincNet, TFN, WKN, and ConvTransformer.
- The smokes completed in `LQ_signal` on dummy data with CPU fallback because
  the current environment reported GPU/NVML unavailable.
- This validates wiring only and does not satisfy the accepted CWRU/XJTU
  multi-seed, GPU-metadata, TOP-representative, or SOTA gates.

## Ablation Evidence Gate

Every ablation must reuse the same datasets, splits, seeds, preprocessing, and
metrics as the baseline suite.

| Gate ID | Ablation | Purpose | Required evidence |
|---|---|---|---|
| A1 | remove symbolic constraints | validates P1 and symbolic contribution | B0 vs no-symbolic config, 5 seeds, reliability/consistency delta |
| A2 | remove physical-consistency constraint | validates P2 | robustness slope and failure cases under the same noise/shift protocol |
| A3 | remove cross-method mapping module | validates framework mapping claim | mapping report difference and downstream metric impact if used in training |
| A4 | constraint-strength sweep | boundary conditions | lambda values including `0`, low, medium, high, plus OOM/failure record |
| A5 | neural-only vs symbolic-only vs neural-symbolic | contribution separation | three matched variants with same data protocol |
| A6 | independent proposition validation | prevents one aggregate demo from proving all claims | separate artifact IDs for P1, P2, P3 |

## 2x4090 Compute Gate

Accepted GPU commands must declare one of:

```bash
CUDA_VISIBLE_DEVICES=0 <command>
CUDA_VISIBLE_DEVICES=1 <command>
CUDA_VISIBLE_DEVICES=0,1 <command>
```

Two-GPU commands are allowed only with a written reason in the run metadata.
Default policy is one GPU per experiment and at most two concurrent jobs.

Each accepted run must record:

- device IDs and GPU model
- GPU count
- seed
- batch size
- precision
- runtime
- config path and overrides
- OOM/resource failure reason, if any

Exact reproduction that needs cloud GPUs, A100/H100 hardware, multi-node
execution, or more than two GPUs is `resource-blocked`.

## SOTA Gate

SOTA wording is not allowed yet.

A SOTA claim may be added only after all conditions are met:

1. B0 through at least six declared baselines have accepted CWRU+XJTU artifacts.
2. At least two TOP-source recent-work representatives are included or explicitly
   marked `resource-blocked`/`literature-only`.
3. All ablations A1-A6 have accepted artifacts or precise blocker records.
4. The proposed method beats every accepted same-protocol baseline on the exact
   claimed axis.
5. If the win is on trustworthiness or constrained diagnosis rather than raw
   accuracy, the manuscript must state that axis and avoid raw-accuracy SOTA
   language.

## Current Blockers

- Local check on 2026-05-11:
  `python simple_validation_demo.py` exits successfully after the JSON fix, but
  records `proposition_2_verified: false`
  (`drop_rate_physics=0.1455`, `drop_rate_standard=0.0805`) in
  `results/theory_validation/validation_summary.json`. This is a current
  boundary/failure case, not support for P2.
- Parent PHM-Vibench smoke checks on 2026-05-11 completed in `LQ_signal` for
  the proposed model and six baselines on dummy data, but the environment
  reported `GPU available: False` and `Can't initialize NVML`; these are not
  accepted GPU-backed evidence.
- `python experiments/proposition2_simple.py` writes a synthetic P2 robustness
  artifact with lower physics-informed sensitivity, but this does not override
  the failed aggregate P2 result above and must be treated as scope-limited
  synthetic evidence until real-data robustness runs exist. The decision rule is
  recorded in `submission_prep/p2_evidence_contract.md`.
- `python code/validate_mapping.py` writes `report/mapping_validation_report.json`
  and `manuscript/figures/mapping_validation.png`; `python scripts/build_source_backed_mapping.py`
  pairs it with `report/source_backed_mapping_report.json` and
  `report/source_backed_mapping_report.md` by checking sibling submodule
  `VIBENCH.md`, matrix, config, and source files. This is not accepted
  mapping-impact evidence.
- Missing final per-baseline configs under `configs/vibench/` for B1-B8.
- Missing real-data CWRU/XJTU multi-seed logs and accepted result tables.
- Missing GPU metadata artifacts for all accepted runs.
- Missing scripts referenced by `plan/EXPERIMENT_PLAN_*.md`:
  `experiments/validate_proposition_1.py`,
  `experiments/validate_proposition_2.py`,
  `experiments/validate_proposition_3.py`, and
  `experiments/validate_across_datasets.py`.
- Current mapping evidence remains non-performance evidence; manuscript claims
  still need explicit negative mappings and accepted train/eval impact artifacts.
- `manuscript/final_tex/main.tex` is now a conservative IEEEtran checkpoint
  rather than final submission text. It still needs accepted result tables and
  final wording after real artifacts exist.

## Ready-To-Run Checks

Run from this submodule root:

```bash
python simple_validation_demo.py
python experiments/proposition2_simple.py
python code/validate_mapping.py
```

Run from the parent repo root:

```bash
python main.py --config paper/UXFD_paper/Neuralsymbolic_theory/configs/vibench/min.yaml --override trainer.num_epochs=1
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/Neuralsymbolic_theory/configs/vibench/min.yaml --override trainer.num_epochs=1 --override trainer.device=cuda --override trainer.gpus=1 --override model.device=cuda
```

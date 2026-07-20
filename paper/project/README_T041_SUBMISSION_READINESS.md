# T041 Submission Readiness Evidence README

Snapshot date: 2026-05-11

This file is the local strict-reviewer binding for the current IEEE Transactions
submission-readiness pass. It records what can be claimed from existing artifacts
and what remains blocked under the 2026-05-11 parent goals.

## Status

- Current verdict: blocked, not submission-ready.
- Main reason: the new command-bound matrix contains dummy-data wiring evidence
  only; it does not contain an accepted CWRU/XJTU six-baseline same-protocol
  matrix or the required real fusion/alignment ablation matrix.
- SOTA wording: blocked until the proposed method beats every declared baseline
  under the same CWRU/XJTU split, seed protocol, preprocessing, and metrics.
- TOP recent-work gate: declared by the parent goal, but not yet bound to local
  exact-run or representative-run artifacts for this paper.

The older local binding at
`results/autoresearch/20260319_193824/manuscript_binding/` says
`submission_ready=true` for the previous local contract. The newer parent gates
are stricter and supersede that local ready flag for T041.

## Canonical Package

- Canonical manuscript entrypoint:
  `paper_draft/NMI_Paper1_Fusion1D2D.tex`
- Non-canonical placeholder:
  `manuscript/final_tex/main.tex`
- Target venue:
  IEEE TII by parent goal, with IEEE TIE or Information Fusion as fallback.
- Current template status:
  the canonical TeX now uses `\documentclass[journal]{IEEEtran}` and
  `IEEEtranN` bibliography style.
- Compile command used for the IEEE Transactions smoke gate:

```bash
cd /home/user/LQ/B_Signal/vibench_fix/PHM-Vibench_fix/paper/UXFD_paper/1D-2D_fusion_explainable
mkdir -p /tmp/uxfd_1d2d_t041_tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=/tmp/uxfd_1d2d_t041_tex paper_draft/NMI_Paper1_Fusion1D2D.tex
```

Known compile blockers to check before accepting the manuscript:

- The smoke compile passes by using placeholder boxes when `architecture.pdf`
  and `gradcam_visualization.pdf` are absent.
- The placeholders must be replaced with accepted figure artifacts before final
  submission.

## Canonical Reproduction Entrypoint

The maintained parent entrypoint is the VIBENCH min config:

```bash
cd /home/user/LQ/B_Signal/vibench_fix/PHM-Vibench_fix
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/1D-2D_fusion_explainable/configs/vibench/min.yaml --override trainer.num_epochs=1 --override data.num_workers=0 --override trainer.device=cuda --override model.device=cuda --override trainer.gpus=1 --override environment.output_dir=paper/UXFD_paper/1D-2D_fusion_explainable/results/t041/vibench_min
```

Current caveat: the local `VIBENCH.md` and several autoresearch artifacts still
record the older `PHM-Vibench copy 2` exec root. T041 should use the current
repo root above unless a human explicitly chooses the old exec root.

Current command-bound checkpoint:

- `submission_prep/baseline_ablation_matrix.yaml` records the proposed
  PHM-Vibench proxy, a no-2D proxy, ResNet, SincNet, TFN, WKN, and
  ConvTransformer dummy-data smokes.
- `scripts/run_minimal_demo.py --use_dummy --num_classes=10` runs the
  paper-local Fusion1D2D demo in `LQ_signal`; `--num_classes=4` fails because
  dummy labels exceed the class range.
- `scripts/run_minimal_demo.py` also has a current PHM-Vibench HDF5 window
  loader smoke path. A tiny THU_018 CPU run loaded 8 windows and completed, but
  it is not accepted CWRU/XJTU evidence.
- STFT/fusion sensitivity smokes run on dummy data; the FFT-only signal-layer
  shape gate now forwards after the TSPN residual path skips incompatible
  length-changing operator outputs.
- These entries validate executable surfaces only and do not satisfy the
  accepted CWRU/XJTU, TOP, GPU, ablation, or SOTA gates.

## Claims Bound To Existing Artifacts

| Claim | Allowed wording | Existing artifact |
|---|---|---|
| Canonical manuscript selected | `paper_draft/NMI_Paper1_Fusion1D2D.tex` is the canonical draft for this cycle. | `results/autoresearch/20260319_193824/manuscript_binding/claim_evidence_map.md` |
| VIBENCH smoke exists | Dummy-data VIBENCH smoke has schema evidence only; it is not final accuracy evidence. | `outputs/RM_DUMMY_VIBENCH/NSN/seed_0/20260319_164150/metrics.json` |
| Synthetic local demo exists | Synthetic demo reports `accuracy=0.93`, `f1_macro=0.8695652173913043`; not a real-data claim. | `outputs/RM_SYNTHETIC_DUMMY/Fusion1D2D/seed_42/20260319_155222/metrics.json` |
| CWRU/XJTU validation slice exists | Current accepted evidence is limited to CWRU and XJTU with mean test accuracy `0.6567793786525726`. | `results/autoresearch/20260319_182200/cross_dataset_generalization/cross_dataset_binding_summary.json` |
| Three-seed stability exists | Three-seed slice reports mean accuracy `0.41413288315137226`, std `0.026909093674508062`, 95% CI `0.03045050605418713`, CV `6.497695490814806`. | `results/autoresearch/manual_20260319_1010/stability_three_seed/stability_metrics_summary.json` |
| Quantitative explainability probe exists | Synthetic attribution probe reports faithfulness `0.0002103795607884725`, stability `0.9987647901238335`, efficiency `63.47314229545494` ms/sample. | `results/autoresearch/20260319_193128/explainability_quant/explainability_metrics_summary.json` |
| Comparison references exist | MoE, TSPN, and OperatorAttention comparison packs exist, but they are not a six-baseline same-protocol SOTA table. | `results/comparison_moe_20260319_175647/`, `results/comparison_tspn_20260319_175647/`, `results/comparison_op_att_20260319_175647/` |
| Strengthened innovation gate is not met | CWRU `>=0.98` attempt failed; XJTU and THU_006 `>=0.98` passes are not accepted. | `outputs/RM_CWRU_FULL_98/Fusion1D2D/seed_42/20260320_143500/metrics.json`, `autoresearch/REVIEW_STATE.json` |

## Six-Baseline Matrix

No accepted six-baseline same-protocol matrix exists in this submodule. The
following matrix is the minimum T041 target and must be run with the same
CWRU/XJTU data split, seeds, preprocessing, metric definitions, and report
format as the proposed method.

| Slot | Required baseline | TOP/relevance mapping | Current local evidence | Status |
|---|---|---|---|---|
| B1 | `ISFM.M_01_ISFM` | strong diagnostic baseline | no CWRU/XJTU same-protocol artifact found | missing |
| B2 | `X_model.NSN` or `X_model.TSPN_UXFD` without 2D fusion | interpretability/paper-specific baseline | dummy smoke only, no accepted real-data baseline | missing |
| B3 | `CNN.ResNet1D` | strong 1D diagnostic baseline | no accepted artifact found | missing |
| B4 | `X_model.Sincnet` | strong signal baseline | no accepted artifact found | missing |
| B5 | `X_model.TFN` | frequency/channel representative for RWTOP2025-CATCH | no accepted artifact found | missing |
| B6 | `X_model.WKN` | frequency/kernel representative for RWTOP2025-CATCH | no accepted artifact found | missing |
| B7 | `Transformer.PatchTST` or `Transformer.ConvTransformer` | representative for RWTOP2024-MOMENT | no accepted artifact found | missing |
| B8 | `CNN.TCN` or multiscale CNN | representative for RWTOP2024-TIMEMIXER | no accepted artifact found | missing |

Required artifact shape for every baseline:

- `outputs/RM_BASELINE_CWRU_XJTU/<method>/seed_<seed>/<run_id>/run_meta.yaml`
- `outputs/RM_BASELINE_CWRU_XJTU/<method>/seed_<seed>/<run_id>/metrics.json`
- `results/t041/baselines/<method>/seed_<seed>/config_snapshot.yaml`
- `results/t041/baselines/<method>/seed_<seed>/logs/*.log`

Required command pattern from the current repo root:

```bash
cd /home/user/LQ/B_Signal/vibench_fix/PHM-Vibench_fix
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/1D-2D_fusion_explainable/configs/vibench/baselines/<method>.yaml --override environment.seed=<seed> --override trainer.num_epochs=50 --override data.num_workers=0 --override trainer.device=cuda --override model.device=cuda --override trainer.gpus=1 --override environment.output_dir=paper/UXFD_paper/1D-2D_fusion_explainable/results/t041/baselines/<method>/seed_<seed>
```

The method-specific baseline config files under
`configs/vibench/baselines/` are also missing and must be created before these
commands can become accepted evidence.

## Ablation Matrix

No accepted fusion/alignment ablation matrix exists under the current T041 gate.
The existing `configs/ablation/` files cover only `1D_only`, `2D_only`, and
`no_statistical`; they remain non-accepted CWRU/XJTU artifacts. The current
`scripts/run_ablation_studies.sh` delegates to `run_ablation_study.py`, resolves
the current repo root, and only permits GPU `0` or `1`.

Required ablations:

| Ablation | Existing evidence | Status |
|---|---|---|
| full proposed model | CWRU/XJTU slice exists, but not as part of same ablation matrix | partial |
| 1D-only branch | config exists, no accepted CWRU/XJTU artifact | missing |
| 2D-only branch | config exists, no accepted CWRU/XJTU artifact | missing |
| fusion without physical alignment | no accepted config/artifact found | missing |
| fusion without semantic/geometric alignment | no accepted config/artifact found | missing |
| late fusion vs progressive fusion | no accepted config/artifact found | missing |
| attribution/explainability module removed | no accepted config/artifact found | missing |

Required command pattern:

```bash
cd /home/user/LQ/B_Signal/vibench_fix/PHM-Vibench_fix
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/1D-2D_fusion_explainable/configs/vibench/ablations/<ablation>.yaml --override environment.seed=<seed> --override trainer.num_epochs=50 --override data.num_workers=0 --override trainer.device=cuda --override model.device=cuda --override trainer.gpus=1 --override environment.output_dir=paper/UXFD_paper/1D-2D_fusion_explainable/results/t041/ablations/<ablation>/seed_<seed>
```

The method-specific ablation config files under `configs/vibench/ablations/`
are missing.

## TOP Recent-Work Binding

Parent goal quota for this paper:

- RWTOP2024-TIMEMIXER: representative-runnable multiscale temporal baseline.
- RWTOP2024-MOMENT: representative-runnable foundation-style representation baseline.
- RWTOP2025-CATCH: representative-runnable channel/frequency baseline.
- RWTOP2025-DADA: representative-runnable bottleneck/anomaly baseline.

Current local status:

- none of the TOP recent-work representatives above is bound to an accepted
  local CWRU/XJTU command, log, and artifact.
- no low-quality venue source is accepted here as a core baseline or SOTA
  comparator.

## Immediate Blockers

1. Replace the canonical TeX placeholder boxes with accepted architecture and
   Grad-CAM figure artifacts, then recompile from the canonical entrypoint.
2. Add and run `configs/vibench/baselines/*.yaml` for at least six baselines
   under CWRU/XJTU, three seeds, local `CUDA_VISIBLE_DEVICES=0` or `1`.
3. Add and run `configs/vibench/ablations/*.yaml` for the fusion/alignment
   ablations above under the same protocol.
4. Keep new accepted reproduction commands bound to the current repo root; old
   `PHM-Vibench copy 2` artifacts remain historical evidence only.
5. Re-run the manuscript binding only after the six-baseline and ablation
   artifacts exist; until then, keep broad SOTA and high-accuracy claims out of
   the manuscript.

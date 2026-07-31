# VIBENCH Mapping And One-Command Reproduction (1D-2D_fusion_explainable)

## Execution Roots

- `paper_root`: `paper/UXFD_paper/1D-2D_fusion_explainable`
- `exec_root`: `.` (nested ViBench repository root)
- Historical `Paper/...` references in older notes map to the real lowercase path under `paper/UXFD_paper/...`.

## Maintained Parent Entry

Config stored in the paper submodule:

- `paper/UXFD_paper/1D-2D_fusion_explainable/configs/vibench/min.yaml`

Maintained smoke command from `exec_root`:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/1D-2D_fusion_explainable/configs/vibench/min.yaml --override trainer.num_epochs=1
```

Submission-readiness and command-bound comparison status are tracked in:

- `README_T041_SUBMISSION_READINESS.md`
- `submission_prep/baseline_ablation_matrix.yaml`
- `submission_prep/ieee_trans_readiness.md`

Current status: blocked for submission. The maintained parent entrypoint and
six baseline commands now have dummy-data smoke evidence, and the paper-local
Fusion1D2D demo can run with dummy data when `--num_classes=10` is used. The
repository also has a non-accepted fusion-ablation smoke runner for FFT-only
and legacy ablation surfaces. These checks are wiring evidence only. The real
CWRU/XJTU baseline matrix, fusion/alignment ablations, TOP representatives,
IEEE TeX compile, and 2x4090 GPU metadata remain missing.

## Local Demo Entry

Run from `paper_root`:

```bash
python scripts/run_minimal_demo.py --use_dummy --output_root "paper/UXFD_paper/1D-2D_fusion_explainable/results/autoresearch/<run_id>/demo"
```

Expected local demo artifacts:

- `demo_results.json`
- `best_model.pth`
- `figures/training_history.png`

## VIBENCH Artifact Expectations

A successful maintained smoke run should leave, under the configured `environment.output_dir`:

- `config_snapshot.yaml`
- `artifacts/manifest.json`
- `artifacts/data_metadata_snapshot.json`
- `artifacts/predictions.npz` when predictions are enabled

These are then wrapped into a Paper2-schema evidence pack under `paper_root/outputs/...` by the nonstop runner.

## Paper 02 command-bound comparison surface

`submission_prep/baseline_ablation_matrix.yaml` records:

- proposed PHM-Vibench NSN proxy with the `signal_processing_2d` config block;
- no-2D proxy, ResNet, SincNet, TFN, WKN, and ConvTransformer baseline smokes;
- paper-local Fusion1D2D dummy demo smoke;
- STFT/fusion sensitivity smokes;
- non-accepted FFT-only and legacy-ablation smoke artifacts from
  `scripts/run_fusion_ablation_smoke.py`, while the true FFT dimensionality
  failure and stale legacy THU/GPU2 paths remain reviewer blockers.

# VIBENCH Mapping And One-Command Reproduction (Explainable_FD_Toolkit)

## Execution Roots

- `paper_root`: `paper/UXFD_paper/Explainable_FD_Toolkit`
- `exec_root`: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench_fix`

## Maintained Parent Entry

Config stored in the paper submodule:

- `paper/UXFD_paper/Explainable_FD_Toolkit/configs/vibench/min.yaml`

Maintained smoke command from `exec_root`:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/Explainable_FD_Toolkit/configs/vibench/min.yaml --override trainer.num_epochs=1
```

## Standalone Benchmark Entry

Run from `paper_root`:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_benchmark_standalone.py --output_dir "results/autoresearch/<run_id>/benchmark"
```

Expected benchmark artifacts:

- `explainability_benchmark_results.json`
- `explainability_benchmark_table.csv`
- `benchmark_analysis_report.md`
- `overall_scores_comparison.png`

## Evidence Chain Rule

Standalone and parent-repo outputs are only promoted into the paper evidence chain after the nonstop runner creates `run_meta.yaml`, `metrics.json`, and a passing `validate_schema.py` report under `paper_root/outputs/...`.

Toolkit ablation smoke artifacts can be generated from `paper_root`:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_toolkit_ablations.py --condition all --output results/toolkit_ablation_smoke --seed 0
```

This command validates artifact shape for schema, metric-family, manifest,
snapshot, and post-hoc-only ablation surfaces. Its outputs are marked
`accepted_evidence=false` and are not accepted reviewer evidence.

## T040 Evidence Pointers

- Manuscript figure source: `results/autoresearch/20260319_090111/benchmark/overall_scores_comparison.png`
- Benchmark table source: `results/autoresearch/20260319_090111/benchmark/explainability_benchmark_table.csv`
- Schema pack: `outputs/RM_MULTI_CWRU_XJTU/ToolkitBenchmark/seed_0/20260319_090111/`
- Remaining IEEE gate blockers: `manuscript/T040_EVIDENCE_README.md`
- Command-bound baseline/ablation checkpoint: `submission_prep/baseline_ablation_matrix.yaml`
- Manuscript checkpoint: `manuscript/final_tex/main.tex` compiles as an
  evidence-bound IEEEtran checkpoint, not final submission text.

Current status: six PHM-Vibench model baselines are dummy-smoke validated in
`LQ_signal`, and Toolkit ablation smoke artifacts can be generated locally.
Accepted CWRU/XJTU evidence, accepted Toolkit-specific ablations, TOP
representatives, complete 2x4090 metadata, and SOTA wording remain blocked.

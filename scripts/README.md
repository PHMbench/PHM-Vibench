# Scripts

Utilities used by PHM‑Vibench tooling and maintenance.

Core commands (maintained):
- `python -m scripts.validate_configs`
- `python -m scripts.config_inspect --config <yaml> --override key=value`
- `python -m scripts.gen_config_atlas`
- `python -m scripts.validate_docs`
- `python -m scripts.paperpack_generative --run_dir <run_dir>`
- `python -m scripts.generative_sweep --config configs/demo/10_generative/dummy_generative_cfm.yaml`
- `python -m scripts.generative_sweep --configs configs/demo/10_generative/dummy_generative_cfm.yaml,configs/demo/10_generative/dummy_generative_ddpm.yaml --seeds 0,1`
- `python -m scripts.generative_benchmark_effect --matrix configs/paper/phm_generative/benchmark_effect_matrix.yaml --dry-run`
- `python -m scripts.generative_benchmark_effect --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml --dry-run --output-dir results/paper/phm_generative/six_dataset_submission_v1/dry_run`
- `eval "$(conda shell.bash hook)" && conda activate LQ_signal && python -m scripts.generative_benchmark_effect --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml --preflight-gpu --dry-run --output-dir results/paper/phm_generative/six_dataset_submission_v1/gpu_preflight`
- `python -m scripts.generative_submission_draft --summary <summary.csv> --manifest <manifest.json> --output specs/002-phm-genbench-frontier/paper/PAPER_DRAFT.md --require-submission-ready`

`scripts.validate_docs` also blocks the deprecated central PHM generative docs
directories `docs/phm_generative/` and `docs/generative/`. Module-specific PHM
generative guidance belongs in the owning module/config README; process notes
belong under the active `specs/<feature>/` directory.

It also validates that `.specify/goals/v2/GOAL-GEN*.md` files expose a
parseable filename-matching `## Goal ID` and the core goal sections used by
the handoff/review queue. When the v2 goal queue is present, it verifies the
required PHM generative module/config README files are present.

Generative paperpack outputs:
- `tables/table_quality_mean_std.csv`
- `tables/table_utility_mean_std.csv`
- `tables/table_efficiency_mean_std.csv`
- `tables/table_leakage.csv`
- `tables/table_ablation.csv`
- `figure_sources/spectra_overlay.csv`
- `figure_sources/temporal_overlay.csv`
- `figure_sources/metric_barplot.csv`
- `figure_sources/dataset_method_heatmap.csv`
- `figure_sources/missing_metric_audit.csv`
- `figure_sources/manifest_index.json`
- `appendix/run_index.csv`
- `appendix/manifest_completeness.csv`
- `appendix/missing_metrics.csv`
- `appendix/missing_metrics.md`

Submission drafts must be generated from completed benchmark evidence. The
draft is `SUBMISSION_READY` only when the evidence covers at least six
benchmark-valid datasets with computable quality and utility rows. Otherwise it
must state `NOT_SUBMISSION_READY` and list evidence gaps. With
`--require-submission-ready`, missing or incomplete evidence returns a non-zero
exit code after writing the blocked draft. The generator writes sidecar
`evidence_gaps.md` and `submission_readiness.md` files next to the draft.

The six-dataset paper matrix must pass GPU preflight on physical GPU 6 and GPU
7 before real training. Run M2 GPU preflight, execution, aggregation, and paper
draft commands from the project `LQ_signal` environment. Check
`CUDA_VISIBLE_DEVICES=6` and `CUDA_VISIBLE_DEVICES=7` individually before the
combined two-GPU preflight. Do not reroute the M2 submission run to CPU.
`--preflight-gpu` writes `gpu_preflight_report.json` under `--output-dir` for
both pass and fail cases, so blocked GPU state is reviewable without relying on
stdout alone. When GPU preflight fails in planning or execution mode, it also
writes `blocked_run_status_ledger.csv` with one row per dataset/method/seed
group and `BLOCKED_GPU_PREFLIGHT` status.
For long M2 execution resumes, use `--skip-existing` to skip completed stage
artifacts and `--max-runs N` to run bounded chunks without retraining completed
jobs. For train rows, `--skip-existing` requires `train_result_0.csv`; a
checkpoint alone is treated as partial evidence from an interrupted run.
`generative_benchmark_effect` treats `--dry-run`, `--execute`, and
`--from-runs` as mutually exclusive primary modes. `--preflight-gpu` may be
combined with `--dry-run` or `--execute`. `--from-runs` requires at least one
run directory with at least one `generative_eval_metrics.csv` record and will
not generate empty aggregation artifacts.
Its benchmark-effect manifest records configured, observed, observed
configured, missing, and unexpected dataset coverage, plus `min_datasets_met`
and `input_gaps`, so a six-dataset claim can be audited against actual metric
records rather than the matrix definition alone. `min_datasets_met` is computed
from configured datasets with evidence; matrix-external datasets do not satisfy
the paper minimum.

UXFD merge utilities:
- `python -m scripts.collect_uxfd_runs --input save/ --out_dir docs/reports/` (collect `artifacts/manifest.json` into CSV)

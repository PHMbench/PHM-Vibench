# Explainable FD Toolkit Autoresearch Program

> paper_root: `paper/UXFD_paper/Explainable_FD_Toolkit`
> exec_root: repository root (`.`)
> Mode: nonstop, baseline-first, no human confirmation between stages

## Contract

- The standalone benchmark inside `paper_root` is the fastest evidence loop.
- The parent-repo `main.py --config ...` smoke run verifies that the toolkit still integrates with the maintained VIBENCH entrypoint.
- Accepted runs only count after Paper2 schema validation passes.

## Stage Order

### Stage 0: Standalone Benchmark Suite

Run from `paper_root`:

```bash
python scripts/run_benchmark_standalone.py --output_dir "paper/UXFD_paper/Explainable_FD_Toolkit/results/autoresearch/<run_id>/benchmark"
```

Acceptance:
- `explainability_benchmark_results.json` exists
- `total_evaluations >= 10`
- `best_overall_score >= 0.60`

### Stage 1: Parent-Repo VIBENCH Smoke

Run from `exec_root`:

```bash
python main.py --config paper/UXFD_paper/Explainable_FD_Toolkit/configs/vibench/min.yaml       --override trainer.num_epochs=1       --override trainer.device=cpu       --override model.device=cpu       --override environment.output_dir=results/uxfd/autoresearch/Explainable_FD_Toolkit/<run_id>
```

Acceptance:
- `artifacts/manifest.json` exists
- schema pack is materialized under `paper_root/outputs/...`
- `validate_schema.py --run_dir <RUN_DIR>` passes

## Loop Policy

- After the parent smoke stage succeeds once, the nonstop runner loops the standalone benchmark stage for additional evidence refreshes.
- Repeated failures or no-progress loops eventually mark this paper `blocked`, but the global daemon does not stop.

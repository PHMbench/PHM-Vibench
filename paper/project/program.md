# MoE Explainable FD Autoresearch Program

> paper_root: `paper/UXFD_paper/MOE_explainable`
> exec_root: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench_fix`
> Mode: nonstop, baseline-first, no human confirmation between stages

## Contract

- `paper_root` owns the MoE demo, results, manuscript notes, and the paper-local contract.
- `exec_root` owns the maintained `main.py --config ...` smoke entrypoint.
- The nonstop loop keeps running without asking for confirmation; a blocked MoE stage does not stop the other papers.

## Stage Order

### Stage 0: Local MoE Demo

Run from `paper_root`:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_minimal_moe_demo.py --output_root "paper/UXFD_paper/MOE_explainable/results/autoresearch/<run_id>/demo"
```

Acceptance:
- `demo_summary.json` exists
- `test_accuracy >= 0.60`
- route statistics are present (`route_entropy`, `top_expert_weight`)

### Stage 1: Parent-Repo VIBENCH Smoke

Run from `exec_root`:

```bash
python main.py --config paper/UXFD_paper/MOE_explainable/configs/vibench/min.yaml       --override trainer.num_epochs=1       --override trainer.device=cpu       --override model.device=cpu       --override environment.output_dir=results/uxfd/autoresearch/MOE_explainable/<run_id>
```

Acceptance:
- `artifacts/manifest.json` exists
- schema pack is materialized under `paper_root/outputs/...`
- `validate_schema.py --run_dir <RUN_DIR>` passes

## Loop Policy

- The local demo is the repeatable stage used for route-quality refreshes.
- If route quality stagnates or import/runtime errors persist beyond budget, mark this paper `blocked` and continue other papers.

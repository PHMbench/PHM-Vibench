# VIBENCH Mapping And One-Command Reproduction (MOE_explainable)

## Execution Roots

- `paper_root`: `paper/UXFD_paper/MOE_explainable`
- `exec_root`: `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench_fix`

## Maintained Parent Entry

Config stored in the paper submodule:

- `paper/UXFD_paper/MOE_explainable/configs/vibench/min.yaml`

Maintained smoke command from `exec_root`:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/MOE_explainable/configs/vibench/min.yaml --override trainer.num_epochs=1
```

## Local Demo Entry

Run from `paper_root`:

```bash
python scripts/run_minimal_moe_demo.py --output_root "paper/UXFD_paper/MOE_explainable/results/autoresearch/<run_id>/demo"
```

Expected local demo artifacts:

- `demo_summary.json`
- `demo_visualizations/moe_demo_results.png`

## Evidence Chain Rule

The nonstop runner promotes only schema-valid runs into the paper evidence chain. Local demo outputs and maintained smoke outputs are wrapped under `paper_root/outputs/...` before they count.

## T043 Matrix Status

The command-bound comparison checkpoint is:

- `submission_prep/baseline_ablation_matrix.yaml`
- `submission_prep/ieee_trans_readiness.md`

Current status:

- Proposed PHM-Vibench proxy and six model baselines are dummy-smoke validated
  in `LQ_signal` with CPU fallback because GPU/NVML is unavailable in this
  sandbox.
- Existing route/expert artifacts remain partial evidence only.
- `scripts/run_moe_ablation_smoke.py` emits non-accepted metadata/metrics for
  no load-balance, no sparsity, router-temperature sweep, expert-family
  removal, and uniform-router surfaces.
- No SOTA wording is allowed until same-protocol CWRU/XJTU or industrial
  baselines, ablations, TOP representatives, and 2x4090 metadata exist.

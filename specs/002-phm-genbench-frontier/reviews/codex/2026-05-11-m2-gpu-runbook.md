# M2 GPU Runbook

## Current Blocker

M2-003 real six-dataset execution is blocked because the current machine does
not expose CUDA to torch:

```text
nvidia-smi -L
-> NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.

CUDA_VISIBLE_DEVICES=6,7 python -c "import torch; ..."
-> cuda_available False
-> device_count 0
```

Do not start six-dataset training until GPU 6 and GPU 7 pass preflight.

Latest audit snapshot:

- latest refresh: `2026-05-16 15:10:59 CST`
- `nvidia-smi -L` still cannot communicate with the NVIDIA driver.
- `CUDA_VISIBLE_DEVICES=6,7` under `LQ_signal` reports
  `cuda_available False`, `device_count 0`, and cannot initialize NVML.
- `python -m pytest test/generative -q` passes with `105 passed, 1 warning`.
- `python -m pytest test/ -q` passes under `LQ_signal` with `220 passed, 1
  warning`.
- Base Python lacks `torchmetrics`; use `LQ_signal` for the full repository
  test gate.
- Latest elevated `scripts.generative_benchmark_effect --preflight-gpu --dry-run`
  recheck passes for GPU 6 and GPU 7 after `nvidia-modprobe -u -c=0`.
- The canonical M2-003 preflight report was refreshed at
  `2026-05-16T07:10:59.524502+00:00` under
  `results/paper/phm_generative/six_dataset_submission_v1/gpu_preflight/`.
- The canonical blocked run-status ledger still has 37 lines: header plus 36
  blocked dataset/method/seed run groups.
- Driver diagnosis at `2026-05-16 15:18:35 CST`: the NVIDIA kernel module is
  loaded and PCI lists eight NVIDIA devices, but sandboxed commands do not see
  `/dev/nvidia*` device nodes. Elevated `nvidia-modprobe -u -c=0` restores GPU
  access for M2-003 commands.
- Train-stage note at `2026-05-16 16:26:39 CST`: the matrix now overrides
  `model.num_fault_classes=32` and `model.num_domains=16`; elevated train
  produced 7 partial checkpoints, including all six CWRU method/seed train jobs
  and XJTU CFM seed 0, then was interrupted to avoid an unattended long run.
- Bounded resume note at `2026-05-16 16:57:53 CST`: train execution with
  `--skip-existing --max-runs 1` skipped the six completed CWRU rows, ran XJTU
  CFM seed 0, produced `train_result_0.csv`, and was interrupted after the
  bounded chunk. Evidence after that chunk was 7 `train_result_0.csv`, 8
  checkpoints, and 6 manifest files; sample/eval/paperpack remained absent.
- Bounded resume note at `2026-05-16 17:24:32 CST`: train execution with
  `--skip-existing --max-runs 1` skipped completed rows through XJTU CFM seed
  0, ran XJTU CFM seed 1, and produced `train_result_0.csv` with
  `train_completed=True`. Current partial evidence is 8 `train_result_0.csv`,
  9 checkpoints, and 6 manifest files; sample/eval/paperpack remain absent.

## Verified Dry-Run Plan

Command plan:

```text
results/paper/phm_generative/six_dataset_submission_v1/dry_run/run_plan.csv
results/paper/phm_generative/six_dataset_submission_v1/dry_run_current_audit/run_plan.csv
```

Plan coverage:

- rows: 144
- datasets: `RM_001_CWRU`, `RM_002_XJTU`, `RM_003_FEMTO`, `RM_008_UNSW`,
  `RM_024_JUST`, `RM_027_PU`
- methods: `cfm_grid`, `rectified_flow_grid`, `ddpm_train_distribution`
- seeds: `0`, `1`
- stages: `train`, `sample`, `eval`, `paperpack`
- GPU IDs: `6`, `7`
- duplicate dataset/method/seed/stage keys: `0`

This covers:

```text
6 datasets x 3 methods x 2 seeds x 4 stages = 144 commands
```

## Resume Gates

Run these before real execution:

```bash
nvidia-smi -L
eval "$(conda shell.bash hook)" && conda activate LQ_signal && \
CUDA_VISIBLE_DEVICES=6 python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
eval "$(conda shell.bash hook)" && conda activate LQ_signal && \
CUDA_VISIBLE_DEVICES=7 python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
eval "$(conda shell.bash hook)" && conda activate LQ_signal && \
CUDA_VISIBLE_DEVICES=6,7 python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

Required result:

- `nvidia-smi -L` lists visible GPUs.
- `torch.cuda.is_available()` is `True` for each individual
  `CUDA_VISIBLE_DEVICES=6` and `CUDA_VISIBLE_DEVICES=7` probe.
- `torch.cuda.device_count()` is exactly `1` for each individual GPU probe.
- `torch.cuda.device_count()` is at least `2` when `CUDA_VISIBLE_DEVICES=6,7`.

Then run benchmark preflight:

```bash
eval "$(conda shell.bash hook)" && conda activate LQ_signal && \
python -m scripts.generative_benchmark_effect \
  --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml \
  --preflight-gpu \
  --dry-run \
  --output-dir results/paper/phm_generative/six_dataset_submission_v1/gpu_preflight
```

## Execution Sequence

After preflight passes, execute one stage at a time:

```bash
eval "$(conda shell.bash hook)" && conda activate LQ_signal && \
python -m scripts.generative_benchmark_effect \
  --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml \
  --execute \
  --preflight-gpu \
  --stages train \
  --output-dir results/paper/phm_generative/six_dataset_submission_v1
```

Then repeat for:

```text
sample
eval
paperpack
```

The stage filter is covered by
`test/generative/test_benchmark_effect.py::test_dry_run_stage_filter_writes_only_requested_stage`.
Misspelled stage names are rejected instead of writing an empty successful run
plan.
Because the M2 matrix sets `resource.require_cuda: true`, each `--execute`
command must include `--preflight-gpu`; otherwise the runner fails before
writing a plan or starting training.
Each `--execute` command must also name exactly one stage. The runner rejects
multi-stage CUDA execution to keep the evidence ledger stage-by-stage.
The runner also rejects mixed primary modes such as `--dry-run --execute`.

For long resume sessions, use bounded chunks and skip completed artifacts:

```bash
eval "$(conda shell.bash hook)" && conda activate LQ_signal && \
python -m scripts.generative_benchmark_effect \
  --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml \
  --execute \
  --preflight-gpu \
  --stages train \
  --skip-existing \
  --max-runs 2 \
  --output-dir results/paper/phm_generative/six_dataset_submission_v1
```

`--skip-existing` skips a train row when `train_result_0.csv` already exists
under that row's `environment.output_dir`; a checkpoint alone is not enough,
because interrupted jobs can leave partial checkpoints. `--max-runs` limits
only non-skipped commands. This lets future sessions resume from the partial
CWRU/XJTU evidence without retraining completed jobs or skipping interrupted
ones.

For `sample`, `eval`, and `paperpack`, the generated plan contains
`<experiment_name>` placeholders because the final run directory is created by
`main.py`. `scripts.generative_benchmark_effect --execute` resolves those
placeholders from the latest completed checkpoint, `samples.pt`, or `iter_0`
directory under the corresponding previous stage. If the prerequisite artifact
does not exist, execution fails instead of silently using an invalid path.

Do not route the paper benchmark to CPU. The matrix requires GPU resources 6
and 7 with `trainer.device=cuda` and `trainer.gpus=1`.

To inspect the next train commands from the dry-run plan without executing
them:

```bash
python - <<'PY'
import csv
from pathlib import Path

plan = Path("results/paper/phm_generative/six_dataset_submission_v1/dry_run/run_plan.csv")
for row in csv.DictReader(plan.open()):
    if row["stage"] == "train":
        print(row["command"])
PY
```

This prints 36 train commands:

```text
6 datasets x 3 methods x 2 seeds
```

The dry-run plan must include GPU-pinned commands for both physical GPUs:

```text
env CUDA_VISIBLE_DEVICES=6 python main.py ...
env CUDA_VISIBLE_DEVICES=7 python main.py ...
```

Each generated `main.py` command must keep `trainer.device=cuda` and
`trainer.gpus=1`. The full dry-run plan remains 144 commands:

```text
6 datasets x 3 methods x 2 seeds x 4 stages
```

Execute only after GPU 6/7 preflight passes.

When resuming from the current partial evidence, keep the matrix embedding-size
overrides:

```text
model.num_fault_classes=32
model.num_domains=16
```

These are required for the real multi-dataset metadata label/domain ranges.

## Evidence Aggregation

After completed runs exist:

```bash
eval "$(conda shell.bash hook)" && conda activate LQ_signal && \
python -m scripts.generative_benchmark_effect \
  --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml \
  --from-runs results/paper/phm_generative/six_dataset_submission_v1/runs \
  --output-dir results/paper/phm_generative/six_dataset_submission_v1/effect
```

If the command fails with:

```text
[FAIL] run_dir does not exist: results/paper/phm_generative/six_dataset_submission_v1/runs
```

then M2-003 real execution has not produced reviewable run directories yet.
Do not generate paper claims from the dry-run plan alone.
Do not call `--from-runs` without at least one run directory; the runner rejects
empty aggregation requests and run directories that contain no
`generative_eval_metrics.csv` records.

Then regenerate the paper draft:

```bash
eval "$(conda shell.bash hook)" && conda activate LQ_signal && \
python -m scripts.generative_submission_draft \
  --summary results/paper/phm_generative/six_dataset_submission_v1/effect/benchmark_effect_summary.csv \
  --manifest results/paper/phm_generative/six_dataset_submission_v1/effect/benchmark_effect_manifest.json \
  --output specs/002-phm-genbench-frontier/paper/PAPER_DRAFT.md \
  --require-submission-ready
```

The draft generator writes `PAPER_DRAFT.md`, `evidence_gaps.md`, and
`submission_readiness.md` together. `SUBMISSION_READY` requires dataset coverage
without missing or unexpected datasets, benchmark-valid quality and utility
rows, and traceable metric/manifest source paths.

## Completion Rule

The active goal remains incomplete until:

- GPU 6/7 preflight passes.
- Six datasets have completed train/sample/eval/paperpack evidence.
- Aggregation produces benchmark-effect summary, report, manifest, and missing
  metric appendix.
- The paper draft is generated from evidence and either becomes
  `SUBMISSION_READY` or explicitly explains remaining gaps in the draft and
  sidecar readiness files.

Task ledger linkage:

- `T047` owns real GPU execution and remains open.
- `T048` owns real aggregation after T047.
- `T049` owns final figures/tables after T048.
- `T050` owns submission draft regeneration after T049.
- `T051` owns final verification and review after T050.

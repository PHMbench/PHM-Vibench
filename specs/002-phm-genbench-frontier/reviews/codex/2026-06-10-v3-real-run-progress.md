# PHM-GenBench v0.3 Real-Run Progress

Date: 2026-06-10

Goal:
`GOAL-V3-008-REAL-SIX-DATASET-RUN`

## Current Decision

Decision: `IN_PROGRESS_LONG_RUN_REQUIRED`

The earlier GPU preflight failure was caused by the default sandbox not exposing
the NVIDIA driver. Unsandboxed execution sees GPUs 6 and 7, and the official
matrix preflight now passes.

## GPU Preflight

Command:

```bash
python -m scripts.generative_benchmark_effect \
  --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml \
  --dry-run \
  --preflight-gpu \
  --output-dir results/paper/phm_generative/six_dataset_submission_v1/gpu_preflight_v3_2026_06_10_unsandboxed
```

Status: passed.

Evidence:
- `results/paper/phm_generative/six_dataset_submission_v1/gpu_preflight_v3_2026_06_10_unsandboxed/gpu_preflight_report.json`
- `results/paper/phm_generative/six_dataset_submission_v1/gpu_preflight_v3_2026_06_10_unsandboxed/run_plan.csv`

## Completed Real Evidence

Output root:
`results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10`

Completed full-chain rows:

| dataset | method | seed | status |
| --- | --- | ---: | --- |
| RM_001_CWRU | cfm_grid | 0 | complete |
| RM_001_CWRU | cfm_grid | 1 | complete |
| RM_001_CWRU | rectified_flow_grid | 0 | complete |
| RM_001_CWRU | rectified_flow_grid | 1 | complete |
| RM_001_CWRU | ddpm_train_distribution | 0 | complete |
| RM_001_CWRU | ddpm_train_distribution | 1 | complete |

Evidence:
- Six `stage_ledger.json` files exist under
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_001_CWRU/`.
- Each completed ledger contains `train`, `sample`, `eval`, and `paperpack`
  stages.
- Each completed train stage has a checkpoint path and `train_result_0.csv`.
- Each completed sample stage has `samples.pt` and `synthetic_data_manifest.json`.
- Each completed eval stage has `generative_eval_metrics.csv` and
  `eval_evidence_manifest.json`.
- Each completed paperpack stage has table CSVs, figure-source CSVs, appendices,
  and a reproducibility statement.
- Batch summary snapshot:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/execution_summary_train_batch_001.csv`
- Current run-status ledger:
  `specs/002-phm-genbench-frontier/reviews/codex/2026-06-10-v3-real-run-ledger.csv`

## Partial Foreground Attempt

The next bounded batch started:

```bash
python -m scripts.generative_benchmark_effect \
  --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml \
  --execute \
  --preflight-gpu \
  --stages train \
  --skip-existing \
  --max-runs 6 \
  --output-dir results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10
```

It reached `RM_002_XJTU / cfm_grid / seed_0`, but that row has about 10,914
steps in epoch 0 and is too long for an interactive foreground Codex run. The
foreground process was stopped with SIGTERM to avoid leaving a stray GPU job.

Partial files exist for XJTU CFM seed 0, including metrics and normalization
artifacts, but there is no completed `stage_ledger.json`, checkpoint, or
`train_result_0.csv` for that row. It must be rerun by the long-run executor.

## Partial Aggregation Check

Command:

```bash
python -m scripts.generative_benchmark_effect \
  --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml \
  --from-runs results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs \
  --output-dir results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/effect_partial_cwru
```

Status: passed.

Result:
- Observed configured datasets: `1`.
- Missing configured datasets: `RM_002_XJTU`, `RM_003_FEMTO`, `RM_008_UNSW`,
  `RM_024_JUST`, `RM_027_PU`.
- `min_datasets_met=false`.
- Summary rows preserve metric, manifest, eval evidence, and paperpack source
  paths.
- All summary rows remain `exploratory`; no partial row is promoted to paper
  evidence.

Submission draft gate:
`results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/effect_partial_cwru/submission_draft.md`

Status: `NOT_SUBMISSION_READY`.

## Required Next Action

The remaining V3-008 stages have been moved into a detached tmux long-running
job context:

```bash
tmux new-session -d -s phm_genbench_v3 \
  "cd /home/user/LQ/B_Signal/vibench_fix/PHM-Vibench_gen_bench && \
   bash scripts/run_phm_genbench_v3_longrun.sh >> \
   results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/logs/v3_longrun_tmux_20260610.log 2>&1"
```

Active session:
`phm_genbench_v3`

Long-run log:
`results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/logs/v3_longrun_tmux_20260610.log`

Status ledger helper:
`python -m scripts.phm_genbench_v3_status --out specs/002-phm-genbench-frontier/reviews/codex/2026-06-10-v3-real-run-ledger.csv`

Current status snapshot:
- `COMPLETE_CHAIN=6`
- `IN_PROGRESS_NO_LEDGER=1`
- `PENDING=29`
- Active row: `RM_002_XJTU / cfm_grid / seed_0 / train`
- Latest active metric snapshot: `epoch=1`, `step=23871`

Updated monitor snapshot at 2026-06-10 20:19 CST:
- `phm_genbench_v3` tmux session is still active.
- Driver process:
  `python -m scripts.generative_benchmark_effect --execute --preflight-gpu --stages train --skip-existing ...`
- Active child:
  `python main.py --config configs/paper/phm_generative/cfm_train_grid_seed0.yaml ...`
- Status ledger remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PENDING=29`.
- Active row remains `RM_002_XJTU / cfm_grid / seed_0 / train`.
- Latest active metric snapshot from the status ledger:
  `epoch=2`, `step=28738`.
- GPU check shows GPU 6 has `513 MiB` allocated and nonzero utilization;
  no evidence of a stopped or failed job was observed.

Updated monitor snapshot at 2026-06-10 20:21 CST:
- `phm_genbench_v3` tmux session is still active.
- Status ledger remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PENDING=29`.
- Active row remains `RM_002_XJTU / cfm_grid / seed_0 / train`.
- Latest active metric snapshot from the status ledger:
  `epoch=2`, `step=37188`.
- Direct metrics tail advanced to `epoch=2`, `step=37676`, confirming the
  active training process is still writing new metrics.
- GPU check still shows GPU 6 allocated at `513 MiB` with nonzero utilization.
- This row is not complete yet because no XJTU `stage_ledger.json` with a
  completed train stage exists.

Updated monitor snapshot at 2026-06-10 20:23 CST:
- `phm_genbench_v3` tmux session is still active.
- Status ledger remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PENDING=29`.
- Active row remains `RM_002_XJTU / cfm_grid / seed_0 / train`.
- Latest active metric snapshot from the status ledger:
  `epoch=3`, `step=39820`.
- Direct metrics tail advanced to `epoch=3`, `step=41745`.
- No `stage_ledger.json`, `train_result_0.csv`, or checkpoint has been
  produced yet for the active XJTU row, so the row remains incomplete evidence.
- GPU check still shows GPU 6 allocated at `513 MiB` with nonzero utilization.
- The root-level
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/benchmark_effect_manifest.json`
  is a dry-run manifest (`mode: dry-run`) and must not be cited as final
  benchmark-effect evidence.
- The current root-level `execution_summary.csv` still reflects the earlier
  CWRU paperpack batch. The long-run script will overwrite/copy stage summaries
  after the active train stage exits.

Updated monitor snapshot at 2026-06-10 20:25 CST:
- `phm_genbench_v3` tmux session is still active.
- Status ledger remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PENDING=29`.
- Active row remains `RM_002_XJTU / cfm_grid / seed_0 / train`.
- Latest active metric snapshot from the status ledger:
  `epoch=3`, `step=49620`.
- Direct metrics tail advanced to `epoch=3`, `step=50235`.
- No `stage_ledger.json`, `train_result_0.csv`, checkpoint,
  `synthetic_data_manifest.json`, `generative_eval_metrics.csv`, or
  `eval_evidence_manifest.json` exists yet for the active XJTU row.
- GPU check still shows GPU 6 allocated at `513 MiB` with nonzero utilization.

Updated monitor snapshot at 2026-06-10 20:26 CST:
- `phm_genbench_v3` tmux session is still active.
- Status ledger remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PENDING=29`.
- Active row remains `RM_002_XJTU / cfm_grid / seed_0 / train`.
- Latest direct metrics tail advanced to `epoch=4`, `step=51406`.
- Post-refresh status ledger captured `epoch=4`, `step=54060`.
- A checkpoint now exists for the active XJTU row:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_002_XJTU/cfm_grid/seed_0/train/metadata.xlsx/M_phm_unet1d/T_generativeconditional_flow_matching_10_200929/iter_0/model-epoch=03-val_loss=1.4103.ckpt`.
- No completed `stage_ledger.json` or `train_result_0.csv` exists yet for the
  active XJTU row, so it remains incomplete evidence.
- The configured train length for this row is `trainer.num_epochs=50` with
  `data.batch_size=2`; early progress is still within the expected train stage,
  not a stage transition.

Updated monitor snapshot at 2026-06-10 20:29 CST:
- `phm_genbench_v3` tmux session is still active.
- Status ledger remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PENDING=29`.
- Active row remains `RM_002_XJTU / cfm_grid / seed_0 / train`.
- Status ledger captured `epoch=4`, `step=59446`.
- Direct metrics tail advanced to `epoch=4`, `step=62764`.
- The only checkpoint observed for the active XJTU row remains
  `model-epoch=03-val_loss=1.4103.ckpt`.
- No completed `stage_ledger.json`, `train_result_0.csv`,
  `synthetic_data_manifest.json`, `generative_eval_metrics.csv`, or
  `eval_evidence_manifest.json` exists yet for the active XJTU row.
- GPU check still shows GPU 6 allocated at `513 MiB` with nonzero utilization.

Updated monitor snapshot at 2026-06-10 20:31 CST:
- `phm_genbench_v3` tmux session is still active.
- Status ledger remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PENDING=29`.
- Active row remains `RM_002_XJTU / cfm_grid / seed_0 / train`.
- Status ledger captured `epoch=4`, `step=62794`.
- Direct metrics tail advanced to `epoch=5`, `step=63789`.
- Post-refresh status ledger captured `epoch=5`, `step=68476`.
- Latest checkpoint observed for the active XJTU row:
  `model-epoch=04-val_loss=1.4088.ckpt`.
- No completed `stage_ledger.json`, `train_result_0.csv`,
  `synthetic_data_manifest.json`, `generative_eval_metrics.csv`, or
  `eval_evidence_manifest.json` exists yet for the active XJTU row.
- GPU check still shows GPU 6 allocated at `513 MiB` with nonzero utilization.

Updated monitor snapshot at 2026-06-10 20:33 CST:
- `phm_genbench_v3` tmux session is still active.
- Status ledger remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PENDING=29`.
- Active row remains `RM_002_XJTU / cfm_grid / seed_0 / train`.
- Status ledger captured `epoch=5`, `step=73979`.
- Direct metrics tail advanced to `epoch=5`, `step=75353`.
- With `trainer.num_epochs=50`, the active XJTU train row is still early in
  the expected train stage, around 12% by epoch count.
- Latest checkpoint observed for the active XJTU row remains
  `model-epoch=04-val_loss=1.4088.ckpt`.
- No completed `stage_ledger.json`, `train_result_0.csv`,
  `synthetic_data_manifest.json`, `generative_eval_metrics.csv`, or
  `eval_evidence_manifest.json` exists yet for the active XJTU row.
- GPU check still shows GPU 6 allocated at `513 MiB` with nonzero utilization.

Updated monitor snapshot at 2026-06-10 20:35 CST:
- `phm_genbench_v3` tmux session is still active.
- Status ledger remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PENDING=29`.
- Active row remains `RM_002_XJTU / cfm_grid / seed_0 / train`.
- Status ledger captured `epoch=6`, `step=75375`.
- Direct metrics tail advanced to `epoch=6`, `step=77592`.
- Post-refresh status ledger captured `epoch=6`, `step=83473`.
- Latest checkpoint observed for the active XJTU row:
  `model-epoch=05-val_loss=1.4044.ckpt`.
- No completed `stage_ledger.json`, `train_result_0.csv`,
  `synthetic_data_manifest.json`, `generative_eval_metrics.csv`, or
  `eval_evidence_manifest.json` exists yet for the active XJTU row.
- Matrix resources declare `gpu_ids: [6, 7]` and `max_parallel_runs: 2`, but
  `scripts.generative_benchmark_effect.execute_plan()` currently executes rows
  serially via `subprocess.run`. This means `max_parallel_runs` rotates planned
  GPU IDs but does not launch concurrent workers.
- Do not start an ad hoc second worker against the same output root without row
  filtering or write isolation; otherwise `execution_summary.csv` and unstarted
  run directories can race with the active long-run driver.

Updated monitor snapshot at 2026-06-10 20:38 CST:
- `phm_genbench_v3` tmux session is still active.
- Status ledger remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PENDING=29`.
- Active row remains `RM_002_XJTU / cfm_grid / seed_0 / train`.
- Status ledger and direct metrics tail captured `epoch=6`, `step=87912`.
- Latest checkpoint observed for the active XJTU row remains
  `model-epoch=05-val_loss=1.4044.ckpt`.
- No completed `stage_ledger.json`, `train_result_0.csv`,
  `synthetic_data_manifest.json`, `generative_eval_metrics.csv`, or
  `eval_evidence_manifest.json` exists yet for the active XJTU row.
- GPU check still shows GPU 6 allocated at `513 MiB` with nonzero utilization.

Updated monitor snapshot at 2026-06-10 20:40 CST:
- `phm_genbench_v3` tmux session is still active.
- Status ledger remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PENDING=29`.
- Active row remains `RM_002_XJTU / cfm_grid / seed_0 / train`.
- Status ledger captured `epoch=7`, `step=93138`.
- Direct metrics tail advanced to `epoch=7`, `step=96306`.
- Latest checkpoint observed for the active XJTU row:
  `model-epoch=06-val_loss=1.4029.ckpt`.
- No completed `stage_ledger.json`, `train_result_0.csv`,
  `synthetic_data_manifest.json`, `generative_eval_metrics.csv`, or
  `eval_evidence_manifest.json` exists yet for the active XJTU row.
- GPU 7 remains effectively idle, but the current executor has no row-level
  lock or isolated execution summary for concurrent workers. Keep the active
  single-driver run unless a safe row-filtered worker is implemented and
  reviewed separately.

Updated monitor snapshot at 2026-06-10 20:42 CST:
- `phm_genbench_v3` tmux session is still active.
- Status ledger remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PENDING=29`.
- Active row remains `RM_002_XJTU / cfm_grid / seed_0 / train`.
- Status ledger and direct metrics tail captured `epoch=7`, `step=100471`.
- Latest checkpoint observed for the active XJTU row remains
  `model-epoch=06-val_loss=1.4029.ckpt`.
- No completed `stage_ledger.json`, `train_result_0.csv`,
  `synthetic_data_manifest.json`, `generative_eval_metrics.csv`, or
  `eval_evidence_manifest.json` exists yet for the active XJTU row.
- GPU check still shows GPU 6 allocated at `513 MiB` with nonzero utilization.

Updated execution action at 2026-06-10 20:44 CST:
- Started an independent direct train worker for
  `RM_002_XJTU / cfm_grid / seed_1 / train` on GPU 7.
- Tmux session:
  `phm_genbench_v3_xjtu_seed1_gpu7`.
- Log:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/logs/direct_xjtu_cfm_seed1_train_gpu7_20260610.log`.
- This worker runs `python main.py --config ...` directly and does not invoke
  `scripts.generative_benchmark_effect --execute`, so it does not write or race
  the main driver's `execution_summary.csv`.
- It uses the matrix-planned output directory and stage ledger path for seed 1:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_002_XJTU/cfm_grid/seed_1/train`
  and
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_002_XJTU/cfm_grid/seed_1/stage_ledger.json`.
- Rationale: seed 1 had no existing run directory and was `PENDING`; GPU 7 was
  idle; the row is disjoint from the active seed 0 run. After successful
  completion, the main driver should skip this row because train skip-existing
  checks for `train_result_0.csv` under the planned train output dir.

Updated monitor snapshot at 2026-06-10 20:45 CST:
- Main tmux `phm_genbench_v3` remains active on seed 0.
- Direct tmux `phm_genbench_v3_xjtu_seed1_gpu7` remains active on seed 1.
- GPU check shows GPU 6 allocated at `513 MiB` and GPU 7 allocated at
  `499 MiB`.
- Seed 0 status ledger captured `epoch=8`, `step=113030`.
- Seed 1 direct metrics tail captured `epoch=0`, `step=2728`.
- The status helper still reports seed 1 as `PENDING`; this appears to lag the
  newly created metrics file and must not be used to claim seed 1 completion.
- Post-refresh status helper now reports `COMPLETE_CHAIN=6`,
  `IN_PROGRESS_NO_LEDGER=2`, `PENDING=28`.
- Post-refresh seed 1 status ledger captured `epoch=0`, `step=5582`.

Updated coordination snapshot at 2026-06-10 20:49 CST:
- Risk identified: the main benchmark-effect parent would move from seed 0 to
  seed 1 as soon as seed 0 finishes. Because the direct GPU7 seed 1 worker
  started later, seed 1 may still be in progress at that point; without a
  completed `train_result_0.csv`, the main driver would start a duplicate seed
  1 train.
- Mitigation applied: paused only the main benchmark-effect parent process
  `PID 3870186` with `SIGSTOP`.
- Confirmed process states after pausing:
  - parent `3870186`: `T+` stopped.
  - seed 0 train child `3870378`: still running.
  - seed 1 direct worker `3902132`: still running.
- Latest observed metrics after pausing parent:
  - seed 0: `epoch=9`, `step=125589`.
  - seed 1: `epoch=1`, `step=14073`.
- Resume condition: after seed 1 produces its completed `train_result_0.csv`,
  resume the parent with `kill -CONT 3870186`. The parent should then skip seed
  1 through the train skip-existing check and continue with the next train row.
- Do not resume the parent while seed 1 is still missing `train_result_0.csv`,
  or the duplicate-run risk returns.

Updated monitor snapshot at 2026-06-10 20:51 CST:
- Parent `PID 3870186` remains stopped (`T+`).
- Seed 0 train child `PID 3870378` remains running.
- Seed 1 direct worker `PID 3902132` remains running.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=2`,
  `PENDING=28`.
- Latest observed metrics:
  - seed 0: `epoch=9`, `step=125589`.
  - seed 1: `epoch=1`, `step=25117`.
- Latest observed checkpoints:
  - seed 0: `model-epoch=08-val_loss=1.4016.ckpt`.
  - seed 1: `model-epoch=00-val_loss=1.4413.ckpt`.
- Neither seed has produced completed `train_result_0.csv`, so the parent
  must remain stopped.

Updated monitor snapshot at 2026-06-10 20:54 CST:
- Parent `PID 3870186` remains stopped (`T+`).
- Seed 0 train child `PID 3870378` remains running under the stopped parent.
- Direct seed 1 worker `PID 3902132` remains running.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=2`,
  `PENDING=28`.
- Latest observed metrics:
  - seed 0: `epoch=9`, `step=125589`.
  - seed 1: `epoch=2`, `step=34302`.
- Latest observed checkpoints:
  - seed 0: `model-epoch=08-val_loss=1.4016.ckpt`.
  - seed 1: `model-epoch=01-val_loss=1.4274.ckpt`.
- Diagnosis: the main driver uses `subprocess.run(..., capture_output=True)`.
  Because only the parent was stopped, seed 0 can reach the end of training but
  then block when the parent is not draining the captured stdout/stderr pipe.
- Conservative action: keep the parent stopped until direct seed 1 writes a
  real `train_result_0.csv`; then resume the parent with `kill -CONT 3870186`.
  This avoids creating any placeholder completion artifact and avoids launching
  a duplicate seed 1 train.

Updated monitor snapshot at 2026-06-10 21:00 CST:
- Parent `PID 3870186` remains stopped (`T+`).
- Seed 0 child `PID 3870378` remains alive under the stopped parent; latest
  seed 0 metrics are still `epoch=9`, `step=125589`.
- Direct seed 1 worker `PID 3902132` remains alive; latest seed 1 metrics are
  `epoch=4`, `step=54558`.
- Latest seed 1 checkpoint is `model-epoch=03-val_loss=1.4123.ckpt`.
- No XJTU CFM seed has produced `train_result_0.csv` or `stage_ledger.json`
  yet, so the parent remains paused.

Updated monitor snapshot at 2026-06-10 21:10 CST:
- Parent `PID 3870186` remains stopped (`T+`).
- Seed 0 child remains alive but has no new completion artifacts; latest seed 0
  metrics remain `epoch=9`, `step=125589`.
- Direct seed 1 worker remains alive; latest seed 1 metrics are `epoch=6`,
  `step=87609`.
- Latest seed 1 checkpoint is `model-epoch=05-val_loss=1.4026.ckpt`.
- No `train_result_0.csv` exists for XJTU CFM seed 0 or seed 1 yet. Resume
  condition is unchanged.

Updated coordination snapshot at 2026-06-10 21:28 CST:
- Rechecked the training config: `cfm_train_grid_seed0.yaml` uses
  `trainer.num_epochs: 50` with early stopping. The earlier assumption that
  epoch 9/10 implied near-completion was wrong.
- A controlled 120-second parent resume window was tested to drain captured
  stdout/stderr and detect seed 0 completion. It timed out without
  `train_result_0.csv`, but confirmed seed 0 continued training:
  `epoch=10`, `step=135023`, with checkpoint
  `model-epoch=09-val_loss=1.4011.ckpt`.
- Direct seed 1 is ahead of seed 0:
  `epoch=11`, `step=149095`, with checkpoint
  `model-epoch=10-val_loss=1.3966.ckpt`.
- Strategy updated: keep parent `PID 3870186` running instead of paused, because
  pausing the parent can repeatedly create captured-output backpressure for seed
  0. Continue monitoring that direct seed 1 remains ahead and produces a real
  `train_result_0.csv` before the parent reaches the seed 1 train row.
- Confirmed after `kill -CONT 3870186`: parent state is `S+`, seed 0 metrics
  advanced to `epoch=10`, `step=136032`, and seed 1 metrics advanced to
  `epoch=11`, `step=150707`.

Updated monitor snapshot at 2026-06-10 21:31 CST:
- Parent `PID 3870186` remains running (`S+`), not paused.
- Seed 0 train child `PID 3870378` remains running; latest seed 0 metrics are
  `epoch=11`, `step=139364`.
- Direct seed 1 worker `PID 3902132` remains running; latest seed 1 metrics are
  `epoch=12`, `step=155101`.
- Latest observed checkpoints:
  - seed 0: `model-epoch=09-val_loss=1.4011.ckpt`.
  - seed 1: `model-epoch=11-val_loss=1.3960.ckpt`.
- Trainer contract rechecked: paper CFM config sets `trainer.num_epochs: 50`;
  base trainer enables early stopping with `patience: 5`. If validation loss
  keeps improving, XJTU CFM train may continue for many epochs. This is a run
  duration risk, not evidence-chain completion.
- Current status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=2`,
  `PENDING=28`; no XJTU CFM `train_result_0.csv` exists yet.

Updated monitor snapshot at 2026-06-10 21:37 CST:
- Parent `PID 3870186` remains running (`S+`).
- Seed 0 train child `PID 3870378` remains running; latest seed 0 metrics are
  `epoch=12`, `step=163001`.
- Direct seed 1 worker `PID 3902132` remains running; latest seed 1 metrics are
  `epoch=13`, `step=175825`.
- Latest observed checkpoints:
  - seed 0: `model-epoch=11-val_loss=1.3939.ckpt`.
  - seed 1: `model-epoch=12-val_loss=1.3913.ckpt`.
- No XJTU CFM `train_result_0.csv` or `stage_ledger.json` exists yet. The
  active train stage is progressing but incomplete.

Updated monitor snapshot at 2026-06-10 21:48 CST:
- Parent `PID 3870186` remains running (`S+`).
- Seed 0 train child `PID 3870378` remains running; latest seed 0 metrics are
  `epoch=15`, `step=195632`.
- Direct seed 1 worker `PID 3902132` remains running; latest seed 1 metrics are
  `epoch=16`, `step=212620`.
- Latest observed checkpoints:
  - seed 0: `model-epoch=13-val_loss=1.3929.ckpt`.
  - seed 1: `model-epoch=13-val_loss=1.3891.ckpt`.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=2`,
  `PENDING=28`; no XJTU CFM `train_result_0.csv` or `stage_ledger.json` exists
  yet.

Updated monitor snapshot at 2026-06-10 22:00 CST:
- Parent `PID 3870186` remains running (`S+`).
- Seed 0 train child `PID 3870378` remains running; latest seed 0 metrics are
  `epoch=18`, `step=238620`.
- Direct seed 1 worker `PID 3902132` remains running; latest seed 1 metrics are
  `epoch=19`, `step=251179`.
- Latest observed checkpoints:
  - seed 0: `model-epoch=17-val_loss=1.3902.ckpt`.
  - seed 1: `model-epoch=18-val_loss=1.3878.ckpt`.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=2`,
  `PENDING=28`; no XJTU CFM `train_result_0.csv` or `stage_ledger.json` exists
  yet.

Updated coordination snapshot at 2026-06-10 23:25 CST:
- XJTU CFM seed 0 completed train and wrote:
  - `seed_0/stage_ledger.json`
  - `seed_0/train/.../train_result_0.csv`
  - checkpoint `model-epoch=18-val_loss=1.3864.ckpt`
- Status helper now reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=1`, `PENDING=28`.
- A duplicate seed 1 race was detected:
  - official main-driver seed 1 child `PID 3987131` on GPU 6, run directory
    `T_generativeconditional_flow_matching_10_222124`;
  - earlier direct seed 1 worker `PID 3902132` on GPU 7, run directory
    `T_generativeconditional_flow_matching_10_204353`.
- Both seed 1 workers used the same configured `seed_1/stage_ledger.json`,
  which would create a ledger overwrite race if both completed.
- Action taken: terminated the non-official direct worker `PID 3902132` with
  `SIGTERM`, preserving the main-driver child `PID 3987131` as the only
  auditable seed 1 train.
- Confirmed after termination:
  - parent `PID 3870186` remains running (`S+`);
  - official seed 1 child `PID 3987131` remains running;
  - GPU 7 is released.
- Note: seed 0's current train ledger is real but minimal; it contains
  `schema_version` and `stages.train`, but not the full top-level metadata
  requested by the v0.3 goal. This must be considered during reviewer closure
  and may require a targeted ledger metadata patch.

Updated implementation snapshot at 2026-06-10 23:31 CST:
- Added a small ledger metadata repair path:
  - `scripts.generative_benchmark_effect.execute_plan()` now writes/updates
    top-level ledger metadata when it starts, completes, fails, or skips a
    planned stage.
  - `scripts.phm_genbench_v3_status --repair-ledger-metadata` now augments
    existing ledgers with `benchmark_id`, dataset, method, seed, config path,
    run root, current stage, and status without adding fake stages.
  - `scripts/run_phm_genbench_v3_longrun.sh` now calls the status helper with
    `--repair-ledger-metadata` after each stage.
- Ran repair mode once. Seed 0 ledger now has v0.3 top-level run metadata and
  still only contains the real `train` stage.
- Validation passed:
  - `python -m pytest test/generative/test_benchmark_effect.py::test_execute_plan_skip_existing_train_artifact test/generative/test_stage_ledger.py`
  - `python -m scripts.validate_docs`
  - `git diff --check`
- Official seed 1 child `PID 3987131` remains running under the main driver;
  latest official seed 1 metrics are `epoch=18`, `step=234820`, with checkpoint
  `T_generativeconditional_flow_matching_10_222124/model-epoch=17-val_loss=1.3885.ckpt`.

Updated monitor snapshot at 2026-06-10 23:35 CST:
- Non-official direct seed 1 worker `PID 3902132` is no longer present.
- Parent `PID 3870186` remains running (`S+`).
- Official seed 1 child `PID 3987131` remains running under the parent.
- Latest official seed 1 metrics are `epoch=19`, `step=247964`.
- Latest official seed 1 checkpoint is
  `T_generativeconditional_flow_matching_10_222124/model-epoch=18-val_loss=1.3878.ckpt`.
- No official seed 1 `train_result_0.csv` or `stage_ledger.json` exists yet,
  so train stage is still incomplete.

Updated monitor snapshot at 2026-06-10 23:42 CST:
- Parent `PID 3870186` remains running (`S+`) in train stage.
- Official seed 1 child `PID 3987131` remains running under the parent.
- Latest official seed 1 metrics are `epoch=21`, `step=270134`.
- Latest official seed 1 checkpoint is
  `T_generativeconditional_flow_matching_10_222124/model-epoch=19-val_loss=1.3877.ckpt`.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=1`, `PENDING=28`.
- No official seed 1 `train_result_0.csv` or `stage_ledger.json` exists yet;
  V3-008 remains in train stage.

Updated monitor snapshot at 2026-06-10 23:54 CST:
- Parent `PID 3870186` remains running (`S+`) in train stage.
- Official seed 1 child `PID 3987131` remains running under the parent.
- Latest official seed 1 metrics are `epoch=24`, `step=310369`.
- Latest official seed 1 checkpoint is
  `T_generativeconditional_flow_matching_10_222124/model-epoch=23-val_loss=1.3860.ckpt`.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=1`, `PENDING=28`.
- No official seed 1 `train_result_0.csv` or `stage_ledger.json` exists yet;
  train stage is still incomplete.

Updated monitor snapshot at 2026-06-11 00:05 CST:
- Parent `PID 3870186` remains running (`S+`) in train stage.
- Official seed 1 child `PID 3987131` remains running under the parent.
- Latest official seed 1 metrics are `epoch=27`, `step=350979`.
- Latest official seed 1 checkpoint is
  `T_generativeconditional_flow_matching_10_222124/model-epoch=25-val_loss=1.3835.ckpt`.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=1`, `PENDING=28`.
- No official seed 1 `train_result_0.csv` or `stage_ledger.json` exists yet;
  train stage remains incomplete.

Updated monitor snapshot at 2026-06-11 00:22 CST:
- Parent `PID 3870186` remains running (`S+`) in train stage.
- Official seed 1 child `PID 3987131` remains running under the parent.
- Latest official seed 1 metrics are `epoch=32`, `step=403754`.
- Latest official seed 1 checkpoint is
  `T_generativeconditional_flow_matching_10_222124/model-epoch=29-val_loss=1.3833.ckpt`.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=1`, `PENDING=28`.
- No official seed 1 `train_result_0.csv` or `stage_ledger.json` exists yet;
  train stage remains incomplete.

Updated monitor snapshot at 2026-06-11 00:40 CST:
- Parent `PID 3870186` remains running (`S+`) in train stage.
- Official seed 1 child `PID 3987131` remains running under the parent.
- Latest official seed 1 metrics are `epoch=36`, `step=464682`.
- Latest official seed 1 checkpoint is
  `T_generativeconditional_flow_matching_10_222124/model-epoch=35-val_loss=1.3827.ckpt`.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=1`, `PENDING=28`.
- No official seed 1 `train_result_0.csv` or `stage_ledger.json` exists yet;
  train stage remains incomplete.

Updated monitor snapshot at 2026-06-11 00:52 CST:
- Parent `PID 3870186` remains running (`S+`) in train stage.
- Official seed 1 child `PID 3987131` remains running under the parent.
- Latest official seed 1 metrics are `epoch=39`, `step=502359`.
- Latest official seed 1 checkpoint is
  `T_generativeconditional_flow_matching_10_222124/model-epoch=38-val_loss=1.3801.ckpt`.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=1`, `PENDING=28`.
- No official seed 1 `train_result_0.csv` or `stage_ledger.json` exists yet;
  train stage remains incomplete.

Updated monitor snapshot at 2026-06-11 00:54 CST:
- Parent `PID 3870186` remains running (`S+`) in train stage.
- Official seed 1 child `PID 3987131` remains the only child under the parent.
- Latest official seed 1 metrics are `epoch=40`, `step=513105`.
- Latest official seed 1 checkpoint is
  `T_generativeconditional_flow_matching_10_222124/model-epoch=39-val_loss=1.3796.ckpt`.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=1`, `PENDING=28`.
- GPU 7 is idle; no duplicate seed 1 worker was observed.
- No official seed 1 `train_result_0.csv` or `stage_ledger.json` exists yet;
  train stage remains incomplete and sample/eval/paperpack must not start.

Updated monitor snapshot at 2026-06-11 00:55 CST:
- Parent `PID 3870186` remains running (`S+`) in train stage.
- Official seed 1 child `PID 3987131` remains the only child under the parent.
- Latest official seed 1 metrics are `epoch=40`, `step=514918`.
- Latest official seed 1 checkpoint remains
  `T_generativeconditional_flow_matching_10_222124/model-epoch=39-val_loss=1.3796.ckpt`.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=1`, `PENDING=28`.
- No official seed 1 `train_result_0.csv` or `stage_ledger.json` exists yet;
  train stage remains incomplete and sample/eval/paperpack must not start.

Updated monitor snapshot at 2026-06-11 00:58 CST:
- Parent `PID 3870186` remains running (`S+`) in train stage.
- Official seed 1 child `PID 3987131` remains the only child under the parent.
- Latest official seed 1 metrics are `epoch=41`, `step=527032`.
- Latest official seed 1 checkpoint remains
  `T_generativeconditional_flow_matching_10_222124/model-epoch=39-val_loss=1.3796.ckpt`.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=1`, `PENDING=28`.
- No official seed 1 `train_result_0.csv` or `stage_ledger.json` exists yet;
  train stage remains incomplete and sample/eval/paperpack must not start.

Updated monitor snapshot at 2026-06-11 00:59 CST:
- Parent `PID 3870186` remains running (`S+`) in train stage.
- Official seed 1 child `PID 3987131` remains the only child under the parent.
- Latest official seed 1 metrics are `epoch=41`, `step=527477`.
- Latest official seed 1 checkpoint remains
  `T_generativeconditional_flow_matching_10_222124/model-epoch=39-val_loss=1.3796.ckpt`.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=1`, `PENDING=28`.
- No official seed 1 `train_result_0.csv` or `stage_ledger.json` exists yet;
  train stage remains incomplete and sample/eval/paperpack must not start.

Updated monitor snapshot at 2026-06-11 01:01 CST:
- Parent `PID 3870186` remains running (`S+`) in train stage.
- Official seed 1 child `PID 3987131` remains the only child under the parent.
- Latest official seed 1 metrics are `epoch=42`, `step=534015`.
- Latest official seed 1 checkpoint is
  `T_generativeconditional_flow_matching_10_222124/model-epoch=41-val_loss=1.3794.ckpt`.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=1`, `PENDING=28`.
- No official seed 1 `train_result_0.csv` or `stage_ledger.json` exists yet;
  train stage remains incomplete and sample/eval/paperpack must not start.

Updated monitor snapshot at 2026-06-11 01:02 CST:
- Parent `PID 3870186` remains running (`S+`) in train stage.
- Official seed 1 child `PID 3987131` remains the only child under the parent.
- Latest official seed 1 metrics are `epoch=42`, `step=540036`.
- Latest official seed 1 checkpoint remains
  `T_generativeconditional_flow_matching_10_222124/model-epoch=41-val_loss=1.3794.ckpt`.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=1`, `PENDING=28`.
- No official seed 1 `train_result_0.csv` or `stage_ledger.json` exists yet;
  train stage remains incomplete and sample/eval/paperpack must not start.

Updated monitor snapshot at 2026-06-11 01:04 CST:
- Parent `PID 3870186` remains running (`S+`) in train stage.
- Official seed 1 child `PID 3987131` remains the only child under the parent.
- Latest official seed 1 metrics are `epoch=43`, `step=541364`.
- Latest official seed 1 checkpoint remains
  `T_generativeconditional_flow_matching_10_222124/model-epoch=41-val_loss=1.3794.ckpt`.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=1`, `PENDING=28`.
- No official seed 1 `train_result_0.csv` or `stage_ledger.json` exists yet;
  train stage remains incomplete and sample/eval/paperpack must not start.

Updated monitor snapshot at 2026-06-11 01:05 CST:
- Parent `PID 3870186` remains running (`S+`) in train stage.
- Official seed 1 child `PID 3987131` remains the only child under the parent;
  process state was `Dl+` during this snapshot, consistent with transient I/O
  wait while metrics continued to advance.
- Latest official seed 1 metrics are `epoch=43`, `step=547087`.
- Latest official seed 1 checkpoint remains
  `T_generativeconditional_flow_matching_10_222124/model-epoch=41-val_loss=1.3794.ckpt`.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=1`, `PENDING=28`.
- No official seed 1 `train_result_0.csv` or `stage_ledger.json` exists yet;
  train stage remains incomplete and sample/eval/paperpack must not start.

Updated monitor snapshot at 2026-06-11 01:06 CST:
- Parent `PID 3870186` remains running (`S+`) in train stage.
- Official seed 1 child `PID 3987131` remains the only child under the parent;
  process state returned to `Rl+`.
- Latest official seed 1 metrics are `epoch=43`, `step=552595`.
- Latest official seed 1 checkpoint remains
  `T_generativeconditional_flow_matching_10_222124/model-epoch=41-val_loss=1.3794.ckpt`.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=1`, `PENDING=28`.
- No official seed 1 `train_result_0.csv` or `stage_ledger.json` exists yet;
  train stage remains incomplete and sample/eval/paperpack must not start.

Updated monitor snapshot at 2026-06-11 01:08 CST:
- Parent `PID 3870186` remains running (`S+`) in train stage.
- Official seed 1 child `PID 3987131` remains the only child under the parent.
- A short poll after a stale metrics tail confirmed metrics advanced again;
  process state was `Rl+`.
- Latest official seed 1 metrics are `epoch=44`, `step=557171`.
- Latest official seed 1 checkpoint remains
  `T_generativeconditional_flow_matching_10_222124/model-epoch=41-val_loss=1.3794.ckpt`.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=1`, `PENDING=28`.
- No official seed 1 `train_result_0.csv` or `stage_ledger.json` exists yet;
  train stage remains incomplete and sample/eval/paperpack must not start.

Updated monitor snapshot at 2026-06-11 01:09 CST:
- Parent `PID 3870186` remains running (`S+`) in train stage.
- Official seed 1 child `PID 3987131` remains the only child under the parent.
- Latest official seed 1 metrics are `epoch=44`, `step=561910`.
- Latest official seed 1 checkpoint remains
  `T_generativeconditional_flow_matching_10_222124/model-epoch=41-val_loss=1.3794.ckpt`.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=1`, `PENDING=28`.
- No official seed 1 `train_result_0.csv` or `stage_ledger.json` exists yet;
  train stage remains incomplete and sample/eval/paperpack must not start.

Updated monitor snapshot at 2026-06-11 01:11 CST:
- Parent `PID 3870186` remains running (`S+`) in train stage.
- Official seed 1 child `PID 3987131` remains the only child under the parent.
- Latest official seed 1 metrics are `epoch=44`, `step=565154`.
- Latest official seed 1 checkpoint remains
  `T_generativeconditional_flow_matching_10_222124/model-epoch=41-val_loss=1.3794.ckpt`.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=1`, `PENDING=28`.
- No official seed 1 `train_result_0.csv` or `stage_ledger.json` exists yet;
  train stage remains incomplete and sample/eval/paperpack must not start.

Updated monitor snapshot at 2026-06-11 01:12 CST:
- Parent `PID 3870186` remains running (`S+`) in train stage.
- Official seed 1 child `PID 3987131` remains the only child under the parent;
  one process sample showed `rq_qos_wait`, consistent with I/O wait while
  training remains in progress.
- Latest official seed 1 metrics are `epoch=45`, `step=569555`.
- Latest official seed 1 checkpoint remains
  `T_generativeconditional_flow_matching_10_222124/model-epoch=41-val_loss=1.3794.ckpt`.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=1`, `PENDING=28`.
- No official seed 1 `train_result_0.csv` or `stage_ledger.json` exists yet;
  train stage remains incomplete and sample/eval/paperpack must not start.

Updated monitor snapshot at 2026-06-11 01:14 CST:
- Parent `PID 3870186` remains running (`S+`) in train stage.
- Official seed 1 child `PID 3987131` remains the only child under the parent.
- Latest official seed 1 metrics are `epoch=45`, `step=577179`.
- Latest official seed 1 checkpoint remains
  `T_generativeconditional_flow_matching_10_222124/model-epoch=41-val_loss=1.3794.ckpt`.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=1`, `PENDING=28`.
- No official seed 1 `train_result_0.csv` or `stage_ledger.json` exists yet;
  train stage remains incomplete and sample/eval/paperpack must not start.

Updated monitor snapshot at 2026-06-11 01:15 CST:
- Parent `PID 3870186` remains running (`S+`) in train stage.
- Official seed 1 child `PID 3987131` remains the only child under the parent.
- Latest official seed 1 metrics are `epoch=45`, `step=577713`.
- Latest official seed 1 checkpoint remains
  `T_generativeconditional_flow_matching_10_222124/model-epoch=41-val_loss=1.3794.ckpt`.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=1`, `PENDING=28`.
- No official seed 1 `train_result_0.csv` or `stage_ledger.json` exists yet;
  train stage remains incomplete and sample/eval/paperpack must not start.

Updated monitor snapshot at 2026-06-11 01:16 CST:
- Parent `PID 3870186` remains running (`S+`) in train stage.
- Official seed 1 child `PID 3987131` remains the only child under the parent;
  one process sample showed `wait_on_page_bit_common`, consistent with I/O wait.
- Latest official seed 1 metrics are `epoch=46`, `step=582268`.
- Latest official seed 1 checkpoint remains
  `T_generativeconditional_flow_matching_10_222124/model-epoch=41-val_loss=1.3794.ckpt`.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=1`, `PENDING=28`.
- No official seed 1 `train_result_0.csv` or `stage_ledger.json` exists yet;
  train stage remains incomplete and sample/eval/paperpack must not start.

Updated monitor snapshot at 2026-06-11 01:17 CST:
- Parent `PID 3870186` remains running (`S+`) in train stage.
- Official seed 1 child `PID 3987131` remains the only child under the parent.
- Latest official seed 1 metrics are `epoch=46`, `step=588028`.
- Latest official seed 1 checkpoint remains
  `T_generativeconditional_flow_matching_10_222124/model-epoch=41-val_loss=1.3794.ckpt`.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=1`, `PENDING=28`.
- No official seed 1 `train_result_0.csv` or `stage_ledger.json` exists yet;
  train stage remains incomplete and sample/eval/paperpack must not start.

Updated monitor snapshot at 2026-06-11 01:20 CST:
- XJTU CFM seed 1 train completed.
- Official seed 1 `train_result_0.csv` exists at
  `runs/RM_002_XJTU/cfm_grid/seed_1/train/metadata.xlsx/M_phm_unet1d/T_generativeconditional_flow_matching_10_222124/iter_0/train_result_0.csv`.
- Official seed 1 `stage_ledger.json` exists at
  `runs/RM_002_XJTU/cfm_grid/seed_1/stage_ledger.json` and records the
  train `run_dir`, `checkpoint_path`, and `train_result_path`.
- The train result reports `train_completed=True`,
  `train_wall_clock_sec=10649.76968259603`, `parameter_count=42290`, and
  `post_train_test_loss_ran=0.0`.
- Parent `PID 3870186` remains running (`S+`) in train stage.
- The current child is `PID 4087668`, running XJTU Rectified Flow seed 0 train:
  `configs/paper/phm_generative/rectified_flow_train_grid_seed0.yaml`.
- Latest Rectified Flow seed 0 metrics observed are `epoch=0`, `step=3747`.
- Status helper now reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=2`, `PENDING=27`.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 01:22 CST:
- Parent `PID 3870186` remains running (`S+`) in train stage.
- Current child `PID 4087668` is still running XJTU Rectified Flow seed 0 train
  with `configs/paper/phm_generative/rectified_flow_train_grid_seed0.yaml`.
- Latest Rectified Flow seed 0 metrics are `epoch=0`, `step=9018`.
- No Rectified Flow seed 0 `train_result_0.csv` or `stage_ledger.json` exists
  yet; train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=2`, `PENDING=27`.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 01:23 CST:
- Parent `PID 3870186` remains running (`S+`) in train stage.
- Current child `PID 4087668` is still running XJTU Rectified Flow seed 0 train
  with `configs/paper/phm_generative/rectified_flow_train_grid_seed0.yaml`.
- Latest Rectified Flow seed 0 metrics are `epoch=0`, `step=12558`.
- No Rectified Flow seed 0 `train_result_0.csv` or `stage_ledger.json` exists
  yet; train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=2`, `PENDING=27`.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 01:24 CST:
- Parent `PID 3870186` remains running (`S+`) in train stage.
- Current child `PID 4087668` is still running XJTU Rectified Flow seed 0 train
  with `configs/paper/phm_generative/rectified_flow_train_grid_seed0.yaml`.
- Latest Rectified Flow seed 0 metrics are `epoch=1`, `step=14673`.
- First Rectified Flow seed 0 checkpoint observed:
  `T_generativerectified_flow_11_011944/iter_0/model-epoch=00-val_loss=1.6054.ckpt`.
- No Rectified Flow seed 0 `train_result_0.csv` or `stage_ledger.json` exists
  yet; train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=2`, `PENDING=27`.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 01:26 CST:
- Parent `PID 3870186` remains running (`S+`) in train stage.
- Current child `PID 4087668` is still running XJTU Rectified Flow seed 0 train
  with `configs/paper/phm_generative/rectified_flow_train_grid_seed0.yaml`.
- Latest Rectified Flow seed 0 metrics are `epoch=1`, `step=19446`.
- Latest Rectified Flow seed 0 checkpoint remains
  `T_generativerectified_flow_11_011944/iter_0/model-epoch=00-val_loss=1.6054.ckpt`.
- No Rectified Flow seed 0 `train_result_0.csv` or `stage_ledger.json` exists
  yet; train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=2`, `PENDING=27`.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 01:27 CST:
- Parent `PID 3870186` remains running (`S+`) in train stage.
- Current child `PID 4087668` is still running XJTU Rectified Flow seed 0 train
  with `configs/paper/phm_generative/rectified_flow_train_grid_seed0.yaml`.
- Latest Rectified Flow seed 0 metrics are `epoch=1`, `step=25030`.
- Latest Rectified Flow seed 0 checkpoint remains
  `T_generativerectified_flow_11_011944/iter_0/model-epoch=00-val_loss=1.6054.ckpt`.
- No Rectified Flow seed 0 `train_result_0.csv` or `stage_ledger.json` exists
  yet; train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=2`, `PENDING=27`.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 01:28 CST:
- Parent `PID 3870186` remains running (`S+`) in train stage.
- Current child `PID 4087668` is still running XJTU Rectified Flow seed 0 train
  with `configs/paper/phm_generative/rectified_flow_train_grid_seed0.yaml`.
- Latest Rectified Flow seed 0 metrics are `epoch=1`, `step=25117`.
- Latest Rectified Flow seed 0 checkpoint remains
  `T_generativerectified_flow_11_011944/iter_0/model-epoch=00-val_loss=1.6054.ckpt`.
- No Rectified Flow seed 0 `train_result_0.csv` or `stage_ledger.json` exists
  yet; train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=2`, `PENDING=27`.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 01:29 CST:
- Parent `PID 3870186` remains running (`S+`) in train stage.
- Current child `PID 4087668` is still running XJTU Rectified Flow seed 0 train
  with `configs/paper/phm_generative/rectified_flow_train_grid_seed0.yaml`.
- Latest Rectified Flow seed 0 metrics are `epoch=2`, `step=29493`.
- Latest Rectified Flow seed 0 checkpoint is
  `T_generativerectified_flow_11_011944/iter_0/model-epoch=01-val_loss=1.5156.ckpt`.
- No Rectified Flow seed 0 `train_result_0.csv` or `stage_ledger.json` exists
  yet; train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=2`, `PENDING=27`.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 01:30 CST:
- Parent `PID 3870186` remains running (`S+`) in train stage.
- Current child `PID 4087668` is still running XJTU Rectified Flow seed 0 train
  with `configs/paper/phm_generative/rectified_flow_train_grid_seed0.yaml`.
- One process sample showed `wait_on_page_bit_common`, consistent with I/O wait
  while training remains in progress.
- Latest Rectified Flow seed 0 metrics are `epoch=2`, `step=34336`.
- Latest Rectified Flow seed 0 checkpoint remains
  `T_generativerectified_flow_11_011944/iter_0/model-epoch=01-val_loss=1.5156.ckpt`.
- No Rectified Flow seed 0 `train_result_0.csv` or `stage_ledger.json` exists
  yet; train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=2`, `PENDING=27`.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 01:31 CST:
- Parent `PID 3870186` remains running (`S+`) in train stage.
- Current child `PID 4087668` is still running XJTU Rectified Flow seed 0 train
  with `configs/paper/phm_generative/rectified_flow_train_grid_seed0.yaml`.
- Latest Rectified Flow seed 0 metrics are `epoch=2`, `step=37676`.
- Latest Rectified Flow seed 0 checkpoint remains
  `T_generativerectified_flow_11_011944/iter_0/model-epoch=01-val_loss=1.5156.ckpt`.
- No Rectified Flow seed 0 `train_result_0.csv` or `stage_ledger.json` exists
  yet; train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=2`, `PENDING=27`.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 01:33 CST:
- Parent `PID 3870186` remains running (`S+`) in train stage.
- Current child `PID 4087668` is still running XJTU Rectified Flow seed 0 train
  with `configs/paper/phm_generative/rectified_flow_train_grid_seed0.yaml`.
- Latest Rectified Flow seed 0 metrics are `epoch=3`, `step=38373`.
- Latest Rectified Flow seed 0 checkpoint is
  `T_generativerectified_flow_11_011944/iter_0/model-epoch=02-val_loss=1.4876.ckpt`.
- No Rectified Flow seed 0 `train_result_0.csv` or `stage_ledger.json` exists
  yet; train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=2`, `PENDING=27`.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 01:34 CST:
- Parent `PID 3870186` remains running (`S+`) in train stage.
- Current child `PID 4087668` is still running XJTU Rectified Flow seed 0 train
  with `configs/paper/phm_generative/rectified_flow_train_grid_seed0.yaml`.
- Latest Rectified Flow seed 0 metrics are `epoch=3`, `step=43553`.
- Latest Rectified Flow seed 0 checkpoint remains
  `T_generativerectified_flow_11_011944/iter_0/model-epoch=02-val_loss=1.4876.ckpt`.
- No Rectified Flow seed 0 `train_result_0.csv` or `stage_ledger.json` exists
  yet; train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=2`, `PENDING=27`.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 01:35 CST:
- Parent `PID 3870186` remains running (`S+`) in train stage.
- Current child `PID 4087668` is still running XJTU Rectified Flow seed 0 train
  with `configs/paper/phm_generative/rectified_flow_train_grid_seed0.yaml`.
- One process sample showed `wait_on_page_bit_common`, consistent with I/O wait
  while training remains in progress.
- Latest Rectified Flow seed 0 metrics are `epoch=3`, `step=48551`.
- Latest Rectified Flow seed 0 checkpoint remains
  `T_generativerectified_flow_11_011944/iter_0/model-epoch=02-val_loss=1.4876.ckpt`.
- No Rectified Flow seed 0 `train_result_0.csv` or `stage_ledger.json` exists
  yet; train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=2`, `PENDING=27`.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 01:36 CST:
- Parent `PID 3870186` remains running (`S+`) in train stage.
- Current child `PID 4087668` is still running XJTU Rectified Flow seed 0 train
  with `configs/paper/phm_generative/rectified_flow_train_grid_seed0.yaml`.
- Latest Rectified Flow seed 0 metrics are `epoch=3`, `step=50235`.
- Latest Rectified Flow seed 0 checkpoint remains
  `T_generativerectified_flow_11_011944/iter_0/model-epoch=02-val_loss=1.4876.ckpt`.
- No Rectified Flow seed 0 `train_result_0.csv` or `stage_ledger.json` exists
  yet; train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=2`, `PENDING=27`.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 01:37 CST:
- Parent `PID 3870186` remains running (`S+`) in train stage.
- Current child `PID 4087668` is still running XJTU Rectified Flow seed 0 train
  with `configs/paper/phm_generative/rectified_flow_train_grid_seed0.yaml`.
- Latest Rectified Flow seed 0 metrics are `epoch=4`, `step=52944`.
- Latest Rectified Flow seed 0 checkpoint is
  `T_generativerectified_flow_11_011944/iter_0/model-epoch=03-val_loss=1.4756.ckpt`.
- No Rectified Flow seed 0 `train_result_0.csv` or `stage_ledger.json` exists
  yet; train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=2`, `PENDING=27`.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 01:40 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `05:31:02`) in train
  stage.
- Current child `PID 4087668` is still running XJTU Rectified Flow seed 0 train
  with `configs/paper/phm_generative/rectified_flow_train_grid_seed0.yaml`
  (`Rl+`, elapsed `00:20:44`, approximately `89.3%` CPU).
- Latest Rectified Flow seed 0 metrics are `epoch=4`, `step=62794`; metrics
  file mtime is `2026-06-11 01:39:54 CST`.
- Latest Rectified Flow seed 0 checkpoint remains
  `T_generativerectified_flow_11_011944/iter_0/model-epoch=03-val_loss=1.4756.ckpt`.
- No Rectified Flow seed 0 `train_result_0.csv` or `stage_ledger.json` exists
  yet; train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=2`, `PENDING=27`.
- GPU snapshot: GPU6 reports `509 MiB`, `3%` utilization, `41C`; metrics are
  still advancing, so this is recorded as in progress rather than failed.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 01:41 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `05:32:15`) in train
  stage.
- Current child `PID 4087668` is still the only child of the driver and is
  running XJTU Rectified Flow seed 0 train (`Dl+`, elapsed `00:21:57`,
  approximately `89.4%` CPU, sampled in `wait_on_page_bit_common`).
- Latest Rectified Flow seed 0 metrics are `epoch=5`, `step=63404`; metrics
  file mtime is `2026-06-11 01:41:26 CST`.
- Latest Rectified Flow seed 0 checkpoint is
  `T_generativerectified_flow_11_011944/iter_0/model-epoch=04-val_loss=1.4670.ckpt`.
- No Rectified Flow seed 0 `train_result_0.csv` or `stage_ledger.json` exists
  yet; train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=2`, `PENDING=27`.
- GPU snapshot: GPU6 reports `509 MiB`, `3%` utilization, `41C`; the newly
  written checkpoint and metrics advance are stronger evidence that the row is
  still in progress.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 01:42 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `05:33:27`) in train
  stage.
- Current child `PID 4087668` is still the only child of the driver and is
  running XJTU Rectified Flow seed 0 train (`Sl+`/`Rl+` samples, elapsed
  `00:23:09`, approximately `89.1%` CPU).
- Latest Rectified Flow seed 0 metrics are `epoch=5`, `step=67901`; metrics
  file mtime is `2026-06-11 01:42:30 CST`.
- Latest Rectified Flow seed 0 checkpoint remains
  `T_generativerectified_flow_11_011944/iter_0/model-epoch=04-val_loss=1.4670.ckpt`.
- No Rectified Flow seed 0 `train_result_0.csv` or `stage_ledger.json` exists
  yet; train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=2`, `PENDING=27`.
- GPU snapshot: GPU6 reports `509 MiB`, `3%` utilization, `41C`; continued
  metrics advancement is the primary evidence that this row is still in
  progress.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 01:43 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `05:34:28`) in train
  stage.
- Current child `PID 4087668` is still the only child of the driver and is
  running XJTU Rectified Flow seed 0 train (`Rl+`, elapsed `00:24:10`,
  approximately `88.7%` CPU).
- Latest Rectified Flow seed 0 metrics are `epoch=5`, `step=72834`; metrics
  file mtime is `2026-06-11 01:43:39 CST`.
- Latest Rectified Flow seed 0 checkpoint remains
  `T_generativerectified_flow_11_011944/iter_0/model-epoch=04-val_loss=1.4670.ckpt`.
- No Rectified Flow seed 0 `train_result_0.csv` or `stage_ledger.json` exists
  yet; train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=2`, `PENDING=27`.
- GPU snapshot: GPU6 reports `509 MiB`, `3%` utilization, `41C`; process and
  metrics evidence still indicate an active train row.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 01:44 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `05:35:28`) in train
  stage.
- Current child `PID 4087668` is still the only child of the driver and is
  running XJTU Rectified Flow seed 0 train (`Rl+`, elapsed `00:25:10`,
  approximately `88.6%` CPU).
- Latest Rectified Flow seed 0 metrics are `epoch=5`, `step=75353`; metrics
  file mtime is `2026-06-11 01:44:17 CST`.
- Latest Rectified Flow seed 0 checkpoint remains
  `T_generativerectified_flow_11_011944/iter_0/model-epoch=04-val_loss=1.4670.ckpt`.
- No Rectified Flow seed 0 `train_result_0.csv` or `stage_ledger.json` exists
  yet; train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=2`, `PENDING=27`.
- GPU snapshot: GPU6 reports `509 MiB`, `4%` utilization, `41C`; process and
  metrics evidence still indicate an active train row.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 01:46 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `05:37:35`) in train
  stage.
- Current child `PID 4087668` is still the only child of the driver and is
  running XJTU Rectified Flow seed 0 train (`Sl+`, elapsed `00:27:17`,
  approximately `88.8%` CPU).
- A short 01:45 sample showed `wait_on_page_bit_common` and no newer metrics
  than `2026-06-11 01:44:17 CST`; a follow-up read at 01:46 showed renewed
  metrics writes, so this is recorded as transient I/O wait rather than a
  stable stall.
- Latest Rectified Flow seed 0 metrics are `epoch=6`, `step=80301`; metrics
  file mtime is `2026-06-11 01:46:45 CST`.
- Latest Rectified Flow seed 0 checkpoint is
  `T_generativerectified_flow_11_011944/iter_0/model-epoch=05-val_loss=1.4610.ckpt`.
- No Rectified Flow seed 0 `train_result_0.csv` or `stage_ledger.json` exists
  yet; train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=2`, `PENDING=27`.
- GPU snapshot: GPU6 reports `509 MiB`, `4%` utilization, `41C`; continued
  process and metrics evidence still indicate an active train row.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 01:47 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `05:38:41`) in train
  stage.
- Current child `PID 4087668` is still the only child of the driver and is
  running XJTU Rectified Flow seed 0 train (`Rl+`, elapsed `00:28:23`,
  approximately `88.4%` CPU).
- Latest Rectified Flow seed 0 metrics are `epoch=6`, `step=85000`; metrics
  file mtime is `2026-06-11 01:47:52 CST`.
- Latest Rectified Flow seed 0 checkpoint remains
  `T_generativerectified_flow_11_011944/iter_0/model-epoch=05-val_loss=1.4610.ckpt`.
- No Rectified Flow seed 0 `train_result_0.csv` or `stage_ledger.json` exists
  yet; train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=2`, `PENDING=27`.
- GPU snapshot: GPU6 reports `509 MiB`, `4%` utilization, `41C`; process and
  metrics evidence still indicate an active train row.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 01:48 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `05:39:46`) in train
  stage.
- Current child `PID 4087668` is still the only child of the driver and is
  running XJTU Rectified Flow seed 0 train (`Rl+`, elapsed `00:29:28`,
  approximately `88.3%` CPU).
- Latest Rectified Flow seed 0 metrics are `epoch=6`, `step=87912`; metrics
  file mtime is `2026-06-11 01:48:33 CST`.
- Latest Rectified Flow seed 0 checkpoint remains
  `T_generativerectified_flow_11_011944/iter_0/model-epoch=05-val_loss=1.4610.ckpt`.
- No Rectified Flow seed 0 `train_result_0.csv` or `stage_ledger.json` exists
  yet; train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=2`, `PENDING=27`.
- GPU snapshot: GPU6 reports `509 MiB`, `4%` utilization, `41C`; process and
  metrics evidence still indicate an active train row.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 01:50 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `05:40:57`) in train
  stage.
- Current child `PID 4087668` is still the only child of the driver and is
  running XJTU Rectified Flow seed 0 train (`Rl+`/`Dl+` samples, elapsed
  `00:30:39`, approximately `88.4%` CPU).
- Latest Rectified Flow seed 0 metrics are `epoch=7`, `step=88562`; metrics
  file mtime is `2026-06-11 01:50:03 CST`.
- Latest Rectified Flow seed 0 checkpoint is
  `T_generativerectified_flow_11_011944/iter_0/model-epoch=06-val_loss=1.4562.ckpt`.
- The train config sets `trainer.num_epochs: 50` and
  `trainer.early_stopping: true`; current metrics therefore show active
  progress, not near-completion evidence.
- No Rectified Flow seed 0 `train_result_0.csv` or `stage_ledger.json` exists
  yet; train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=2`, `PENDING=27`.
- GPU snapshot: GPU6 reports `509 MiB`, `2%` utilization, `41C`; process and
  metrics evidence still indicate an active train row.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 01:51 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `05:42:09`) in train
  stage.
- Current child `PID 4087668` is still the only child of the driver and is
  running XJTU Rectified Flow seed 0 train (`Rl+`/`Sl+` samples, elapsed
  `00:31:51`, approximately `88.1%` CPU).
- Latest Rectified Flow seed 0 metrics are `epoch=7`, `step=93833`; metrics
  file mtime is `2026-06-11 01:51:21 CST`.
- Latest Rectified Flow seed 0 checkpoint remains
  `T_generativerectified_flow_11_011944/iter_0/model-epoch=06-val_loss=1.4562.ckpt`.
- No Rectified Flow seed 0 `train_result_0.csv` or `stage_ledger.json` exists
  yet; train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=2`, `PENDING=27`.
- GPU snapshot: GPU6 reports `509 MiB`, `4%` utilization, `41C`; process and
  metrics evidence still indicate an active train row.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 01:52 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `05:43:35`) in train
  stage.
- Current child `PID 4087668` is still the only child of the driver and is
  running XJTU Rectified Flow seed 0 train (`Rl+`, elapsed `00:33:17`,
  approximately `87.8%` CPU).
- Latest Rectified Flow seed 0 metrics are `epoch=7`, `step=99908`; metrics
  file mtime is `2026-06-11 01:52:44 CST`.
- Latest Rectified Flow seed 0 checkpoint remains
  `T_generativerectified_flow_11_011944/iter_0/model-epoch=06-val_loss=1.4562.ckpt`.
- No Rectified Flow seed 0 `train_result_0.csv` or `stage_ledger.json` exists
  yet; train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=2`, `PENDING=27`.
- GPU snapshot: GPU6 reports `509 MiB`, `4%` utilization, `41C`; process and
  metrics evidence still indicate an active train row.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 01:53 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `05:44:38`) in train
  stage.
- Current child `PID 4087668` is still the only child of the driver and is
  running XJTU Rectified Flow seed 0 train (`Rl+`, elapsed `00:34:20`,
  approximately `88.0%` CPU).
- Latest Rectified Flow seed 0 metrics are `epoch=7`, `step=100471`; metrics
  file mtime is `2026-06-11 01:52:52 CST`.
- Latest Rectified Flow seed 0 checkpoint remains
  `T_generativerectified_flow_11_011944/iter_0/model-epoch=06-val_loss=1.4562.ckpt`.
- No Rectified Flow seed 0 `train_result_0.csv` or `stage_ledger.json` exists
  yet; train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=2`, `PENDING=27`.
- GPU snapshot: GPU6 reports `509 MiB`, `4%` utilization, `41C`; process and
  metrics evidence still indicate an active train row.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 01:54 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `05:45:44`) in train
  stage.
- Current child `PID 4087668` is still the only child of the driver and is
  running XJTU Rectified Flow seed 0 train (`Rl+`/`Dl+` samples, elapsed
  `00:35:26`, approximately `87.8%` CPU).
- Latest Rectified Flow seed 0 metrics are `epoch=8`, `step=103235`; metrics
  file mtime is `2026-06-11 01:54:54 CST`.
- Latest Rectified Flow seed 0 checkpoint remains
  `T_generativerectified_flow_11_011944/iter_0/model-epoch=06-val_loss=1.4562.ckpt`.
- No Rectified Flow seed 0 `train_result_0.csv` or `stage_ledger.json` exists
  yet; train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=2`, `PENDING=27`.
- GPU snapshot: GPU6 reports `509 MiB`, `3%` utilization, `41C`; recent metrics
  writes and the official child process indicate an active train row despite a
  sampled I/O wait state.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 01:56 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `05:46:49`) in train
  stage.
- Current child `PID 4087668` is still the only child of the driver and is
  running XJTU Rectified Flow seed 0 train (`Sl+`/`Rl+` samples, elapsed
  `00:36:31`, approximately `87.6%` CPU).
- Latest Rectified Flow seed 0 metrics are `epoch=8`, `step=107797`; metrics
  file mtime is `2026-06-11 01:56:00 CST`.
- Latest Rectified Flow seed 0 checkpoint remains
  `T_generativerectified_flow_11_011944/iter_0/model-epoch=06-val_loss=1.4562.ckpt`.
- No Rectified Flow seed 0 `train_result_0.csv` or `stage_ledger.json` exists
  yet; train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=2`, `PENDING=27`.
- GPU snapshot: GPU6 reports `509 MiB`, `3%` utilization, `41C`; process and
  metrics evidence still indicate an active train row.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 01:57 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `05:47:51`) in train
  stage.
- Current child `PID 4087668` is still the only child of the driver and is
  running XJTU Rectified Flow seed 0 train (`Rl+`/`Dl+` samples, elapsed
  `00:37:33`, approximately `87.3%` CPU).
- Latest Rectified Flow seed 0 metrics are `epoch=8`, `step=112073`; metrics
  file mtime is `2026-06-11 01:57:02 CST`.
- Latest Rectified Flow seed 0 checkpoint remains
  `T_generativerectified_flow_11_011944/iter_0/model-epoch=06-val_loss=1.4562.ckpt`.
- No Rectified Flow seed 0 `train_result_0.csv` or `stage_ledger.json` exists
  yet; train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=2`, `PENDING=27`.
- GPU snapshot: GPU6 reports `509 MiB`, `0%` utilization, `41C`; recent metrics
  writes and the official child process indicate an active train row despite a
  sampled I/O wait state.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 01:58 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `05:48:53`) in train
  stage.
- Current child `PID 4087668` is still the only child of the driver and is
  running XJTU Rectified Flow seed 0 train (`Rl+`, elapsed `00:38:35`,
  approximately `87.5%` CPU).
- Latest Rectified Flow seed 0 metrics are `epoch=8`, `step=113030`; metrics
  file mtime is `2026-06-11 01:57:15 CST`.
- Latest Rectified Flow seed 0 checkpoint remains
  `T_generativerectified_flow_11_011944/iter_0/model-epoch=06-val_loss=1.4562.ckpt`.
- No Rectified Flow seed 0 `train_result_0.csv` or `stage_ledger.json` exists
  yet; train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=2`, `PENDING=27`.
- GPU snapshot: GPU6 reports `509 MiB`, `5%` utilization, `41C`; process and
  metrics evidence still indicate an active train row.
- Monitoring note: until completion artifacts appear or a larger milestone is
  reached, further progress recording should use lower-frequency snapshots to
  avoid minute-level duplicate evidence.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 01:59 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `05:49:59`) in train
  stage.
- Current child `PID 4087668` is still the only child of the driver and is
  running XJTU Rectified Flow seed 0 train (`Rl+`, elapsed `00:39:41`,
  approximately `87.4%` CPU).
- Latest Rectified Flow seed 0 metrics are `epoch=9`, `step=115323`; metrics
  file mtime is `2026-06-11 01:59:10 CST`.
- Latest Rectified Flow seed 0 checkpoint remains
  `T_generativerectified_flow_11_011944/iter_0/model-epoch=06-val_loss=1.4562.ckpt`.
- No Rectified Flow seed 0 `train_result_0.csv` or `stage_ledger.json` exists
  yet; train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=2`, `PENDING=27`.
- GPU snapshot: GPU6 reports `509 MiB`, `2%` utilization, `41C`; process and
  metrics evidence still indicate an active train row.
- This is recorded as a cross-epoch milestone; subsequent updates should wait
  for completion artifacts, a new checkpoint, or another larger milestone.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 02:03 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `05:54:07`) in train
  stage.
- Current child `PID 4087668` is still the only child of the driver and is
  running XJTU Rectified Flow seed 0 train (`Rl+`/`Sl+` samples, elapsed
  `00:43:49`, approximately `87.2%` CPU).
- Latest Rectified Flow seed 0 metrics are `epoch=10`, `step=126723`;
  metrics file mtime is `2026-06-11 02:03:18 CST`.
- Latest Rectified Flow seed 0 checkpoint is
  `T_generativerectified_flow_11_011944/iter_0/model-epoch=09-val_loss=1.4561.ckpt`.
- No Rectified Flow seed 0 `train_result_0.csv` or `stage_ledger.json` exists
  yet; train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=2`, `PENDING=27`.
- GPU snapshot: GPU6 reports `509 MiB`, `2%` utilization, `41C`; process,
  metrics, and checkpoint evidence still indicate an active train row.
- This is recorded as a cross-epoch and new-checkpoint milestone; subsequent
  updates should wait for completion artifacts, a new checkpoint, or another
  larger milestone.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 02:07 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `05:58:42`) in train
  stage.
- Current child `PID 4087668` is still the only child of the driver and is
  running XJTU Rectified Flow seed 0 train (`Rl+`/`Dl+` samples, elapsed
  `00:48:24`, approximately `86.8%` CPU).
- Latest Rectified Flow seed 0 metrics are `epoch=11`, `step=139596`;
  metrics file mtime is `2026-06-11 02:07:52 CST`.
- Latest Rectified Flow seed 0 checkpoint is
  `T_generativerectified_flow_11_011944/iter_0/model-epoch=10-val_loss=1.4533.ckpt`.
- No Rectified Flow seed 0 `train_result_0.csv` or `stage_ledger.json` exists
  yet; train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=2`, `PENDING=27`.
- GPU snapshot: GPU6 reports `509 MiB`, `3%` utilization, `41C`; process,
  metrics, and checkpoint evidence still indicate an active train row.
- This is recorded as a cross-epoch and new-checkpoint milestone; subsequent
  updates should wait for completion artifacts, a new checkpoint, or another
  larger milestone.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 02:12 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `06:03:28`) in train
  stage.
- Current child `PID 4087668` is still the only child of the driver and is
  running XJTU Rectified Flow seed 0 train (`Rl+`/`Dl+` samples, elapsed
  `00:53:10`, approximately `86.4%` CPU).
- Latest Rectified Flow seed 0 metrics are `epoch=12`, `step=152996`;
  metrics file mtime is `2026-06-11 02:12:38 CST`.
- Latest Rectified Flow seed 0 checkpoint remains
  `T_generativerectified_flow_11_011944/iter_0/model-epoch=10-val_loss=1.4533.ckpt`.
- No Rectified Flow seed 0 `train_result_0.csv` or `stage_ledger.json` exists
  yet; train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=2`, `PENDING=27`.
- GPU snapshot: GPU6 reports `509 MiB`, `4%` utilization, `41C`; process and
  metrics evidence still indicate an active train row.
- This is recorded as a cross-epoch milestone without a new checkpoint;
  subsequent updates should wait for completion artifacts, a new checkpoint,
  or another larger milestone.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 02:17 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `06:08:20`) in train
  stage.
- Current child `PID 4087668` is still the only child of the driver and is
  running XJTU Rectified Flow seed 0 train (`Rl+`, elapsed `00:58:02`,
  approximately `86.1%` CPU).
- Latest Rectified Flow seed 0 metrics are `epoch=13`, `step=167268`;
  metrics file mtime is `2026-06-11 02:17:29 CST`.
- Latest Rectified Flow seed 0 checkpoint is
  `T_generativerectified_flow_11_011944/iter_0/model-epoch=12-val_loss=1.4511.ckpt`.
- No Rectified Flow seed 0 `train_result_0.csv` or `stage_ledger.json` exists
  yet; train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=2`, `PENDING=27`.
- GPU snapshot: GPU6 reports `509 MiB`, `2%` utilization, `41C`; process,
  metrics, and checkpoint evidence still indicate an active train row.
- This is recorded as a cross-epoch and new-checkpoint milestone; subsequent
  updates should wait for completion artifacts, a new checkpoint, or another
  larger milestone.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 02:21 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `06:12:24`) in train
  stage.
- Current child `PID 4087668` is still the only child of the driver and is
  running XJTU Rectified Flow seed 0 train (`Sl+`/`Dl+` samples, elapsed
  `01:02:06`, approximately `85.9%` CPU).
- Latest Rectified Flow seed 0 metrics are `epoch=14`, `step=178063`;
  metrics file mtime is `2026-06-11 02:21:34 CST`.
- Latest Rectified Flow seed 0 checkpoint is
  `T_generativerectified_flow_11_011944/iter_0/model-epoch=13-val_loss=1.4493.ckpt`.
- No Rectified Flow seed 0 `train_result_0.csv` or `stage_ledger.json` exists
  yet; train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=2`, `PENDING=27`.
- GPU snapshot: GPU6 reports `509 MiB`, `4%` utilization, `41C`; process,
  metrics, and checkpoint evidence still indicate an active train row.
- This is recorded as a cross-epoch and new-checkpoint milestone; subsequent
  updates should wait for completion artifacts, a new checkpoint, or another
  larger milestone.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 02:25 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `06:16:39`) in train
  stage.
- Current child `PID 4087668` is still the only child of the driver and is
  running XJTU Rectified Flow seed 0 train (`Rl+`/`Sl+` samples, elapsed
  `01:06:21`, approximately `85.8%` CPU).
- Latest Rectified Flow seed 0 metrics are `epoch=15`, `step=189746`;
  metrics file mtime is `2026-06-11 02:25:47 CST`.
- Latest Rectified Flow seed 0 checkpoint remains
  `T_generativerectified_flow_11_011944/iter_0/model-epoch=13-val_loss=1.4493.ckpt`.
- No Rectified Flow seed 0 `train_result_0.csv` or `stage_ledger.json` exists
  yet; train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=2`, `PENDING=27`.
- GPU snapshot: GPU6 reports `509 MiB`, `4%` utilization, `41C`; process and
  metrics evidence still indicate an active train row.
- This is recorded as a cross-epoch milestone without a new checkpoint;
  subsequent updates should wait for completion artifacts, a new checkpoint,
  or another larger milestone.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 02:33 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `06:24:26`) in train
  stage.
- Current child `PID 4087668` is still the only child of the driver and is
  running XJTU Rectified Flow seed 0 train (`Rl+`, elapsed `01:14:08`,
  approximately `86.2%` CPU).
- Latest Rectified Flow seed 0 metrics are `epoch=16`, `step=213502`;
  metrics file mtime is `2026-06-11 02:32:31 CST`.
- Latest Rectified Flow seed 0 checkpoint is
  `T_generativerectified_flow_11_011944/iter_0/model-epoch=15-val_loss=1.4493.ckpt`.
- No Rectified Flow seed 0 `train_result_0.csv` or `stage_ledger.json` exists
  yet; train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=2`, `PENDING=27`.
- GPU snapshot: GPU6 reports `509 MiB`, `4%` utilization, `41C`; process,
  metrics, and checkpoint evidence still indicate an active train row.
- This is recorded as a cross-epoch and new-checkpoint milestone; subsequent
  updates should wait for completion artifacts, a new checkpoint, or another
  larger milestone.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 02:38 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `06:29:27`) in train
  stage.
- Current child `PID 4087668` is still the only child of the driver and is
  running XJTU Rectified Flow seed 0 train (`Rl+`/`Sl+` samples, elapsed
  `01:19:09`, approximately `86.4%` CPU).
- Latest Rectified Flow seed 0 metrics are `epoch=18`, `step=226923`;
  metrics file mtime is `2026-06-11 02:38:37 CST`.
- Latest Rectified Flow seed 0 checkpoint is
  `T_generativerectified_flow_11_011944/iter_0/model-epoch=16-val_loss=1.4461.ckpt`.
- No Rectified Flow seed 0 `train_result_0.csv` or `stage_ledger.json` exists
  yet; train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=2`, `PENDING=27`.
- GPU snapshot: GPU6 reports `509 MiB`, `4%` utilization, `41C`; process,
  metrics, and checkpoint evidence still indicate an active train row.
- This is recorded as a cross-epoch and new-checkpoint milestone; subsequent
  updates should wait for completion artifacts, a new checkpoint, or another
  larger milestone.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 02:48 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `06:38:56`) in train
  stage.
- Current child `PID 4087668` is still the only child of the driver and is
  running XJTU Rectified Flow seed 0 train (`Rl+`, elapsed `01:28:38`,
  approximately `86.8%` CPU).
- Latest Rectified Flow seed 0 metrics are `epoch=20`, `step=255850`;
  metrics file mtime is `2026-06-11 02:48:06 CST`.
- Latest Rectified Flow seed 0 checkpoint remains
  `T_generativerectified_flow_11_011944/iter_0/model-epoch=16-val_loss=1.4461.ckpt`.
- No Rectified Flow seed 0 `train_result_0.csv` or `stage_ledger.json` exists
  yet; train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=2`, `PENDING=27`.
- GPU snapshot: GPU6 reports `509 MiB`, `3%` utilization, `41C`; process and
  metrics evidence still indicate an active train row.
- This is recorded as a round-epoch cross-epoch milestone without a new
  checkpoint; subsequent updates should wait for completion artifacts, a new
  checkpoint, or another larger milestone.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 02:59 CST:
- XJTU Rectified Flow seed 0 train completed and wrote official completion
  artifacts at `2026-06-11 02:55:41 CST`.
- Seed 0 train result:
  `runs/RM_002_XJTU/rectified_flow_grid/seed_0/train/metadata.xlsx/M_phm_dit1d/T_generativerectified_flow_11_011944/iter_0/train_result_0.csv`.
- Seed 0 stage ledger:
  `runs/RM_002_XJTU/rectified_flow_grid/seed_0/stage_ledger.json`.
- Seed 0 ledger `stages.train` points to checkpoint
  `T_generativerectified_flow_11_011944/iter_0/model-epoch=16-val_loss=1.4461.ckpt`
  and the same train result path above.
- Seed 0 train result row reports `train_completed=True`,
  `train_wall_clock_sec=5716.814129904029`, `parameter_count=19522`, and
  `post_train_test_loss_ran=0.0`.
- Status helper now reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=3`, `PENDING=26`.
- Parent `PID 3870186` remains running (`S+`, elapsed `06:50:24`) in train
  stage.
- The driver has advanced to XJTU Rectified Flow seed 1 train with child
  `PID 4160041` (`Rl+`, elapsed `00:04:03`, approximately `96.3%` CPU).
- Current seed 1 run directory is
  `runs/RM_002_XJTU/rectified_flow_grid/seed_1/train/metadata.xlsx/M_phm_dit1d/T_generativerectified_flow_11_025547/iter_0`.
- Seed 1 metrics are present at `epoch=0`, `step=12558`; no seed 1 checkpoint,
  `train_result_0.csv`, or `stage_ledger.json` exists yet.
- GPU snapshot: GPU6 reports `509 MiB`, `4%` utilization, `40C`; process and
  metrics evidence indicate the next official train row is active.
- This is a train-stage completion for one row only. It is not a complete
  train/sample/eval/paperpack chain, and sample/eval/paperpack must not start
  until the full train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 03:06 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `06:56:57`) in train
  stage.
- Current child `PID 4160041` is still the only child of the driver and is
  running XJTU Rectified Flow seed 1 train (`Rl+`, elapsed `00:10:36`,
  approximately `93.2%` CPU).
- Latest Rectified Flow seed 1 metrics are `epoch=2`, `step=30526`; metrics
  file mtime is `2026-06-11 03:06:06 CST`.
- Latest Rectified Flow seed 1 checkpoint is
  `T_generativerectified_flow_11_025547/iter_0/model-epoch=01-val_loss=1.5158.ckpt`.
- No Rectified Flow seed 1 `train_result_0.csv` or `stage_ledger.json` exists
  yet; train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=3`, `PENDING=26`.
- GPU snapshot: GPU6 reports `509 MiB`, `4%` utilization, `40C`; process,
  metrics, and checkpoint evidence indicate the official seed 1 train row is
  active.
- This is recorded as a seed 1 new-checkpoint milestone. Subsequent updates
  should wait for completion artifacts, a newer checkpoint, or another larger
  milestone.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 03:12 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `07:02:50`) in train
  stage.
- Current child `PID 4160041` is still the only child of the driver and is
  running XJTU Rectified Flow seed 1 train (`Rl+`, elapsed `00:16:29`,
  approximately `92.4%` CPU).
- Latest Rectified Flow seed 1 metrics are `epoch=3`, `step=50235`; metrics
  file mtime is `2026-06-11 03:11:58 CST`.
- Latest Rectified Flow seed 1 checkpoint is
  `T_generativerectified_flow_11_025547/iter_0/model-epoch=02-val_loss=1.4823.ckpt`.
- No Rectified Flow seed 1 `train_result_0.csv` or `stage_ledger.json` exists
  yet; train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=3`, `PENDING=26`.
- GPU snapshot: GPU6 reports `509 MiB`, `3%` utilization, `40C`; process,
  metrics, and checkpoint evidence indicate the official seed 1 train row is
  active.
- This is recorded as a seed 1 new-checkpoint milestone. Subsequent updates
  should wait for completion artifacts, a newer checkpoint, or another larger
  milestone.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 03:17 CST:
- Parent `PID 3870186` remains running (`R+`, elapsed `07:08:47`) in train
  stage.
- Current child `PID 4160041` is still the only child of the driver and is
  running XJTU Rectified Flow seed 1 train (`Rl+`, elapsed `00:22:26`,
  approximately `92.2%` CPU).
- Latest Rectified Flow seed 1 metrics are `epoch=5`, `step=63120`; metrics
  file mtime is `2026-06-11 03:17:52 CST`.
- Latest Rectified Flow seed 1 checkpoint is
  `T_generativerectified_flow_11_025547/iter_0/model-epoch=04-val_loss=1.4651.ckpt`.
- No Rectified Flow seed 1 `train_result_0.csv` or `stage_ledger.json` exists
  yet; train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=3`, `PENDING=26`.
- GPU snapshot: GPU6 reports `509 MiB`, `3%` utilization, `40C`; process,
  metrics, and checkpoint evidence indicate the official seed 1 train row is
  active.
- This is recorded as a seed 1 new-checkpoint milestone. Subsequent updates
  should wait for completion artifacts, a newer checkpoint, or another larger
  milestone.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 03:23 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `07:14:36`) in train
  stage.
- Current child `PID 4160041` is still the only child of the driver and is
  running XJTU Rectified Flow seed 1 train (`Dl+` samples on write/page-cache
  paths, elapsed `00:28:15`, approximately `91.7%` CPU).
- Latest Rectified Flow seed 1 metrics are `epoch=6`, `step=82742`; metrics
  file mtime is `2026-06-11 03:23:47 CST`.
- Latest Rectified Flow seed 1 checkpoint is
  `T_generativerectified_flow_11_025547/iter_0/model-epoch=05-val_loss=1.4624.ckpt`.
- No Rectified Flow seed 1 `train_result_0.csv` or `stage_ledger.json` exists
  yet; train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=3`, `PENDING=26`.
- GPU snapshot: GPU6 reports `509 MiB`, `1%` utilization, `40C`; the newly
  written checkpoint plus metrics evidence indicate the official seed 1 train
  row is still active.
- This is recorded as a seed 1 new-checkpoint milestone. Subsequent updates
  should wait for completion artifacts, a newer checkpoint, or another larger
  milestone.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 03:26 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `07:17:27`) in train
  stage.
- Current child `PID 4160041` is still the only child of the driver and is
  running XJTU Rectified Flow seed 1 train (`Sl+`, elapsed `00:31:06`,
  approximately `91.3%` CPU).
- Latest Rectified Flow seed 1 metrics are `epoch=7`, `step=88505`; metrics
  file mtime is `2026-06-11 03:26:37 CST`.
- Latest Rectified Flow seed 1 checkpoint is
  `T_generativerectified_flow_11_025547/iter_0/model-epoch=06-val_loss=1.4568.ckpt`.
- No Rectified Flow seed 1 `train_result_0.csv` or `stage_ledger.json` exists
  yet; train stage for this row remains incomplete.
- GPU snapshot: GPU6 reports `509 MiB`, `4%` utilization, `40C`; process,
  metrics, and checkpoint evidence indicate the official seed 1 train row is
  active.
- This is recorded as a seed 1 new-checkpoint milestone after the previous
  write-state sample; subsequent updates should wait for completion artifacts,
  a newer checkpoint, or another larger milestone.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 03:37 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `07:28:50`) in train
  stage.
- Current child `PID 4160041` is still the only child of the driver and is
  running XJTU Rectified Flow seed 1 train (`Rl+`/`Sl+` samples, elapsed
  `00:42:29`, approximately `91.0%` CPU).
- Latest Rectified Flow seed 1 metrics are `epoch=9`, `step=125239`; metrics
  file mtime is `2026-06-11 03:37:58 CST`.
- Latest Rectified Flow seed 1 checkpoint is
  `T_generativerectified_flow_11_025547/iter_0/model-epoch=08-val_loss=1.4521.ckpt`,
  improving on the previously recorded `val_loss=1.4568` checkpoint.
- No Rectified Flow seed 1 `train_result_0.csv` or `stage_ledger.json` exists
  yet; train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=3`, `PENDING=26`.
- GPU snapshot: GPU6 reports `509 MiB`, `3%` utilization, `40C`; process,
  metrics, and checkpoint evidence indicate the official seed 1 train row is
  active.
- This is recorded as a seed 1 best-checkpoint update. Subsequent updates
  should wait for completion artifacts, a substantially newer best checkpoint,
  or another larger milestone.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 03:48 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `07:39:45`) in train
  stage.
- Current child `PID 4160041` is still the only child of the driver and is
  running XJTU Rectified Flow seed 1 train (`Rl+`/`Dl+` samples, elapsed
  `00:53:24`, approximately `91.2%` CPU).
- Latest Rectified Flow seed 1 metrics are `epoch=12`, `step=154210`;
  metrics file mtime is `2026-06-11 03:48:55 CST`.
- Latest Rectified Flow seed 1 checkpoint is
  `T_generativerectified_flow_11_025547/iter_0/model-epoch=11-val_loss=1.4463.ckpt`,
  improving on the previously recorded `val_loss=1.4521` checkpoint.
- No Rectified Flow seed 1 `train_result_0.csv` or `stage_ledger.json` exists
  yet; train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=3`, `PENDING=26`.
- GPU snapshot: GPU6 reports `509 MiB`, `2%` utilization, `40C`; process,
  metrics, and checkpoint evidence indicate the official seed 1 train row is
  active.
- This is recorded as a seed 1 round-epoch/best-checkpoint milestone.
  Subsequent updates should wait for completion artifacts, a substantially
  newer best checkpoint, or another larger milestone.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 04:23 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `08:13:58`) in train
  stage.
- Current child `PID 4160041` is still the only child of the driver and is
  running XJTU Rectified Flow seed 1 train (`Sl+`/`Rl+` samples, elapsed
  `01:27:37`, approximately `91.2%` CPU).
- Latest Rectified Flow seed 1 metrics are `epoch=20`, `step=251726`;
  metrics file mtime is `2026-06-11 04:23:03 CST`.
- Latest Rectified Flow seed 1 checkpoint is
  `T_generativerectified_flow_11_025547/iter_0/model-epoch=19-val_loss=1.4441.ckpt`,
  improving on the previously recorded `val_loss=1.4463` checkpoint.
- No Rectified Flow seed 1 `train_result_0.csv` or `stage_ledger.json` exists
  yet; train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=3`, `PENDING=26`.
- GPU snapshot: GPU6 reports `509 MiB`, `4%` utilization, `40C`; process,
  metrics, and checkpoint evidence indicate the official seed 1 train row is
  active.
- This is recorded as a seed 1 round-epoch/best-checkpoint milestone.
  Subsequent updates should wait for completion artifacts, a substantially
  newer best checkpoint, or another larger milestone.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 05:02 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `08:53:43`) in train
  stage.
- Current child `PID 4160041` is still the only child of the driver and is
  running XJTU Rectified Flow seed 1 train (`Rl+`, elapsed `02:07:22`,
  approximately `91.4%` CPU).
- Latest Rectified Flow seed 1 metrics are `epoch=29`, `step=376769`;
  metrics file mtime is `2026-06-11 05:02:10 CST`.
- Latest Rectified Flow seed 1 checkpoint is
  `T_generativerectified_flow_11_025547/iter_0/model-epoch=26-val_loss=1.4399.ckpt`,
  improving on the previously recorded `val_loss=1.4441` checkpoint.
- No Rectified Flow seed 1 `train_result_0.csv` or `stage_ledger.json` exists
  yet; train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=3`, `PENDING=26`.
- GPU snapshot: GPU6 reports `509 MiB`, `4%` utilization, `40C`; process,
  metrics, and checkpoint evidence indicate the official seed 1 train row is
  active.
- This is recorded as a seed 1 best-checkpoint milestone. Subsequent updates
  should wait for completion artifacts, a round epoch milestone, or another
  substantially newer best checkpoint.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 05:06 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `08:57:36`) in train
  stage.
- Current child `PID 4160041` is still the only child of the driver and is
  running XJTU Rectified Flow seed 1 train (`Rl+`, elapsed `02:11:15`,
  approximately `91.4%` CPU).
- Latest Rectified Flow seed 1 metrics are `epoch=30`, `step=389328`;
  metrics file mtime is `2026-06-11 05:06:11 CST`.
- Latest Rectified Flow seed 1 checkpoint remains
  `T_generativerectified_flow_11_025547/iter_0/model-epoch=26-val_loss=1.4399.ckpt`.
- No Rectified Flow seed 1 `train_result_0.csv` or `stage_ledger.json` exists
  yet; train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=3`, `PENDING=26`.
- GPU snapshot: GPU6 reports `509 MiB`, `3%` utilization, `40C`; process and
  metrics evidence indicate the official seed 1 train row is active.
- This is recorded as a seed 1 round-epoch milestone. Subsequent updates
  should wait for completion artifacts, another round epoch milestone, or a
  substantially newer best checkpoint.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 05:17 CST:
- XJTU Rectified Flow seed 1 train completed and wrote official completion
  artifacts at `2026-06-11 05:11:29 CST`.
- Seed 1 train result:
  `runs/RM_002_XJTU/rectified_flow_grid/seed_1/train/metadata.xlsx/M_phm_dit1d/T_generativerectified_flow_11_025547/iter_0/train_result_0.csv`.
- Seed 1 stage ledger:
  `runs/RM_002_XJTU/rectified_flow_grid/seed_1/stage_ledger.json`.
- Seed 1 ledger `stages.train` points to checkpoint
  `T_generativerectified_flow_11_025547/iter_0/model-epoch=26-val_loss=1.4399.ckpt`
  and the same train result path above.
- Seed 1 train result row reports `train_completed=True`,
  `train_wall_clock_sec=8102.370414295001`, `parameter_count=19522`, and
  `post_train_test_loss_ran=0.0`.
- Status helper now reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=4`, `PENDING=25`.
- Parent `PID 3870186` remains running (`S+`, elapsed `09:08:43`) in train
  stage.
- The driver has advanced to XJTU DDPM seed 0 train with child `PID 48103`
  (`Rl+`, elapsed `00:06:33`, approximately `93.9%` CPU).
- Current DDPM seed 0 run directory is
  `runs/RM_002_XJTU/ddpm_train_distribution/seed_0/train/metadata.xlsx/M_mamba1d_backbone/T_generativeddpm_epsilon_11_051137/iter_0`.
- DDPM seed 0 metrics are present at `epoch=1`, `step=25117`; latest
  checkpoint is
  `T_generativeddpm_epsilon_11_051137/iter_0/model-epoch=00-val_loss=0.2640.ckpt`.
- GPU snapshot: GPU6 reports `509 MiB`, `4%` utilization, `40C`; process,
  metrics, and checkpoint evidence indicate the next official train row is
  active.
- This is a train-stage completion for one row only. It is not a complete
  train/sample/eval/paperpack chain, and sample/eval/paperpack must not start
  until the full train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 05:47 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `09:37:57`) in train
  stage.
- Current child `PID 48103` is still the only child of the driver and is
  running XJTU DDPM seed 0 train (`Rl+`/`Sl+` samples, elapsed `00:35:47`,
  approximately `92.0%` CPU).
- Latest DDPM seed 0 metrics are `epoch=10`, `step=131122`; metrics file
  mtime is `2026-06-11 05:47:06 CST`.
- Latest DDPM seed 0 checkpoint is
  `T_generativeddpm_epsilon_11_051137/iter_0/model-epoch=09-val_loss=0.2557.ckpt`.
- No DDPM seed 0 `train_result_0.csv` or `stage_ledger.json` exists yet;
  train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=4`, `PENDING=25`.
- GPU snapshot: GPU6 reports `509 MiB`, `3%` utilization, `40C`; process,
  metrics, and checkpoint evidence indicate the official DDPM seed 0 train row
  is active.
- This is recorded as a DDPM seed 0 round-epoch milestone. Subsequent updates
  should wait for completion artifacts, another round epoch milestone, or a
  substantially newer best checkpoint.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 06:21 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `10:12:14`) in train
  stage.
- Current child `PID 48103` is still the only child of the driver and is
  running XJTU DDPM seed 0 train (`Rl+`/`Dl+` samples, elapsed `01:10:04`,
  approximately `91.7%` CPU).
- Latest DDPM seed 0 metrics are `epoch=20`, `step=257705`; metrics file
  mtime is `2026-06-11 06:21:22 CST`.
- Latest DDPM seed 0 checkpoint is
  `T_generativeddpm_epsilon_11_051137/iter_0/model-epoch=19-val_loss=0.2530.ckpt`.
- No DDPM seed 0 `train_result_0.csv` or `stage_ledger.json` exists yet;
  train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=4`, `PENDING=25`.
- GPU snapshot: GPU6 reports `509 MiB`, `3%` utilization, `40C`; process,
  metrics, and checkpoint evidence indicate the official DDPM seed 0 train row
  is active.
- This is recorded as a DDPM seed 0 round-epoch milestone. Subsequent updates
  should wait for completion artifacts, another round epoch milestone, or a
  substantially newer best checkpoint.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 06:42 CST:
- XJTU DDPM seed 0 train completed and wrote official completion artifacts at
  `2026-06-11 06:37:20 CST`.
- DDPM seed 0 train result:
  `runs/RM_002_XJTU/ddpm_train_distribution/seed_0/train/metadata.xlsx/M_mamba1d_backbone/T_generativeddpm_epsilon_11_051137/iter_0/train_result_0.csv`.
- DDPM seed 0 stage ledger:
  `runs/RM_002_XJTU/ddpm_train_distribution/seed_0/stage_ledger.json`.
- DDPM seed 0 ledger `stages.train` points to checkpoint
  `T_generativeddpm_epsilon_11_051137/iter_0/model-epoch=19-val_loss=0.2530.ckpt`
  and the same train result path above.
- DDPM seed 0 train result row reports `train_completed=True`,
  `train_wall_clock_sec=5101.880791144038`, `parameter_count=3090`, and
  `post_train_test_loss_ran=0.0`.
- Status helper now reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=5`, `PENDING=24`.
- Parent `PID 3870186` remains running (`S+`, elapsed `10:34:05`) in train
  stage.
- The driver has advanced to XJTU DDPM seed 1 train with child `PID 100069`
  (`Rl+`, elapsed `00:06:04`, approximately `94.0%` CPU).
- Current DDPM seed 1 run directory is
  `runs/RM_002_XJTU/ddpm_train_distribution/seed_1/train/metadata.xlsx/M_mamba1d_backbone/T_generativeddpm_epsilon_11_063727/iter_0`.
- DDPM seed 1 metrics are present at `epoch=1`, `step=24646`; latest
  checkpoint is
  `T_generativeddpm_epsilon_11_063727/iter_0/model-epoch=00-val_loss=0.2711.ckpt`.
- GPU snapshot: GPU6 reports `509 MiB`, `3%` utilization, `40C`; process,
  metrics, and checkpoint evidence indicate the next official train row is
  active.
- This is a train-stage completion for one row only. It is not a complete
  train/sample/eval/paperpack chain, and sample/eval/paperpack must not start
  until the full train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 07:12 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `11:03:46`) in train
  stage.
- Current child `PID 100069` is still the only child of the driver and is
  running XJTU DDPM seed 1 train (`Rl+`/`Dl+` samples, elapsed `00:35:45`,
  approximately `91.8%` CPU).
- Latest DDPM seed 1 metrics are `epoch=10`, `step=130482`; metrics file
  mtime is `2026-06-11 07:12:44 CST`.
- Latest DDPM seed 1 checkpoint is
  `T_generativeddpm_epsilon_11_063727/iter_0/model-epoch=07-val_loss=0.2543.ckpt`.
- No DDPM seed 1 `train_result_0.csv` or `stage_ledger.json` exists yet;
  train stage for this row remains incomplete.
- Status helper remains `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=5`, `PENDING=24`.
- GPU snapshot: GPU6 reports `509 MiB`, `3%` utilization, `41C`; process,
  metrics, and checkpoint evidence indicate the official DDPM seed 1 train row
  is active.
- This is recorded as a DDPM seed 1 round-epoch milestone. Subsequent updates
  should wait for completion artifacts, another round epoch milestone, or a
  substantially newer best checkpoint.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 07:36 CST:
- XJTU DDPM seed 1 train completed and wrote official completion artifacts at
  `2026-06-11 07:35:51 CST`.
- DDPM seed 1 train result:
  `runs/RM_002_XJTU/ddpm_train_distribution/seed_1/train/metadata.xlsx/M_mamba1d_backbone/T_generativeddpm_epsilon_11_063727/iter_0/train_result_0.csv`.
- DDPM seed 1 stage ledger:
  `runs/RM_002_XJTU/ddpm_train_distribution/seed_1/stage_ledger.json`.
- DDPM seed 1 ledger `stages.train` points to checkpoint
  `T_generativeddpm_epsilon_11_063727/iter_0/model-epoch=11-val_loss=0.2529.ckpt`
  and the same train result path above.
- DDPM seed 1 train result row reports `train_completed=True`,
  `train_wall_clock_sec=3462.688279768976`, `parameter_count=3090`, and
  `post_train_test_loss_ran=0.0`.
- Status helper now reports `COMPLETE_CHAIN=6`, `PARTIAL_STAGE_LEDGER=6`,
  `PENDING=24`.
- Parent `PID 3870186` remains running (`S+`, elapsed `11:27:22`) in train
  stage.
- The driver has advanced to FEMTO CFM seed 0 train with child `PID 137510`
  (`Dl+`, elapsed `00:50`, approximately `72.7%` CPU).
- Current FEMTO CFM seed 0 run directory is
  `runs/RM_003_FEMTO/cfm_grid/seed_0/train/metadata.xlsx/M_phm_unet1d/T_generativeconditional_flow_matching_11_073558`.
- At this snapshot no FEMTO CFM seed 0 `metrics.csv`, checkpoint,
  `train_result_0.csv`, or `stage_ledger.json` exists yet; the new train row
  has only just started.
- GPU snapshot: GPU6 reports `11 MiB`, `0%` utilization, `35C`; process state
  indicates an early data/loading or filesystem phase before metrics output.
- This completes all six XJTU train rows as train-only stage ledgers. It is not
  a complete train/sample/eval/paperpack chain, and sample/eval/paperpack must
  not start until the full train queue completes and ledger evidence is
  reviewed.

Updated monitor snapshot at 2026-06-11 07:51 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `11:42:39`) in train
  stage.
- Current child `PID 137510` is still the only child of the driver and is
  running FEMTO CFM seed 0 train (`Rl+`, elapsed `00:16:07`,
  approximately `90.5%` CPU).
- Latest FEMTO CFM seed 0 metrics are `epoch=1`, `step=49741`; metrics file
  mtime is `2026-06-11 07:50:05 CST`.
- Latest FEMTO CFM seed 0 checkpoint is
  `T_generativeconditional_flow_matching_11_073558/iter_0/model-epoch=00-val_loss=1.4348.ckpt`.
- No FEMTO CFM seed 0 `train_result_0.csv` or `stage_ledger.json` exists yet;
  train stage for this row remains incomplete.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=6`, `PENDING=23`.
- GPU snapshot: GPU6 reports `513 MiB`, `6%` utilization, `41C`; process,
  metrics, and checkpoint evidence indicate the official FEMTO CFM seed 0
  train row is active.
- This is recorded as a FEMTO CFM seed 0 first-checkpoint milestone.
  Subsequent updates should wait for completion artifacts, another round epoch
  milestone, or a substantially newer best checkpoint.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 08:03 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `11:54:14`) in train
  stage.
- Current child `PID 137510` is still the only child of the driver and is
  running FEMTO CFM seed 0 train (`Sl+`/`Dl+` samples, elapsed `00:27:42`,
  approximately `91.3%` CPU).
- Latest FEMTO CFM seed 0 metrics are `epoch=3`, `step=91084`; metrics file
  mtime is `2026-06-11 08:03:18 CST`.
- Latest FEMTO CFM seed 0 checkpoint is
  `T_generativeconditional_flow_matching_11_073558/iter_0/model-epoch=02-val_loss=1.4150.ckpt`,
  improving on the previously recorded `val_loss=1.4348` checkpoint.
- No FEMTO CFM seed 0 `train_result_0.csv` or `stage_ledger.json` exists yet;
  train stage for this row remains incomplete.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=6`, `PENDING=23`.
- GPU snapshot: GPU6 reports `513 MiB`, `4%` utilization, `41C`; process,
  metrics, and checkpoint evidence indicate the official FEMTO CFM seed 0
  train row is active.
- This is recorded as a FEMTO CFM seed 0 best-checkpoint milestone.
  Subsequent updates should wait for completion artifacts, another round epoch
  milestone, or a substantially newer best checkpoint.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 08:55 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `12:46:10`) in train
  stage.
- Current child `PID 137510` is still the only child of the driver and is
  running FEMTO CFM seed 0 train (`Rl+`, elapsed `01:19:38`,
  approximately `91.8%` CPU).
- Latest FEMTO CFM seed 0 metrics are `epoch=10`, `step=264188`; metrics file
  mtime is `2026-06-11 08:55:11 CST`.
- Latest FEMTO CFM seed 0 checkpoint is
  `T_generativeconditional_flow_matching_11_073558/iter_0/model-epoch=09-val_loss=1.4009.ckpt`,
  improving on the previously recorded `val_loss=1.4150` checkpoint.
- No FEMTO CFM seed 0 `train_result_0.csv` or `stage_ledger.json` exists yet;
  train stage for this row remains incomplete.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=6`, `PENDING=23`.
- GPU snapshot: GPU6 reports `513 MiB`, `5%` utilization, `41C`; process,
  metrics, and checkpoint evidence indicate the official FEMTO CFM seed 0
  train row is active.
- This is recorded as a FEMTO CFM seed 0 round-epoch/best-checkpoint milestone.
  Subsequent updates should wait for completion artifacts, another round epoch
  milestone, or a substantially newer best checkpoint.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 10:08 CST:
- Parent `PID 3870186` remains running (`R+`, elapsed `13:59:48`) in train
  stage.
- Current child `PID 137510` is still the only child of the driver and is
  running FEMTO CFM seed 0 train (`Rl+`, elapsed `02:33:16`,
  approximately `92.1%` CPU).
- Latest FEMTO CFM seed 0 metrics are `epoch=20`, `step=508352`; metrics file
  mtime is `2026-06-11 10:08:52 CST`.
- Latest FEMTO CFM seed 0 checkpoint remains
  `T_generativeconditional_flow_matching_11_073558/iter_0/model-epoch=15-val_loss=1.3953.ckpt`.
- No FEMTO CFM seed 0 `train_result_0.csv` or `stage_ledger.json` exists yet;
  train stage for this row remains incomplete.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=6`, `PENDING=23`.
- GPU snapshot: GPU6 reports `513 MiB`, `4%` utilization, `41C`; process,
  metrics, and checkpoint evidence indicate the official FEMTO CFM seed 0
  train row is active.
- This is recorded as a FEMTO CFM seed 0 round-epoch milestone. Subsequent
  updates should wait for completion artifacts, another round epoch milestone,
  or a substantially newer best checkpoint.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 10:53 CST:
- FEMTO CFM seed 0 train completed. The row now has
  `PARTIAL_STAGE_LEDGER` status with completed stage `train`; this is not a
  complete train/sample/eval/paperpack chain.
- Completion artifacts:
  - `runs/RM_003_FEMTO/cfm_grid/seed_0/stage_ledger.json` mtime
    `2026-06-11 10:52:25 CST`; ledger `created_at`/`updated_at`
    `2026-06-11T02:53:03Z`.
  - `runs/RM_003_FEMTO/cfm_grid/seed_0/train/metadata.xlsx/M_phm_unet1d/T_generativeconditional_flow_matching_11_073558/iter_0/train_result_0.csv`
    reports `train_completed=True`, `train_wall_clock_sec=11669.435398249014`,
    `parameter_count=42290`, and `post_train_test_loss_ran=0.0`.
  - Ledger train checkpoint path is
    `T_generativeconditional_flow_matching_11_073558/iter_0/model-epoch=20-val_loss=1.3931.ckpt`.
- Status helper now reports `COMPLETE_CHAIN=6`, `PARTIAL_STAGE_LEDGER=7`,
  `PENDING=23`.
- The official driver advanced to FEMTO CFM seed 1 train. New child
  `PID 260416` is running with `environment.seed=1` (`Rl+`, elapsed `00:01:02`,
  approximately `112%` CPU) under parent `PID 3870186` (`S+`, elapsed
  `14:44:09`).
- At this snapshot, FEMTO CFM seed 1 has no train artifacts yet; the repaired
  ledger still lists `RM_003_FEMTO,cfm_grid,1` as `PENDING`.
- GPU snapshot: GPU6 is idle (`11 MiB`, `0%`, `35C`) after seed 0 completion;
  the new seed 1 child is still in early startup.
- The run remains in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 11:04 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `14:55:22`) in train
  stage.
- Current child `PID 260416` is the only child of the driver and is running
  FEMTO CFM seed 1 train (`Dl+`, elapsed `00:12:15`, approximately `94.5%`
  CPU).
- Latest FEMTO CFM seed 1 metrics are `epoch=1`, `step=45793`; metrics file
  mtime is `2026-06-11 11:04:30 CST`.
- First FEMTO CFM seed 1 checkpoint is
  `T_generativeconditional_flow_matching_11_105233/iter_0/model-epoch=00-val_loss=1.4343.ckpt`.
- No FEMTO CFM seed 1 `train_result_0.csv` or `stage_ledger.json` exists yet;
  train stage for this row remains incomplete.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=7`, `PENDING=22`.
- GPU snapshot: GPU6 reports `513 MiB`, `4%` utilization, `41C`; process,
  metrics, and checkpoint evidence indicate the official FEMTO CFM seed 1
  train row is active.
- This is recorded as a FEMTO CFM seed 1 first-checkpoint milestone.
  Subsequent updates should wait for completion artifacts, a round epoch
  milestone, or a substantially newer best checkpoint.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 11:15 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `15:06:27`) in train
  stage.
- Current child `PID 260416` is still the only child of the driver and is
  running FEMTO CFM seed 1 train (`Rl+`/`Dl+` samples, elapsed `00:23:20`,
  approximately `93.4%` CPU).
- Latest FEMTO CFM seed 1 metrics are `epoch=3`, `step=75849`; metrics file
  mtime is `2026-06-11 11:15:35 CST`.
- Latest FEMTO CFM seed 1 checkpoint is
  `T_generativeconditional_flow_matching_11_105233/iter_0/model-epoch=02-val_loss=1.4183.ckpt`,
  improving on the previously recorded `val_loss=1.4343` checkpoint.
- No FEMTO CFM seed 1 `train_result_0.csv` or `stage_ledger.json` exists yet;
  train stage for this row remains incomplete.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=7`, `PENDING=22`.
- GPU snapshot: GPU6 reports `513 MiB`, `3%` utilization, `41C`; process,
  metrics, and checkpoint evidence indicate the official FEMTO CFM seed 1
  train row is active.
- This is recorded as a FEMTO CFM seed 1 best-checkpoint milestone. Subsequent
  updates should wait for completion artifacts, a round epoch milestone, or a
  substantially newer best checkpoint.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 12:10 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `16:01:51`) in train
  stage.
- Current child `PID 260416` is still the only child of the driver and is
  running FEMTO CFM seed 1 train (`Rl+`, elapsed `01:18:44`, approximately
  `93.1%` CPU).
- Latest FEMTO CFM seed 1 metrics are `epoch=10`, `step=273580`; metrics file
  mtime is `2026-06-11 12:09:46 CST`.
- Latest FEMTO CFM seed 1 checkpoint is
  `T_generativeconditional_flow_matching_11_105233/iter_0/model-epoch=09-val_loss=1.4022.ckpt`.
- No FEMTO CFM seed 1 `train_result_0.csv` or `stage_ledger.json` exists yet;
  train stage for this row remains incomplete.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=7`, `PENDING=22`.
- GPU snapshot: GPU6 reports `513 MiB`, `5%` utilization, `42C`; process,
  metrics, and checkpoint evidence indicate the official FEMTO CFM seed 1
  train row is active.
- This is recorded as a FEMTO CFM seed 1 round-epoch milestone. Subsequent
  updates should wait for completion artifacts, another round epoch milestone,
  or a substantially newer best checkpoint.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 13:18 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `17:09:31`) in train
  stage.
- Current child `PID 260416` is still the only child of the driver and is
  running FEMTO CFM seed 1 train (`Sl+`/`Rl+` samples, elapsed `02:26:24`,
  approximately `93.2%` CPU).
- Latest FEMTO CFM seed 1 metrics are `epoch=20`, `step=509848`; metrics file
  mtime is `2026-06-11 13:18:40 CST`.
- Latest FEMTO CFM seed 1 checkpoint is
  `T_generativeconditional_flow_matching_11_105233/iter_0/model-epoch=19-val_loss=1.3948.ckpt`.
- No FEMTO CFM seed 1 `train_result_0.csv` or `stage_ledger.json` exists yet;
  train stage for this row remains incomplete.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=7`, `PENDING=22`.
- GPU snapshot: GPU6 reports `513 MiB`, `4%` utilization, `42C`; process,
  metrics, and checkpoint evidence indicate the official FEMTO CFM seed 1
  train row is active.
- This is recorded as a FEMTO CFM seed 1 round-epoch milestone. Subsequent
  updates should wait for completion artifacts, another round epoch milestone,
  or a substantially newer best checkpoint.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 14:28 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `18:19:01`) in train
  stage.
- Current child `PID 260416` is still the only child of the driver and is
  running FEMTO CFM seed 1 train (`Rl+`, elapsed `03:35:54`, approximately
  `93.2%` CPU).
- Latest FEMTO CFM seed 1 metrics are `epoch=30`, `step=747612`; metrics file
  mtime is `2026-06-11 14:28:09 CST`.
- Latest FEMTO CFM seed 1 checkpoint remains
  `T_generativeconditional_flow_matching_11_105233/iter_0/model-epoch=25-val_loss=1.3916.ckpt`.
- No FEMTO CFM seed 1 `train_result_0.csv` or `stage_ledger.json` exists yet;
  train stage for this row remains incomplete.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=7`, `PENDING=22`.
- GPU snapshot: GPU6 reports `513 MiB`, `5%` utilization, `42C`; process,
  metrics, and checkpoint evidence indicate the official FEMTO CFM seed 1
  train row is active.
- This is recorded as a FEMTO CFM seed 1 round-epoch milestone. Subsequent
  updates should wait for completion artifacts, another round epoch milestone,
  or a substantially newer best checkpoint.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 14:40 CST:
- FEMTO CFM seed 1 train completed. The row now has
  `PARTIAL_STAGE_LEDGER` status with completed stage `train`; this is not a
  complete train/sample/eval/paperpack chain.
- Completion artifacts:
  - `runs/RM_003_FEMTO/cfm_grid/seed_1/stage_ledger.json` mtime
    `2026-06-11 14:35:06 CST`; ledger `created_at`/`updated_at`
    `2026-06-11T06:39:22Z`.
  - `runs/RM_003_FEMTO/cfm_grid/seed_1/train/metadata.xlsx/M_phm_unet1d/T_generativeconditional_flow_matching_11_105233/iter_0/train_result_0.csv`
    reports `train_completed=True`, `train_wall_clock_sec=13283.659325939952`,
    `parameter_count=42290`, and `post_train_test_loss_ran=0.0`.
  - Ledger train checkpoint path is
    `T_generativeconditional_flow_matching_11_105233/iter_0/model-epoch=25-val_loss=1.3916.ckpt`.
- Status helper now reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=8`, `PENDING=21`.
- The official driver advanced to FEMTO rectified-flow seed 0 train. New child
  `PID 378964` is running
  `configs/paper/phm_generative/rectified_flow_train_grid_seed0.yaml` with
  `environment.seed=0` (`Sl+`, elapsed `00:04:54`, approximately `98.2%` CPU)
  under parent `PID 3870186` (`S+`, elapsed `18:30:42`).
- FEMTO rectified-flow seed 0 is currently `IN_PROGRESS_NO_LEDGER`; latest
  status-ledger metrics row is `epoch=0`, `step=15044`.
- GPU snapshot: GPU6 reports `509 MiB`, `4%` utilization, `42C`; process and
  metrics evidence indicate the official FEMTO rectified-flow seed 0 train row
  is active.
- The run remains in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 14:51 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `18:41:54`) in train
  stage.
- Current child `PID 378964` is the only child of the driver and is running
  FEMTO rectified-flow seed 0 train (`Rl+`, elapsed `00:16:06`,
  approximately `94.3%` CPU).
- Latest FEMTO rectified-flow seed 0 metrics are `epoch=1`, `step=49741`;
  metrics file mtime is `2026-06-11 14:49:12 CST`.
- First FEMTO rectified-flow seed 0 checkpoint is
  `T_generativerectified_flow_11_143513/iter_0/model-epoch=00-val_loss=1.5282.ckpt`.
- No FEMTO rectified-flow seed 0 `train_result_0.csv` or
  `stage_ledger.json` exists yet; train stage for this row remains incomplete.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=8`, `PENDING=21`.
- GPU snapshot: GPU6 reports `509 MiB`, `4%` utilization, `42C`; process,
  metrics, and checkpoint evidence indicate the official FEMTO rectified-flow
  seed 0 train row is active.
- This is recorded as a FEMTO rectified-flow seed 0 first-checkpoint
  milestone. Subsequent updates should wait for completion artifacts, a round
  epoch milestone, or a substantially newer best checkpoint.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 15:03 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `18:54:16`) in train
  stage.
- Current child `PID 378964` is still the only child of the driver and is
  running FEMTO rectified-flow seed 0 train (`Rl+`/`Sl+` samples, elapsed
  `00:28:28`, approximately `94.1%` CPU).
- Latest FEMTO rectified-flow seed 0 metrics are `epoch=3`, `step=93189`;
  metrics file mtime is `2026-06-11 15:03:17 CST`.
- Latest FEMTO rectified-flow seed 0 checkpoint is
  `T_generativerectified_flow_11_143513/iter_0/model-epoch=02-val_loss=1.4818.ckpt`,
  improving on the previously recorded `val_loss=1.5282` checkpoint.
- No FEMTO rectified-flow seed 0 `train_result_0.csv` or
  `stage_ledger.json` exists yet; train stage for this row remains incomplete.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=8`, `PENDING=21`.
- GPU snapshot: GPU6 reports `509 MiB`, `4%` utilization, `42C`; process,
  metrics, and checkpoint evidence indicate the official FEMTO rectified-flow
  seed 0 train row is active.
- This is recorded as a FEMTO rectified-flow seed 0 best-checkpoint milestone.
  Subsequent updates should wait for completion artifacts, a round epoch
  milestone, or a substantially newer best checkpoint.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 15:54 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `19:45:24`) in train
  stage.
- Current child `PID 378964` is still the only child of the driver and is
  running FEMTO rectified-flow seed 0 train (`Rl+`, elapsed `01:19:36`,
  approximately `93.5%` CPU).
- Latest FEMTO rectified-flow seed 0 metrics are `epoch=10`, `step=251616`;
  metrics file mtime is `2026-06-11 15:54:26 CST`.
- Latest FEMTO rectified-flow seed 0 checkpoint is
  `T_generativerectified_flow_11_143513/iter_0/model-epoch=09-val_loss=1.4638.ckpt`.
- No FEMTO rectified-flow seed 0 `train_result_0.csv` or
  `stage_ledger.json` exists yet; train stage for this row remains incomplete.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=8`, `PENDING=21`.
- GPU snapshot: GPU6 reports `509 MiB`, `4%` utilization, `42C`; process,
  metrics, and checkpoint evidence indicate the official FEMTO rectified-flow
  seed 0 train row is active.
- This is recorded as a FEMTO rectified-flow seed 0 round-epoch milestone.
  Subsequent updates should wait for completion artifacts, another round epoch
  milestone, or a substantially newer best checkpoint.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 17:17 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `21:08:02`) in train
  stage.
- Current child `PID 378964` is still the only child of the driver and is
  running FEMTO rectified-flow seed 0 train (`Sl+`, elapsed `02:42:14`,
  approximately `93.3%` CPU).
- Latest FEMTO rectified-flow seed 0 metrics are `epoch=20`, `step=500269`;
  metrics file mtime is `2026-06-11 17:17:05 CST`.
- Latest FEMTO rectified-flow seed 0 checkpoint remains
  `T_generativerectified_flow_11_143513/iter_0/model-epoch=18-val_loss=1.4563.ckpt`.
- No FEMTO rectified-flow seed 0 `train_result_0.csv` or
  `stage_ledger.json` exists yet; train stage for this row remains incomplete.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=8`, `PENDING=21`.
- GPU snapshot: GPU6 reports `509 MiB`, `4%` utilization, `42C`; process,
  metrics, and checkpoint evidence indicate the official FEMTO rectified-flow
  seed 0 train row is active.
- This is recorded as a FEMTO rectified-flow seed 0 round-epoch milestone.
  Subsequent updates should wait for completion artifacts, another round epoch
  milestone, or a substantially newer best checkpoint.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 18:38 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `22:30:05`) in train
  stage.
- Current child `PID 378964` is still the only child of the driver and is
  running FEMTO rectified-flow seed 0 train (`Rl+`, elapsed `04:04:17`,
  approximately `93.3%` CPU).
- Latest FEMTO rectified-flow seed 0 metrics are `epoch=30`, `step=768464`;
  metrics file mtime is `2026-06-11 18:38:52 CST`.
- Latest FEMTO rectified-flow seed 0 checkpoint remains
  `T_generativerectified_flow_11_143513/iter_0/model-epoch=27-val_loss=1.4516.ckpt`.
- No FEMTO rectified-flow seed 0 `train_result_0.csv` or
  `stage_ledger.json` exists yet; train stage for this row remains incomplete.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=8`, `PENDING=21`.
- GPU snapshot: GPU6 reports `509 MiB`, `4%` utilization, `43C`; process,
  metrics, and checkpoint evidence indicate the official FEMTO rectified-flow
  seed 0 train row is active.
- This is recorded as a FEMTO rectified-flow seed 0 round-epoch milestone.
  Subsequent updates should wait for completion artifacts, another round epoch
  milestone, or a substantially newer best checkpoint.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 19:02 CST:
- FEMTO rectified-flow seed 0 train completed. The row now has
  `PARTIAL_STAGE_LEDGER` status with completed stage `train`; this is not a
  complete train/sample/eval/paperpack chain.
- Completion artifacts:
  - `runs/RM_003_FEMTO/rectified_flow_grid/seed_0/stage_ledger.json` mtime
    `2026-06-11 18:57:30 CST`; ledger `created_at`/`updated_at`
    `2026-06-11T11:01:35Z`.
  - `runs/RM_003_FEMTO/rectified_flow_grid/seed_0/train/metadata.xlsx/M_phm_dit1d/T_generativerectified_flow_11_143513/iter_0/train_result_0.csv`
    reports `train_completed=True`, `train_wall_clock_sec=15669.165808198974`,
    `parameter_count=19522`, and `post_train_test_loss_ran=0.0`.
  - Ledger train checkpoint path is
    `T_generativerectified_flow_11_143513/iter_0/model-epoch=27-val_loss=1.4516.ckpt`.
- Status helper now reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=9`, `PENDING=20`.
- The official driver advanced to FEMTO rectified-flow seed 1 train. New child
  `PID 523644` is running
  `configs/paper/phm_generative/rectified_flow_train_grid_seed0.yaml` with
  `environment.seed=1` (`Sl+`, elapsed `00:04:50`, approximately `97.6%` CPU)
  under parent `PID 3870186` (`S+`, elapsed `22:53:02`).
- FEMTO rectified-flow seed 1 is currently `IN_PROGRESS_NO_LEDGER`; latest
  status-ledger metrics row is `epoch=0`, `step=13711`.
- GPU snapshot: GPU6 reports `509 MiB`, `2%` utilization, `42C`; process and
  metrics evidence indicate the official FEMTO rectified-flow seed 1 train row
  is active.
- The run remains in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 19:13 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `23:04:25`) in train
  stage.
- Current child `PID 523644` is the only child of the driver and is running
  FEMTO rectified-flow seed 1 train (`Sl+`/`Rl+` samples, elapsed `00:16:13`,
  approximately `94.1%` CPU).
- Latest FEMTO rectified-flow seed 1 metrics are `epoch=1`, `step=49741`;
  metrics file mtime is `2026-06-11 19:11:54 CST`.
- First FEMTO rectified-flow seed 1 checkpoint is
  `T_generativerectified_flow_11_185739/iter_0/model-epoch=00-val_loss=1.5318.ckpt`.
- No FEMTO rectified-flow seed 1 `train_result_0.csv` or
  `stage_ledger.json` exists yet; train stage for this row remains incomplete.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=9`, `PENDING=20`.
- GPU snapshot: GPU6 reports `509 MiB`, `4%` utilization, `43C`; process,
  metrics, and checkpoint evidence indicate the official FEMTO rectified-flow
  seed 1 train row is active.
- This is recorded as a FEMTO rectified-flow seed 1 first-checkpoint
  milestone. Subsequent updates should wait for completion artifacts, a round
  epoch milestone, or a substantially newer best checkpoint.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 19:24 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `23:15:47`) in train
  stage.
- Current child `PID 523644` is still the only child of the driver and is
  running FEMTO rectified-flow seed 1 train (`Rl+`, elapsed `00:27:35`,
  approximately `93.9%` CPU).
- Latest FEMTO rectified-flow seed 1 metrics are `epoch=3`, `step=86405`;
  metrics file mtime is `2026-06-11 19:24:41 CST`.
- Latest FEMTO rectified-flow seed 1 checkpoint is
  `T_generativerectified_flow_11_185739/iter_0/model-epoch=02-val_loss=1.4794.ckpt`,
  improving on the previously recorded `val_loss=1.5318` checkpoint.
- No FEMTO rectified-flow seed 1 `train_result_0.csv` or
  `stage_ledger.json` exists yet; train stage for this row remains incomplete.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=9`, `PENDING=20`.
- GPU snapshot: GPU6 reports `509 MiB`, `4%` utilization, `43C`; process,
  metrics, and checkpoint evidence indicate the official FEMTO rectified-flow
  seed 1 train row is active.
- This is recorded as a FEMTO rectified-flow seed 1 best-checkpoint milestone.
  Subsequent updates should wait for completion artifacts, a round epoch
  milestone, or a substantially newer best checkpoint.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 20:20 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `1-00:11:59`) in train
  stage.
- Current child `PID 523644` is still the only child of the driver and is
  running FEMTO rectified-flow seed 1 train (`Rl+`, elapsed `01:23:47`,
  approximately `93.8%` CPU).
- Latest FEMTO rectified-flow seed 1 metrics are `epoch=10`, `step=268797`;
  metrics file mtime is `2026-06-11 20:20:57 CST`.
- Latest FEMTO rectified-flow seed 1 checkpoint is
  `T_generativerectified_flow_11_185739/iter_0/model-epoch=09-val_loss=1.4647.ckpt`.
- No FEMTO rectified-flow seed 1 `train_result_0.csv` or
  `stage_ledger.json` exists yet; train stage for this row remains incomplete.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=9`, `PENDING=20`.
- GPU snapshot: GPU6 reports `509 MiB`, `2%` utilization, `43C`; process,
  metrics, and checkpoint evidence indicate the official FEMTO rectified-flow
  seed 1 train row is active.
- This is recorded as a FEMTO rectified-flow seed 1 round-epoch milestone.
  Subsequent updates should wait for completion artifacts, another round epoch
  milestone, or a substantially newer best checkpoint.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 21:36 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `1-01:27:42`) in train
  stage.
- Current child `PID 523644` is still the only child of the driver and is
  running FEMTO rectified-flow seed 1 train (`Sl+`, elapsed `02:39:30`,
  approximately `93.6%` CPU).
- Latest FEMTO rectified-flow seed 1 metrics are `epoch=20`, `step=502642`;
  metrics file mtime is `2026-06-11 21:36:40 CST`.
- Latest FEMTO rectified-flow seed 1 checkpoint is
  `T_generativerectified_flow_11_185739/iter_0/model-epoch=19-val_loss=1.4563.ckpt`.
- No FEMTO rectified-flow seed 1 `train_result_0.csv` or
  `stage_ledger.json` exists yet; train stage for this row remains incomplete.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=9`, `PENDING=20`.
- GPU snapshot: GPU6 reports `509 MiB`, `4%` utilization, `43C`; process,
  metrics, and checkpoint evidence indicate the official FEMTO rectified-flow
  seed 1 train row is active.
- This is recorded as a FEMTO rectified-flow seed 1 round-epoch milestone.
  Subsequent updates should wait for completion artifacts, another round epoch
  milestone, or a substantially newer best checkpoint.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 22:34 CST:
- FEMTO rectified-flow seed 1 train completed as a train-only
  `PARTIAL_STAGE_LEDGER`.
- Stage ledger:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_003_FEMTO/rectified_flow_grid/seed_1/stage_ledger.json`;
  file mtime is `2026-06-11 22:30:41 CST`, with ledger
  `created_at`/`updated_at` of `2026-06-11T14:34:38Z`.
- Train result:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_003_FEMTO/rectified_flow_grid/seed_1/train/metadata.xlsx/M_phm_dit1d/T_generativerectified_flow_11_185739/iter_0/train_result_0.csv`;
  `train_completed=True`, `train_wall_clock_sec=12714.324233171996`,
  `parameter_count=19522`, `post_train_test_loss_ran=0.0`.
- Ledger checkpoint path is
  `T_generativerectified_flow_11_185739/iter_0/model-epoch=21-val_loss=1.4542.ckpt`.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=10`, `PENDING=19`.
- Parent `PID 3870186` remains running (`S+`, elapsed `1-02:25:30`) in train
  stage. The driver has advanced to FEMTO DDPM train-distribution seed 0:
  child `PID 658438`, `Dl+`, elapsed `00:04:08`, approximately `96.5%` CPU.
- GPU snapshot: GPU1 reports `18839 MiB`, `94%` utilization, `84C`; GPU3
  `18837 MiB`, `95%`, `85C`; GPU4 `17701 MiB`, `85%`, `74C`; GPU5
  `17701 MiB`, `88%`, `83C`; GPU6 remains at `509 MiB`, `3%`, `42C`.
- This is a train-stage completion snapshot only. The full
  `train/sample/eval/paperpack` chain is not complete, and sample/eval/paperpack
  must not start until the train queue completes and ledger evidence is
  reviewed.

Updated monitor snapshot at 2026-06-11 22:47 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `1-02:37:58`) in train
  stage.
- Current child `PID 658438` is the only child of the driver and is running
  FEMTO DDPM train-distribution seed 0 train (`Rl+`, elapsed `00:16:36`,
  approximately `91.9%` CPU).
- Latest FEMTO DDPM train-distribution seed 0 metrics are `epoch=2`,
  `step=55334`; metrics file mtime is `2026-06-11 22:47:07 CST`.
- First FEMTO DDPM train-distribution seed 0 checkpoint is
  `T_generativeddpm_epsilon_11_223048/iter_0/model-epoch=00-val_loss=0.2567.ckpt`.
- No FEMTO DDPM train-distribution seed 0 `train_result_0.csv` or
  `stage_ledger.json` exists yet; train stage for this row remains incomplete.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=10`, `PENDING=19`.
- GPU snapshot: GPU1 reports `18839 MiB`, `94%` utilization, `83C`; GPU3
  `18837 MiB`, `95%`, `85C`; GPU4 `17701 MiB`, `80%`, `74C`; GPU5
  `17701 MiB`, `92%`, `82C`; GPU6 remains at `509 MiB`, `1%`, `42C`.
- This is recorded as the FEMTO DDPM train-distribution seed 0 first-checkpoint
  milestone. Subsequent updates should wait for completion artifacts, a round
  epoch milestone, or a substantially newer best checkpoint.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 22:58 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `1-02:48:56`) in train
  stage.
- Current child `PID 658438` is still the only child of the driver and is
  running FEMTO DDPM train-distribution seed 0 train (`Rl+`, elapsed
  `00:27:34`, approximately `91.7%` CPU).
- Latest FEMTO DDPM train-distribution seed 0 metrics are `epoch=3`,
  `step=99483`; metrics file mtime is `2026-06-11 22:57:44 CST`.
- Latest FEMTO DDPM train-distribution seed 0 checkpoint is
  `T_generativeddpm_epsilon_11_223048/iter_0/model-epoch=02-val_loss=0.2548.ckpt`.
- No FEMTO DDPM train-distribution seed 0 `train_result_0.csv` or
  `stage_ledger.json` exists yet; train stage for this row remains incomplete.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=10`, `PENDING=19`.
- GPU snapshot: GPU1 reports `18839 MiB`, `94%` utilization, `82C`; GPU3
  `18837 MiB`, `96%`, `86C`; GPU4 `17701 MiB`, `82%`, `80C`; GPU5
  `17701 MiB`, `93%`, `85C`; GPU6 remains at `509 MiB`, `3%`, `42C`.
- This is recorded as a FEMTO DDPM train-distribution seed 0 early best
  checkpoint milestone. Subsequent updates should wait for completion artifacts,
  a round epoch milestone, or a substantially newer best checkpoint.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-11 23:42 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `1-03:33:52`) in train
  stage.
- Current child `PID 658438` is still the only child of the driver and is
  running FEMTO DDPM train-distribution seed 0 train (`Dl+`, elapsed
  `01:12:30`, approximately `92.2%` CPU).
- Latest FEMTO DDPM train-distribution seed 0 metrics are `epoch=10`,
  `step=249975`; metrics file mtime is `2026-06-11 23:42:47 CST`.
- Latest FEMTO DDPM train-distribution seed 0 checkpoint is
  `T_generativeddpm_epsilon_11_223048/iter_0/model-epoch=09-val_loss=0.2514.ckpt`.
- No FEMTO DDPM train-distribution seed 0 `train_result_0.csv` or
  `stage_ledger.json` exists yet; train stage for this row remains incomplete.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=10`, `PENDING=19`.
- GPU snapshot: GPU1 reports `18839 MiB`, `48%` utilization, `73C`; GPU3
  `18837 MiB`, `27%`, `74C`; GPU4 `17701 MiB`, `87%`, `71C`; GPU5
  `17721 MiB`, `89%`, `83C`; GPU6 remains at `509 MiB`, `2%`, `42C`.
- This is recorded as a FEMTO DDPM train-distribution seed 0 round-epoch
  milestone. Subsequent updates should wait for completion artifacts, another
  round epoch milestone, or a substantially newer best checkpoint.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-12 00:50 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `1-04:41:55`) in train
  stage.
- Current child `PID 658438` is still the only child of the driver and is
  running FEMTO DDPM train-distribution seed 0 train (`Rl+`, elapsed
  `02:20:33`, approximately `92.3%` CPU).
- Latest FEMTO DDPM train-distribution seed 0 metrics are `epoch=20`,
  `step=499875`; metrics file mtime is `2026-06-12 00:50:57 CST`.
- Latest FEMTO DDPM train-distribution seed 0 checkpoint is
  `T_generativeddpm_epsilon_11_223048/iter_0/model-epoch=18-val_loss=0.2503.ckpt`.
- No FEMTO DDPM train-distribution seed 0 `train_result_0.csv` or
  `stage_ledger.json` exists yet; train stage for this row remains incomplete.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=10`, `PENDING=19`.
- GPU snapshot: GPU1 reports `18839 MiB`, `95%` utilization, `85C`; GPU3
  `18837 MiB`, `95%`, `88C`; GPU6 remains at `509 MiB`, `1%`, `40C`; GPUs
  4 and 5 have dropped to `11 MiB`, `0%`.
- This is recorded as a FEMTO DDPM train-distribution seed 0 round-epoch
  milestone. Subsequent updates should wait for completion artifacts, another
  round epoch milestone, or a substantially newer best checkpoint.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-12 02:02 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `1-05:52:53`) in train
  stage.
- Current child `PID 658438` is still the only child of the driver and is
  running FEMTO DDPM train-distribution seed 0 train (`Rl+`, elapsed
  `03:31:32`, approximately `92.4%` CPU).
- Latest FEMTO DDPM train-distribution seed 0 metrics are `epoch=30`,
  `step=771000`; metrics file mtime is `2026-06-12 02:00:49 CST`.
- Latest FEMTO DDPM train-distribution seed 0 checkpoint is
  `T_generativeddpm_epsilon_11_223048/iter_0/model-epoch=25-val_loss=0.2494.ckpt`.
- No FEMTO DDPM train-distribution seed 0 `train_result_0.csv` or
  `stage_ledger.json` exists yet; train stage for this row remains incomplete.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=10`, `PENDING=19`.
- GPU snapshot: GPU1 reports `18839 MiB`, `96%` utilization, `83C`; GPU3
  `18837 MiB`, `96%`, `85C`; GPU6 remains at `509 MiB`, `3%`, `40C`; GPUs
  4 and 5 remain idle at `11 MiB`, `0%`.
- This is recorded as a FEMTO DDPM train-distribution seed 0 round-epoch
  milestone. Subsequent updates should wait for completion artifacts, another
  round epoch milestone, or a substantially newer best checkpoint.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-12 02:12 CST:
- FEMTO DDPM train-distribution seed 0 train completed as a train-only
  `PARTIAL_STAGE_LEDGER`.
- Stage ledger:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_003_FEMTO/ddpm_train_distribution/seed_0/stage_ledger.json`;
  file mtime is `2026-06-12 02:03:30 CST`, with ledger
  `created_at`/`updated_at` of `2026-06-11T18:13:02Z`.
- Train result:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_003_FEMTO/ddpm_train_distribution/seed_0/train/metadata.xlsx/M_mamba1d_backbone/T_generativeddpm_epsilon_11_223048/iter_0/train_result_0.csv`;
  `train_completed=True`, `train_wall_clock_sec=12695.317004651995`,
  `parameter_count=3090`, `post_train_test_loss_ran=0.0`.
- Ledger checkpoint path is
  `T_generativeddpm_epsilon_11_223048/iter_0/model-epoch=25-val_loss=0.2494.ckpt`.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=11`, `PENDING=18`.
- Parent `PID 3870186` remains running (`S+`, elapsed `1-06:03:54`) in train
  stage. The driver has advanced to FEMTO DDPM train-distribution seed 1:
  child `PID 751704`, `Rl+`, elapsed `00:09:43`, approximately `95.0%` CPU.
- GPU snapshot: GPU1 reports `18839 MiB`, `95%` utilization, `85C`; GPU3
  `18837 MiB`, `96%`, `87C`; GPU6 remains at `509 MiB`, `3%`, `40C`; GPUs
  4 and 5 remain idle at `11 MiB`, `0%`.
- This is a train-stage completion snapshot only. The full
  `train/sample/eval/paperpack` chain is not complete, and sample/eval/paperpack
  must not start until the train queue completes and ledger evidence is
  reviewed.

Updated monitor snapshot at 2026-06-12 02:14 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `1-06:05:10`) in train
  stage.
- Current child `PID 751704` is the only child of the driver and is running
  FEMTO DDPM train-distribution seed 1 train (`Rl+`, elapsed `00:10:59`,
  approximately `93.1%` CPU).
- Latest FEMTO DDPM train-distribution seed 1 metrics are `epoch=1`,
  `step=42804`; metrics file mtime is `2026-06-12 02:14:05 CST`.
- First FEMTO DDPM train-distribution seed 1 checkpoint is
  `T_generativeddpm_epsilon_12_020338/iter_0/model-epoch=00-val_loss=0.2591.ckpt`.
- No FEMTO DDPM train-distribution seed 1 `train_result_0.csv` or
  `stage_ledger.json` exists yet; train stage for this row remains incomplete.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=11`, `PENDING=18`.
- GPU snapshot: GPU1 reports `18839 MiB`, `94%` utilization, `85C`; GPU3
  `18837 MiB`, `96%`, `88C`; GPU6 remains at `509 MiB`, `3%`, `40C`; GPUs
  4 and 5 remain idle at `11 MiB`, `0%`.
- This is recorded as the FEMTO DDPM train-distribution seed 1 first-checkpoint
  milestone. Subsequent updates should wait for completion artifacts, a round
  epoch milestone, or a substantially newer best checkpoint.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-12 02:25 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `1-06:16:06`) in train
  stage.
- Current child `PID 751704` is still the only child of the driver and is
  running FEMTO DDPM train-distribution seed 1 train (`Sl+`, elapsed
  `00:21:55`, approximately `92.3%` CPU).
- Latest FEMTO DDPM train-distribution seed 1 metrics are `epoch=3`,
  `step=75712`; metrics file mtime is `2026-06-12 02:25:14 CST`.
- Latest FEMTO DDPM train-distribution seed 1 checkpoint is
  `T_generativeddpm_epsilon_12_020338/iter_0/model-epoch=01-val_loss=0.2554.ckpt`.
- No FEMTO DDPM train-distribution seed 1 `train_result_0.csv` or
  `stage_ledger.json` exists yet; train stage for this row remains incomplete.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=11`, `PENDING=18`.
- GPU snapshot: GPU1 reports `18839 MiB`, `95%` utilization, `85C`; GPU3
  `18837 MiB`, `96%`, `88C`; GPU6 remains at `509 MiB`, `3%`, `40C`; GPUs
  4 and 5 remain idle at `11 MiB`, `0%`.
- This is recorded as a FEMTO DDPM train-distribution seed 1 early best
  checkpoint milestone. Subsequent updates should wait for completion artifacts,
  a round epoch milestone, or a substantially newer best checkpoint.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-12 03:12 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `1-07:02:57`) in train
  stage.
- Current child `PID 751704` is still the only child of the driver and is
  running FEMTO DDPM train-distribution seed 1 train (`Sl+`, elapsed
  `01:08:46`, approximately `92.9%` CPU).
- Latest FEMTO DDPM train-distribution seed 1 metrics are `epoch=10`,
  `step=260902`; metrics file mtime is `2026-06-12 03:12:07 CST`.
- Latest FEMTO DDPM train-distribution seed 1 checkpoint is
  `T_generativeddpm_epsilon_12_020338/iter_0/model-epoch=08-val_loss=0.2515.ckpt`.
- No FEMTO DDPM train-distribution seed 1 `train_result_0.csv` or
  `stage_ledger.json` exists yet; train stage for this row remains incomplete.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=11`, `PENDING=18`.
- GPU snapshot: GPU1 reports `18839 MiB`, `95%` utilization, `85C`; GPU3
  `18837 MiB`, `95%`, `86C`; GPU6 remains at `509 MiB`, `4%`, `40C`; GPUs
  4 and 5 remain idle at `11 MiB`, `0%`.
- This is recorded as a FEMTO DDPM train-distribution seed 1 round-epoch
  milestone. Subsequent updates should wait for completion artifacts, another
  round epoch milestone, or a substantially newer best checkpoint.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-12 04:16 CST:
- Parent `PID 3870186` remains running (`S+`, elapsed `1-08:07:40`) in train
  stage.
- Current child `PID 751704` is still the only child of the driver and is
  running FEMTO DDPM train-distribution seed 1 train (`Dl+`, elapsed
  `02:13:29`, approximately `92.8%` CPU).
- Latest FEMTO DDPM train-distribution seed 1 metrics are `epoch=20`,
  `step=505000`; metrics file mtime is `2026-06-12 04:16:49 CST`.
- Latest FEMTO DDPM train-distribution seed 1 checkpoint is
  `T_generativeddpm_epsilon_12_020338/iter_0/model-epoch=15-val_loss=0.2510.ckpt`.
- No FEMTO DDPM train-distribution seed 1 `train_result_0.csv` or
  `stage_ledger.json` exists yet; train stage for this row remains incomplete.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=11`, `PENDING=18`.
- GPU snapshot: GPU1 reports `18839 MiB`, `92%` utilization, `83C`; GPU3
  `18837 MiB`, `92%`, `85C`; GPU6 remains at `509 MiB`, `2%`, `40C`; GPUs
  4 and 5 remain idle at `11 MiB`, `0%`.
- This is recorded as a FEMTO DDPM train-distribution seed 1 round-epoch
  milestone. Subsequent updates should wait for completion artifacts, another
  round epoch milestone, or a substantially newer best checkpoint.
- The run is still in the train-stage queue; sample/eval/paperpack must not
  start until the train queue completes and ledger evidence is reviewed.

Updated monitor snapshot at 2026-06-12 04:27 CST:
- FEMTO DDPM train-distribution seed 1 train completed as a train-only
  `PARTIAL_STAGE_LEDGER`. FEMTO now has all six train rows completed as
  train-only ledgers.
- Stage ledger:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_003_FEMTO/ddpm_train_distribution/seed_1/stage_ledger.json`;
  file mtime is `2026-06-12 04:22:25 CST`, with ledger
  `created_at`/`updated_at` of `2026-06-11T20:28:01Z`.
- Train result:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_003_FEMTO/ddpm_train_distribution/seed_1/train/metadata.xlsx/M_mamba1d_backbone/T_generativeddpm_epsilon_12_020338/iter_0/train_result_0.csv`;
  `train_completed=True`, `train_wall_clock_sec=8260.470404354972`,
  `parameter_count=3090`, `post_train_test_loss_ran=0.0`.
- Ledger checkpoint path is
  `T_generativeddpm_epsilon_12_020338/iter_0/model-epoch=15-val_loss=0.2510.ckpt`.
- Status helper reports `COMPLETE_CHAIN=6`, `PARTIAL_STAGE_LEDGER=12`,
  `PENDING=18`.
- Parent `PID 3870186` remains running (`S+`, elapsed `1-08:19:02`) in train
  stage. The driver has advanced to UNSW CFM grid seed 0: child `PID 805158`,
  `Rl+`, elapsed `00:05:55`, approximately `43.2%` CPU.
- GPU snapshot: GPU1 reports `18839 MiB`, `92%` utilization, `85C`; GPU3
  `18837 MiB`, `94%`, `87C`; GPU6 has dropped to `11 MiB`, `0%`.
- This is a train-stage completion snapshot only. The full
  `train/sample/eval/paperpack` chain is not complete, and sample/eval/paperpack
  must not start until the train queue completes and ledger evidence is
  reviewed.

Updated monitor snapshot at 2026-06-12 05:21 CST:
- The official train driver has exited; parent `PID 3870186` is no longer
  present and there is no child process under that PID.
- `execution_summary.csv` was updated at `2026-06-12 05:03:20 CST` and
  contains 19 executed rows. The last row is `RM_008_UNSW / cfm_grid / seed=0 /
  train`, `gpu_id=6`, `returncode=1`, `wall_clock_sec=2451.000778`.
- No UNSW CFM grid seed 0 `metrics.csv`, checkpoint, `train_result_0.csv`, or
  `stage_ledger.json` exists. Only the output directory scaffold exists under
  `T_generativeconditional_flow_matching_12_042233/iter_0`.
- The failure occurred while attaching normalization artifacts:
  `_build_normalization_params -> _to_ncl` raised
  `ValueError: expected channel axis with channels=2 in [N,C,L] or [N,L,C], got
  shape=(2, 128, 6)`.
- Status helper reports `COMPLETE_CHAIN=6`, `PARTIAL_STAGE_LEDGER=12`,
  `PENDING=18`.
- This is a train-stage failure snapshot. The full train queue did not complete,
  and sample/eval/paperpack must not start until the failure is diagnosed,
  fixed, and the missing train evidence is produced.

Fix note at 2026-06-12 05:21 CST:
- Root cause: the six-dataset matrix inherited generative base model defaults
  with `model.in_channels=2` for every dataset, while metadata shows dataset
  channel counts are CWRU/XJTU/FEMTO=`2`, UNSW=`6`, JUST/PU=`3`.
- Fix applied: `configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml`
  now explicitly sets `model.in_channels` in each dataset override.
- Regression guard added:
  `test/generative/test_six_dataset_submission.py::test_repository_six_dataset_matrix_builds_complete_run_plan`
  asserts that train/sample/eval commands carry the expected dataset-specific
  `model.in_channels` override.
- Validation passed:
  `python -m pytest test/generative/test_six_dataset_submission.py::test_repository_six_dataset_matrix_builds_complete_run_plan`,
  `python -m scripts.validate_configs`, `python -m scripts.validate_docs`, and
  `git diff --check`.
- Dry-run check passed: the repaired UNSW CFM grid seed 0 train command contains
  `--override model.in_channels=6`.
- The failed train queue has not been restarted yet in this note. Restart must
  stay train-only with `--skip-existing`; sample/eval/paperpack must remain
  blocked until the train queue completes and ledger evidence is reviewed.

Restart snapshot at 2026-06-12 05:27 CST:
- The failed `execution_summary.csv` was preserved at
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/logs/execution_summary_failed_unsw_cfm_seed0_20260612_050320.csv`
  before restarting, because the runner rewrites `execution_summary.csv`.
- Train-only recovery was started with the same output root and `--skip-existing`:
  `python -m scripts.generative_benchmark_effect --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml --execute --preflight-gpu --stages train --skip-existing --output-dir results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10`.
- Initial sandbox execution failed CUDA preflight with `torch cuda unavailable`;
  the same command was restarted with CUDA-visible permissions.
- New recovery driver `PID 831355` is running. Child `PID 831544` is rerunning
  `RM_008_UNSW / cfm_grid / seed=0 / train` and the command contains
  `--override model.in_channels=6`.
- New UNSW run directory scaffold:
  `T_generativeconditional_flow_matching_12_052618/iter_0`. No metrics,
  checkpoint, `train_result_0.csv`, or train-stage ledger artifact exists yet.
- Current UNSW stage ledger only has runner metadata (`status=PENDING`,
  `stages={}`), so the row is not complete.
- GPU snapshot: GPU1 reports `18839 MiB`, `96%` utilization, `85C`; GPU3
  `18837 MiB`, `94%`, `87C`.

Recovery monitor snapshot at 2026-06-12 05:37 CST:
- Recovery driver `PID 831355` remains running (`Ss`, elapsed `00:11:41`) in
  train-only mode. Child `PID 831544` is rerunning UNSW CFM grid seed 0 with
  `model.in_channels=6` (`Rl`, elapsed `00:11:37`, approximately `75.0%` CPU).
- The repaired UNSW CFM grid seed 0 run has produced its first files under
  `T_generativeconditional_flow_matching_12_052618/iter_0`.
- Latest metrics are `epoch=22`, `step=21665`; metrics file mtime is
  `2026-06-12 05:37:44 CST`.
- Latest checkpoint is
  `T_generativeconditional_flow_matching_12_052618/iter_0/model-epoch=21-val_loss=1.0558.ckpt`.
- No `train_result_0.csv` exists yet, and the stage ledger still has
  `stages={}` with status `IN_PROGRESS_NO_LEDGER`; this row remains incomplete.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=12`, `PENDING=17`.
- GPU snapshot: GPU1 reports `18839 MiB`, `95%` utilization, `85C`; GPU3
  `18837 MiB`, `96%`, `88C`; GPU6 reports `513 MiB`, `0%`, `39C`.
- This is recorded as a recovery first-artifact checkpoint. Subsequent updates
  should wait for completion artifacts, another round epoch milestone, or a
  substantially newer best checkpoint.

Recovery monitor snapshot at 2026-06-12 05:52 CST:
- UNSW CFM grid seed 0 train completed as a train-only `PARTIAL_STAGE_LEDGER`.
- Stage ledger:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_008_UNSW/cfm_grid/seed_0/stage_ledger.json`;
  file mtime is `2026-06-12 05:49:10 CST`, with ledger
  `updated_at` of `2026-06-11T21:52:14Z` and `last_returncode=0`.
- Train result:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_008_UNSW/cfm_grid/seed_0/train/metadata.xlsx/M_phm_unet1d/T_generativeconditional_flow_matching_12_052618/iter_0/train_result_0.csv`;
  `train_completed=True`, `train_wall_clock_sec=1220.239231960033`,
  `parameter_count=43062`, `post_train_test_loss_ran=0.0`.
- Ledger checkpoint path is
  `T_generativeconditional_flow_matching_12_052618/iter_0/model-epoch=48-val_loss=0.9544.ckpt`.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=13`, `PENDING=16`.
- Recovery driver `PID 831355` remains running. The driver has advanced to
  UNSW CFM grid seed 1: child `PID 842641`, `Rl`, elapsed `00:03:17`,
  approximately `78.0%` CPU, with `--override model.in_channels=6`.
- GPU snapshot: GPU1 reports `18839 MiB`, `95%` utilization, `85C`; GPU3
  `18837 MiB`, `91%`, `87C`; GPU6 reports `513 MiB`, `4%`, `39C`.
- This is a train-stage completion snapshot only. The full
  `train/sample/eval/paperpack` chain is not complete, and sample/eval/paperpack
  must not start until the train queue completes and ledger evidence is
  reviewed.

Recovery monitor snapshot at 2026-06-12 05:53 CST:
- Recovery driver `PID 831355` remains running (`Ss`, elapsed `00:27:13`) in
  train-only mode. Child `PID 842641` is running UNSW CFM grid seed 1 with
  `model.in_channels=6` (`Rl`, elapsed `00:04:08`, approximately `77.1%` CPU).
- UNSW CFM grid seed 1 has produced first artifacts under
  `T_generativeconditional_flow_matching_12_054924/iter_0`.
- Latest metrics are `epoch=5`, `step=5651`; metrics file mtime is
  `2026-06-12 05:53:05 CST`.
- Latest checkpoint is
  `T_generativeconditional_flow_matching_12_054924/iter_0/model-epoch=04-val_loss=1.2506.ckpt`.
- No `train_result_0.csv` exists yet; the row remains incomplete.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=13`, `PENDING=16`.
- GPU snapshot: GPU1 reports `18839 MiB`, `96%` utilization, `85C`; GPU3
  `18837 MiB`, `96%`, `87C`; GPU6 reports `513 MiB`, `0%`, `39C`.
- This is recorded as an UNSW CFM grid seed 1 first-artifact checkpoint.
  Subsequent updates should wait for completion artifacts, a round epoch
  milestone, or a substantially newer best checkpoint.

Recovery monitor snapshot at 2026-06-12 06:04 CST:
- Recovery driver `PID 831355` remains running (`Ss`, elapsed `00:38:08`) in
  train-only mode. Child `PID 842641` is still running UNSW CFM grid seed 1 with
  `model.in_channels=6` (`Rl`, elapsed `00:15:03`, approximately `82.1%` CPU).
- Latest UNSW CFM grid seed 1 metrics are `epoch=30`, `step=29201`; metrics
  file mtime is `2026-06-12 06:04:00 CST`.
- Latest checkpoint is
  `T_generativeconditional_flow_matching_12_054924/iter_0/model-epoch=30-val_loss=0.9771.ckpt`.
- No `train_result_0.csv` exists yet; the row remains incomplete.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=13`, `PENDING=16`.
- GPU snapshot: GPU1 reports `18839 MiB`, `96%` utilization, `86C`; GPU3
  `18837 MiB`, `96%`, `87C`; GPU6 reports `513 MiB`, `0%`, `39C`.
- This is recorded as an UNSW CFM grid seed 1 round-epoch milestone. Subsequent
  updates should wait for completion artifacts, another round epoch milestone,
  or a substantially newer best checkpoint.

Recovery monitor snapshot at 2026-06-12 06:15 CST:
- UNSW CFM grid seed 1 train completed as a train-only `PARTIAL_STAGE_LEDGER`.
- Stage ledger:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_008_UNSW/cfm_grid/seed_1/stage_ledger.json`;
  file mtime is `2026-06-12 06:12:13 CST`, with ledger
  `updated_at` of `2026-06-11T22:15:17Z` and `last_returncode=0`.
- Train result:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_008_UNSW/cfm_grid/seed_1/train/metadata.xlsx/M_phm_unet1d/T_generativeconditional_flow_matching_12_054924/iter_0/train_result_0.csv`;
  `train_completed=True`, `train_wall_clock_sec=1286.3342537159915`,
  `parameter_count=43062`, `post_train_test_loss_ran=0.0`.
- Ledger checkpoint path is
  `T_generativeconditional_flow_matching_12_054924/iter_0/model-epoch=47-val_loss=0.9404.ckpt`.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=14`, `PENDING=15`.
- Recovery driver `PID 831355` remains running. The driver has advanced to
  UNSW rectified-flow grid seed 0: child `PID 854202`, `Sl`, elapsed `00:03:02`,
  approximately `93.5%` CPU, with `--override model.in_channels=6`.
- GPU snapshot: GPU1 reports `18839 MiB`, `96%` utilization, `83C`; GPU3
  `18837 MiB`, `95%`, `86C`; GPU6 reports `509 MiB`, `0%`, `39C`.
- This is a train-stage completion snapshot only. The full
  `train/sample/eval/paperpack` chain is not complete, and sample/eval/paperpack
  must not start until the train queue completes and ledger evidence is
  reviewed.

Recovery monitor snapshot at 2026-06-12 06:16 CST:
- Recovery driver `PID 831355` remains running (`Ss`, elapsed `00:50:20`) in
  train-only mode. Child `PID 854202` is running UNSW rectified-flow grid seed 0
  with `model.in_channels=6` (`Sl`, elapsed `00:04:12`, approximately `90.0%`
  CPU).
- UNSW rectified-flow grid seed 0 has produced first artifacts under
  `T_generativerectified_flow_12_061218/iter_0`.
- Latest metrics are `epoch=7`, `step=7198`; metrics file mtime is
  `2026-06-12 06:16:03 CST`.
- Latest checkpoint is
  `T_generativerectified_flow_12_061218/iter_0/model-epoch=05-val_loss=1.5142.ckpt`.
- No `train_result_0.csv` exists yet; the row remains incomplete.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=14`, `PENDING=15`.
- GPU snapshot: GPU1 reports `18839 MiB`, `97%` utilization, `85C`; GPU3
  `18837 MiB`, `96%`, `87C`; GPU6 reports `509 MiB`, `4%`, `39C`.
- This is recorded as an UNSW rectified-flow grid seed 0 first-artifact
  checkpoint. Subsequent updates should wait for completion artifacts, a round
  epoch milestone, or a substantially newer best checkpoint.

Recovery monitor snapshot at 2026-06-12 06:27 CST:
- Recovery driver `PID 831355` remains running (`Ss`, elapsed `01:01:18`) in
  train-only mode. Child `PID 854202` is still running UNSW rectified-flow grid
  seed 0 with `model.in_channels=6` (`Sl`, elapsed `00:15:10`, approximately
  `86.7%` CPU).
- Latest UNSW rectified-flow grid seed 0 metrics are `epoch=31`, `step=30143`;
  metrics file mtime is `2026-06-12 06:27:01 CST`.
- Latest checkpoint is
  `T_generativerectified_flow_12_061218/iter_0/model-epoch=30-val_loss=1.3634.ckpt`.
- No `train_result_0.csv` exists yet; the row remains incomplete.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=14`, `PENDING=15`.
- GPU snapshot: GPU1 reports `18839 MiB`, `96%` utilization, `85C`; GPU3
  `18837 MiB`, `96%`, `87C`; GPU6 reports `509 MiB`, `4%`, `39C`.
- This is recorded as an UNSW rectified-flow grid seed 0 round-epoch milestone.
  Subsequent updates should wait for completion artifacts, another round epoch
  milestone, or a substantially newer best checkpoint.

Recovery monitor snapshot at 2026-06-12 06:35 CST:
- UNSW rectified-flow grid seed 0 train completed as a train-only
  `PARTIAL_STAGE_LEDGER`.
- Stage ledger:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_008_UNSW/rectified_flow_grid/seed_0/stage_ledger.json`;
  file mtime is `2026-06-12 06:29:32 CST`, with ledger
  `updated_at` of `2026-06-11T22:35:08Z` and `last_returncode=0`.
- Train result:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_008_UNSW/rectified_flow_grid/seed_0/train/metadata.xlsx/M_phm_dit1d/T_generativerectified_flow_12_061218/iter_0/train_result_0.csv`;
  `train_completed=True`, `train_wall_clock_sec=984.4574503329932`,
  `parameter_count=21574`, `post_train_test_loss_ran=0.0`.
- Ledger checkpoint path is
  `T_generativerectified_flow_12_061218/iter_0/model-epoch=31-val_loss=1.3594.ckpt`.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=15`, `PENDING=14`.
- Recovery driver `PID 831355` remains running. The driver has advanced to
  UNSW rectified-flow grid seed 1: child `PID 862780`, `Dl`, elapsed `00:05:50`,
  approximately `89.5%` CPU, with `--override model.in_channels=6`.
- GPU snapshot: GPU1 reports `18839 MiB`, `96%` utilization, `85C`; GPU3
  `18837 MiB`, `94%`, `87C`; GPU6 reports `509 MiB`, `3%`, `39C`.
- This is a train-stage completion snapshot only. The full
  `train/sample/eval/paperpack` chain is not complete, and sample/eval/paperpack
  must not start until the train queue completes and ledger evidence is
  reviewed.

Recovery monitor snapshot at 2026-06-12 06:36 CST:
- Recovery driver `PID 831355` remains running (`Ss`, elapsed `01:10:08`) in
  train-only mode. Child `PID 862780` is running UNSW rectified-flow grid seed 1
  with `model.in_channels=6` (`Rl`, elapsed `00:06:40`, approximately `88.1%`
  CPU).
- UNSW rectified-flow grid seed 1 has produced first artifacts under
  `T_generativerectified_flow_12_062938/iter_0`.
- Latest metrics are `epoch=12`, `step=12245`; metrics file mtime is
  `2026-06-12 06:36:00 CST`.
- Latest checkpoint is
  `T_generativerectified_flow_12_062938/iter_0/model-epoch=11-val_loss=1.4280.ckpt`.
- No `train_result_0.csv` exists yet; the row remains incomplete.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=15`, `PENDING=14`.
- GPU snapshot: GPU1 reports `18839 MiB`, `96%` utilization, `84C`; GPU3
  `18837 MiB`, `94%`, `85C`; GPU6 reports `509 MiB`, `0%`, `39C`.
- This is recorded as an UNSW rectified-flow grid seed 1 first-artifact
  checkpoint. Subsequent updates should wait for completion artifacts, a round
  epoch milestone, or a substantially newer best checkpoint.

Recovery monitor snapshot at 2026-06-12 06:43 CST:
- Recovery driver `PID 831355` remains running (`Ss`, elapsed `01:18:03`) in
  train-only mode. Child `PID 862780` is still running UNSW rectified-flow grid
  seed 1 with `model.in_channels=6` (`Dl`, elapsed `00:14:35`, approximately
  `86.1%` CPU).
- Latest UNSW rectified-flow grid seed 1 metrics are `epoch=30`, `step=28406`;
  metrics file mtime is `2026-06-12 06:43:57 CST`.
- Latest checkpoint is
  `T_generativerectified_flow_12_062938/iter_0/model-epoch=28-val_loss=1.3659.ckpt`.
- No `train_result_0.csv` exists yet; the row remains incomplete.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=15`, `PENDING=14`.
- GPU snapshot: GPU1 reports `18839 MiB`, `96%` utilization, `83C`; GPU3
  `18837 MiB`, `95%`, `85C`; GPU6 reports `509 MiB`, `4%`, `39C`.
- This is recorded as an UNSW rectified-flow grid seed 1 round-epoch milestone.
  Subsequent updates should wait for completion artifacts, another round epoch
  milestone, or a substantially newer best checkpoint.

Recovery monitor snapshot at 2026-06-12 06:51 CST:
- UNSW rectified-flow grid seed 1 train completed as a train-only
  `PARTIAL_STAGE_LEDGER`.
- Stage ledger:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_008_UNSW/rectified_flow_grid/seed_1/stage_ledger.json`;
  file mtime is `2026-06-12 06:51:27 CST`, with ledger
  `updated_at` of `2026-06-11T22:52:00Z` and `last_returncode=0`.
- Train result:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_008_UNSW/rectified_flow_grid/seed_1/train/metadata.xlsx/M_phm_dit1d/T_generativerectified_flow_12_062938/iter_0/train_result_0.csv`;
  `train_completed=True`, `train_wall_clock_sec=1259.6426614840166`,
  `parameter_count=21574`, `post_train_test_loss_ran=0.0`.
- Ledger checkpoint path is
  `T_generativerectified_flow_12_062938/iter_0/model-epoch=41-val_loss=1.3471.ckpt`.
- Status helper reports `COMPLETE_CHAIN=6`, `PARTIAL_STAGE_LEDGER=16`,
  `PENDING=14`.
- Recovery driver `PID 831355` remains running. The driver has advanced to
  UNSW DDPM train-distribution seed 0: child `PID 873527`, `Rl`, elapsed
  `00:00:47`, approximately `111%` CPU, with `--override model.in_channels=6`.
- GPU snapshot: GPU1 reports `18839 MiB`, `96%` utilization, `85C`; GPU3
  `18837 MiB`, `94%`, `87C`; GPU6 reports `509 MiB`, `3%`, `37C`.
- This is a train-stage completion snapshot only. The full
  `train/sample/eval/paperpack` chain is not complete, and sample/eval/paperpack
  must not start until the train queue completes and ledger evidence is
  reviewed.

Recovery monitor snapshot at 2026-06-12 06:53 CST:
- Recovery driver `PID 831355` remains running (`Ss`, elapsed `01:27:13`) in
  train-only mode. Child `PID 873527` is running UNSW DDPM train-distribution
  seed 0 with `model.in_channels=6` (`Rl`, elapsed `00:01:51`, approximately
  `94.0%` CPU).
- UNSW DDPM train-distribution seed 0 has produced first artifacts under
  `T_generativeddpm_epsilon_12_065132/iter_0`.
- Latest metrics are `epoch=2`, `step=2344`; metrics file mtime is
  `2026-06-12 06:53:00 CST`.
- First checkpoint is
  `T_generativeddpm_epsilon_12_065132/iter_0/model-epoch=00-val_loss=0.3839.ckpt`.
- No `train_result_0.csv` exists yet; the row remains incomplete.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=16`, `PENDING=13`.
- GPU snapshot: GPU1 reports `18839 MiB`, `96%` utilization, `83C`; GPU3
  `18837 MiB`, `96%`, `86C`; GPU6 reports `509 MiB`, `3%`, `39C`.
- This is recorded as an UNSW DDPM train-distribution seed 0 first-checkpoint
  milestone. Subsequent updates should wait for completion artifacts, a round
  epoch milestone, or a substantially newer best checkpoint.

Recovery monitor snapshot at 2026-06-12 07:04 CST:
- UNSW DDPM train-distribution seed 0 train completed as a train-only
  `PARTIAL_STAGE_LEDGER`.
- Stage ledger:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_008_UNSW/ddpm_train_distribution/seed_0/stage_ledger.json`;
  file mtime is `2026-06-12 07:03:40 CST`, with ledger
  `updated_at` of `2026-06-11T23:04:05Z` and `last_returncode=0`.
- Train result:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_008_UNSW/ddpm_train_distribution/seed_0/train/metadata.xlsx/M_mamba1d_backbone/T_generativeddpm_epsilon_12_065132/iter_0/train_result_0.csv`;
  `train_completed=True`, `train_wall_clock_sec=676.3024906970095`,
  `parameter_count=3350`, `post_train_test_loss_ran=0.0`.
- Ledger checkpoint path is
  `T_generativeddpm_epsilon_12_065132/iter_0/model-epoch=24-val_loss=0.1938.ckpt`.
- Status helper reports `COMPLETE_CHAIN=6`, `PARTIAL_STAGE_LEDGER=17`,
  `PENDING=13`.
- Recovery driver `PID 831355` remains running. The driver has advanced to
  UNSW DDPM train-distribution seed 1: child `PID 879630`, `Rl`, elapsed
  `00:00:43`, approximately `117%` CPU, with `--override model.in_channels=6`.
- GPU snapshot: GPU1 reports `16647 MiB`, `29%` utilization, `69C`; GPU3
  `16665 MiB`, `58%`, `71C`; GPU6 reports `465 MiB`, `0%`, `36C`.
- This is a train-stage completion snapshot only. The full
  `train/sample/eval/paperpack` chain is not complete, and sample/eval/paperpack
  must not start until the train queue completes and ledger evidence is
  reviewed.

Recovery monitor snapshot at 2026-06-12 07:05 CST:
- Recovery driver `PID 831355` remains running (`Ss`, elapsed `01:39:13`) in
  train-only mode. Child `PID 879630` is running UNSW DDPM train-distribution
  seed 1 with `model.in_channels=6` (`Sl`, elapsed `00:01:38`, approximately
  `93.8%` CPU).
- UNSW DDPM train-distribution seed 1 has produced first artifacts under
  `T_generativeddpm_epsilon_12_070344/iter_0`.
- Latest metrics are `epoch=1`, `step=1883`; metrics file mtime is
  `2026-06-12 07:05:05 CST`.
- First checkpoint is
  `T_generativeddpm_epsilon_12_070344/iter_0/model-epoch=00-val_loss=0.3465.ckpt`.
- No `train_result_0.csv` exists yet; the row remains incomplete.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=17`, `PENDING=12`.
- GPU snapshot: GPU1 reports `18839 MiB`, `96%` utilization, `85C`; GPU3
  `18837 MiB`, `96%`, `88C`; GPU6 reports `509 MiB`, `4%`, `39C`.
- This is recorded as an UNSW DDPM train-distribution seed 1 first-checkpoint
  milestone. Subsequent updates should wait for completion artifacts, a round
  epoch milestone, or a substantially newer best checkpoint.

Recovery monitor snapshot at 2026-06-12 07:16 CST:
- UNSW DDPM train-distribution seed 1 train completed as a train-only
  `PARTIAL_STAGE_LEDGER`. UNSW now has all six train rows completed as
  train-only ledgers.
- Stage ledger:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_008_UNSW/ddpm_train_distribution/seed_1/stage_ledger.json`;
  file mtime is `2026-06-12 07:14:41 CST`, with ledger
  `updated_at` of `2026-06-11T23:16:06Z` and `last_returncode=0`.
- Train result:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_008_UNSW/ddpm_train_distribution/seed_1/train/metadata.xlsx/M_mamba1d_backbone/T_generativeddpm_epsilon_12_070344/iter_0/train_result_0.csv`;
  `train_completed=True`, `train_wall_clock_sec=606.6107365359785`,
  `parameter_count=3350`, `post_train_test_loss_ran=0.0`.
- Ledger checkpoint path is
  `T_generativeddpm_epsilon_12_070344/iter_0/model-epoch=21-val_loss=0.1974.ckpt`.
- Status helper reports `COMPLETE_CHAIN=6`, `PARTIAL_STAGE_LEDGER=18`,
  `PENDING=12`.
- Recovery driver `PID 831355` remains running. The driver has advanced to JUST
  CFM grid seed 0: child `PID 884998`, `Dl`, elapsed `00:01:43`, approximately
  `33.1%` CPU, with `--override model.in_channels=3`.
- GPU snapshot: GPU1 reports `18839 MiB`, `97%` utilization, `85C`; GPU3
  `18837 MiB`, `96%`, `87C`; GPU6 has dropped to `11 MiB`, `0%`, `32C`.
- This is a train-stage completion snapshot only. The full
  `train/sample/eval/paperpack` chain is not complete, and sample/eval/paperpack
  must not start until the train queue completes and ledger evidence is
  reviewed.

Recovery failure snapshot at 2026-06-12 07:29 CST:
- The recovery train-only driver exited with return code 1 at JUST CFM grid
  seed 0. This row did not produce `train_result_0.csv`; only
  `runs/RM_024_JUST/cfm_grid/seed_0/stage_ledger.json` exists.
- The current execution summary has been backed up to
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/logs/execution_summary_failed_just_cfm_seed0_20260612_072955.csv`
  before any further restart.
- Failed row:
  `dataset=RM_024_JUST`, `method=cfm_grid`, `seed=0`, `stage=train`,
  `gpu_id=6`, `returncode=1`, `wall_clock_sec=913.470984`.
- The failed command carried `--override model.in_channels=3`. The traceback
  failed in `_build_normalization_params -> _to_ncl` with
  `ValueError: expected channel axis with channels=3 in [N,C,L] or [N,L,C], got shape=(2, 128, 7)`.
- Direct HDF5 evidence shows the prior metadata-channel assumption was wrong for
  JUST: `RM_024_JUST` cache samples have shape `(N, 7, 1)`, so processed
  dataloader windows are `[batch, length, 7]`. The same HDF5 check confirmed
  CWRU/XJTU/FEMTO use 2 channels, UNSW uses 6 channels, and PU uses 3 channels.
- Minimal fix applied: `configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml`
  now sets `RM_024_JUST` `model.in_channels: 7`; the run-plan test expectation
  in `test/generative/test_six_dataset_submission.py` was updated to match.
- Validation completed:
  `python -m pytest test/generative/test_six_dataset_submission.py::test_repository_six_dataset_matrix_builds_complete_run_plan`,
  `python -m scripts.validate_configs`, and a train-stage dry-run whose JUST
  CFM seed 0 command contains `--override model.in_channels=7`.
- Status helper still reports `COMPLETE_CHAIN=6`, `PARTIAL_STAGE_LEDGER=18`,
  `PENDING=12`. This is still only a partial train-stage recovery state; the
  full `train/sample/eval/paperpack` chain is not complete.

Recovery restart snapshot at 2026-06-12 07:36 CST:
- The train-only recovery queue was restarted with the same output root and
  `--skip-existing` after the JUST channel fix.
- New recovery driver: `PID 894712`, command
  `python -m scripts.generative_benchmark_effect --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml --execute --preflight-gpu --stages train --skip-existing --output-dir results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10`.
- Current child process: `PID 894881`, `RM_024_JUST / cfm_grid / seed=0 /
  train`, with `--override model.in_channels=7`.
- JUST CFM seed 0 has only refreshed
  `runs/RM_024_JUST/cfm_grid/seed_0/stage_ledger.json` so far; no checkpoint or
  `train_result_0.csv` exists yet.
- This restart is still train-only. Do not start sample/eval/paperpack until the
  train queue completes and the train ledger is reviewed.

Recovery monitor snapshot at 2026-06-12 07:48 CST:
- Recovery driver `PID 894712` remains running (`Ss`, elapsed `00:11:52`) in
  train-only mode. Child `PID 894881` is running JUST CFM grid seed 0 with
  `model.in_channels=7` (`Rl`, elapsed `00:11:47`, approximately `36.6%` CPU).
- JUST CFM seed 0 has produced first artifacts under
  `T_generativeconditional_flow_matching_12_073642/iter_0`.
- First observed checkpoint:
  `T_generativeconditional_flow_matching_12_073642/iter_0/model-epoch=09-val_loss=1.6875.ckpt`.
- Metrics file mtime is `2026-06-12 07:48:22 CST`; latest row observed is
  `epoch=12`, `step=3509`, and latest non-empty validation metric is
  `epoch=11`, `val_loss=1.6815791130065918`.
- No `train_result_0.csv` exists yet; the row remains incomplete.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=18`, `PENDING=11`.
- GPU snapshot: GPU1 reports `18839 MiB`, `95%` utilization, `85C`; GPU3
  `18837 MiB`, `95%`, `87C`; GPU6 reports `513 MiB`, `5%`, `39C`.
- This is recorded as a JUST CFM seed 0 first-checkpoint milestone. It is not a
  train-stage completion.

Recovery monitor snapshot at 2026-06-12 07:59 CST:
- JUST CFM grid seed 0 train completed as a train-only `PARTIAL_STAGE_LEDGER`.
- Stage ledger:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_024_JUST/cfm_grid/seed_0/stage_ledger.json`;
  file mtime is `2026-06-12 07:58:15 CST`, with ledger
  `updated_at` of `2026-06-11T23:59:39Z` and `last_returncode=0`.
- Train result:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_024_JUST/cfm_grid/seed_0/train/metadata.xlsx/M_phm_unet1d/T_generativeconditional_flow_matching_12_073642/iter_0/train_result_0.csv`;
  `train_completed=True`, `train_wall_clock_sec=777.828290771984`,
  `parameter_count=43255`, `post_train_test_loss_ran=0.0`.
- Ledger checkpoint path is
  `T_generativeconditional_flow_matching_12_073642/iter_0/model-epoch=48-val_loss=1.5457.ckpt`.
- Status helper reports `COMPLETE_CHAIN=6`, `PARTIAL_STAGE_LEDGER=19`,
  `PENDING=11`.
- Recovery driver `PID 894712` remains running. The driver has advanced to JUST
  CFM grid seed 1: child `PID 910475`, `Dl`, elapsed `00:01:21`, approximately
  `42.5%` CPU, with `--override model.in_channels=7`.
- JUST CFM seed 1 currently has only
  `runs/RM_024_JUST/cfm_grid/seed_1/stage_ledger.json`; no checkpoint or
  `train_result_0.csv` exists yet.
- This is a train-stage completion snapshot only. The full
  `train/sample/eval/paperpack` chain is not complete, and sample/eval/paperpack
  must not start until the train queue completes and ledger evidence is
  reviewed.

Recovery monitor snapshot at 2026-06-12 08:11 CST:
- Recovery driver `PID 894712` remains running (`Ss`, elapsed `00:34:29`) in
  train-only mode. Child `PID 910475` is running JUST CFM grid seed 1 with
  `model.in_channels=7` (`Rl`, elapsed `00:12:44`, approximately `24.2%` CPU).
- JUST CFM seed 1 has produced first artifacts under
  `T_generativeconditional_flow_matching_12_075824/iter_0`.
- First observed checkpoint:
  `T_generativeconditional_flow_matching_12_075824/iter_0/model-epoch=02-val_loss=1.8099.ckpt`.
- Metrics file mtime is `2026-06-12 08:11:00 CST`; latest row observed is
  `epoch=5`, `step=1619`, and latest non-empty validation metric is
  `epoch=4`, `val_loss=1.7640423774719238`.
- No `train_result_0.csv` exists yet; the row remains incomplete.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=19`, `PENDING=10`.
- GPU snapshot: GPU1 reports `18839 MiB`, `95%` utilization, `85C`; GPU3
  `18837 MiB`, `93%`, `86C`; GPU6 reports `513 MiB`, `0%`, `37C`.
- This is recorded as a JUST CFM seed 1 first-checkpoint milestone. It is not a
  train-stage completion.

Recovery monitor snapshot at 2026-06-12 08:22 CST:
- Recovery driver `PID 894712` remains running (`Ss`, elapsed `00:45:55`) in
  train-only mode. Child `PID 910475` is still running JUST CFM grid seed 1
  with `model.in_channels=7` (`Dl`, elapsed `00:24:11`, approximately `46.4%`
  CPU).
- JUST CFM seed 1 has advanced to a substantially newer checkpoint:
  `T_generativeconditional_flow_matching_12_075824/iter_0/model-epoch=43-val_loss=1.5501.ckpt`.
- Metrics file mtime is `2026-06-12 08:22:26 CST`; latest row observed is
  `epoch=45`, `step=12246`, and latest non-empty validation metric is
  `epoch=44`, `val_loss=1.5413905382156372`.
- No `train_result_0.csv` exists yet; the row remains incomplete.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=19`, `PENDING=10`.
- GPU snapshot: GPU1 reports `18839 MiB`, `96%` utilization, `85C`; GPU3
  `18837 MiB`, `96%`, `88C`; GPU6 reports `513 MiB`, `0%`, `39C`.
- This is not a train-stage completion.

Recovery monitor snapshot at 2026-06-12 08:28 CST:
- JUST CFM grid seed 1 train completed as a train-only `PARTIAL_STAGE_LEDGER`.
  JUST now has both CFM train rows completed as train-only ledgers.
- Stage ledger:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_024_JUST/cfm_grid/seed_1/stage_ledger.json`;
  file mtime is `2026-06-12 08:24:21 CST`, with ledger
  `updated_at` of `2026-06-12T00:28:24Z` and `last_returncode=0`.
- Train result:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_024_JUST/cfm_grid/seed_1/train/metadata.xlsx/M_phm_unet1d/T_generativeconditional_flow_matching_12_075824/iter_0/train_result_0.csv`;
  `train_completed=True`, `train_wall_clock_sec=851.4084321500268`,
  `parameter_count=43255`, `post_train_test_loss_ran=0.0`.
- Ledger checkpoint path is
  `T_generativeconditional_flow_matching_12_075824/iter_0/model-epoch=49-val_loss=1.4993.ckpt`.
- Status helper reports `COMPLETE_CHAIN=6`, `PARTIAL_STAGE_LEDGER=20`,
  `PENDING=10`.
- Recovery driver `PID 894712` remains running. The driver has advanced to JUST
  rectified-flow grid seed 0: child `PID 951425`, `Dl`, elapsed `00:04:01`,
  approximately `25.8%` CPU, with `--override model.in_channels=7`.
- JUST rectified-flow seed 0 currently has only
  `runs/RM_024_JUST/rectified_flow_grid/seed_0/stage_ledger.json`; no
  checkpoint or `train_result_0.csv` exists yet.
- This is a train-stage completion snapshot only. The full
  `train/sample/eval/paperpack` chain is not complete, and sample/eval/paperpack
  must not start until the train queue completes and ledger evidence is
  reviewed.

Recovery monitor snapshot at 2026-06-12 08:39 CST:
- Recovery driver `PID 894712` remains running (`Ss`, elapsed `01:02:54`) in
  train-only mode. Child `PID 951425` is running JUST rectified-flow grid seed 0
  with `model.in_channels=7` (`Dl`, elapsed `00:15:04`, approximately `52.0%`
  CPU).
- JUST rectified-flow seed 0 has produced artifacts under
  `T_generativerectified_flow_12_082513/iter_0`.
- Latest observed checkpoint:
  `T_generativerectified_flow_12_082513/iter_0/model-epoch=25-val_loss=1.7223.ckpt`.
- Metrics file mtime is `2026-06-12 08:39:31 CST`; latest row observed is
  `epoch=29`, `step=7903`, and latest non-empty validation metric is
  `epoch=28`, `val_loss=1.7251511812210083`.
- No `train_result_0.csv` exists yet; the row remains incomplete.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=20`, `PENDING=9`.
- GPU snapshot: GPU6 reports `5251 MiB`, `0%` utilization, `43C`; GPU7 reports
  `4757 MiB`, `1%`, `41C`.
- This is not a train-stage completion.

Recovery monitor snapshot at 2026-06-12 08:52 CST:
- JUST rectified-flow grid seed 0 train completed as a train-only
  `PARTIAL_STAGE_LEDGER`.
- Stage ledger:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_024_JUST/rectified_flow_grid/seed_0/stage_ledger.json`;
  file mtime is `2026-06-12 08:52:20 CST`, with ledger
  `updated_at` of `2026-06-12T00:52:42Z` and `last_returncode=0`.
- Train result:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_024_JUST/rectified_flow_grid/seed_0/train/metadata.xlsx/M_phm_dit1d/T_generativerectified_flow_12_082513/iter_0/train_result_0.csv`;
  `train_completed=True`, `train_wall_clock_sec=1245.5134858200327`,
  `parameter_count=22087`, `post_train_test_loss_ran=0.0`.
- Ledger checkpoint path is
  `T_generativerectified_flow_12_082513/iter_0/model-epoch=47-val_loss=1.6856.ckpt`.
- Status helper reports `COMPLETE_CHAIN=6`, `PARTIAL_STAGE_LEDGER=21`,
  `PENDING=9`.
- Recovery driver `PID 894712` remains running. The driver has advanced to JUST
  rectified-flow grid seed 1: child `PID 987390`, `Dl`, elapsed `00:00:20`,
  approximately `64.0%` CPU, with `--override model.in_channels=7`.
- JUST rectified-flow seed 1 currently has only
  `runs/RM_024_JUST/rectified_flow_grid/seed_1/stage_ledger.json`; no
  checkpoint or `train_result_0.csv` exists yet.
- This is a train-stage completion snapshot only. The full
  `train/sample/eval/paperpack` chain is not complete, and sample/eval/paperpack
  must not start until the train queue completes and ledger evidence is
  reviewed.

Recovery failure snapshot at 2026-06-12 09:20 CST:
- The recovery train-only driver exited with return code 1 at JUST
  rectified-flow grid seed 1. This row did not produce `train_result_0.csv`;
  only normalization artifacts and
  `runs/RM_024_JUST/rectified_flow_grid/seed_1/stage_ledger.json` exist.
- The current execution summary has been backed up to
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/logs/execution_summary_failed_just_rectified_seed1_oom_20260612_091134.csv`
  before any further restart.
- Failed row:
  `dataset=RM_024_JUST`, `method=rectified_flow_grid`, `seed=1`,
  `stage=train`, `gpu_id=6`, `returncode=1`, `wall_clock_sec=1152.595826`.
- The failed command carried `--override model.in_channels=7`; the failure is a
  CUDA resource failure, not a channel-shape failure. The stderr tail ends with
  `RuntimeError: CUDA error: out of memory` during Lightning sanity checking.
- GPU process evidence at the time showed the matrix GPUs 6 and 7 were occupied
  by other Python processes with approximately `23750 MiB` and `23910 MiB`
  allocated. GPU2 was effectively idle (`11 MiB` used, `24199 MiB` free).
- To avoid encoding transient machine occupancy into the canonical paper matrix,
  a run-local recovery matrix was created at
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/logs/six_dataset_benchmark_matrix_gpu2_recovery_20260612.yaml`.
  It is identical to the canonical matrix except `resource.gpu_ids: [2]` and
  `resource.max_parallel_runs: 1`.
- Validation completed:
  `python -m scripts.validate_configs` and a train-stage dry-run whose JUST
  rectified-flow seed 1 command contains `CUDA_VISIBLE_DEVICES=2` and
  `--override model.in_channels=7`.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=21`, `PENDING=8`. This remains a train-only partial
  recovery state; sample/eval/paperpack are not started.

Recovery restart snapshot at 2026-06-12 09:21 CST:
- The train-only recovery queue was restarted with the run-local GPU2 recovery
  matrix:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/logs/six_dataset_benchmark_matrix_gpu2_recovery_20260612.yaml`.
- New recovery driver: `PID 1021743`, command
  `python -m scripts.generative_benchmark_effect --matrix results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/logs/six_dataset_benchmark_matrix_gpu2_recovery_20260612.yaml --execute --preflight-gpu --stages train --skip-existing --output-dir results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10`.
- Current child process after CUDA preflight: `PID 1022389`,
  `RM_024_JUST / rectified_flow_grid / seed=1 / train`, with
  `CUDA_VISIBLE_DEVICES=2` and `--override model.in_channels=7`.
- JUST rectified-flow seed 1 has not produced a checkpoint or
  `train_result_0.csv` yet after restart.
- This restart is still train-only. Do not start sample/eval/paperpack until the
  train queue completes and the train ledger is reviewed.

Recovery monitor snapshot at 2026-06-12 09:48 CST:
- GPU2 recovery driver `PID 1021743` remains running (`Ss`, elapsed `00:23:51`)
  in train-only mode. Child `PID 1022389` is running JUST rectified-flow grid
  seed 1 with `CUDA_VISIBLE_DEVICES=2` and `model.in_channels=7` (`Rl`, elapsed
  `00:23:04`, approximately `28.3%` CPU).
- JUST rectified-flow seed 1 has produced first artifacts under
  `T_generativerectified_flow_12_092553/iter_0`.
- First observed checkpoint:
  `T_generativerectified_flow_12_092553/iter_0/model-epoch=03-val_loss=1.8299.ckpt`.
- Metrics file mtime is `2026-06-12 09:48:26 CST`; latest row observed is
  `epoch=6`, `step=1889`, and latest non-empty validation metric is
  `epoch=5`, `val_loss=1.8143771886825562`.
- No `train_result_0.csv` exists yet; the row remains incomplete.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=21`, `PENDING=8`.
- GPU snapshot: GPU2 reports `509 MiB`, `23700 MiB` free, `0%` utilization,
  `39C`. Other GPUs remain occupied by non-recovery processes.
- This is recorded as a JUST rectified-flow seed 1 first-checkpoint milestone.
  It is not a train-stage completion.

Recovery monitor snapshot at 2026-06-12 10:00 CST:
- JUST rectified-flow grid seed 1 train completed as a train-only
  `PARTIAL_STAGE_LEDGER`. JUST now has both rectified-flow train rows completed
  as train-only ledgers.
- Stage ledger:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_024_JUST/rectified_flow_grid/seed_1/stage_ledger.json`;
  file mtime is `2026-06-12 10:00:07 CST`, with ledger
  `updated_at` of `2026-06-12T02:00:19Z` and `last_returncode=0`.
- Train result:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_024_JUST/rectified_flow_grid/seed_1/train/metadata.xlsx/M_phm_dit1d/T_generativerectified_flow_12_092553/iter_0/train_result_0.csv`;
  `train_completed=True`, `train_wall_clock_sec=776.6951018869877`,
  `parameter_count=22087`, `post_train_test_loss_ran=0.0`.
- Ledger checkpoint path is
  `T_generativerectified_flow_12_092553/iter_0/model-epoch=37-val_loss=1.6933.ckpt`.
- Status helper reports `COMPLETE_CHAIN=6`, `PARTIAL_STAGE_LEDGER=22`,
  `PENDING=8`.
- GPU2 recovery driver `PID 1021743` remains running. The driver has advanced to
  JUST DDPM train-distribution seed 0: child `PID 1051770`, `Dl`, elapsed
  `00:00:06`, approximately `145%` CPU, with `CUDA_VISIBLE_DEVICES=2` and
  `--override model.in_channels=7`.
- JUST DDPM seed 0 currently has only
  `runs/RM_024_JUST/ddpm_train_distribution/seed_0/stage_ledger.json`; no
  checkpoint or `train_result_0.csv` exists yet.
- This is a train-stage completion snapshot only. The full
  `train/sample/eval/paperpack` chain is not complete, and sample/eval/paperpack
  must not start until the train queue completes and ledger evidence is
  reviewed.

Recovery monitor snapshot at 2026-06-12 10:32 CST:
- GPU2 recovery driver `PID 1021743` remains running (`Ss`, elapsed `01:08:10`)
  in train-only mode. Child `PID 1051770` is running JUST DDPM
  train-distribution seed 0 with `CUDA_VISIBLE_DEVICES=2` and
  `model.in_channels=7` (`Rl`, elapsed `00:32:39`, approximately `43.0%` CPU).
- JUST DDPM seed 0 has produced artifacts under
  `T_generativeddpm_epsilon_12_100034/iter_0`.
- Latest observed checkpoint:
  `T_generativeddpm_epsilon_12_100034/iter_0/model-epoch=18-val_loss=0.2814.ckpt`.
- Metrics file mtime is `2026-06-12 10:32:46 CST`; latest row observed is
  `epoch=25`, `step=7019`, and latest non-empty validation metric is
  `epoch=24`, `val_loss=0.27304521203041077`.
- No `train_result_0.csv` exists yet; the row remains incomplete.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=22`, `PENDING=7`.
- GPU snapshot: GPU2 reports `511 MiB`, `23698 MiB` free, `0%` utilization,
  `41C`. Other GPUs remain occupied by non-recovery processes.
- This is not a train-stage completion.

Recovery monitor snapshot at 2026-06-12 10:40 CST:
- JUST DDPM train-distribution seed 0 train completed as a train-only
  `PARTIAL_STAGE_LEDGER`.
- Stage ledger:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_024_JUST/ddpm_train_distribution/seed_0/stage_ledger.json`;
  file mtime is `2026-06-12 10:35:57 CST`, with ledger
  `updated_at` of `2026-06-12T02:40:08Z` and `last_returncode=0`.
- Train result:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_024_JUST/ddpm_train_distribution/seed_0/train/metadata.xlsx/M_mamba1d_backbone/T_generativeddpm_epsilon_12_100034/iter_0/train_result_0.csv`;
  `train_completed=True`, `train_wall_clock_sec=538.163270229008`,
  `parameter_count=3415`, `post_train_test_loss_ran=0.0`.
- Ledger checkpoint path is
  `T_generativeddpm_epsilon_12_100034/iter_0/model-epoch=25-val_loss=0.2660.ckpt`.
- Status helper reports `COMPLETE_CHAIN=6`, `PARTIAL_STAGE_LEDGER=23`,
  `PENDING=7`.
- GPU2 recovery driver `PID 1021743` remains running. The driver has advanced to
  JUST DDPM train-distribution seed 1: child `PID 1082788`, `Dl`, elapsed
  `00:04:09`, approximately `17.7%` CPU, with `CUDA_VISIBLE_DEVICES=2` and
  `--override model.in_channels=7`.
- JUST DDPM seed 1 currently has only
  `runs/RM_024_JUST/ddpm_train_distribution/seed_1/stage_ledger.json`; no
  checkpoint or `train_result_0.csv` exists yet.
- This is a train-stage completion snapshot only. The full
  `train/sample/eval/paperpack` chain is not complete, and sample/eval/paperpack
  must not start until the train queue completes and ledger evidence is
  reviewed.

Recovery monitor snapshot at 2026-06-12 11:13 CST:
- JUST DDPM train-distribution seed 1 train completed as a train-only
  `PARTIAL_STAGE_LEDGER`. JUST now has all six train rows completed as
  train-only ledgers.
- Stage ledger:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_024_JUST/ddpm_train_distribution/seed_1/stage_ledger.json`;
  file mtime is `2026-06-12 11:11:41 CST`, with ledger
  `updated_at` of `2026-06-12T03:13:44Z` and `last_returncode=0`.
- Train result:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_024_JUST/ddpm_train_distribution/seed_1/train/metadata.xlsx/M_mamba1d_backbone/T_generativeddpm_epsilon_12_103618/iter_0/train_result_0.csv`;
  `train_completed=True`, `train_wall_clock_sec=384.3311318019987`,
  `parameter_count=3415`, `post_train_test_loss_ran=0.0`.
- Ledger checkpoint path is
  `T_generativeddpm_epsilon_12_103618/iter_0/model-epoch=11-val_loss=0.2958.ckpt`.
- Status helper reports `COMPLETE_CHAIN=6`, `PARTIAL_STAGE_LEDGER=24`,
  `PENDING=6`.
- GPU2 recovery driver `PID 1021743` remains running. The driver has advanced to
  PU CFM grid seed 0: child `PID 1120002`, `Dl`, elapsed `00:01:48`,
  approximately `26.0%` CPU, with `CUDA_VISIBLE_DEVICES=2` and
  `--override model.in_channels=3`.
- This is a train-stage completion snapshot only. The full
  `train/sample/eval/paperpack` chain is not complete, and sample/eval/paperpack
  must not start until the train queue completes and ledger evidence is
  reviewed.

Recovery monitor snapshot at 2026-06-12 11:24 CST:
- GPU2 recovery driver `PID 1021743` remains running (`Ss`, elapsed `02:00:08`)
  in train-only mode. Child `PID 1120002` is running PU CFM grid seed 0 with
  `CUDA_VISIBLE_DEVICES=2` and `model.in_channels=3` (`Rl`, elapsed `00:13:04`,
  approximately `35.6%` CPU).
- PU CFM seed 0 has produced first artifacts under
  `T_generativeconditional_flow_matching_12_111217/iter_0`.
- First observed checkpoint:
  `T_generativeconditional_flow_matching_12_111217/iter_0/model-epoch=01-val_loss=1.0813.ckpt`.
- Metrics file mtime is `2026-06-12 11:24:45 CST`; latest row observed is
  `epoch=2`, `step=11513`, and latest non-empty validation metric is
  `epoch=1`, `val_loss=1.0812854766845703`.
- No `train_result_0.csv` exists yet; the row remains incomplete.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=24`, `PENDING=5`.
- GPU snapshot: GPU2 reports `513 MiB`, `23696 MiB` free, `4%` utilization,
  `42C`. Other GPUs remain occupied by non-recovery processes.
- This is recorded as a PU CFM seed 0 first-checkpoint milestone. It is not a
  train-stage completion.

Recovery monitor snapshot at 2026-06-12 11:46 CST:
- GPU2 recovery driver `PID 1021743` remains running (`Ss`, elapsed `02:21:42`)
  in train-only mode. Child `PID 1120002` is still running PU CFM grid seed 0
  with `CUDA_VISIBLE_DEVICES=2` and `model.in_channels=3` (`Dl`, elapsed
  `00:34:38`, approximately `57.0%` CPU).
- PU CFM seed 0 has advanced to a newer checkpoint:
  `T_generativeconditional_flow_matching_12_111217/iter_0/model-epoch=14-val_loss=0.8602.ckpt`.
- Metrics file mtime is `2026-06-12 11:46:17 CST`; latest row observed is
  `epoch=15`, `step=58272`, and latest non-empty validation metric is
  `epoch=14`, `val_loss=0.8601890802383423`.
- No `train_result_0.csv` exists yet; the row remains incomplete.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=24`, `PENDING=5`.
- GPU snapshot: GPU2 reports `513 MiB`, `23696 MiB` free, `2%` utilization,
  `42C`. Other GPUs remain occupied by non-recovery processes.
- This is not a train-stage completion.

Recovery monitor snapshot at 2026-06-12 11:57 CST:
- GPU2 recovery driver `PID 1021743` remains running (`Ss`, elapsed `02:33:04`)
  in train-only mode. Child `PID 1120002` is still running PU CFM grid seed 0
  with `CUDA_VISIBLE_DEVICES=2` and `model.in_channels=3` (`Sl`, elapsed
  `00:46:00`, approximately `62.0%` CPU).
- PU CFM seed 0 has advanced beyond the epoch 20 milestone:
  `T_generativeconditional_flow_matching_12_111217/iter_0/model-epoch=21-val_loss=0.8434.ckpt`.
- Metrics file mtime is `2026-06-12 11:57:35 CST`; latest row observed is
  `epoch=22`, `step=88273`, and latest non-empty validation metric is
  `epoch=21`, `val_loss=0.8433608412742615`.
- No `train_result_0.csv` exists yet; the row remains incomplete.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=24`, `PENDING=5`.
- GPU snapshot: GPU2 reports `513 MiB`, `23696 MiB` free, `5%` utilization,
  `42C`. Other GPUs remain occupied by non-recovery processes.
- This is not a train-stage completion.

Recovery monitor snapshot at 2026-06-12 12:09 CST:
- GPU2 recovery driver `PID 1021743` remains running (`Ss`, elapsed `02:44:31`)
  in train-only mode. Child `PID 1120002` is still running PU CFM grid seed 0
  with `CUDA_VISIBLE_DEVICES=2` and `model.in_channels=3` (`Dl`, elapsed
  `00:57:27`, approximately `65.7%` CPU).
- PU CFM seed 0 has advanced to the epoch 30 milestone:
  `T_generativeconditional_flow_matching_12_111217/iter_0/model-epoch=29-val_loss=0.8360.ckpt`.
- Metrics file mtime is `2026-06-12 12:09:08 CST`; latest row observed is
  `epoch=31`, `step=119988`, and latest non-empty validation metric is
  `epoch=30`, `val_loss=0.8384023904800415`.
- No `train_result_0.csv` exists yet; the row remains incomplete.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=24`, `PENDING=5`.
- GPU snapshot: GPU2 reports `513 MiB`, `23696 MiB` free, `6%` utilization,
  `42C`. Other GPUs remain occupied by non-recovery processes.
- This is not a train-stage completion.

Recovery monitor snapshot at 2026-06-12 12:30 CST:
- GPU2 recovery driver `PID 1021743` remains running (`Ss`, elapsed `03:06:05`)
  in train-only mode. Child `PID 1120002` is still running PU CFM grid seed 0
  with `CUDA_VISIBLE_DEVICES=2` and `model.in_channels=3` (`Dl`, elapsed
  `01:19:01`, approximately `69.5%` CPU).
- PU CFM seed 0 has advanced beyond the epoch 40 milestone:
  `T_generativeconditional_flow_matching_12_111217/iter_0/model-epoch=43-val_loss=0.8235.ckpt`.
- Metrics file mtime is `2026-06-12 12:30:42 CST`; latest row observed is
  `epoch=46`, `step=179186`, and latest non-empty validation metric is
  `epoch=45`, `val_loss=0.8266474008560181`.
- No `train_result_0.csv` exists yet; the row remains incomplete.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=24`, `PENDING=5`.
- GPU snapshot: GPU2 reports `513 MiB`, `23696 MiB` free, `1%` utilization,
  `41C`. Other GPUs remain occupied by non-recovery processes.
- This is not a train-stage completion.

Recovery monitor snapshot at 2026-06-12 12:37 CST:
- PU CFM grid seed 0 completed its train stage as a train-only
  `PARTIAL_STAGE_LEDGER`; this is not a full `train/sample/eval/paperpack`
  completion.
- Stage ledger:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_027_PU/cfm_grid/seed_0/stage_ledger.json`
  with mtime `2026-06-12 12:36:38 CST`, `updated_at=2026-06-12T04:37:03Z`,
  and `last_returncode=0`.
- Train result:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_027_PU/cfm_grid/seed_0/train/metadata.xlsx/M_phm_unet1d/T_generativeconditional_flow_matching_12_111217/iter_0/train_result_0.csv`
  reports `train_completed=True`, `train_wall_clock_sec=4547.600654121023`,
  `parameter_count=42483`, and `post_train_test_loss_ran=0.0`.
- Final checkpoint observed:
  `T_generativeconditional_flow_matching_12_111217/iter_0/model-epoch=49-val_loss=0.8188.ckpt`.
- Status helper reports `COMPLETE_CHAIN=6`, `PARTIAL_STAGE_LEDGER=25`,
  `PENDING=5`.
- GPU2 recovery driver `PID 1021743` remains running and has advanced to PU CFM
  grid seed 1. Child `PID 1187911` is running with `CUDA_VISIBLE_DEVICES=2`
  and `model.in_channels=3` (`Sl`, elapsed `00:00:23`, approximately `79.9%`
  CPU at observation time).
- No sample/eval/paperpack rows were started manually.

Recovery monitor snapshot at 2026-06-12 12:43 CST:
- PU CFM grid seed 1 has moved past initialization and is actively training;
  this is a train-stage progress milestone, not a completion.
- First checkpoint observed:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_027_PU/cfm_grid/seed_1/train/metadata.xlsx/M_phm_unet1d/T_generativeconditional_flow_matching_12_123656/iter_0/model-epoch=00-val_loss=1.1316.ckpt`
  with mtime `2026-06-12 12:43:30 CST`.
- Metrics file:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_027_PU/cfm_grid/seed_1/train/metadata.xlsx/M_phm_unet1d/T_generativeconditional_flow_matching_12_123656/iter_0/logs/version_0/metrics.csv`
  has mtime `2026-06-12 12:43:30 CST`; latest observed row is `epoch=1`,
  `step=4232`, `train_loss_step=0.9225966334342957`.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=25`, `PENDING=4`.
- GPU2 recovery driver `PID 1021743` remains running (`Ss`, elapsed
  `03:18:54`). Child `PID 1187911` is running PU CFM grid seed 1 with
  `CUDA_VISIBLE_DEVICES=2` and `model.in_channels=3` (`Rl`, elapsed
  `00:06:52`, approximately `32.4%` CPU at observation time).
- GPU snapshot: GPU2 reports `513 MiB`, `23696 MiB` free, `0%` utilization,
  `39C`. Other GPUs remain occupied by non-recovery processes.

Recovery monitor snapshot at 2026-06-12 12:58 CST:
- PU CFM grid seed 1 has reached the epoch 10 training milestone; this is not a
  train-stage completion.
- Latest checkpoint observed:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_027_PU/cfm_grid/seed_1/train/metadata.xlsx/M_phm_unet1d/T_generativeconditional_flow_matching_12_123656/iter_0/model-epoch=09-val_loss=0.8843.ckpt`
  with mtime `2026-06-12 12:57:38 CST`.
- Metrics file mtime is `2026-06-12 12:58:38 CST`; latest observed row is
  `epoch=10`, `step=41964`, `train_loss_step=1.1308691501617432`.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=25`, `PENDING=4`.
- GPU2 recovery driver `PID 1021743` remains running. Child `PID 1187911` is
  running PU CFM grid seed 1 with `CUDA_VISIBLE_DEVICES=2` and
  `model.in_channels=3` (`Rl`, elapsed `00:22:00`, approximately `60.2%` CPU at
  observation time).
- GPU snapshot: GPU2 reports `513 MiB`, `23696 MiB` free, `5%` utilization,
  `41C`. Other GPUs remain occupied by non-recovery processes.

Recovery monitor snapshot at 2026-06-12 13:14 CST:
- PU CFM grid seed 1 has passed the epoch 20 training milestone in metrics;
  this is not a train-stage completion.
- Metrics file mtime is `2026-06-12 13:14:08 CST`; latest observed row is
  `epoch=21`, `step=82366`, `train_loss_step=1.0249017477035522`.
- Latest checkpoint retained at observation time:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_027_PU/cfm_grid/seed_1/train/metadata.xlsx/M_phm_unet1d/T_generativeconditional_flow_matching_12_123656/iter_0/model-epoch=17-val_loss=0.8523.ckpt`
  with mtime `2026-06-12 13:09:35 CST`.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=25`, `PENDING=4`.
- GPU2 recovery driver `PID 1021743` remains running. Child `PID 1187911` is
  running PU CFM grid seed 1 with `CUDA_VISIBLE_DEVICES=2` and
  `model.in_channels=3` (`Dl`, elapsed `00:37:29`, approximately `66.4%` CPU at
  observation time).
- GPU snapshot: GPU2 reports `513 MiB`, `23696 MiB` free, `1%` utilization,
  `42C`. Other GPUs remain occupied by non-recovery processes.

Recovery monitor snapshot at 2026-06-12 13:28 CST:
- PU CFM grid seed 1 has reached the epoch 30 training milestone; this is not a
  train-stage completion.
- Metrics file mtime is `2026-06-12 13:28:24 CST`; latest observed row is
  `epoch=30`, `step=118738`, `train_loss_step=0.8984953165054321`.
- Latest checkpoint observed:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_027_PU/cfm_grid/seed_1/train/metadata.xlsx/M_phm_unet1d/T_generativeconditional_flow_matching_12_123656/iter_0/model-epoch=29-val_loss=0.8369.ckpt`
  with mtime `2026-06-12 13:27:12 CST`.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=25`, `PENDING=4`.
- GPU2 recovery driver `PID 1021743` remains running. Child `PID 1187911` is
  running PU CFM grid seed 1 with `CUDA_VISIBLE_DEVICES=2` and
  `model.in_channels=3` (`Dl`, elapsed `00:51:46`, approximately `67.9%` CPU at
  observation time).
- GPU snapshot: GPU2 reports `513 MiB`, `23696 MiB` free, `0%` utilization,
  `42C`. Other GPUs remain occupied by non-recovery processes.

Recovery monitor snapshot at 2026-06-12 13:42 CST:
- PU CFM grid seed 1 has reached the epoch 40 training milestone; this is not a
  train-stage completion.
- Metrics file mtime is `2026-06-12 13:42:21 CST`; latest observed row is
  `epoch=40`, `step=157357`, `train_loss_step=0.7091207504272461`.
- Latest checkpoint observed:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_027_PU/cfm_grid/seed_1/train/metadata.xlsx/M_phm_unet1d/T_generativeconditional_flow_matching_12_123656/iter_0/model-epoch=38-val_loss=0.8290.ckpt`
  with mtime `2026-06-12 13:40:06 CST`.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=25`, `PENDING=4`.
- GPU2 recovery driver `PID 1021743` remains running. Child `PID 1187911` is
  running PU CFM grid seed 1 with `CUDA_VISIBLE_DEVICES=2` and
  `model.in_channels=3` (`Rl`, elapsed `01:06:01`, approximately `70.4%` CPU at
  observation time).
- GPU snapshot: GPU2 reports `513 MiB`, `23696 MiB` free, `6%` utilization,
  `42C`. Other GPUs remain occupied by non-recovery processes.

Recovery monitor snapshot at 2026-06-12 13:53 CST:
- PU CFM grid seed 1 completed its train stage as a train-only
  `PARTIAL_STAGE_LEDGER`; this is not a full `train/sample/eval/paperpack`
  completion.
- Stage ledger:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_027_PU/cfm_grid/seed_1/stage_ledger.json`
  with mtime `2026-06-12 13:53:28 CST`, `updated_at=2026-06-12T05:53:36Z`,
  and `last_returncode=0`.
- Train result:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_027_PU/cfm_grid/seed_1/train/metadata.xlsx/M_phm_unet1d/T_generativeconditional_flow_matching_12_123656/iter_0/train_result_0.csv`
  reports `train_completed=True`, `train_wall_clock_sec=4276.657334582007`,
  `parameter_count=42483`, and `post_train_test_loss_ran=0.0`.
- Final checkpoint recorded in the ledger:
  `T_generativeconditional_flow_matching_12_123656/iter_0/model-epoch=43-val_loss=0.8216.ckpt`.
- Status helper reports `COMPLETE_CHAIN=6`, `PARTIAL_STAGE_LEDGER=26`,
  `PENDING=4`.
- GPU2 recovery driver `PID 1021743` remains running and has advanced to PU
  rectified flow grid seed 0. Child `PID 1240573` is running with
  `CUDA_VISIBLE_DEVICES=2` and `model.in_channels=3` (`Dl`, elapsed
  `00:00:24`, approximately `82.5%` CPU at observation time).
- No sample/eval/paperpack rows were started manually.

Recovery monitor snapshot at 2026-06-12 14:08 CST:
- PU rectified flow grid seed 0 has moved past initialization and produced its
  first retained checkpoint; this is a train-stage progress milestone, not a
  completion.
- Latest checkpoint observed:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_027_PU/rectified_flow_grid/seed_0/train/metadata.xlsx/M_phm_dit1d/T_generativerectified_flow_12_135351/iter_0/model-epoch=06-val_loss=1.1255.ckpt`
  with mtime `2026-06-12 14:07:52 CST`.
- Metrics file mtime is `2026-06-12 14:08:33 CST`; latest observed row is
  `epoch=7`, `step=30091`, `train_loss_step=0.9679874181747437`.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=26`, `PENDING=3`.
- GPU2 recovery driver `PID 1021743` remains running. Child `PID 1240573` is
  running PU rectified flow grid seed 0 with `CUDA_VISIBLE_DEVICES=2` and
  `model.in_channels=3` (`Dl`, elapsed `00:14:58`, approximately `66.3%` CPU at
  observation time).
- GPU snapshot: GPU2 reports `509 MiB`, `23700 MiB` free, `4%` utilization,
  `40C`. Other GPUs remain occupied by non-recovery processes.

Recovery monitor snapshot at 2026-06-12 14:15 CST:
- PU rectified flow grid seed 0 has passed the epoch 10 training milestone;
  this is not a train-stage completion.
- Metrics file mtime is `2026-06-12 14:15:13 CST`; latest observed row is
  `epoch=12`, `step=49893`, `train_loss_step=1.0542397499084473`.
- Latest checkpoint observed:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_027_PU/rectified_flow_grid/seed_0/train/metadata.xlsx/M_phm_dit1d/T_generativerectified_flow_12_135351/iter_0/model-epoch=11-val_loss=1.0700.ckpt`
  with mtime `2026-06-12 14:14:25 CST`.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=26`, `PENDING=3`.
- GPU2 recovery driver `PID 1021743` remains running. Child `PID 1240573` is
  running PU rectified flow grid seed 0 with `CUDA_VISIBLE_DEVICES=2` and
  `model.in_channels=3` (`Rl`, elapsed `00:21:57`, approximately `73.3%` CPU at
  observation time).
- GPU snapshot: GPU2 reports `509 MiB`, `23700 MiB` free, `2%` utilization,
  `39C`. Other GPUs remain occupied by non-recovery processes.

Recovery monitor snapshot at 2026-06-12 14:30 CST:
- PU rectified flow grid seed 0 has reached the epoch 20 training milestone;
  this is not a train-stage completion.
- Metrics file mtime is `2026-06-12 14:30:08 CST`; latest observed row is
  `epoch=20`, `step=79613`, `train_loss_step=1.0049978494644165`.
- Latest checkpoint observed:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_027_PU/rectified_flow_grid/seed_0/train/metadata.xlsx/M_phm_dit1d/T_generativerectified_flow_12_135351/iter_0/model-epoch=18-val_loss=1.0267.ckpt`
  with mtime `2026-06-12 14:25:44 CST`.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=26`, `PENDING=3`.
- GPU2 recovery driver `PID 1021743` remains running. Child `PID 1240573` is
  running PU rectified flow grid seed 0 with `CUDA_VISIBLE_DEVICES=2` and
  `model.in_channels=3` (`Dl`, elapsed `00:36:33`, approximately `69.0%` CPU at
  observation time).
- GPU snapshot: GPU2 reports `509 MiB`, `23700 MiB` free, `3%` utilization,
  `42C`. Other GPUs remain occupied by non-recovery processes.

Recovery monitor snapshot at 2026-06-12 14:47 CST:
- PU rectified flow grid seed 0 has reached the epoch 30 training milestone;
  this is not a train-stage completion.
- Metrics file mtime is `2026-06-12 14:47:44 CST`; latest observed row is
  `epoch=30`, `step=115473`, `train_loss_step=1.0906922817230225`.
- Latest checkpoint observed:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_027_PU/rectified_flow_grid/seed_0/train/metadata.xlsx/M_phm_dit1d/T_generativerectified_flow_12_135351/iter_0/model-epoch=29-val_loss=0.9968.ckpt`
  with mtime `2026-06-12 14:47:36 CST`.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=26`, `PENDING=3`.
- GPU2 recovery driver `PID 1021743` remains running. Child `PID 1240573` is
  running PU rectified flow grid seed 0 with `CUDA_VISIBLE_DEVICES=2` and
  `model.in_channels=3` (`Dl`, elapsed `00:54:09`, approximately `69.6%` CPU at
  observation time).
- GPU snapshot: GPU2 reports `509 MiB`, `23700 MiB` free, `3%` utilization,
  `43C`. Other GPUs remain occupied by non-recovery processes.

Recovery monitor snapshot at 2026-06-12 15:09 CST:
- PU rectified flow grid seed 0 has passed the epoch 40 training milestone;
  this is not a train-stage completion.
- Metrics file mtime is `2026-06-12 15:09:55 CST`; latest observed row is
  `epoch=42`, `step=164264`, `train_loss_step=0.9661216735839844`.
- Latest checkpoint observed:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_027_PU/rectified_flow_grid/seed_0/train/metadata.xlsx/M_phm_dit1d/T_generativerectified_flow_12_135351/iter_0/model-epoch=41-val_loss=0.9597.ckpt`
  with mtime `2026-06-12 15:09:14 CST`.
- Status helper reports `COMPLETE_CHAIN=6`, `IN_PROGRESS_NO_LEDGER=1`,
  `PARTIAL_STAGE_LEDGER=26`, `PENDING=3`.
- GPU2 recovery driver `PID 1021743` remains running. Child `PID 1240573` is
  running PU rectified flow grid seed 0 with `CUDA_VISIBLE_DEVICES=2` and
  `model.in_channels=3` (`Rl`, elapsed `01:16:20`, approximately `69.4%` CPU at
  observation time).
- GPU snapshot: GPU2 reports `509 MiB`, `23700 MiB` free, `0%` utilization,
  `42C`. Other GPUs remain occupied by non-recovery processes.

Recovery monitor snapshot at 2026-06-12 15:24 CST:
- PU rectified flow grid seed 0 completed its train stage as a train-only
  `PARTIAL_STAGE_LEDGER`; this is not a full `train/sample/eval/paperpack`
  completion.
- Stage ledger:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_027_PU/rectified_flow_grid/seed_0/stage_ledger.json`
  with mtime `2026-06-12 15:23:12 CST`, `updated_at=2026-06-12T07:24:37Z`,
  and `last_returncode=0`.
- Train result:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_027_PU/rectified_flow_grid/seed_0/train/metadata.xlsx/M_phm_dit1d/T_generativerectified_flow_12_135351/iter_0/train_result_0.csv`
  reports `train_completed=True`, `train_wall_clock_sec=5190.448037490016`,
  `parameter_count=20035`, and `post_train_test_loss_ran=0.0`.
- Final checkpoint recorded in the ledger:
  `T_generativerectified_flow_12_135351/iter_0/model-epoch=48-val_loss=0.9551.ckpt`.
- Status helper reports `COMPLETE_CHAIN=6`, `PARTIAL_STAGE_LEDGER=27`,
  `PENDING=3`.
- GPU2 recovery driver `PID 1021743` remains running and has advanced to PU
  rectified flow grid seed 1. Child `PID 1318673` is running with
  `CUDA_VISIBLE_DEVICES=2` and `model.in_channels=3` (`Dl`, elapsed
  `00:01:45`, approximately `28.5%` CPU at observation time).
- GPU snapshot: GPU2 reports `11 MiB`, `24199 MiB` free, `0%` utilization,
  `35C`; seed 1 is still in its startup phase at this observation point. Other
  GPUs remain occupied by non-recovery processes.
- No sample/eval/paperpack rows were started manually.

Recovery monitor snapshot at 2026-06-12 21:14 CST:
- The train-only GPU2 recovery driver exited with return code 0. Its stdout
  reported CUDA preflight success and run-plan generation:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/run_plan.csv`.
- Status helper reports `COMPLETE_CHAIN=6`, `PARTIAL_STAGE_LEDGER=30`, with no
  `PENDING` rows. The 30 partial ledgers are train-only rows; they are not full
  `train/sample/eval/paperpack` chains.
- CSV parsing of
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/execution_summary.csv`
  found 36 rows, all with `stage=train` and `returncode=0`.
- PU rectified flow grid seed 1 completed train:
  `train_completed=True`, `train_wall_clock_sec=4635.820114518981`,
  `parameter_count=20035`, `post_train_test_loss_ran=0.0`, checkpoint
  `T_generativerectified_flow_12_152319/iter_0/model-epoch=42-val_loss=0.9635.ckpt`.
- PU DDPM train-distribution seed 0 completed train:
  `train_completed=True`, `train_wall_clock_sec=1561.8682563849725`,
  `parameter_count=3155`, `post_train_test_loss_ran=0.0`, checkpoint
  `T_generativeddpm_epsilon_12_164640/iter_0/model-epoch=14-val_loss=0.1755.ckpt`.
- PU DDPM train-distribution seed 1 completed train:
  `train_completed=True`, `train_wall_clock_sec=1540.1583214289858`,
  `parameter_count=3155`, `post_train_test_loss_ran=0.0`, checkpoint
  `T_generativeddpm_epsilon_12_171839/iter_0/model-epoch=13-val_loss=0.1767.ckpt`.
- `ps -p 1021743` and `ps --ppid 1021743` show no remaining recovery driver or
  child training process.
- No sample/eval/paperpack rows were started manually during train recovery.

Recovery monitor snapshot at 2026-06-12 21:32 CST:
- GOAL-V3-008 requires the real queue to cover `train/sample/eval/paperpack`.
  After train-only recovery completed and ledgers were reviewed, a dry-run for
  `sample,eval,paperpack` against the GPU2 recovery matrix wrote
  `run_plan.csv` with 108 non-train rows: 36 `sample`, 36 `eval`, and 36
  `paperpack`.
- Demo preflight passed:
  `python main.py --config configs/demo/10_generative/dummy_generative_cfm.yaml --preflight-only`.
- A sandboxed `sample` execution attempt failed before stage execution with
  `torch cuda unavailable`; an escalated CUDA visibility check with
  `CUDA_VISIBLE_DEVICES=2` reported `torch.cuda.is_available() == True` and one
  visible device.
- The sample-stage runner was then started in the non-sandbox environment:
  `python -m scripts.generative_benchmark_effect --matrix results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/logs/six_dataset_benchmark_matrix_gpu2_recovery_20260612.yaml --execute --preflight-gpu --stages sample --skip-existing --output-dir results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10`.
- Sample runner `PID 1690846` is active. Child `PID 1690955` is running
  `RM_002_XJTU / cfm_grid / seed_0 / sample` on `CUDA_VISIBLE_DEVICES=2`,
  using the resolved checkpoint
  `T_generativeconditional_flow_matching_10_200929/iter_0/model-epoch=18-val_loss=1.3864.ckpt`.
- Existing CWRU sample manifests remain present and were skipped by
  `--skip-existing`; no eval or paperpack stage has been manually started.

Recovery monitor snapshot at 2026-06-12 21:34 CST:
- `RM_002_XJTU / cfm_grid / seed_0 / sample` completed with stage ledger
  `status=succeeded` and `last_returncode=0`.
- Stage ledger sample paths:
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_002_XJTU/cfm_grid/seed_0/sample/metadata.xlsx/M_phm_unet1d/T_generativeconditional_flow_matching_12_212214/iter_0/synthetic/samples.pt`
  and
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/runs/RM_002_XJTU/cfm_grid/seed_0/sample/metadata.xlsx/M_phm_unet1d/T_generativeconditional_flow_matching_12_212214/iter_0/synthetic/synthetic_data_manifest.json`.
- Manifest evidence: `sampling.num_samples=256`, `sampling.shape=[256, 2, 128]`,
  `condition_sampling_policy=grid`, and four condition cells with 64 samples
  each. The manifest remains `validity.status=exploratory` because eval metric
  evidence is not present yet.
- Sample runner advanced to `RM_002_XJTU / cfm_grid / seed_1 / sample` on
  `CUDA_VISIBLE_DEVICES=2`, using checkpoint
  `T_generativeconditional_flow_matching_10_222124/iter_0/model-epoch=41-val_loss=1.3794.ckpt`.
- Status helper still reports `COMPLETE_CHAIN=6`, `PARTIAL_STAGE_LEDGER=30`;
  this is expected because the new XJTU sample row is not a full chain until
  eval and paperpack complete.

Recovery monitor snapshot at 2026-06-12 21:36 CST:
- XJTU sample stage has produced all 6 expected `synthetic_data_manifest.json`
  files for CFM, rectified flow, and DDPM across seeds 0 and 1.
- Global sample manifest count is now 12: the 6 pre-existing CWRU full-chain
  sample manifests plus 6 new XJTU sample manifests from this recovery stage.
- The sample runner advanced to `RM_003_FEMTO / cfm_grid / seed_0 / sample` on
  `CUDA_VISIBLE_DEVICES=2`, using checkpoint
  `T_generativeconditional_flow_matching_11_073558/iter_0/model-epoch=20-val_loss=1.3931.ckpt`.
- This is still sample-stage evidence only for XJTU; no eval or paperpack stage
  has run for XJTU in this continuation.

Recovery monitor snapshot at 2026-06-12 21:42 CST:
- FEMTO sample stage has produced all 6 expected `synthetic_data_manifest.json`
  files for CFM, rectified flow, and DDPM across seeds 0 and 1.
- Global sample manifest count is now 18: CWRU 6, XJTU 6, and FEMTO 6.
- The sample runner advanced to `RM_008_UNSW / cfm_grid / seed_0 / sample` on
  `CUDA_VISIBLE_DEVICES=2`, using checkpoint
  `T_generativeconditional_flow_matching_12_052618/iter_0/model-epoch=48-val_loss=0.9544.ckpt`.
- The UNSW sample command includes `model.in_channels=6`, matching the channel
  repair used for train recovery.
- This is still sample-stage evidence only for FEMTO; no eval or paperpack
  stage has run for FEMTO in this continuation.

Recovery monitor snapshot at 2026-06-12 21:51 CST:
- `RM_008_UNSW / cfm_grid / seed_0 / sample` is still running and has not yet
  produced `samples.pt` or `synthetic_data_manifest.json`.
- Child process `PID 1726168` is in `D` state at elapsed `00:11:16`, with no
  data-loader worker children visible. GPU2 reports `11 MiB` used and no
  meaningful utilization, so this is likely still in an I/O or setup path.
- The stage ledger is present and says `current_stage=sample`,
  `status=running`, checkpoint
  `T_generativeconditional_flow_matching_12_052618/iter_0/model-epoch=48-val_loss=0.9544.ckpt`.
- This is a runtime observation, not a failure classification. No process was
  interrupted.

Recovery monitor snapshot at 2026-06-12 21:58 CST:
- The earlier UNSW CFM seed 0 sample setup delay resolved without intervention.
  `samples.pt` and `synthetic_data_manifest.json` were written under
  `T_generativeconditional_flow_matching_12_214003/iter_0/synthetic/`.
- The sample runner advanced to `RM_008_UNSW / cfm_grid / seed_1 / sample` on
  `CUDA_VISIBLE_DEVICES=2`, using checkpoint
  `T_generativeconditional_flow_matching_12_054924/iter_0/model-epoch=47-val_loss=0.9404.ckpt`.
- No process was interrupted and no manual artifact path was supplied.

Recovery monitor snapshot at 2026-06-12 22:33 CST:
- UNSW sample stage has produced all 6 expected `synthetic_data_manifest.json`
  files for CFM, rectified flow, and DDPM across seeds 0 and 1.
- Global sample manifest count is now 24: CWRU, XJTU, FEMTO, and UNSW each have
  6 sample manifests.
- The sample runner advanced to `RM_024_JUST / cfm_grid / seed_0 / sample` on
  `CUDA_VISIBLE_DEVICES=2`, using checkpoint
  `T_generativeconditional_flow_matching_12_073642/iter_0/model-epoch=48-val_loss=1.5457.ckpt`.
- The JUST sample command includes `model.in_channels=7`, matching the channel
  repair used for train recovery.
- This is still sample-stage evidence only for UNSW; no eval or paperpack
  stage has run for UNSW in this continuation.

Recovery monitor snapshot at 2026-06-12 23:17 CST:
- JUST sample stage has produced all 6 expected `synthetic_data_manifest.json`
  files for CFM, rectified flow, and DDPM across seeds 0 and 1.
- Global sample manifest count is now 30; only PU remains in the sample stage.
- The sample runner advanced to `RM_027_PU / cfm_grid / seed_0 / sample` on
  `CUDA_VISIBLE_DEVICES=2`, using checkpoint
  `T_generativeconditional_flow_matching_12_111217/iter_0/model-epoch=49-val_loss=0.8188.ckpt`.
- The PU sample command includes `model.in_channels=3`, matching the channel
  repair used for train recovery.
- This is still sample-stage evidence only for JUST; no eval or paperpack stage
  has run for JUST in this continuation.

Recovery monitor snapshot at 2026-06-12 23:24 CST:
- PU sample stage has produced all 6 expected `synthetic_data_manifest.json`
  files for CFM, rectified flow, and DDPM across seeds 0 and 1.
- The sample-stage runner exited with return code 0. Its stdout reported CUDA
  preflight success and run-plan generation.
- Global sample manifest count is now 36, matching 6 datasets x 3 methods x 2
  seeds.
- CSV parsing of
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/execution_summary.csv`
  found 36 rows, all with `stage=sample` and `returncode=0`; the last row is
  `RM_027_PU / ddpm_train_distribution / seed_1 / sample`.
- Status helper still reports `COMPLETE_CHAIN=6`, `PARTIAL_STAGE_LEDGER=30`.
  This helper does not promote rows to full chain until eval and paperpack are
  also present, so sample-stage completion is evidenced by the sample manifests
  and sample execution summary above.
- No eval or paperpack rows were manually started during the sample-stage run.

Recovery monitor snapshot at 2026-06-12 23:26 CST:
- The eval-stage runner was started after sample-stage completion:
  `python -m scripts.generative_benchmark_effect --matrix results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/logs/six_dataset_benchmark_matrix_gpu2_recovery_20260612.yaml --execute --preflight-gpu --stages eval --skip-existing --output-dir results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10`.
- Eval runner `PID 1843194` is active. Child `PID 1843282` is running
  `RM_002_XJTU / cfm_grid / seed_0 / eval` on `CUDA_VISIBLE_DEVICES=2`.
- The eval command resolved `generated_path` to
  `T_generativeconditional_flow_matching_12_212214/iter_0/synthetic/samples.pt`,
  confirming stage-ledger/sample path handoff for the first non-CWRU eval row.
- Existing CWRU eval metrics and eval evidence manifests remain present and
  were skipped by `--skip-existing`.

Recovery monitor snapshot at 2026-06-12 23:34 CST:
- XJTU eval stage has produced all 6 expected `generative_eval_metrics.csv`
  files and all 6 expected `eval_evidence_manifest.json` files for CFM,
  rectified flow, and DDPM across seeds 0 and 1.
- Global eval metrics count is now 12: CWRU 6 plus XJTU 6.
- The eval runner advanced to `RM_003_FEMTO / cfm_grid / seed_0 / eval` on
  `CUDA_VISIBLE_DEVICES=2`, with `generated_path` resolved to the FEMTO CFM
  seed0 sample artifact.
- This is still eval-stage evidence only for XJTU; paperpack has not run for
  XJTU in this continuation.

Recovery monitor snapshot at 2026-06-12 23:43 CST:
- FEMTO eval stage has produced all 6 expected `generative_eval_metrics.csv`
  files and all 6 expected `eval_evidence_manifest.json` files for CFM,
  rectified flow, and DDPM across seeds 0 and 1.
- Global eval metrics count is now 18: CWRU, XJTU, and FEMTO each have 6.
- The eval runner advanced to `RM_008_UNSW / cfm_grid / seed_0 / eval` on
  `CUDA_VISIBLE_DEVICES=2`, with `model.in_channels=6` and `generated_path`
  resolved to the UNSW CFM seed0 sample artifact.
- This is still eval-stage evidence only for FEMTO; paperpack has not run for
  FEMTO in this continuation.

Recovery monitor snapshot at 2026-06-13 00:02 CST:
- UNSW eval stage has produced all 6 expected `generative_eval_metrics.csv`
  files and all 6 expected `eval_evidence_manifest.json` files for CFM,
  rectified flow, and DDPM across seeds 0 and 1.
- Global eval metrics count is now 24: CWRU, XJTU, FEMTO, and UNSW each have 6.
- The eval runner advanced to `RM_024_JUST / cfm_grid / seed_0 / eval` on
  `CUDA_VISIBLE_DEVICES=2`, with `model.in_channels=7` and `generated_path`
  resolved to the JUST CFM seed0 sample artifact.
- This is still eval-stage evidence only for UNSW; paperpack has not run for
  UNSW in this continuation.

Recovery monitor snapshot at 2026-06-13 00:14 CST:
- `RM_024_JUST / cfm_grid / seed_0 / eval` is still running and has not yet
  produced `generative_eval_metrics.csv` or `eval_evidence_manifest.json`.
- Child process `PID 1878419` is active at elapsed `00:19:02`. The stage ledger
  says `current_stage=eval`, `status=running`, and contains train/sample
  artifact paths, including the resolved sample `samples.pt`.
- This is a runtime observation, not a failure classification. No process was
  interrupted.

Recovery monitor snapshot at 2026-06-13 00:53 CST:
- The eval-stage recovery driver exited successfully with return code 0.
- Eval artifacts are complete for the real six-dataset matrix: 36
  `generative_eval_metrics.csv` files and 36 `eval_evidence_manifest.json`
  files under the run tree.
- The current `execution_summary.csv` contains 36 rows, all with `stage=eval`
  and `returncode=0`, distributed as 6 eval rows each for dataset ids 1, 2, 3,
  8, 18, and 20.
- `scripts.phm_genbench_v3_status --repair-ledger-metadata` rewrote the status
  ledger and still reports `COMPLETE_CHAIN=6 PARTIAL_STAGE_LEDGER=30`, which is
  expected before non-CWRU paperpack rows are generated. Paperpack has not been
  run in this continuation.

Recovery monitor snapshot at 2026-06-13 00:56 CST:
- The paperpack-stage recovery driver exited successfully with return code 0.
- The current `execution_summary.csv` contains 36 rows, all with
  `stage=paperpack` and `returncode=0`, distributed as 6 paperpack rows each
  for dataset ids 1, 2, 3, 8, 18, and 20.
- Paperpack artifacts are complete by per-run count: 36
  `paperpack/figure_sources/manifest_index.json` files, 36
  `paperpack/reproducibility_statement.md` files, and 36
  `paperpack/tables/table_utility_mean_std.csv` files.
- `scripts.phm_genbench_v3_status --repair-ledger-metadata` now reports
  `COMPLETE_CHAIN=36`.

Recovery monitor snapshot at 2026-06-13 00:58 CST:
- Full benchmark-effect aggregation was generated under
  `results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/effect/`
  with:
  - `benchmark_effect_summary.csv`
  - `benchmark_effect_report.md`
  - `benchmark_effect_manifest.json`
  - `missing_metrics.md`
- The aggregate manifest reports 6 configured datasets, 6 observed configured
  datasets, no missing datasets, no unexpected datasets, `min_datasets=6`, and
  `min_datasets_met=True`.
- The aggregate summary contains 2490 rows across all six datasets: CWRU 441,
  XJTU 337, FEMTO 493, UNSW 441, JUST 285, and PU 493. All 2490 rows remain
  `benchmark_status=exploratory`.
- `scripts.generative_submission_draft` generated
  `effect/submission_draft.md`, `effect/submission_readiness.md`, and
  `effect/evidence_gaps.md`. The draft status is `NOT_SUBMISSION_READY`.
- The explicit readiness gate
  `python -m scripts.generative_submission_draft ... --require-submission-ready`
  failed with return code 2, as expected for this evidence state. The recorded
  blockers are: at least 6 datasets with benchmark-valid quality and utility
  evidence required but 0 found, all contributing rows must be benchmark-valid,
  no computable quality metrics found, and no computable utility metrics found.

Recovery monitor snapshot at 2026-06-13 01:05 CST:
- GOAL-V3-009 canonical outputs were regenerated from the real run dirs under
  `results/paper/phm_generative/six_dataset_submission_v1/`.
- Canonical outputs now include `benchmark_effect_summary.csv`,
  `benchmark_effect_report.md`, `benchmark_effect_manifest.json`,
  `missing_metrics.md`, and `paper_evidence_package/`.
- The canonical manifest reports `summary_rows=2490`,
  `benchmark_status_counts={"exploratory": 2490}`,
  `benchmark_valid_row_count=0`, `exploratory_row_count=2490`,
  `observed_configured_dataset_count=6`, and `min_datasets_met=True`.
- `specs/002-phm-genbench-frontier/paper/PAPER_DRAFT.md`,
  `evidence_gaps.md`, and `submission_readiness.md` were regenerated from the
  canonical summary and manifest. The draft remains `NOT_SUBMISSION_READY`.
- `paper_evidence_package/package_manifest.json` indexes 36 per-run paperpack
  directories and combines table, figure-source, appendix run-index, manifest
  completeness, and missing-metric audit CSVs without promoting exploratory
  rows.
- The explicit canonical `--require-submission-ready` gate still fails with
  return code 2 for the same evidence reasons: 0 benchmark-valid datasets with
  both quality and utility evidence, all rows exploratory, no computable quality
  metrics found by the ready gate, and no computable utility metrics found by
  the ready gate.

The driver runs:

```bash
python -m scripts.generative_benchmark_effect \
  --matrix configs/paper/phm_generative/six_dataset_benchmark_matrix.yaml \
  --execute \
  --preflight-gpu \
  --stages train \
  --skip-existing \
  --output-dir results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10
```

for `train`, `sample`, `eval`, and `paperpack`, then runs benchmark-effect
aggregation and submission-draft generation. Do not treat the run as complete
until the tmux session exits successfully and the final `effect/` manifest
passes review. As of 2026-06-13 00:58 CST, the full stage chain and aggregation
exist, but the final draft remains `NOT_SUBMISSION_READY` by gate.

Monitor commands:

```bash
tmux ls
tail -f results/paper/phm_generative/six_dataset_submission_v1/real_run_v3_2026_06_10/logs/v3_longrun_tmux_20260610.log
python -m scripts.phm_genbench_v3_status --out specs/002-phm-genbench-frontier/reviews/codex/2026-06-10-v3-real-run-ledger.csv
```

## Guardrails

- Do not treat the partial XJTU row as complete.
- Do not use CWRU-only results as six-dataset benchmark evidence.
- Do not mark the generated submission draft `SUBMISSION_READY` while the
  aggregate rows remain `benchmark_status=exploratory` and the readiness gate
  returns nonzero.

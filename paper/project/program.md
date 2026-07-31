# 1D-2D Fusion Autoresearch Program

> paper_root: `paper/UXFD_paper/1D-2D_fusion_explainable`
> exec_root: `.` (nested ViBench repository root)
> mode: staged execution, evidence-gated, no SOTA wording before accepted artifacts

## Contract

- `paper_root` owns the Paper02 manuscript, planning docs, paper-local scripts, and paper-local evidence references.
- `exec_root` owns the maintained `python main.py --config ...` PHM-Vibench entrypoint.
- This program coordinates candidate runs only. A run is not accepted until the artifact gate validates its metadata and result files.
- Paper02 must stay within the local 2x4090 policy: use `CUDA_VISIBLE_DEVICES=0` or `CUDA_VISIBLE_DEVICES=1` for normal runs; use `CUDA_VISIBLE_DEVICES=0,1` only with a recorded reason.

## Stage 0: Resource Preflight

Run from `exec_root`:

```bash
nvidia-smi -L
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

Acceptance:

- GPUs `0` and `1` are visible as RTX 4090-class devices.
- PyTorch prints `True` and `2`.
- If this stage fails, stop GPU execution and record the blocker. Do not create accepted evidence.

## Stage 1: Maintained VIBENCH Smoke

Run from `exec_root` after Stage 0 passes:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config paper/UXFD_paper/1D-2D_fusion_explainable/configs/vibench/min.yaml \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

Acceptance:

- The command exits successfully.
- The run records command, seed, config path, device ID, GPU model, runtime, batch size, precision, and output path.
- The output is labelled smoke evidence unless it uses the same CWRU/XJTU protocol as the final table.

## Stage 2: Baseline Matrix

Run P00 and B01-B06 from `submission_prep/baseline_ablation_matrix.yaml` under the same CWRU/XJTU split, preprocessing, metrics, and at least three seeds.

Acceptance:

- Each run emits `run_meta.yaml`, metrics, log, config pointer, and failure reason if blocked.
- Result aggregation computes mean, standard deviation, and confidence interval from accepted runs only.
- The proposed method is compared against all six baselines before any performance claim enters the manuscript.

## Stage 3: Fusion Ablations

Run A01-A07 from the same matrix.

Acceptance:

- A01, A04, A06, and A07 must directly test the claimed fusion/alignment contribution.
- A05 remains sanity-only until rewritten to emit the same accepted artifact contract.
- Any missing ablation must be represented by an explicit blocker record, not silently omitted.

## Stage 4: TOP Representative Binding

Bind TOP-Q2 for `RWTOP2026-GTM` through B04/B05/A06 and keep the broader Paper02 TOP pool mapped in `paper/UXFD_paper/goal/08_recent_work_citation_readme.md`.

Acceptance:

- Exact reproduction, representative run, or resource-blocked status is recorded for each TOP method used in the paper.
- Representative runs are labelled as representative, not exact.
- No SOTA claim depends on a missing or resource-blocked exact baseline.

## Stage 5: Artifact Gate And Manuscript Update

Run from `exec_root` after candidate artifacts are produced:

```bash
python -m scripts.uxfd_artifact_gate paper/UXFD_paper/results/accepted_runs --format markdown
python -m scripts.uxfd_submission_gate --format markdown
```

Acceptance:

- Artifact gate passes for the Paper02 accepted runs.
- Submission gate shows Paper02 blockers reduced by actual evidence.
- Manuscript tables and figures are updated only from accepted artifacts.

## Reporting Contract

Every candidate or accepted loop must record:

1. repo root and branch/worktree tag
2. submodule SHA and parent SHA when available
3. exact command
4. dataset split, seed, preprocessing signature, and metrics
5. GPU device binding and runtime metadata
6. artifact paths or explicit blocker/failure mode
7. smallest next step

## Stop Policy

- Stop GPU work when Stage 0 fails.
- Stop SOTA wording when any declared baseline, ablation, or TOP representative lacks accepted same-protocol evidence.
- Do not commit generated result artifacts as accepted evidence until the artifact gate passes.

# UXFD Accepted-Run Artifact Action Packet

Status: artifact-promotion response packet only. This file is not accepted
experiment evidence and not a submission-readiness gate.

Purpose: describe the minimum package needed to promote a real Q0-passed run
into `paper/UXFD_paper/results/accepted_runs` after the experiment launch gate
passes on local RTX 4090 GPUs.

## Current Blocker

The accepted-run root currently has zero accepted records. No SOTA aggregate,
TOP representative evidence, ablation table, or submission-ready claim may use
smoke outputs, templates, failed preflight logs, or dirty submodule result files
as a substitute.

## Promotion Preconditions

1. `python -m scripts.uxfd_experiment_launch_gate --format markdown` exits
   with code `0` without `--allow-not-ready`.
2. `python -m scripts.uxfd_gpu_queue --format markdown --live-preflight --require-preflight`
   exits with code `0`.
3. The launched queue row is from `paper/UXFD_paper/goal/09_gpu_execution_queue.yaml`.
4. The source tree and relevant paper submodule are clean before the run is
   recorded as accepted evidence.
5. The run uses local RTX 4090 device `0`, device `1`, or documented `0,1`
   binding according to the queue command.

## Required Per-Run Files

Each accepted run directory must contain:

- `run_meta.yaml`
- `metrics.json` or `metrics.csv`
- `run.log`
- the YAML config evidence referenced by `config_path`

`metrics.json` or `metrics.csv` must contain at least one finite numeric metric
and must not contain TODO, NaN, or infinite payloads. `run.log` must be
non-empty and contain no TODO placeholders. The config evidence must be a
parseable, non-empty YAML mapping with no TODO placeholders.

## Required `run_meta.yaml` Fields

The metadata must include all queue-bound fields enforced by
`scripts.uxfd_artifact_gate`: `source_queue_id`, `paper_id`, `phase`,
`entry_id`, `cuda_visible_devices`, `gpu_model`, `gpu_count`, `seed`,
`dataset_split`, `preprocessing_signature`, `batch_size`, `precision`,
`runtime`, `evidence_level`, `command`, `git_sha_or_submodule_sha`,
`source_tree_status`, `config_path`, `log_path`, and `metrics_path`.

The values must satisfy:

- `accepted_evidence: true`
- `gpu_model` contains RTX 4090 and does not contain nonlocal GPU markers.
- `seed` is a non-negative integer.
- `batch_size` is a positive integer.
- `runtime` is positive `HH:MM:SS`.
- `precision` is one of `fp32`, `tf32`, `fp16`, `bf16`, or `amp`.
- `evidence_level` is `accepted_same_protocol`.
- `preprocessing_signature` matches `sha256:<64 lowercase hex>`.
- `source_tree_status` is `clean`.
- SHA provenance is concrete and contains no dirty, modified, unknown, or
  uncommitted marker.
- referenced `config_path`, `log_path`, and `metrics_path` stay inside the run
  directory.

## Acceptance Command

After adding candidate runs, execute:

```bash
python -m scripts.uxfd_artifact_gate paper/UXFD_paper/results/accepted_runs --require-queue-coverage --format markdown
```

The command must pass without `--allow-not-ready` before any accepted run can
feed TOP representative evidence, SOTA aggregates, or paper submission-ready
claims.

## Non-Evidence Boundary

This packet is a checklist only. It does not create accepted artifacts, does not
turn templates into evidence, and does not override GPU preflight, owner-review,
source-tree-clean, artifact, SOTA, submission, or objective gates.

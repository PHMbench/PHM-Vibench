# UXFD Codex XHigh Subagent Test Log

Date: 2026-05-11

## Subagent Audit Evidence

- Six Codex xhigh read-only subagents launched; see
  `CODEX_SUBAGENT_LAUNCH.md`.
- Subagent reports were received for Paper02/07, Paper01/03, Paper04/05,
  Paper06, TOP recent-work policy, and cross-paper execution gates.

## Commands Run By Main Codex Thread

- `python -m py_compile scripts/run_moe_ablation_smoke.py`
- `CUDA_VISIBLE_DEVICES=0 python scripts/run_moe_ablation_smoke.py --condition all --output /tmp/uxfd_paper04_moe_ablation_smoke --seed 0`
- `python -m unittest -q scripts/test_moe_ablation_smoke.py`
- `python -m py_compile scripts/run_mapping_ablation_smoke.py`
- `CUDA_VISIBLE_DEVICES=0 python scripts/run_mapping_ablation_smoke.py --condition all --output /tmp/uxfd_paper06_mapping_ablation_smoke --seed 0`
- `python -m unittest -q scripts/test_mapping_ablation_smoke.py`
- `python -m py_compile scripts/run_fusion_ablation_smoke.py`
- `CUDA_VISIBLE_DEVICES=0 python scripts/run_fusion_ablation_smoke.py --condition all --output /tmp/uxfd_paper02_fusion_ablation_smoke --seed 0`
- `python -m unittest -q scripts/test_fusion_ablation_smoke.py`
- `python -m pytest -q test/test_uxfd_objective_audit.py test/test_uxfd_recent_work_gate.py test/test_uxfd_artifact_gate.py test/test_uxfd_submission_gate.py test/test_uxfd_gpu_queue.py test/test_uxfd_paper_alignment_contract.py test/test_collect_uxfd_runs.py test/test_baseline_mapping_contract.py`
- `python -m scripts.uxfd_gpu_queue --format json`
- `python -m scripts.uxfd_submission_gate --format json --allow-not-ready`
- `python -m scripts.uxfd_objective_audit --format json --allow-not-achieved`
- `nvidia-smi -L`

## Observed Results

- Focused UXFD test suite passed after the smoke-runner updates: `54 passed,
  1 warning`.
- GPU queue is structurally valid and has zero command-level blocked rows, but
  `can_execute=false` because resource preflight is blocked.
- Submission gate remains `ready=false`.
- Objective audit remains `achieved=false`.
- `nvidia-smi -L` fails because the current session cannot communicate with the
  NVIDIA driver.

## Subagent Read-Only Commands

Subagents reported read-only file inspection with `find`, `sed`, `rg`,
`git status --short`, YAML parsing, and gate checks. No subagent edited files.

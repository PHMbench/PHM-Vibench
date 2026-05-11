# Session Handoff: UXFD IEEE Transactions Submission Readiness

**Date:** 2026-05-11
**Project:** `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench_fix`
**Session Duration:** current Codex implementation turn

## Current State

**Task:** Implement parent-level goal/spec workflow for seven UXFD IEEE Transactions paper submissions.
**Phase:** planning-to-implementation setup
**Progress:** parent control-plane artifacts created; all seven papers now have partial paper-local evidence checkpoints and none is submission-ready.

## What We Did

Created the seven-paper goal package under `paper/UXFD_paper/goal/` and the new Spec Kit feature under `specs/006-uxfd-ieee-trans-submission-readiness/`. Prepared a read-only Claude Code Team task spec and this handoff so future sessions can continue without rediscovering decisions. The goal package was then upgraded with six-baseline, ablation, SOTA-gate, TOP-venue recent-work citation/reproduction requirements, and a hard 2x4090 compute gate.
It now also includes a 2026 ICLR main-conference TOP-method addendum in the citation README and all seven paper-specific TOP quotas; these entries are citation/proxy planning evidence only until exact command/log/artifact evidence is generated.
Documentation validation was run after creation and passed.
Claude Team launch was attempted but blocked by external-service policy; the local task spec and launch log remain available.
The first paper-local execution milestone was started for Paper 07
`TII_operator_attention`: synthetic validation and the minimal PHM-Vibench
contract now run, and the submodule has a local milestone commit.
Then six parallel xhigh workers produced paper-local readiness checkpoints for
Papers 01-06. Each checkpoint is a narrow submodule commit that records
evidence and remaining blockers; none of these commits makes a paper
submission-ready.
Follow-up reproduction-contract commits ensured all seven paper submodule SHAs
tracked both `VIBENCH.md` and `configs/vibench/min.yaml`.
Parent contract tests now enforce that those two files are tracked inside each
paper submodule and that each `VIBENCH.md` declares a local
`CUDA_VISIBLE_DEVICES` binding.
After all seven paper-local matrices were created, the parent readiness matrix
was extended with a completion audit and the contract tests now verify that all
seven matrix files exist, each has at least six baselines, at least six
ablations, strict blockers, local 4090 policy, and `submission_ready: false`.
The latest resource preflight also confirms this session cannot produce accepted
GPU evidence: `nvidia-smi -L` cannot communicate with the NVIDIA driver, and
PyTorch reports `cuda_available False` with `device_count 0`. The readiness
matrix now records a blocked 2x4090 execution queue that must start with GPU
visibility validation before any real baseline, ablation, TOP representative,
or SOTA gate run is accepted.
The execution queue is also available as
`paper/UXFD_paper/goal/09_gpu_execution_queue.yaml`, which references all seven
paper-local matrices, enforces Q0-Q8 ordering, records two-GPU scheduler
limits, and lists the accepted run metadata required before SOTA or
submission-ready wording.
It also binds one 2026 TOP representative per paper to local proxy matrix
entries, but every binding remains `pending_gpu_and_artifacts`; none is exact
reproduction or accepted SOTA evidence yet.
`python -m scripts.uxfd_gpu_queue --format markdown` expands the queue as a
dry-run command manifest; `--require-preflight` is expected to fail until the
two local 4090 GPUs are visible.
The expander also supports `--output <path>` for writing markdown/json dry-run
manifests without shell redirection.
The dry-run payload includes summary counts by phase and paper, including
blocked entries and TOP representative bindings, so future sessions can inspect
execution scope before running anything.
Use `--live-preflight` when rerunning the expander after GPU access changes; it
records current `nvidia-smi -L` and PyTorch CUDA visibility and can be combined
with `--require-preflight` to fail before experiments if GPUs `0,1` are still
not accepted.
`python -m scripts.uxfd_submission_gate` emits a cross-paper submission gate
report and returns non-zero while any matrix is `submission_ready: false` or the
GPU queue remains blocked; use `--allow-not-ready` only for audit export.
The gate report includes queue-derived `next_actions` so each paper has a
visible unblock condition tied back to `09_gpu_execution_queue.yaml`.
It also includes an `objective_checklist` that maps named goal files, Claude
Team artifacts, seven matrix files, baseline/ablation coverage, the GPU queue,
and final submission readiness to concrete evidence.
`python -m scripts.uxfd_artifact_gate <artifact_root>` validates future accepted
run artifacts without executing experiments; it requires `run_meta.yaml`, local
4090 GPU metadata, and existing config/log/metrics paths.
The submission gate now invokes that artifact metadata gate by default against
`paper/UXFD_paper/results/accepted_runs`; use `--artifact-root` to point at a
specific accepted evidence bundle after real runs complete.
The artifact gate includes a tested field map from
`09_gpu_execution_queue.yaml` metadata labels to `run_meta.yaml` field names,
with OOM/failure reason treated as conditional metadata.

## Decisions Made

- **Seven independent papers** - The target is seven IEEE Transactions submissions, not one merged mega-paper.
- **Default per-paper journals** - Each paper has a default IEEE Transactions target and alternate, to be adjusted later if needed.
- **Parent-level goal package** - Cross-paper coordination lives in the parent repo; paper-specific content remains inside each submodule.
- **Read-only Claude Team first** - Parallel Claude review is useful, but implementation teams wait until dirty submodule ownership is clear.
- **Submodule milestone commits** - Important paper-specific updates must be committed inside the owning submodule before parent gitlink updates.
- **TOP recent work only** - Core related work, baseline, novelty, and SOTA positioning must come from TOP journals or computer-science top conferences. Low-tier sources are excluded from accepted method pools.
- **2x4090 only** - Experiment goals may use only local GPUs `0` and `1`, both RTX 4090-class cards. No cloud/A100/H100/multi-node assumption is valid without a later approved resource update.

## Code Changes

**Files added:**

- `paper/UXFD_paper/goal/README.md` - goal package index and status legend.
- `paper/UXFD_paper/goal/00_overall_goal.md` - shared seven-paper objective and strict-reviewer rubric.
- `paper/UXFD_paper/goal/01_explainable_fd_toolkit.md` through `07_tii_operator_attention.md` - per-paper readiness goals.
- `paper/UXFD_paper/goal/08_recent_work_citation_readme.md` - 2024-2026 TOP-source related-work citations and reproduction status policy.
- `paper/UXFD_paper/goal/99_submission_readiness_matrix.md` - cross-paper readiness matrix.
- `specs/006-uxfd-ieee-trans-submission-readiness/` - Spec Kit feature artifacts.
- `.codex/claude-team-runs/20260511-uxfd-ieee-trans-review/TASK_SPEC.md` - read-only Claude Team review spec.
- `.codex/claude-team-runs/20260511-uxfd-ieee-trans-review/LAUNCH_LOG.md` - launch attempt and policy-block record.
- `.claude/handoffs/2026-05-11-uxfd-ieee-trans-submission-readiness.md` - this handoff.

**Files modified:**

- `.specify/feature.json` - active feature pointer set to `specs/006-uxfd-ieee-trans-submission-readiness`.
- `AGENTS.md` - current Spec Kit plan pointer updated to the new feature plan.
- `test/test_uxfd_paper_alignment_contract.py` - added TOP-source quota and low-tier exclusion checks.
- `scripts/uxfd_recent_work_gate.py` and `test/test_uxfd_recent_work_gate.py` - added a non-executing TOP recent-work gate that separates citation-policy readiness from pending TOP representative artifact evidence.
- `scripts/uxfd_objective_audit.py` and `test/test_uxfd_objective_audit.py` - added a non-executing prompt-to-artifact audit that maps the active objective to goal files, Spec Kit files, handoff, team evidence, paper matrices, TOP/GPU/artifact gates, and final readiness.
- `specs/006-uxfd-ieee-trans-submission-readiness/` - updated spec, plan, data model, contract, checklist, quickstart, and tasks for TOP-source recent-work gates.
- `paper/UXFD_paper/goal/00_overall_goal.md` and all seven paper goal files - added compute budget and `resource-blocked` exact-reproduction policy.
- `test/test_uxfd_paper_alignment_contract.py` - added regression tests for tracked submodule reproduction contracts and local GPU binding declarations.
- `test/test_uxfd_paper_alignment_contract.py` - added a cross-paper matrix audit test for all seven paper-local baseline/ablation matrices.
- `paper/UXFD_paper/Explainable_FD_Toolkit` - submodule milestone commit `b76b5d8` replaces the missing placeholder figure/table in `manuscript/final_tex/main.tex`, adds `manuscript/T040_EVIDENCE_README.md`, and updates `VIBENCH.md`; follow-up commit `39b6a06` tracks `configs/vibench/min.yaml`.
- `paper/UXFD_paper/Explainable_FD_Toolkit` - follow-up submodule commit `40ea419` adds `submission_prep/baseline_ablation_matrix.yaml` and `submission_prep/ieee_trans_readiness.md`, fixes the stale `VIBENCH.md` exec root, records six command-bound PHM-Vibench baselines, one explain-extension ablation, and five blocked Toolkit ablation hooks.
- `paper/UXFD_paper/1D-2D_fusion_explainable` - submodule milestone commit `ecdae0a` adds `README_T041_SUBMISSION_READINESS.md`; follow-up commit `d548f11` tracks `VIBENCH.md` and `configs/vibench/min.yaml` with the current repo root.
- `paper/UXFD_paper/1D-2D_fusion_explainable` - follow-up submodule commit `f5c3cd3` adds `submission_prep/baseline_ablation_matrix.yaml` and `submission_prep/ieee_trans_readiness.md`, records six command-bound PHM-Vibench baselines, a paper-local Fusion1D2D dummy demo, STFT/fusion sensitivity smokes, and FFT/legacy ablation blockers.
- `paper/UXFD_paper/1D-2D_fusion_explainable` - current uncommitted follow-up adds `scripts/run_fusion_ablation_smoke.py` and `scripts/test_fusion_ablation_smoke.py`, converting FFT-only and legacy ablation surfaces into non-accepted smoke runners while keeping true Fusion1D2D accepted evidence blocked.
- `paper/UXFD_paper/LLM_Explainable_FD_Toolkit` - submodule milestone commit `dc014de` adds `SUBMISSION_READINESS.md` and updates `VIBENCH.md`; follow-up commit `9a5b141` tracks `configs/vibench/min.yaml`.
- `paper/UXFD_paper/LLM_Explainable_FD_Toolkit` - follow-up submodule commit `cfb4321` adds `submission_prep/baseline_ablation_matrix.yaml` and `submission_prep/ieee_trans_readiness.md`, records PHM-Vibench baseline smokes, standalone template LLM demos, and the earlier package-level `llm.llm_explainer` import blockers. Current uncommitted follow-up fixes the package smoke gate with local template/knowledge fallback, adds a conservative IEEE TeX compile checkpoint, emits non-accepted smoke `run_meta.yaml`/`metrics.json`, adds a non-accepted hallucination/context/latency smoke runner, and keeps accepted evidence packages blocked.
- `paper/UXFD_paper/Explainable_FD_Toolkit` - current uncommitted follow-up adds `scripts/run_toolkit_ablations.py` and `scripts/test_toolkit_ablation_smoke.py`, converting schema/metric/manifest/snapshot/post-hoc ablation blockers into non-accepted smoke runners while keeping accepted same-protocol Toolkit ablation evidence blocked.
- `paper/UXFD_paper/MOE_explainable` - submodule milestone commit `c2adc5a` adds `T043_SUBMISSION_READINESS_EVIDENCE.md`; follow-up commit `6992839` tracks `VIBENCH.md` and `configs/vibench/min.yaml` with the current repo root.
- `paper/UXFD_paper/MOE_explainable` - follow-up submodule commit `3dfc989` adds `submission_prep/baseline_ablation_matrix.yaml` and `submission_prep/ieee_trans_readiness.md`, records six command-bound PHM-Vibench baselines, partial expert-count ablation evidence, and five missing MoE ablation hooks.
- `paper/UXFD_paper/MOE_explainable` - current uncommitted follow-up adds `scripts/run_moe_ablation_smoke.py` and `scripts/test_moe_ablation_smoke.py`, converting load-balance, sparsity, router-temperature, expert-family, and uniform-router ablation blockers into non-accepted smoke runners while keeping accepted same-protocol MoE ablation evidence blocked.
- `paper/UXFD_paper/Paper_fuzzy_XFD` - submodule milestone commit `53e6d1b` adds a compilable evidence snapshot, updates `VIBENCH.md` and `configs/vibench/min.yaml`, adds `doc/T044_submission_readiness_evidence.md`, and fixes the NumPy bool serializer in `scripts/run_fuzzy_baseline.py`.
- `paper/UXFD_paper/Paper_fuzzy_XFD` - follow-up submodule commit `b82c05f` adds `submission_prep/baseline_ablation_matrix.yaml` and `submission_prep/ieee_trans_readiness.md`, records seven command-bound baselines, six supported fuzzy ablations, dummy-smoke metrics, and the remaining hard-threshold/safety/no-rule-output blockers.
- `paper/UXFD_paper/Neuralsymbolic_theory` - submodule milestone commit `9139307` adds `report/T045_evidence_readiness.md`, updates `VIBENCH.md`, and fixes `simple_validation_demo.py` so failed P2 evidence is recorded as a boundary case; follow-up commit `e3e268d` tracks `configs/vibench/min.yaml`.
- `paper/UXFD_paper/Neuralsymbolic_theory` - follow-up submodule commit `bea8a4a` adds `submission_prep/baseline_ablation_matrix.yaml` and `submission_prep/ieee_trans_readiness.md`, records six command-bound PHM-Vibench baselines, P1/P2/P3 proposition hooks, a scripted mapping hook, logic-strength sensitivity ablations, and the remaining P2/source-backed/GPU/TOP/SOTA blockers.
- `paper/UXFD_paper/Neuralsymbolic_theory` - current uncommitted follow-up adds `scripts/run_mapping_ablation_smoke.py` and `scripts/test_mapping_ablation_smoke.py`, converting the remove-cross-method-mapping ablation blocker into a non-accepted smoke runner while keeping source-backed mapping and real train/eval impact evidence blocked.
- `paper/UXFD_paper/TII_operator_attention` - submodule milestone commit `10a3d16` adds/updates `VIBENCH.md`, `configs/vibench/min.yaml`, `code/synthetic_verification.py`, synthetic validation outputs, rejection-recovery notes, and `submission_prep/ieee_trans_readiness.md`; follow-up commit `e8f8994` expands synthetic validation to eight signal classes.
- `paper/UXFD_paper/TII_operator_attention` - follow-up submodule commit `dd40adc` adds normalized canonical TeX entrypoint `manuscript/final_tex/main.tex`, updates `VIBENCH.md`, and records the compile gate in `submission_prep/ieee_trans_readiness.md`; follow-up commit `4315617` records the full `pdflatex`/`bibtex`/`pdflatex`/`pdflatex` compile flow; follow-up commit `e106fe8` fixes the five empty-year BibTeX warnings; follow-up commit `23990c0` binds a seven-baseline and six-ablation command matrix in `submission_prep/baseline_ablation_matrix.yaml`; follow-up commit `c5e960b` records dummy-smoke pass notes for B01/A01 and B02; follow-up commit `f306832` records B03-B05/B07 dummy-smoke pass notes and B06 Transformer import blocker; follow-up commit `0e037d9` records A02-A06 dummy-smoke pass notes so all six local ablation commands have dummy executable evidence; follow-up commit `2cae464` records B06 ConvTransformer dummy-smoke pass after restoring legacy model registration compatibility in the parent repo.
- `.codex/claude-team-runs/20260511-uxfd-ieee-trans-review/CODEX_SUBAGENT_LAUNCH.md` - records six Codex xhigh read-only subagents after the external Claude Team launch path was policy-blocked.
- `paper/UXFD_paper/goal/09_gpu_execution_queue.yaml` - current uncommitted follow-up changes Paper03 TOP representative binding from literature-only `RWTOP2026-CALTSFM` to representative-runnable `RWTOP2026-TIMESEG`; `scripts/uxfd_recent_work_gate.py` now rejects literature-only IDs in TOP representative bindings.

**Validation:**

- `python -m scripts.validate_docs` - passed: `[OK] Documentation checks passed (128 files scanned).`
- `python -m pytest -q test/test_uxfd_paper_alignment_contract.py test/test_collect_uxfd_runs.py` - passed: `9 passed in 0.32s`.
- `python -m scripts.phm_literature_matrix --min-count 50` - passed with 58 entries from 2025-2026.
- `python -m scripts.baseline_mapping` - passed and rendered the current baseline mapping.
- `python -m pytest -q test/test_uxfd_paper_alignment_contract.py test/test_collect_uxfd_runs.py test/test_phm_literature_matrix.py test/test_baseline_mapping_contract.py` - passed: `23 passed in 2.51s`.
- `python -m pytest -q test/test_model_registry_contract.py test/test_x_model_smoke.py` - failed in the base Python environment because `pytorch_lightning` is missing.
- `eval "$(conda shell.bash hook)" && conda activate LQ_signal && python -m pytest -q test/test_model_registry_contract.py test/test_x_model_smoke.py` - passed: `23 passed, 1 skipped`.
- `.specify/scripts/bash/check-prerequisites.sh --json --require-tasks --include-tasks` - passed and resolved the active feature directory to `specs/006-uxfd-ieee-trans-submission-readiness`.
- Cross-artifact consistency pass - no critical/high issues found; unchecked tasks are intentional future Claude launch and paper-production backlog.
- Claude Team launch - attempted, then blocked by policy because it can transmit repository/submodule contents to an external service.
- After TOP-source gate update:
  - `.specify/scripts/bash/check-prerequisites.sh --json --require-tasks --include-tasks` - passed and resolved `FEATURE_DIR` to `specs/006-uxfd-ieee-trans-submission-readiness`.
  - `python -m scripts.validate_docs` - passed: `[OK] Documentation checks passed (128 files scanned).`
  - `python -m scripts.baseline_mapping` - passed and rendered the current baseline mapping.
  - `python -m pytest -q test/test_uxfd_paper_alignment_contract.py test/test_collect_uxfd_runs.py test/test_baseline_mapping_contract.py` - passed: `19 passed in 2.10s`.
  - `python -m pytest -q test/test_uxfd_recent_work_gate.py` - passed: `4 passed in 0.21s`; the gate reports `policy_ready=true` and `evidence_ready=false` because the seven TOP representative bindings are still `pending_gpu_and_artifacts`.
  - `python -m pytest -q test/test_uxfd_objective_audit.py` - passed: `4 passed in 3.71s`; the audit reports `achieved=false` because team execution reports, accepted TOP artifacts, GPU preflight, accepted run metadata, and final submission readiness are not met.
  - Base-env representative model smoke still fails because `pytorch_lightning` is missing.
  - `eval "$(conda shell.bash hook)" && conda activate LQ_signal && python -m pytest -q test/test_model_registry_contract.py test/test_x_model_smoke.py` - passed: `23 passed, 1 skipped`.
- After 2x4090 compute-gate update:
  - `.specify/scripts/bash/check-prerequisites.sh --json --require-tasks --include-tasks` - passed and resolved `FEATURE_DIR` to `specs/006-uxfd-ieee-trans-submission-readiness`.
  - `python -m scripts.validate_docs` - passed: `[OK] Documentation checks passed (128 files scanned).`
  - `python -m scripts.baseline_mapping` - passed and rendered the current baseline mapping.
  - `python -m pytest -q test/test_uxfd_paper_alignment_contract.py` - passed: `16 passed in 0.27s`.
  - `python -m pytest -q test/test_uxfd_paper_alignment_contract.py test/test_collect_uxfd_runs.py test/test_baseline_mapping_contract.py` - passed: `21 passed in 2.22s`.
  - `git diff --check` on the touched goal/spec/test/handoff/doc paths - passed.
- After tracked reproduction-contract tests:
  - `python -m pytest -q test/test_uxfd_paper_alignment_contract.py test/test_collect_uxfd_runs.py test/test_baseline_mapping_contract.py` - passed: `23 passed in 2.34s`.
- Seven-paper minimal VIBENCH entrypoint check:
  - `eval "$(conda shell.bash hook)" && conda activate LQ_signal && for cfg in paper/UXFD_paper/*/configs/vibench/min.yaml; do CUDA_VISIBLE_DEVICES=0 python main.py --config "$cfg" --override trainer.num_epochs=1 --override data.num_workers=0; done` - all seven tracked paper configs completed as one-epoch dummy-data smoke runs.
  - PyTorch reported `GPU available: False` / NVML unavailable in the current sandbox, so this check is entrypoint/wiring evidence only, not accepted GPU feasibility or SOTA evidence.
- Paper 07 `TII_operator_attention` milestone:
  - `python code/synthetic_verification.py` from the submodule - passed and generated submodule-local figures, JSON, and report; current coverage is 8 signal classes, mean physics consistency `0.999`, mean explainability `0.261`.
  - `CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/TII_operator_attention/configs/vibench/min.yaml --override trainer.num_epochs=1` - failed in base env because `pytorch_lightning` is missing.
  - `eval "$(conda shell.bash hook)" && conda activate LQ_signal && CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/TII_operator_attention/configs/vibench/min.yaml --override trainer.num_epochs=1` - passed as dummy-data smoke; PyTorch reported GPU unavailable in this sandbox, so this is wiring evidence, not GPU industrial proof.
  - `pdflatex -interaction=nonstopmode manuscript/final_tex/main.tex`; `bibtex main`; `pdflatex -interaction=nonstopmode manuscript/final_tex/main.tex`; `pdflatex -interaction=nonstopmode manuscript/final_tex/main.tex` from the submodule root - all four commands passed and produced `main.pdf`; the final pass did not show undefined citation/reference warnings. The five BibTeX empty-year warnings were fixed by adding explicit `year` fields to the affected `ref.bib` entries; remaining warnings are routine layout and IEEEtran language-hyphenation warnings, so this is a normalized-entrypoint compile gate only, not final paper readiness.
  - `submission_prep/baseline_ablation_matrix.yaml` now contains seven command-bound baselines and six command-bound ablations. The entries were checked with `python -m scripts.config_inspect` target resolution; this is command-binding evidence only and does not replace accepted industrial GPU runs.
  - `eval "$(conda shell.bash hook)" && conda activate LQ_signal && CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/TII_operator_attention/configs/vibench/min.yaml --override model.name=NSN --override model.uxfd.operator_attention.enable=false --override trainer.num_epochs=1 --override data.num_workers=0` - B01/A01 dummy smoke passed, CPU fallback, `test_loss=0.7205665111541748`, `test_acc_Dummy_Data=0.0`.
  - `eval "$(conda shell.bash hook)" && conda activate LQ_signal && CUDA_VISIBLE_DEVICES=0 python main.py --config paper/UXFD_paper/TII_operator_attention/configs/vibench/min.yaml --override model.name=Resnet --override trainer.num_epochs=1 --override data.num_workers=0` - B02 dummy smoke passed, CPU fallback, `test_loss=1.1218299865722656`, `test_acc_Dummy_Data=0.0`.
  - B03 `Sincnet`, B04 `TFN`, B05 `WKN`, and B07 `AttentionCNN` dummy smokes also passed in `LQ_signal` with CPU fallback. B07 required `--override model.input_dim=2`.
  - B06 `Transformer.ConvTransformer` initially failed before training because importing `src/model_factory/Transformer/__init__.py` raised `ImportError: cannot import name 'register_model' from 'src.model_factory'`; restoring a legacy-compatible `register_model` decorator in `src/model_factory/__init__.py` and adding `--override model.input_dim=2` made the B06 dummy smoke pass with CPU fallback, `test_loss=7.736400604248047`, `test_acc_Dummy_Data=0.0`.
  - A02 identity-only, A03 Hilbert-only, A04 FFT-only, A05 temperature 0.5, and A06 temperature 2.0 dummy smokes all passed in `LQ_signal` with CPU fallback; A01 shares the B01 run.
  - `python -m pytest -q test/test_uxfd_paper_alignment_contract.py test/test_collect_uxfd_runs.py test/test_baseline_mapping_contract.py` - passed after adding the Paper 07 matrix contract test: `24 passed in 2.58s`.
- Paper 01 `Explainable_FD_Toolkit` milestone:
  - Worker schema checks passed for benchmark, unified matrix, Captum, SHAP/LIME, and THU018 packs.
  - `pdflatex` failed on pre-existing Chinese Unicode/inputenc handling.
  - `xelatex` passed and produced `/tmp/uxfd_toolkit_texcheck/main.pdf`.
  - P00 and B01-B06 PHM-Vibench dummy smokes passed in `LQ_signal` with CPU fallback because GPU/NVML was unavailable. B06 ConvTransformer required `--override model.input_dim=2`.
  - `submission_prep/baseline_ablation_matrix.yaml` records six command-bound baselines, one smoke-level ablation for disabling the PHM-Vibench explain extension, and five blocked Toolkit ablation hooks: schema removal, metric-family removal, standardized manifest off, fixed-seed/config-snapshot off, and post-hoc-only comparator mode.
  - `python -m pytest -q test/test_uxfd_paper_alignment_contract.py` passed after adding the Toolkit matrix contract test: `22 passed in 0.55s`.
  - Remaining blockers: no same-protocol CWRU/XJTU six-baseline result matrix, five Toolkit ablation hooks missing, no TOP proxy mapping, incomplete compute metadata, SOTA blocked, broader manuscript placeholders remain.
- Paper 02 `1D-2D_fusion_explainable` milestone:
  - `git diff --check -- README_T041_SUBMISSION_READINESS.md` passed.
  - `python scripts/truth_audit.py --paper-root . --output-dir /tmp/uxfd_1d2d_t041_truth_audit` passed and reported 3 blocking issues.
  - `latexmk ... paper_draft/NMI_Paper1_Fusion1D2D.tex` failed because `NatureMi.cls` is missing.
  - `CUDA_VISIBLE_DEVICES=0 python scripts/run_minimal_demo.py --use_dummy --num_epochs=1 --batch_size=8 --input_dim=128 --num_classes=4 --output_root /tmp/uxfd_paper02_minimal_demo` failed with `IndexError: Target 8 is out of bounds`; rerunning with `--num_classes=10` passed and produced dummy `test_accuracy=0.39`, `test_f1_macro=0.23883535636476813`.
  - P00 proposed PHM-Vibench proxy plus B01 no-2D proxy, B02 ResNet, B03 SincNet, B04 TFN, B05 WKN, and B06 ConvTransformer dummy smokes passed in `LQ_signal` with CPU fallback because GPU/NVML was unavailable.
  - A02/A03 STFT sensitivity and A04 concat-fusion dummy smokes passed in `LQ_signal`; A06 FFT-only signal-layer stress failed with a skip-connection dimensionality mismatch. Current uncommitted follow-up binds non-accepted smoke runners for A06 and the legacy A07 surface; this does not fix the true FFT or stale legacy runner paths.
  - `submission_prep/baseline_ablation_matrix.yaml` now contains six command-bound baselines, seven ablation/demo/smoke rows, TOP recent-work blocker statuses, and strict blockers for real Fusion1D2D ablations, stale THU/GPU2 legacy runner assumptions, missing TOP representatives, missing 2x4090 metadata, TeX, and SOTA.
  - `python -m py_compile scripts/run_fusion_ablation_smoke.py`, `CUDA_VISIBLE_DEVICES=0 python scripts/run_fusion_ablation_smoke.py --condition all --output /tmp/uxfd_paper02_fusion_ablation_smoke --seed 0`, and `python -m unittest -q scripts/test_fusion_ablation_smoke.py` passed for the current fusion-ablation smoke runner.
  - Parent focused UXFD gates passed after the current Paper 02 update: `python -m pytest -q test/test_uxfd_objective_audit.py test/test_uxfd_recent_work_gate.py test/test_uxfd_artifact_gate.py test/test_uxfd_submission_gate.py test/test_uxfd_gpu_queue.py test/test_uxfd_paper_alignment_contract.py test/test_collect_uxfd_runs.py test/test_baseline_mapping_contract.py` -> `54 passed, 1 warning`.
  - Remaining blockers: no accepted CWRU/XJTU six-baseline matrix, no true 1D-only/2D-only/no-stat/no-alignment ablation package, no local TOP representatives, no accepted 2x4090 metadata, TeX not compiling, SOTA blocked.
- Paper 03 `LLM_Explainable_FD_Toolkit` milestone:
  - `CUDA_VISIBLE_DEVICES=0 python experiments/scripts/run_minimal_llm_demo_standalone.py --mode pipeline` passed.
  - `python -m pytest -q code/tests/test_basic_functionality.py` now passes with `14 passed` after local template/knowledge fallback and timestamp normalization fixes.
  - `CUDA_VISIBLE_DEVICES=0 python experiments/scripts/run_minimal_llm_demo_standalone.py --mode single --case 0` passed and emitted four template dialogue responses.
  - `CUDA_VISIBLE_DEVICES=0 python experiments/scripts/run_minimal_llm_demo.py --mode pipeline --save --output /tmp/uxfd_paper03_template_llm_artifacts` now passes and writes demo artifacts under `/tmp`.
  - P00 PHM-Vibench agent-enabled smoke plus B01 no-agent, B03 ResNet, B04 SincNet, B05 TFN, B06 WKN, and B07 ConvTransformer dummy smokes passed in `LQ_signal` with CPU fallback because GPU/NVML was unavailable.
  - `submission_prep/baseline_ablation_matrix.yaml` now contains seven baselines, seven ablation/demo/blocker rows, TOP recent-work blocker statuses, and strict blockers for missing IEEE TeX, `results/llm_evidence` packages, hallucination/retrieval/latency runners, TOP representatives, 2x4090 metadata, and SOTA.
  - `python -m pytest -q test/test_uxfd_paper_alignment_contract.py test/test_uxfd_submission_gate.py test/test_uxfd_objective_audit.py` passed after updating the Paper 03 matrix contract: `36 passed in 5.98s`.
  - Remaining blockers: conservative IEEE TeX exists but final evidence-bearing manuscript content is missing; package demo and ablation runner emit non-accepted smoke metadata only; no accepted `results/llm_evidence/**/{run_meta.yaml,metrics.json}` packages, no same-protocol LLM baseline/ablation/TOP/latency/anti-hallucination evidence, no accepted 2x4090 metadata, SOTA blocked.
- Paper 04 `MOE_explainable` milestone:
  - Worker `jq` checks passed for seed stability, routing analysis, expert ablation, and CWRU/XJTU bridge artifacts.
  - `python scripts/bind_submission_ready_evidence.py --mode review-evidence --paper-root . --output-dir /tmp/moe_t043_review_evidence_validate` passed, but the new T043 gate records why the older local `ready: true` does not satisfy the strict IEEE gate.
  - P00 and B01-B06 PHM-Vibench dummy smokes passed in `LQ_signal` with CPU fallback because GPU/NVML was unavailable. B06 ConvTransformer required `--override model.input_dim=2`.
  - `submission_prep/baseline_ablation_matrix.yaml` records six command-bound baselines plus partial existing expert-count ablation evidence. Current uncommitted follow-up binds non-accepted smoke runners for no-load-balance, no-sparsity, router-temperature, expert-family-removal, and uniform-router surfaces.
  - `python -m py_compile scripts/run_moe_ablation_smoke.py`, `CUDA_VISIBLE_DEVICES=0 python scripts/run_moe_ablation_smoke.py --condition all --output /tmp/uxfd_paper04_moe_ablation_smoke --seed 0`, and `python -m unittest -q scripts/test_moe_ablation_smoke.py` passed for the current non-accepted smoke runner.
  - Parent focused UXFD gates passed after the current MoE smoke-runner update: `python -m pytest -q test/test_uxfd_objective_audit.py test/test_uxfd_recent_work_gate.py test/test_uxfd_artifact_gate.py test/test_uxfd_submission_gate.py test/test_uxfd_gpu_queue.py test/test_uxfd_paper_alignment_contract.py test/test_collect_uxfd_runs.py test/test_baseline_mapping_contract.py` -> `54 passed, 1 warning`.
  - Remaining blockers: no full CWRU/XJTU multi-seed matrix, no accepted same-protocol MoE ablation artifacts, incomplete 2x4090 metadata, no TOP representative artifacts, SOTA blocked.
- Paper 05 `Paper_fuzzy_XFD` milestone:
  - `pdflatex` passed for `manuscript/final_tex/main.tex` and wrote `/tmp/fuzzy_xfd_tex/main.pdf`.
  - YAML parse and `py_compile` passed.
  - Synthetic fuzzy baseline smoke passed after replacing deprecated `np.bool`.
  - Base-env root smoke failed due to missing `pytorch_lightning`; `conda run -n LQ_signal ... main.py ...` passed as dummy smoke, with GPU unavailable in this sandbox.
  - `submission_prep/baseline_ablation_matrix.yaml` now contains seven command-bound baselines and six command-bound fuzzy ablations. P00, B01-B06, A02-A06 ran in `LQ_signal` as one-epoch dummy-data smokes with CPU fallback; A01 shares B01. The paper-local classical fuzzy baseline script also ran with script-generated demo data because the feature file was absent.
  - `python -m pytest -q test/test_uxfd_paper_alignment_contract.py` passed after adding the Paper 05 matrix contract test: `20 passed in 0.51s`.
  - Remaining blockers: no CWRU/XJTU 3-seed real-data results, no accepted baseline/ablation artifacts, no rule metrics, no safety cases, no TOP representative artifact, hard-threshold/safety/no-rule-output ablation hooks missing, SOTA blocked.
- Paper 06 `Neuralsymbolic_theory` milestone:
  - `python -m py_compile simple_validation_demo.py experiments/proposition2_simple.py code/validate_mapping.py` passed.
  - `python simple_validation_demo.py` passed and correctly records P1 pass, P2 fail, P3 pass, `overall_theory_supported=false`.
  - `python -m scripts.config_inspect --config paper/UXFD_paper/Neuralsymbolic_theory/configs/vibench/min.yaml --override trainer.num_epochs=1` resolved config/data paths but failed importing the pipeline because `pytorch_lightning` is missing.
  - P00 proposed constrained NSN/TSPN_UXFD plus B01 no-symbolic NSN/TSPN_UXFD, B02 ResNet, B03 SincNet, B04 TFN, B05 WKN, and B06 ConvTransformer dummy smokes passed in `LQ_signal` with CPU fallback because GPU/NVML was unavailable.
  - A03/A04 logic `logit_scale=0.1` and `logit_scale=1.0` dummy smokes passed in `LQ_signal` with CPU fallback.
  - `python experiments/proposition2_simple.py` passed and rewrote `experiments/results/proposition2_12_14/simple_results.json` with a synthetic lower physics-informed sensitivity artifact; this does not override the failed P2 aggregate validation demo.
  - `python code/validate_mapping.py` passed and generated `report/mapping_validation_report.json` plus `manuscript/figures/mapping_validation.png`; the matrix records this as scripted mapping only, not source-backed evidence.
  - `submission_prep/baseline_ablation_matrix.yaml` now contains six command-bound baselines, seven ablation/proposition/mapping rows, TOP recent-work blocker statuses, and strict blockers for P2, source-backed mapping, GPU metadata, TOP representatives, manuscript placeholders, and SOTA. Current uncommitted follow-up binds a non-accepted remove-mapping smoke runner for A07.
  - `python -m py_compile scripts/run_mapping_ablation_smoke.py`, `CUDA_VISIBLE_DEVICES=0 python scripts/run_mapping_ablation_smoke.py --condition all --output /tmp/uxfd_paper06_mapping_ablation_smoke --seed 0`, and `python -m unittest -q scripts/test_mapping_ablation_smoke.py` passed for the current mapping-ablation smoke runner.
  - Parent focused UXFD gates passed after the current Paper 06 update: `python -m pytest -q test/test_uxfd_objective_audit.py test/test_uxfd_recent_work_gate.py test/test_uxfd_artifact_gate.py test/test_uxfd_submission_gate.py test/test_uxfd_gpu_queue.py test/test_uxfd_paper_alignment_contract.py test/test_collect_uxfd_runs.py test/test_baseline_mapping_contract.py` -> `54 passed, 1 warning`.
  - Remaining blockers: placeholder TeX, no accepted CWRU/XJTU multi-seed baseline/ablation artifacts, failed/inconsistent P2 support, source-backed mapping evidence missing, no TOP representative artifacts, no accepted 2x4090 metadata, SOTA blocked.

## Open Questions

- [ ] Should the read-only Claude Code Team be launched now, or only after a human reviews the newly created goal package?
- [ ] Which paper should be the first implementation milestone: MoE, because compile previously passed, or Toolkit, because it is upstream infrastructure?
- [ ] Should each paper adopt a normalized `manuscript/final_tex/main.tex` convention, or allow paper-specific canonical entrypoints?

## Blockers / Issues

- Parent worktree was already heavily dirty before this task.
- Several UXFD submodules have uncommitted or untracked content; treat all of it as existing user work until attributed.
- Most papers are blocked for submission readiness by placeholder TeX, missing canonical entrypoints, incomplete evidence, or unverified claim maps.
- TOP recent related work is now citation-mapped, but exact reproduction still requires paper-local commands, logs, and artifacts before it can count as exact SOTA evidence.
- `scripts/uxfd_recent_work_gate.py` currently passes the TOP-source policy layer but intentionally fails final readiness because all seven TOP representative bindings are still pending GPU/artifact evidence.
- `scripts/uxfd_objective_audit.py` currently fails the active-objective completion audit because the TOP/GPU/artifact gates are not accepted, all seven papers remain non-submission-ready, and six paper submodules still have dirty working trees. The audit treats dirty paper submodules as a blocker before parent handoff; local Codex xhigh subagent launch evidence and synthesized `report.md`/`risks.md`/`test-log.md` are present.
- `scripts/uxfd_gpu_queue.py --format shell --output paper/UXFD_paper/results/queue_launch_plan.sh --shard-dir paper/UXFD_paper/results/queue_launch_shards` now generates a non-executing 2x4090 launch plan with live preflight guards, device 0/1 round-robin binding, paper-local workdir wrappers for submodule-relative commands, and per-device shard scripts. The combined script has 97 launchable commands; `gpu0.sh` has 49 and `gpu1.sh` has 48. These scripts must not be treated as accepted evidence until the preflight passes and accepted run metadata is collected.
- `scripts/uxfd_artifact_scaffold.py --output-root paper/UXFD_paper/results/accepted_run_templates` now generates 97 `run_meta.template.yaml` files plus a manifest for post-run metadata collection. The template root intentionally contains zero `run_meta.yaml` files, and `scripts/uxfd_artifact_gate.py` now rejects `accepted_evidence: false` and `TODO` placeholder values if a template is accidentally promoted without being filled.
- `scripts/uxfd_artifact_gate.py --require-queue-coverage` now requires accepted `run_meta.yaml` files to cover every launchable queue row. `scripts/uxfd_submission_gate.py` uses that mode, so a partial artifact set cannot satisfy the final evidence gate.
- `paper/UXFD_paper/results/artifact_gate_queue_coverage.md` records the current per-paper queue coverage summary. It currently shows `0/97` accepted queue rows covered because `paper/UXFD_paper/results/accepted_runs` does not exist yet.
- `paper/UXFD_paper/results/GPU_EXECUTION_RUNBOOK.md` now gives the concrete GPU execution order: live preflight, regenerate launch shards/templates, run `gpu0.sh`/`gpu1.sh`, promote filled metadata into `accepted_runs`, and run artifact/recent-work/submission/objective gates.
- `paper/UXFD_paper/results/gpu_queue_live_preflight.json` records the current live preflight result: `accepted=false`, `nvidia_smi_ok=false`, `torch_cuda_available=false`, and `torch_cuda_device_count=0`.
- Five verified non-accepted smoke/evidence gate updates were committed inside their owning submodules:
  - `paper/UXFD_paper/Explainable_FD_Toolkit` commit `b9c82e5` records the Toolkit ablation smoke gate.
  - `paper/UXFD_paper/1D-2D_fusion_explainable` commit `e6f9b58` records the fusion ablation smoke gate.
  - `paper/UXFD_paper/LLM_Explainable_FD_Toolkit` commit `f40255f` records the LLM evidence smoke gate and conservative IEEE TeX source checkpoint.
  - `paper/UXFD_paper/MOE_explainable` commit `e85c246` records the MoE ablation smoke gate.
  - `paper/UXFD_paper/Neuralsymbolic_theory` commit `fb9b98d` records the mapping ablation smoke gate.
- `paper/UXFD_paper/results/objective_audit_current.md` and `.json` persist the latest prompt-to-artifact audit after the submodule commits and dirty-triage report. Current summary remains `achieved=false`, `met=46`, `not_met=11`, `blocked=1`; dirty submodule counts are now `Explainable_FD_Toolkit:38`, `1D-2D_fusion_explainable:28`, `LLM_Explainable_FD_Toolkit:2`, `MOE_explainable:25`, `Paper_fuzzy_XFD:1`, and `Neuralsymbolic_theory:1`.
- `scripts/uxfd_submodule_dirty_triage.py` and `paper/UXFD_paper/results/submodule_dirty_triage.md` now classify the remaining 95 dirty submodule entries by review policy. The report is blocker triage only, not accepted experiment evidence; it marks result artifacts for promotion only through `scripts.uxfd_artifact_gate` and marks drafts/scripts/plans for owner review before any commit.
- The broad PHM literature inventory may still contain low-tier sources; it is not sufficient for UXFD TOP submission positioning.
- Exact reproduction of large TOP methods may be `resource-blocked` under 2x4090; use labelled representative runs instead of calling them exact baselines.
- Claude Team launch is prepared but not run because the external-service launch was policy-blocked.
- Paper 07 remains partial, not submission-ready: the 8-signal synthetic gate is now passed, but no accepted 6+ same-protocol baseline matrix exists yet, ablation evidence is not command-bound, TOP representatives are not mapped to local runs, industrial GPU proof is missing, and SOTA wording is still blocked.
- Papers 01-06 also remain partial or blocked. The new submodule commits are
  evidence-gate checkpoints, not final paper-package commits.

## Context to Remember

The new artifacts are a production system, not a claim that the seven papers are ready. Minimal root configs previously ran in Slice 4, but submission readiness requires canonical manuscripts, compile gates, claim-to-evidence maps, and submodule-local milestone commits.

## Next Steps

1. [ ] Review the goal package for wording and target-journal fit.
2. [ ] For each paper, select which TOP-source recent works are exact-runnable versus representative-runnable baselines.
3. [ ] After GPU/NVML is restored, run `paper/UXFD_paper/results/queue_launch_shards/gpu0.sh` and `gpu1.sh` in separate terminals; do not bypass their `nvidia-smi -L` and PyTorch CUDA preflight.
4. [ ] For each completed run, copy the matching `run_meta.template.yaml` from `paper/UXFD_paper/results/accepted_run_templates` into the accepted run directory as `run_meta.yaml`, fill every TODO field, attach `run.log`, `metrics.json`, and config evidence, then run `python -m scripts.uxfd_artifact_gate paper/UXFD_paper/results/accepted_runs --require-queue-coverage`.
5. [ ] Run `python -m scripts.uxfd_recent_work_gate --format markdown` before allowing any TOP/SOTA wording to enter a manuscript.
6. [ ] Run `python -m scripts.uxfd_objective_audit --format markdown` before any completion claim.
7. [ ] Decide whether to pursue an explicitly approved external Claude review or use local-only review.
8. [ ] Pick the first paper milestone and work only inside that submodule.
9. [ ] Resolve or intentionally commit paper-specific dirty submodule work before recording any parent gitlink update; do not commit broad submodule dirt without attributing the files.
10. [ ] Re-run `python -m scripts.uxfd_objective_audit --format markdown --output paper/UXFD_paper/results/objective_audit_current.md --allow-not-achieved` after each material readiness update.

## Files to Review on Resume

- `paper/UXFD_paper/goal/00_overall_goal.md` - shared goal and strict-reviewer rubric.
- `paper/UXFD_paper/goal/08_recent_work_citation_readme.md` - TOP recent-work citation and reproduction status map.
- `paper/UXFD_paper/goal/99_submission_readiness_matrix.md` - current cross-paper status.
- `specs/006-uxfd-ieee-trans-submission-readiness/tasks.md` - completed setup and remaining production backlog.
- `.codex/claude-team-runs/20260511-uxfd-ieee-trans-review/TASK_SPEC.md` - prepared Claude Team review task.
- `.codex/claude-team-runs/20260511-uxfd-ieee-trans-review/LAUNCH_LOG.md` - why the team was not launched.
- `paper/UXFD_paper/results/queue_launch_plan.sh` - generated two-GPU launch plan; not accepted evidence.
- `paper/UXFD_paper/results/queue_launch_shards/gpu0.sh` and `gpu1.sh` - generated per-GPU launch shards; not accepted evidence.
- `paper/UXFD_paper/results/accepted_run_templates/manifest.json` - generated accepted-run metadata scaffold; templates only, not accepted evidence.
- `paper/UXFD_paper/results/artifact_gate_queue_coverage.md` - current accepted-artifact coverage summary by paper; currently all missing.
- `paper/UXFD_paper/results/GPU_EXECUTION_RUNBOOK.md` - one-page execution procedure for the 2x4090 batch and evidence gates.
- `paper/UXFD_paper/results/gpu_queue_live_preflight.json` - current live GPU preflight snapshot; not accepted experiment evidence.
- `paper/UXFD_paper/results/objective_audit_current.md` - latest prompt-to-artifact audit, including the dirty-submodule gate.
- `paper/UXFD_paper/results/submodule_dirty_triage.md` - current residual dirty-submodule inventory and recommended handling.

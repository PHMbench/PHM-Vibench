# Session Handoff: UXFD IEEE Transactions Submission Readiness

**Date:** 2026-05-11
**Project:** `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench_fix`
**Session Duration:** current Codex implementation turn

## Current State

**Task:** Implement parent-level goal/spec workflow for seven UXFD IEEE Transactions paper submissions.
**Phase:** planning-to-implementation setup
**Progress:** parent control-plane artifacts created; paper-local manuscript/evidence work not started.

## What We Did

Created the seven-paper goal package under `paper/UXFD_paper/goal/` and the new Spec Kit feature under `specs/006-uxfd-ieee-trans-submission-readiness/`. Prepared a read-only Claude Code Team task spec and this handoff so future sessions can continue without rediscovering decisions. The goal package was then upgraded with six-baseline, ablation, SOTA-gate, TOP-venue recent-work citation/reproduction requirements, and a hard 2x4090 compute gate.
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
- `specs/006-uxfd-ieee-trans-submission-readiness/` - updated spec, plan, data model, contract, checklist, quickstart, and tasks for TOP-source recent-work gates.
- `paper/UXFD_paper/goal/00_overall_goal.md` and all seven paper goal files - added compute budget and `resource-blocked` exact-reproduction policy.
- `test/test_uxfd_paper_alignment_contract.py` - added regression tests for tracked submodule reproduction contracts and local GPU binding declarations.
- `paper/UXFD_paper/Explainable_FD_Toolkit` - submodule milestone commit `b76b5d8` replaces the missing placeholder figure/table in `manuscript/final_tex/main.tex`, adds `manuscript/T040_EVIDENCE_README.md`, and updates `VIBENCH.md`; follow-up commit `39b6a06` tracks `configs/vibench/min.yaml`.
- `paper/UXFD_paper/1D-2D_fusion_explainable` - submodule milestone commit `ecdae0a` adds `README_T041_SUBMISSION_READINESS.md`; follow-up commit `d548f11` tracks `VIBENCH.md` and `configs/vibench/min.yaml` with the current repo root.
- `paper/UXFD_paper/LLM_Explainable_FD_Toolkit` - submodule milestone commit `dc014de` adds `SUBMISSION_READINESS.md` and updates `VIBENCH.md`; follow-up commit `9a5b141` tracks `configs/vibench/min.yaml`.
- `paper/UXFD_paper/MOE_explainable` - submodule milestone commit `c2adc5a` adds `T043_SUBMISSION_READINESS_EVIDENCE.md`; follow-up commit `6992839` tracks `VIBENCH.md` and `configs/vibench/min.yaml` with the current repo root.
- `paper/UXFD_paper/Paper_fuzzy_XFD` - submodule milestone commit `53e6d1b` adds a compilable evidence snapshot, updates `VIBENCH.md` and `configs/vibench/min.yaml`, adds `doc/T044_submission_readiness_evidence.md`, and fixes the NumPy bool serializer in `scripts/run_fuzzy_baseline.py`.
- `paper/UXFD_paper/Paper_fuzzy_XFD` - follow-up submodule commit `b82c05f` adds `submission_prep/baseline_ablation_matrix.yaml` and `submission_prep/ieee_trans_readiness.md`, records seven command-bound baselines, six supported fuzzy ablations, dummy-smoke metrics, and the remaining hard-threshold/safety/no-rule-output blockers.
- `paper/UXFD_paper/Neuralsymbolic_theory` - submodule milestone commit `9139307` adds `report/T045_evidence_readiness.md`, updates `VIBENCH.md`, and fixes `simple_validation_demo.py` so failed P2 evidence is recorded as a boundary case; follow-up commit `e3e268d` tracks `configs/vibench/min.yaml`.
- `paper/UXFD_paper/TII_operator_attention` - submodule milestone commit `10a3d16` adds/updates `VIBENCH.md`, `configs/vibench/min.yaml`, `code/synthetic_verification.py`, synthetic validation outputs, rejection-recovery notes, and `submission_prep/ieee_trans_readiness.md`; follow-up commit `e8f8994` expands synthetic validation to eight signal classes.
- `paper/UXFD_paper/TII_operator_attention` - follow-up submodule commit `dd40adc` adds normalized canonical TeX entrypoint `manuscript/final_tex/main.tex`, updates `VIBENCH.md`, and records the compile gate in `submission_prep/ieee_trans_readiness.md`; follow-up commit `4315617` records the full `pdflatex`/`bibtex`/`pdflatex`/`pdflatex` compile flow; follow-up commit `e106fe8` fixes the five empty-year BibTeX warnings; follow-up commit `23990c0` binds a seven-baseline and six-ablation command matrix in `submission_prep/baseline_ablation_matrix.yaml`; follow-up commit `c5e960b` records dummy-smoke pass notes for B01/A01 and B02; follow-up commit `f306832` records B03-B05/B07 dummy-smoke pass notes and B06 Transformer import blocker; follow-up commit `0e037d9` records A02-A06 dummy-smoke pass notes so all six local ablation commands have dummy executable evidence; follow-up commit `2cae464` records B06 ConvTransformer dummy-smoke pass after restoring legacy model registration compatibility in the parent repo.

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
  - Remaining blockers: only five same-protocol diagnostic models, no ablations, no TOP proxy mapping, incomplete compute metadata, SOTA blocked, broader manuscript placeholders remain.
- Paper 02 `1D-2D_fusion_explainable` milestone:
  - `git diff --check -- README_T041_SUBMISSION_READINESS.md` passed.
  - `python scripts/truth_audit.py --paper-root . --output-dir /tmp/uxfd_1d2d_t041_truth_audit` passed and reported 3 blocking issues.
  - `latexmk ... paper_draft/NMI_Paper1_Fusion1D2D.tex` failed because `NatureMi.cls` is missing.
  - Remaining blockers: no accepted six-baseline matrix, no accepted fusion/alignment ablations, no local TOP representatives, TeX not compiling.
- Paper 03 `LLM_Explainable_FD_Toolkit` milestone:
  - `CUDA_VISIBLE_DEVICES=0 python experiments/scripts/run_minimal_llm_demo_standalone.py --mode pipeline` passed.
  - `python -m pytest -q code/tests/test_basic_functionality.py` failed on `ModuleNotFoundError: No module named 'llm'`.
  - Remaining blockers: no final IEEE TeX entrypoint, no accepted `results/llm_evidence/**` packages, no baseline/ablation/TOP/latency/anti-hallucination evidence.
- Paper 04 `MOE_explainable` milestone:
  - Worker `jq` checks passed for seed stability, routing analysis, expert ablation, and CWRU/XJTU bridge artifacts.
  - `python scripts/bind_submission_ready_evidence.py --mode review-evidence --paper-root . --output-dir /tmp/moe_t043_review_evidence_validate` passed, but the new T043 gate records why the older local `ready: true` does not satisfy the strict IEEE gate.
  - Remaining blockers: no six-baseline matrix, no full CWRU/XJTU multi-seed matrix, incomplete 2x4090 metadata, SOTA blocked.
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
  - Remaining blockers: placeholder TeX, missing validation scripts/configs, no accepted CWRU/XJTU multi-seed baseline/ablation artifacts, source-backed mapping evidence missing.

## Open Questions

- [ ] Should the read-only Claude Code Team be launched now, or only after a human reviews the newly created goal package?
- [ ] Which paper should be the first implementation milestone: MoE, because compile previously passed, or Toolkit, because it is upstream infrastructure?
- [ ] Should each paper adopt a normalized `manuscript/final_tex/main.tex` convention, or allow paper-specific canonical entrypoints?

## Blockers / Issues

- Parent worktree was already heavily dirty before this task.
- Several UXFD submodules have uncommitted or untracked content; treat all of it as existing user work until attributed.
- Most papers are blocked for submission readiness by placeholder TeX, missing canonical entrypoints, incomplete evidence, or unverified claim maps.
- TOP recent related work is now citation-mapped, but exact reproduction still requires paper-local commands, logs, and artifacts before it can count as exact SOTA evidence.
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
3. [ ] For each paper, bind baseline/ablation commands to `CUDA_VISIBLE_DEVICES=0` or `CUDA_VISIBLE_DEVICES=1` before running.
4. [ ] Decide whether to pursue an explicitly approved external Claude review or use local-only review.
5. [ ] Pick the first paper milestone and work only inside that submodule.
6. [ ] Commit paper-specific changes inside the submodule before recording any parent gitlink update.

## Files to Review on Resume

- `paper/UXFD_paper/goal/00_overall_goal.md` - shared goal and strict-reviewer rubric.
- `paper/UXFD_paper/goal/08_recent_work_citation_readme.md` - TOP recent-work citation and reproduction status map.
- `paper/UXFD_paper/goal/99_submission_readiness_matrix.md` - current cross-paper status.
- `specs/006-uxfd-ieee-trans-submission-readiness/tasks.md` - completed setup and remaining production backlog.
- `.codex/claude-team-runs/20260511-uxfd-ieee-trans-review/TASK_SPEC.md` - prepared Claude Team review task.
- `.codex/claude-team-runs/20260511-uxfd-ieee-trans-review/LAUNCH_LOG.md` - why the team was not launched.

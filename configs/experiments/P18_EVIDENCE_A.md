# Goal — P18 fixed-checkpoint source-fit and reference-reader diagnosis

This file specifies one bounded P18 evaluation for the existing research-configuration owner. It is not a status file, model registry or A0–A3 work queue. The paper repository's `paper/STATUS.md` alone decides current work; its `paper/experiments.md` owns subsequent scientific decisions. This contract neither enables P18 in tracked dev nor replaces a missing native configuration.

**Core claim / essential gap:** C2/C3 need a reliable classification chain and interpretable reference use. Existing low target scores cannot distinguish source fitting from generation/reader failure. Evaluate the original MFPT-held-out seed101 block before any new fit.

**Question and endpoints:** Per-source recording-balanced accuracy, window accuracy, macro-F1, balanced accuracy, class recall/confusion, predicted class shares and entropy; fixed-reader reference versus generated-input differences. The 95% source-training threshold is diagnostic, not a claim of generalization or a requirement to repeat training until success.

**Reuse:** Original v1 selected checkpoints, splits/windows, selection records and compatible saved predictions. Preserve prior CWRU/MFPT/JNU results, all registered seeds and failed runs. Do not repeat completed model training or full-Q equivalence studies.

## Inputs and protocol

- **Native implementation leads:** `src/model_factory/P18/QualifiedHSE.py`, `src/task_factory/task/DG/p18_inference.py`, `configs/experiments/p18/gate3_protocol.md`. These are recorded local research paths, not an assertion of availability in tracked dev.
- **Selected run:** `results/p18/gate3/preparation/mfpt_seed101_all_source_selection.json`, followed by its actual native configuration, split, audit and selected checkpoint references. Do not substitute a later/last checkpoint.
- **Data:** Existing PHMFactory Data Factory, user-supplied PHM-Vibench root, metadata and README. MFPT is excluded from fitting; CWRU/JNU retain the complete original source-training cohorts and original 48-window-per-record construction. Read the actual split rather than guessing record membership.
- **Comparison:** Selected OBS and S-L observed / true-reference-access / generated-deployment chains. Ref/gen use the same frozen reference reader and observed code. True reference enters only the isolated diagnostic reader, never the deployed generator.
- **Fixed:** original weights, ontology, preprocessing, groups/windows, evaluation-mode deterministic HSE positions, sampling seed and32 draws; no target fitting, relabeling, model substitution, source-normalization refit or selection.
- **Statistical unit:** original recording or verified physical group, not window. One existing optimization seed101 is the bounded diagnostic, not qualification of all seeds. Report complete source-training fit descriptively; use the frozen paired-group analysis separately for any already-saved held-out predictions.
- **Budget and availability:** zero optimization updates and zero HPO; at most one complete inference pass per required chain. Reuse compatible saved predictions. Local LQ_signal, one GPU excluding physical GPU2; leave existing jobs unchanged. Data, weights and the local P18 worktree are not mounted in the writing environment.

## Execution

Work in the actual PHMFactory research worktree. The following read-only commands inspect the required inputs; they are **not** a D0 inference command:

```bash
git status --short
git branch --show-current
git rev-parse HEAD
test -r src/model_factory/P18/QualifiedHSE.py
test -r src/task_factory/task/DG/p18_inference.py
test -r results/p18/gate3/preparation/mfpt_seed101_all_source_selection.json
```

The P18 Task was not retrievable on inspected child dev `3e0ac28`; that does not prove it is absent locally. The exact evaluation invocation remains **UNVERIFIED** until the local native Task/evaluator, resolved config and selected weights are read. Do not invent `--split train`, a future YAML or an alternative trainer. `scripts/run_tii_fit95.sh` is a separate S/DLinear seed0, 10000-update fit and is not a substitute.

1. Read the existing owner implementation and original selected-run manifest. Record the actual evaluation-only command and runtime resolution, including necessary uncommitted research code. Stop if the original experiment cannot be reconstructed; do not rebuild it from the paper's prose.
2. Let the existing Data Factory independently enumerate the complete expected source-training cohort. Keep training, source-validation and target populations separate.
3. Restore each original selected checkpoint in evaluation mode. Verify zero optimizer activity, then evaluate OBS and the three S-L chains without changing inference inputs or selecting another checkpoint.
4. Preserve raw probabilities before aggregation. Use the existing evaluator and exact correct/count convention by source; missing, duplicate or extra cohort rows must not yield a passing partial score.
5. On identical instances, report the following finite-reader differences with the frozen group-balanced F1 utility U:

$$
\Delta_{ref}=U(p_{ref})-U(p_{obs}),\qquad
\Delta_{gen}=U(p_{gen})-U(p_{obs}),\qquad
\Delta_{reader}=U(p_{ref})-U(p_{gen}).
$$

Ref/gen share a reader; comparison to OBS also changes the learned reader. Reference access is extra-information diagnosis, **not a Bayes upper bound**. Do not divide by `max(Delta_ref,epsilon)` or infer that a nonpositive finite-reader difference proves the target has no label information. Keep source-training and held-out analyses separate; do not open a new untouched target in this block.

## Artifacts and validation

Use the existing native artifact format. Planned location: `results/p18/diagnostics/d0_mfpt_seed101/` (not an existing result). Retain raw probabilities, integer labels, ontology order, source/recording/window keys, independently enumerated cohort, exact per-record correct/count, per-source per-chain metrics, paired differences, sampling settings, real command/config, code and selected-checkpoint/selection references, elapsed time, necessary logs and failures. Large weights/draws follow existing authorized storage with real references rather than compulsory Git upload.

Validate full cohort identity, original selected checkpoint, no optimizer steps, no hidden reference in deployed inference, finite normalized probabilities, unchanged metric/group weighting and common-instance comparisons. A complete below-95% result is valid negative diagnostic evidence; leakage, wrong labels or an incomplete scored population invalidate the affected comparison. A low score alone does not.

Map validated output to the paper's `tab:d0`, `sec:reference-utility` and existing baseline/STATUS entries. This contract does not write Results, activate method changes or qualify baselines from a single seed. Absence of one required resource is recorded as BLOCKED/UNKNOWN, not repaired with synthetic predictions.

## Failure and sync

If native inputs cannot be recovered, record exact missing paths and stop inference. Independent recovery of already-existing paper evidence can continue only as specified by the paper's current experiment table; no new training queue originates here. Preserve failures and protocol deviations. Environment fixes may preserve the original experiment; any scientific change returns to design rather than silently replacing v1.

Validated native changes and allowed artifacts use the existing focused branch/PR-to-dev process. This documentation-only contract does not validate a runtime or advance the parent gitlink. The parent links the accepted evidence after exact-child acceptance where required. No force push, master/main changes, unrelated merges or branch deletion. Return the actual binding/command, per-source per-chain results or blockers, validity and fit verdicts separately, artifact paths and synchronization status; stop before retraining or Results drafting.

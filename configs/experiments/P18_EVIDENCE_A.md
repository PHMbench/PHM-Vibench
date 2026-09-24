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
- **Statistical unit:** original recording or verified physical group, not window. One existing optimization seed101 is the bounded diagnostic, not qualification of all seeds. Report complete source-training fit descriptively; use the frozen paired-group analysis separately for any already-saved held-out predictions. Do not add seeds to this diagnostic block or bootstrap a training-set score into a generalization claim.
- **Budget and availability:** zero optimization updates and zero HPO; at most one complete inference pass per required chain, reusing compatible saved predictions. The path check below permits at most two additional replays of one fixed source batch, not two extra full-cohort runs. Local LQ_signal, one GPU excluding physical GPU2; leave existing jobs unchanged. Data, weights and the local P18 worktree are not mounted in the writing environment.

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

The P18 Task was not retrievable on inspected child dev `1902d839`; that does not prove it is absent locally. The exact evaluation invocation remains **UNVERIFIED** until the local native Task/evaluator, resolved config and selected weights are read. Do not invent `--split train`, a future YAML or an alternative trainer. `scripts/run_tii_fit95.sh` is a separate S/DLinear seed0, 10000-update fit and is not a substitute.

1. Read the existing owner implementation and original selected-run manifest. Record the actual evaluation-only command and runtime resolution, including necessary uncommitted research code. Stop if the original experiment cannot be reconstructed; do not rebuild it from the paper's prose.
2. Let the existing Data Factory independently enumerate the complete expected source-training cohort. Keep training, source-validation and target populations separate.
3. Restore each original selected checkpoint in evaluation mode. Bind the four paths below to the actual native functions and weights. Verify zero optimizer activity and the bounded path checks, then evaluate the complete cohorts without selecting another checkpoint.
4. Preserve raw probabilities before aggregation. Use the existing evaluator and exact correct/count convention by source; missing, duplicate or extra cohort rows must not yield a passing partial score.
5. On identical instances, report the following finite-reader differences with the frozen group-balanced F1 utility U:

$$
\Delta_{ref}=U(p_{ref})-U(p_{obs}),\qquad
\Delta_{gen}=U(p_{gen})-U(p_{obs}),\qquad
\Delta_{reader}=U(p_{ref})-U(p_{gen}).
$$

Ref/gen share a reader; comparison to OBS also changes the learned reader. Reference access is extra-information diagnosis, **not a Bayes upper bound**. Do not divide by `max(Delta_ref,epsilon)` or infer that a nonpositive finite-reader difference proves the target has no label information. Keep source-training and held-out analyses separate; do not open a new untouched target in this block.

### Bind the four actual prediction paths

| Output name | Checkpoint and consumed inputs | Scientific role |
|---|---|---|
| `p_obs` | Original source-selected standalone OBS, including its selected HSE/ResNet choice; deployment observation only | Direct-diagnosis reference, not the S-L auxiliary head |
| `p_anchor` | Observed classification head and R from the original selected S-L checkpoint | Metrics-only `S-L observed` row; no new primary contrast or checkpoint candidate |
| `p_ref` | Original selected S-L reference reader and observed code, supplied with the true missing reference in this isolated reader branch | Finite-reader reference-access control |
| `p_gen` | The same S-L reference reader and observed code; original deployed generator, sampler and32 draws | Deployed classification, not a teacher-forced or auxiliary prediction |

Record the resolved stage/head and actual input ownership in existing diagnostics; a YAML branch name is not proof that the corresponding head was evaluated. No absent or incompatible head may be silently substituted. In a paired/noisy extension ref/gen must share the same allowed observed-state conditioning; do not introduce new noise into the current point-observed v1 task.

### Validate the path before interpreting a difference

Reuse compatible prior path checks at the exact implementation/configuration first. Otherwise use only the first batch in the original source-cohort order, with its original batch size; never choose a batch based on labels or the size/sign of an effect. Reuse its unmodified predictions from the ordinary D0 pass.

- **Assigned and received:** trace the configured path to the native head, actual reference tensor, observed condition, eligibility basis and class order. Verify that ref/gen use identical reference-reader weights and that generator inputs contain only allowed observations. A branch that resolves to another head, hidden teacher forcing or the wrong cohort is an implementation defect.
- **Deterministic replay and leakage check:** at most two additional replays of this one S-L batch are allowed, with identical weights, observed inputs and restored sampling RNG. First repeat the unchanged batch to record the native repeatability floor. Then change only the hidden missing-reference tensor at an existing native evaluation seam, after observation construction, without changing the original observed signal/condition or recomputing it from the modified reference. Use a fixed cyclic re-pairing of the batch references; retain the paired keys and changed-entry count. The deployed probabilities should remain within the existing reproducibility criterion of the unchanged replay. A dependence on withheld truth is a validity failure. If no safe native seam is available, or the tensor does not vary in this batch, record the check as UNVERIFIED or UNINFORMATIVE rather than manufacturing evidence or adding another runtime.
- **Reader consumption versus outcome:** the same cyclic replacement can be passed to the isolated true-reference reader without another generator call. Record its probability change and argmax-change count separately. Unchanged argmax can coexist with changed probabilities; unchanged probabilities on this batch can reflect a fitted reader that ignores the reference or an uninformative probe. Neither observation alone proves a hard bug or that the reference has no diagnostic information. A changed output establishes sensitivity to this replacement, not useful information, correct calibration or positive performance.

These replacement outputs are `PROBE_ONLY`: exclude them from cohort scores, fitting verdicts, primary contrasts and method selection. Keep their cost separate. Do not require a positive reaction or rerun with more dramatic perturbations until a difference appears. Existing cold-start or head-reload checks can be reused; this document adds no optimizer, gradient test suite or model family.

## Artifacts and validation

Use the existing native artifact format. Planned location: `results/p18/diagnostics/d0_mfpt_seed101/` (not an existing result). Retain raw probabilities, integer labels, ontology order, source/recording/window keys, independently enumerated cohort, exact per-record correct/count, per-source per-chain metrics, paired differences, sampling settings, real command/config, code and selected-checkpoint/selection references, elapsed time, necessary logs and failures. Include all four output names and their resolved heads; retain bounded path-check facts in the same native diagnostics rather than a second registry. Large weights/draws follow existing authorized storage with real references rather than compulsory Git upload.

Validate full cohort identity, original selected checkpoint, no optimizer steps, no hidden reference in deployed inference, finite normalized probabilities, unchanged metric/group weighting and common-instance comparisons. A complete below-95% result is valid negative diagnostic evidence; leakage, wrong labels or an incomplete scored population invalidate the affected comparison. A low score alone does not.

Separate **hard defects** (wrong head/data/label/split, unused configured route, leakage or incorrect metric) from **behavioral limitations** (a correctly fitted path ignores an available reference, constant-class decisions, inadequate optimization or excess cost). A behavioral limitation may be a valid negative result, not a reason to erase a run; an unqualified baseline still cannot support a superiority claim. Nominally different but identical full-Q interventions are non-discriminating for a restriction claim, even when their execution is valid. Record execution, protocol validity, fit verdict and scientific interpretation separately. Do not certify all four paths when one remains unverified; independently valid paths may retain their measurements.

Map validated output to the paper's `tab:d0`, `sec:reference-utility` and existing baseline/STATUS entries. This contract does not write Results, activate method changes or qualify baselines from a single seed. Absence of one required resource is recorded as BLOCKED/UNKNOWN, not repaired with synthetic predictions.

## Failure and sync

If native inputs cannot be recovered, record exact missing paths and stop inference. Independent recovery of already-existing paper evidence can continue only as specified by the paper's current experiment table; no new training queue originates here. Preserve failures and protocol deviations. Environment fixes may preserve the original experiment; any scientific change returns to design rather than silently replacing v1.

Validated native changes and allowed artifacts use the existing focused branch/PR-to-dev process. This documentation-only contract does not validate a runtime or advance the parent gitlink. The parent links the accepted evidence after exact-child acceptance where required. No force push, master/main changes, unrelated merges or branch deletion. Return the actual binding/command, per-source per-chain results or blockers, validity and fit verdicts separately, artifact paths and synchronization status; stop before retraining or Results drafting.

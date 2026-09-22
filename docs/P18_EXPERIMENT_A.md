# Goal — P18 essential evidence A: recover, then measure the fixed source checkpoint

This is the implementation-side execution handoff for P18 HSE–LLapDiff, not PHMFactory's own Benchmark paper or the S/DLinear `tii_one_model` study. The parent paper's `paper/STATUS.md` remains the only current scientific judgment; `paper/experiments.md` maps this Goal to the manuscript. No new trainer, reader, configuration manager or result ledger is introduced.

**Core claims and essential gap:** C2/C3 require a reliable classification chain and a traceable comparison against observed-only, point, Gaussian, S-T and CSDI. Existing summaries report mixed results; they do not establish baseline qualification or a stable S-L advantage. The first unresolved action is still the original MFPT-held-out, seed101 D0 block. C1 is presently a formulation and identification boundary, not an empirically validated qualification advantage.

**Question / metrics:** Does the original source-selected model fit each source, and does its fitted reader use the actual missing reference more effectively than generated features? Report per-source recording-balanced accuracy, window accuracy, macro-F1, balanced accuracy, class recalls, confusion, class shares and probability entropy. The 95% training threshold is diagnostic; it is neither a generalization theorem nor an instruction to retrain until passing.

**Completed material to reuse:** The recorded CWRU/MFPT five-seed comparisons, JNU pilot, original split and selection files, original checkpoints, fixed numerical witnesses and the exact-threshold reducer already exist as records/prototypes. Verify their actual local versions and outputs before reusing them. Do not repeat completed training, full-Q equivalence runs or unchanged mathematical tests merely to fill a log.

## Inputs and protocol

- **Paper:** `liq22/P-18-HSE-Laplace-Unified-Representation`, canonical `paper/main.tex`, existing `paper/contributions.md`, `paper/experiments.md` and `paper/STATUS.md`.
- **Implementation:** this PHMFactory repository through its existing Data/Model/Task/Trainer path. Historical P18 paths are `src/model_factory/P18/QualifiedHSE.py`, `src/task_factory/task/DG/p18_inference.py` and `configs/experiments/p18/gate3_protocol.md`. These are evidence leads, not a claim that the current tracked dev contains the local research implementation.
- **Selected manifest:** `results/p18/gate3/preparation/mfpt_seed101_all_source_selection.json`, followed by the actual selected-config, audit and checkpoint references it contains. Never select a convenient newer or last checkpoint.
- **Data:** the user-supplied local PHM-Vibench root, `metadata.xlsx` and its README, read through the existing Data Factory. MFPT is held out; CWRU/JNU are the source-training cohorts. Keep the original recording roles and 48-window-per-record construction. Verify the manifest, rather than inferring membership from a filename.
- **Models:** the selected OBS model and selected S-L chain. For S-L, evaluate its observed head, true-reference-access head and generated/deployed head separately. The true-reference and generated paths must use the same frozen reader and the same observed code; do not substitute the M-track head for a selected P-track head.
- **Fixed conditions:** original checkpoint, labels/ontology, source records/windows, preprocessing, eval mode, deterministic HSE patch positions, sampling seed and 32 draws. No target fitting, loss change, method substitution, normalization refit or checkpoint reselection.
- **Unit / repetition:** highest trustworthy original recording or physical group, never window as an independent environment. One existing optimization seed (101) for this bounded diagnosis; it does not qualify every seed. Preserve the frozen analysis convention. Per-source training fit is descriptive; any paired interval for held-out data uses common group resampling and is kept separate from optimization-seed variability.
- **Budget:** zero optimization updates, zero new HPO trials, one pass per required frozen chain over the complete source-training cohorts. Reuse compatible saved predictions/draws. Offline restoration of other existing main-table artifacts may proceed independently; it authorizes no missing training runs. Use `LQ_signal`, one GPU per experiment excluding physical GPU2; do not stop or reconfigure existing jobs.

## Execution

### A0 — recover the existing evidence, without manufacturing a new run

Start read-only in both working trees. These commands inspect availability; they do not perform D0 inference:

```bash
git status --short
git branch --show-current
git rev-parse HEAD
git remote -v
# Run from the existing PHMFactory working tree:
test -r src/model_factory/P18/QualifiedHSE.py
test -r src/task_factory/task/DG/p18_inference.py
test -r results/p18/gate3/preparation/mfpt_seed101_all_source_selection.json
```

Read `AGENTS.md`, `CORE.md`, the relevant task/evaluator and the selected manifest. Record the actual import resolution, original configuration, group split, selection metric/trial, trained checkpoint and runtime revision, including any uncommitted research modifications needed to reconstruct that run. Never call a historical working-tree experiment reproducible from a clean commit alone when those modifications are missing.

**Access state at this handoff:** the P18 Task was not retrievable on inspected child dev `3e0ac28`; the parent snapshot describes a local research implementation and outputs. A 404 establishes inaccessible content at that reference, not that local work never occurred. The exact P18 D0 inference command is therefore **UNVERIFIED**. Bind it only after reading the actual local Task/evaluator and selected configuration. Do not invent `--split train`, a future YAML or a new training wrapper.

Do not use `scripts/run_tii_fit95.sh` as a substitute: that path trains a different S/DLinear model for 10,000 updates with seed0. Do not rebuild P18 from the paper merely to bypass unavailable original code. If the original implementation, cohort or selected weights cannot be recovered, record the precise blocker and stop A1.

### A1 — fixed-checkpoint source fit and reference-reader diagnosis

1. Have the existing Data Factory enumerate the complete expected source-training cohort independently of model predictions. Keep source-train, source-validation and target records separate.
2. Bind the real evaluation-only invocation to the original selected checkpoints. Confirm that no optimizer step or fit callback is called and that the same model state is retained.
3. Run OBS and the three S-L classification chains on that cohort. Reference truth may enter only the isolated reference-reader control; it never enters the deployed generator. Reuse existing compatible samples instead of drawing repeatedly until scores improve.
4. Use the existing evaluator and exact correct/count convention for each source. Do not average sources or seeds to hide a failure. Missing/duplicated/extra cohort rows require an explicit completeness failure rather than a partial score.
5. On each common cohort, calculate the three utility differences below. Save probabilities before aggregation. Do not open a new untouched target during this diagnostic block.

Let U be the frozen group-balanced macro-F1, `ref` the true-reference prediction, `gen` the generated prediction through that same reader, and `obs` the separately trained observed-only model:

$$
\Delta_{ref}=U(ref)-U(obs),\qquad
\Delta_{gen}=U(gen)-U(obs),\qquad
\Delta_{readout}=U(ref)-U(gen).
$$

The first two include a difference in trained readers. The third shares the reader but supplies different information. A finite reference reader is **not a Bayes upper bound**. Do not report `Delta_gen/max(Delta_ref,epsilon)` as a recovery fraction: it is unstable near zero and has no general upper-bound interpretation. A nonpositive reference difference means no demonstrated benefit for these readers/protocol, not absence of label information. Any already-saved source-validation or explored-target analysis is labelled separately and cannot tune the original model.

## Artifacts and validation

Use existing artifact conventions and direct paths. The planned output location is `results/p18/diagnostics/d0_mfpt_seed101/`; this path is not represented as an existing completed experiment.

Retain the actual command, configuration and overrides; original model/reader checkpoint references; code/runtime and selection provenance; the data-owned cohort; raw probabilities, integer labels, ontology order and source/recording/window keys; sampling seed/draw count; exact per-record correct/count; per-source per-chain metrics and the three paired utility differences; necessary logs, elapsed time and every failure. Large weights/draws follow the existing authorized storage policy, with accessible references rather than forced Git uploads.

Check: cohort identity/completeness; selected-checkpoint identity; no optimizer activity; source/target separation; unchanged deployment inputs; normalized finite probabilities; frozen metric convention; group weighting; common-instance comparison; and explicit absent-class/failed-row handling. Source-training accuracy >=0.95 is reported as a fit verdict, not the experiment-completion requirement. A valid below-threshold run is completed negative evidence. Leakage, wrong labels or an incomplete scored population invalidate the relevant comparison; a poor score alone does not.

Map outputs to the paper's reserved D0 table, `sec:reference-utility`, the existing baseline-qualification table and STATUS E. This execution Goal does not write Results or promote C1/C2/C3 to SUPPORTED.

## Subsequent essential evidence — not an automatic launch queue

| Item | Why required / smallest next treatment |
|---|---|
| A2: traceable competitive comparison | Recover/recompute existing P/M tables from selected native configurations, split, actual code, trials, checkpoints and raw outputs. Retain all frozen methods/seeds/failures and separate HPO cost from replay. Propose only genuinely missing essential cells after A1; no new training budget is granted here. |
| A3: empirical qualification or restriction advantage | Required only if that stronger claim is retained. First inspect the already recorded controlled qualification results. Full-Q cannot supply this contrast. A legal new contrast needs both eligible and unestimated missing components, known evaluation truth, common scored target and comparable information. Target-specific fitting, score slicing, and selecting coordinates from valid joint samples are different procedures; do not defeat a baseline by giving it an ill-defined target or denying its valid marginalization. |
| Independent confirmation | Required for a confirmatory claim about a revised method after using CWRU/MFPT/JNU to design it. Freeze the revision before opening a genuinely untouched population. No ten-dataset quota and no automatic new acquisition budget. |
| Optional only | Further fixed-output-class theory/parameterization experiments, exhaustive H_M/time-branch ablations, few-shot, ten-fold expansion and a full step/draw Pareto grid. Promote only when needed for a retained core claim. |

Laplace and H_M are inherited/optional hypotheses, not established improvements. Do not select S-T as a new primary method merely because the already observed targets favor it. A prospective change uses source evidence and is recorded as a new version; old results remain unchanged. Failure to beat a baseline, absence of significance, and a valid equivalence result are different outcomes.

## Failure and sync

If A1 is blocked, continue only independent offline recovery of already-existing A2 artifacts. If neither is possible, stop with the exact missing resource. Do not fill tables from the review text, infer accuracy from loss, replace data, remove failed seeds, rerun until favorable, or silently relax the protocol. Environment repair may preserve the same experiment; scientific changes return to design.

Sync validated implementation/configuration/evidence through a focused child branch and normal PR to dev. This documentation-only handoff does not certify an implementation revision or advance the paper's accepted gitlink. The parent updates its evidence map and, only after exact-child acceptance when needed, its gitlink in a separate PR. No force push, main/master edits, unrelated merges or branch deletion.

Return: actual local binding and commands; PASS/FAIL/BLOCKED by validity and by fit; per-source per-chain measurements and artifact paths; restored versus missing evidence; exact child/parent synchronization status; and the single next decision. Stop before Results writing or a new training campaign.

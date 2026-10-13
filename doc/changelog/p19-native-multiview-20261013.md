# P19 native multiview study — 2026-10-13

P19 now has a PHMFactory-native staged path for the existing fixed signal-view correction. The numerical method is preserved; the original 13 TSPN/readout outputs gain 12 single-view outputs from the same frozen export. No new backbone, routing method or geometry experiment is introduced.

## Behavior and ownership

- Public `phmfactory research` resolves the same explicit config/local-config/CLI layers and exposes preflight, metadata sanity, native baseline/TSPN smoke, source overfit, source tuning, readout freeze, source comparison, target evaluation, saved-prediction analysis and CSV-only plots. The task is `classification.multiview_signal`; normal factories still own loading, preprocessing, model/task construction, training and checkpoint selection.
- Paderborn preparation consumes the actual physical-role map, checks labels/rate/class support and writes derived source-only/evaluation metadata and native split manifests. Source factories cannot eagerly load the target population. Joint condition/entity folds exclude the unused cross combinations explicitly. Original raw files and metadata stay read-only.
- OP1 preserves its fixed three fits and 0.80 tune-Accuracy stop gate. DG adds four equal-budget optimizer trials at seed 11 plus the selected seeds 23/37 per model/fold; TSPN, ResNet1D and TCN are configured. Trial selection and temperature/readout/guard fitting are source-only; incomplete searches/freezes fail rather than silently reduce the cohort.
- Public predecessor artifacts must contain exactly the configured model/seed family. Empty, truncated, extra or duplicate cohorts fail before readout fitting, source replay or target access; completion is measured against the configured family rather than a supplied subset.
- Native features and logits are captured in one forward pass. Export holds parameters and running buffers frozen, enforces checkpoint replay and uses evaluation batch size 1. This is required before adding a batch-statistic-dependent model such as TimesNet to the comparison.
- CUDA execution binds the UUID obtained by querying physical GPU 0 before CUDA initialization and requires exactly one visible device. Ambiguous numeric visibility, another UUID and configuration overrides of the verified UUID fail explicitly. CPU diagnostics hide all CUDA devices; no GPU workload or real device-identity verification was performed for this documentation change.
- Saved analysis uses complete registered seeds/models, equal-entity Brier/Accuracy and weighted-confusion Macro-F1. Paired whole-entity resampling preserves seed dependence and nonlinear Macro-F1 semantics. Raw, projected and guarded identities remain separate; guarded MV Brier is the primary.

## Dependency correction: constant TSPN features

Native TSPN `CustomBatchNorm` previously computed population standard deviation via `var.sqrt()`. At a constant feature, the square-root derivative at zero could produce a nonfinite backward gradient even when the forward normalized value was finite. The training expression now uses the centered vector norm divided by the square root of batch size. This preserves the positive-variance population-standard-deviation formula, epsilon placement, running-state update and evaluation formula, while giving a finite derivative for a zero norm.

This is a real native dependency behavior change. It can alter training trajectories that previously encountered constant coordinates; it is not evidence that historical runs/checkpoints reproduce identically. Focused checks cover positive-variance values, constant-coordinate backward behavior, running updates and evaluation; native export additionally checks actual checkpoint logits/features before reuse.

## Validation and remaining boundary

Synthetic CPU tests exercise signal coordinates/units, 25 output identities, readout/state replay, weighted metrics, source/target isolation, native training/checkpoint/export and the zero-variance gradient correction. A complete three-seed stage fixture additionally checks calibration, readout freeze, source comparison, target output and analysis with mocked native export boundaries. These are software checks, not PHM performance measurements. The existing CPU grouped-representation workflow runs these targeted tests.

Local focused tests, documentation/config validation, public Dummy demo and the declared TSPN profile's CPU forward/backward checks passed. The once-at-final-review full `python -m pytest test/ -q` attempt stopped during collection: an installed `examples` package shadowed the checkout's namespace and `streamlit` was absent. No full-suite pass is claimed, and no unrelated UI/dependency changes were made to suppress those failures. Documentation validation checks links, not execution or scientific validity.

**NOT RUN for this implementation:** real PHM metadata/data qualification, local WSL/CUDA/GPU validation, full tuning/benchmark, formal multiseed results and real mechanism/ablation comparisons. One-epoch smoke is not a competitive baseline or tiny-set overfit result. An official strong external TSC reference is not yet protocol-qualified. The required second PHM dataset still needs physical-entity/domain/class/profile qualification; HAR still needs channel/subject/profile/native-dimension binding and cannot replace it.

Follow the sole [Local Execution SOP](../../docs/P19_TSPN_SIGNAL_STUDY.md) and [configuration index](../../configs/experiments/p19/README.md). Keep private arrays and raw data local. No scientific gain, baseline-valid status, remote merge or complete multi-PHM reproduction is claimed by this source change.

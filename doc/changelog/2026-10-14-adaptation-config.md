# Config-driven frozen-source and Tent evaluation

## Verified gap and change

The existing B01/B02 Python runtimes could not be selected through a TTA task configuration.
Bind both to the existing `phmfactory research` resolver/dispatcher with one small
`TTA/stream` task and a Data Factory reader for explicitly prepared target-only NPZ rows.
The model, strict source loader, protocol, adaptation implementation, metric accumulation
and result-summary writer are reused. No new method, CLI, Trainer, registry or dependency.

Two experimental YAMLs select frozen source or one-step Tent. Method/regime mismatches,
unknown task/data/trainer options, label-bearing updates, extra passes, implicit devices,
malformed NPZ observations and partial outputs are rejected. Row order is explicit and
must not be called chronological without upstream acquisition evidence. Only x reaches
the model. Full-population Macro-F1 is not averaged from batch F1.

## Literature and compatibility

The existing model catalogue now links one shared bibliography and an original technical
synopsis map with 26 primary-source records spanning generative objectives, TTA/SFDA,
online learning and limited PHM applications. This is not a full 187-candidate disposition
or SOTA ranking. Unfrozen code/license/reproduction dependencies remain reference-only.
The original candidate enum and model catalogue, generative runtime and all existing
TTA algorithms are unchanged. No SAR or online-supervised lifecycle is introduced.

## Validation and boundary

Local reconstructed affected-source tests passed 217 cases: 34 new Config/archive/CLI
checks plus the unchanged B00/B01/B02 tests. The two CLI processes used constructed target
observations and synthetic ResNet1D source checkpoints; no source training or real PHM
experiment ran. Existing installed-wheel CI already executes the extended Tent test file;
its current-head result must be read separately before merge. Local source-slice tests are
not a full current-checkout or clean-install claim.

An initial metric-binding attempt passed a list where the existing metric API requires a
DataFrame; the binding was corrected. A later test expected a nonexistent summary field;
it now checks the existing `iterations` field without changing the writer. Both failed
local logs are preserved. No test tolerance, original test or algorithm was weakened.

The archive is loaded in CPU memory. `validate` is configuration-only, not architecture,
data or source-checkpoint qualification. This Config entry does not save adapted weights,
implement whole-experiment resume, or measure adaptation speed. Source checkpoint paths,
build/stdout records and real-data split/preprocessing validation remain study-owned.
Rollback: revert the bounded squash commit. Do not change main, release/support claims,
unrelated PRs or scientific benchmark status.

# Explicit physical groups and frozen target readouts

Added a PCG64 physical-group assignment with explicit group counts and no hash
or manifest dependency. The source and held-out episode paths share the native
H5 reader and physical projection. The held-out episode uses the fixed five-shot
record draw and excludes every support physical group from query.

The source task now admits ordinary patch and native HSE baselines through the
same source-normalized union. A frozen-feature path and source-only alpha grid
support equal-group query NLL, macro-F1, accuracy and entropy, with both raw and
source-training-feature-RMS scales and saved probabilities. The physics impact
simulator and its bounded constructed-data checks accept explicit input and
output paths; result directories must be outside this repository.

The caller remains responsible for truthful checkpoint, source and episode
provenance and for a scientifically admissible support declaration. These
changes do not qualify industrial datasets or establish model convergence.

The float64 L-BFGS-B head uses an invertible ridge-regularized SVD coordinate
change and a constant objective shift/scale for large correlated features. The
original regularized objective, output logits, success requirement and
original-coordinate gradient threshold are unchanged; failed prior fits remain
failed historical results.

After L-BFGS-B, every head receives one Newton polishing step in the same
preconditioned coordinates. This addresses observed relative-objective stops
whose original gradients still exceeded the acceptance threshold. The step is
checked against the same objective; it does not change the regularizer or gate.

The frozen episode exporter can optionally save the matched tokenizer's common
branch before the backbone, without modifying its forward computation. A paired
class-stratified group bootstrap utility operates on already-aggregated physical
groups; its interval is conditional on the supplied fits and support episode.
The paired-results plotting command reads saved tables and exports editable SVG
and vector PDF without loading any encoder or industrial data.

The frozen-episode path additionally accepts the official MIT-licensed FISHER
inference implementation and an explicit caller-supplied checkpoint. The same
DataFactory, source RMS, episode and target readout remain in use; the report
discloses the original log-STFT preprocessing and any discarded coordinates.
The optional `fisher` dependency includes OmegaConf. This is an inference path,
not an implementation of the original EMA pretraining or a validated baseline.

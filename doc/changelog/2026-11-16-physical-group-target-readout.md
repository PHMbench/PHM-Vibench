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

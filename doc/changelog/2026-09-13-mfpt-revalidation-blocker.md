# Distinguish MFPT revalidation debt from an invalid baseline reference

The MFPT registry row already remains `sanity_ok / smoke_only`: bounded software execution
is retained, while the earlier `baseline_valid` scientific claim is suspended. Release
readiness now reports `BASELINE_REVALIDATION_REQUIRED` for that intentional state instead
of misclassifying the structurally valid row as `BASELINE_VALID_REFERENCE_INVALID`.

A malformed or missing registry row still reports `BASELINE_VALID_REFERENCE_INVALID`.
Audit mode displays the revalidation reason and exits successfully; release mode remains
blocked until reviewed current-source evidence restores `protocol_status=baseline_valid`.
No MFPT configuration, data split, model, hyperparameter, runtime, metric definition or
benchmark result changes in this correction.

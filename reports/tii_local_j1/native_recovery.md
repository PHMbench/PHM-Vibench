# Native M0 fixture recovery

PASS — constructed source fixtures, not J1-C industrial evidence.

Command: `python -m scripts.tii_verify_native_fixture --fixture reports/tii_local_j1/native_fixture_v1 --output reports/tii_local_j1`

CPU verification: 0.716 s; peak RSS 703020 KiB.

Both completed CLI checkpoints have global_step=20, 16 populated AdamW parameter states and only source heads 1/2.
Fresh native Data/Model/Task Factory reconstruction preserves all 20 sampled rounds, source RMS, optimizer state,
and tokens/features/logits bitwise across save/load. Source validation checkpoint scores match recomputation.

The 32 rows in native_source_fixture_predictions.csv are source_val windows (16 per arm), not target queries.

Constructed source fixtures; no industrial qualification, target head or query evaluation.
Replays saved states and the fixed round sampler; does not resume Trainer loops or take a new optimizer step.
Only CPU bitwise equality is established.

The two saved native fixture runs and this recovery check precede the validation
float64 correction. Their checkpoint monitor was stored as float32; the earlier
check does not establish 1e-8 selection precision for those runs. The separate
Task/callback regression checks float64 NLL logging and checkpoint selection on
the corrected code without retraining or replacing these preserved artifacts.

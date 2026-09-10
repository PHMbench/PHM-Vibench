# Streamlit Advanced fields follow the current YAML draft

Date: 2026-09-08

## User-visible correction

Advanced mode no longer initializes its common-field controls from the original template after the user edits the full YAML draft. The current YAML is parsed first; common fields are then created from those values. Editing the YAML creates a new UI-only widget revision so an older widget value cannot silently overwrite the new YAML on the next validation.

The revision counter is only Streamlit widget state. It is not a run identity, configuration digest, manifest, or scientific authority.

If the Advanced YAML is invalid, common-field controls are withheld until the YAML is corrected. The public configuration inspector remains the authority for the approved experiment.

## Validation

Focused tests cover unchanged and changed YAML drafts, strict parse failure, Quick Start isolation, and widget revision changes. The real Streamlit AppTest changes `trainer.num_epochs` in the YAML, verifies the common field shows that value, applies a new common-field value through validation, then edits YAML again and verifies the previous widget cannot override it.

No PHMFactory backend configuration, Factory, Pipeline, split, objective, metric, checkpoint, or experiment YAML logic changes in this correction.

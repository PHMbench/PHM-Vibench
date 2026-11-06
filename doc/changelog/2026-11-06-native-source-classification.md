# Native source-only classification

- `data.load_test=false` excludes test records before caching and materialization;
  `trainer.test_after_fit=true` is rejected with that setting. Default loading is unchanged.
- `data.source_h5_dir` explicitly reuses prepared immutable H5 inputs; selected records
  are copied into a separate writable cache, with no raw-reader fallback.
- DG classification trains and evaluates TSPN_fusion candidate logits, including after
  checkpoint restoration, while keeping its reference frozen.
- Operator-residual heads accept an optional `residual_group_count` for fixed full-model
  scaling in deletion controls. Omission preserves active-group scaling and state keys.

These are runtime capabilities, not evidence of diagnostic improvement.

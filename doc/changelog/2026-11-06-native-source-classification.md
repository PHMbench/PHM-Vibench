# Native source-only classification

- `data.load_test=false` excludes test records before caching and materialization;
  `trainer.test_after_fit=true` is rejected with that setting. Default loading is unchanged.
- `data.source_h5_dir` explicitly reuses prepared immutable H5 inputs; selected records
  are copied into a separate writable cache, with no raw-reader fallback.
- DG classification trains and evaluates TSPN_fusion candidate logits, including after
  checkpoint restoration, while keeping its reference frozen.

These are runtime capabilities, not evidence of diagnostic improvement.

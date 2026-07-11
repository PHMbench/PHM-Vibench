# PHM-Vibench v0.2.0 Release Notes

## Status

This is a v0.2.0 release-candidate state for the maintained demo surface.
Machine evidence currently shows:

- 7/7 maintained public demo smoke commands pass.
- 3/3 invalid-combination smoke commands fail as expected.
- Maintained tests pass in the `LQ_signal` environment.

Final publication still requires an independent reviewer verdict. The executor
does not self-approve the release.

## Supported Scope

The supported public surface is the set of maintained demo combinations in
`configs/config_registry.csv` where `category=demo` and `status=sanity_ok`.
See `SUPPORTED_COMBINATIONS.md` for the exact table.

The stable entrypoint remains:

```bash
python main.py --config <yaml> --override key=value
```

Use `configs/demo/00_smoke/dummy_dg.yaml` for an offline smoke run with
repo-shipped dummy data. The other maintained demos require PHM-Vibench metadata
and raw data supplied via `data.data_dir`.

## Evidence

Cycle-03 local evidence:

- Public demo smoke matrix:
  `reports/smoke_matrix_cycle03_public_demo_after_main_fix/summary.json`
- Invalid-combination matrix:
  `reports/invalid_matrix_cycle03_after_main_fix/summary.json`
- Maintained test evidence:
  `reports/test_evidence_cycle03_maintained_only.json`
- Pipeline coverage:
  `reports/pipeline_coverage_cycle03.json`
- Combination coverage:
  `reports/combination_coverage_cycle03.json`

Those report paths live in the external v0.2.0 goal pack, not in the repository.

## Notable Changes

- Top-level `pipeline` CLI overrides are now respected by `main.py`.
- `FS,classification` is explicitly registered for the config-first few-shot demo.
- The cross-system few-shot base task registry description now reflects its `GFS`
  task type.

## Boundaries

- `Pipeline_02_pretrain_fewshot` is release-supported only for the maintained
  single-stage pretrain demo unless a config provides a validated `stages:` plan.
- `Pipeline_03` is not part of the v0.2.0 supported public surface.
- Registry discovery alone is not treated as runtime support.


# Demo: Dummy DG Smoke (`demo_00_smoke_dummy_dg`)

Purpose: one-command end-to-end run using repo-shipped dummy metadata + raw CSV.

## Minimal Run

```bash
python main.py --config configs/demo/00_smoke/dummy_dg.yaml
```

## Expected Outputs

- Output base dir: `results/demo/dummy_dg_smoke/`
- A subfolder `{experiment_name}/iter_0/` with Lightning logs/checkpoints.

## Recommended Overrides

```bash
python main.py --config configs/demo/00_smoke/dummy_dg.yaml --override trainer.num_epochs=1
```

## Common Pitfalls

1) Running from a different working directory (paths are repo-relative).
2) Deleting `data/metadata_dummy.csv`.
3) Using a very large `data.batch_size` (the smoke dataset is tiny; keep it small, e.g. `4`).
4) Increasing `data.window_size` beyond the dummy signal length (when using real dummy CSVs).

Note:
- If `data/raw/Dummy_Data/*.csv` is missing, `Dummy_Data` will generate synthetic signals so the smoke demo still runs.

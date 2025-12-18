# `configs/base/data/`

## 1) What This Block Controls

`data` controls:
- where metadata lives (`data_dir`, `metadata_file`)
- dataset windowing / normalization
- dataloader worker settings

## 2) Minimal Example (YAML Snippet)

```yaml
data:
  data_dir: "/path/to/PHM-Vibench"
  metadata_file: "metadata.xlsx"
  batch_size: 256
  num_workers: 8
  window_size: 4096
  stride: 5
```

## 3) Key Fields

| Field | Type | Notes |
|---|---:|---|
| `data.data_dir` | str | Root dir containing `metadata_file` and `raw/` (and optional caches). |
| `data.metadata_file` | str | CSV/XLSX metadata filename. |
| `data.window_size` | int | Sliding window length. |
| `data.stride` | int | Sliding window stride. |
| `data.num_workers` | int | PyTorch DataLoader workers. |

## 4) Typical Overrides

```bash
python main.py --config <yaml> --override data.data_dir=/path/to/PHM-Vibench
python main.py --config <yaml> --override data.metadata_file=metadata.xlsx
python main.py --config <yaml> --override data.num_workers=0
```

## 5) Coupling Notes

- Metadata is loaded by `src/data_factory/data_factory.py:_init_metadata`.
- Raw files are expected under `{data_dir}/raw/<Name>/<File>` where `Name` and `File` come from metadata.

## 6) How to Extend

- Add a new dataset reader under `src/data_factory/reader/` and ensure metadata `Name` matches the module name.
- For custom data factories, register a new factory name in `src/data_factory/data_factory.py` and set `data.factory_name`.

## 7) Common Pitfalls

1) Metadata file missing → the loader may attempt network download (use local files for reproducibility).
2) Wrong `Name` in metadata → `import src.data_factory.reader.<Name>` fails.
3) Raw file path mismatch (`raw/<Name>/<File>`).
4) `window_size` larger than signal length → zero samples.
5) Too many `num_workers` on small datasets → slower or unstable.


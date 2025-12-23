# data_factory/reader

Dataset readers: each file implements how to load a dataset's raw file into a NumPy array.

Conventions:
- Reader module name matches the `Name` field in metadata (e.g. `Dummy_Data`, `RM_006_THU`).
- Each reader provides a `read(file_path, args_data, *...) -> np.ndarray` function.
- Returned array is typically `(L, C)` and may be expanded by the factory to `(L, C, 1)` as needed.

Smoke/demo note:
- `Dummy_Data` supports generating **synthetic data** when raw files are not present, so
  `configs/demo/00_smoke/dummy_dg.yaml` can run offline.


"""Repair RM_006_THU metadata and H5 cache entries.

Default mode is dry-run. Use ``--apply`` only after reviewing the printed plan.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import h5py
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_factory.reader import RM_006_THU


DATASET_NAME = "RM_006_THU"


def _load_thu006_arrays(data_dir: Path, rows: pd.DataFrame) -> dict[str, object]:
    arrays = {}
    for _, row in rows.iterrows():
        key = str(int(row["Id"]))
        raw_path = data_dir / "raw" / DATASET_NAME / str(row["File"])
        arrays[key] = RM_006_THU.read(raw_path)
    return arrays


def _canonical_metadata(df: pd.DataFrame, arrays: dict[str, object]) -> pd.DataFrame:
    is_thu = df["Name"].astype(str).eq(DATASET_NAME)
    is_canonical = is_thu & df["File"].astype(str).str.startswith("vibration/")
    out = df[~is_thu | is_canonical].copy()
    for key, array in arrays.items():
        idx = out["Id"].astype(str).eq(key)
        out.loc[idx, "Sample_rate"] = 20480
        out.loc[idx, "Sample_lenth"] = int(array.shape[0])
        out.loc[idx, "Channel"] = int(array.shape[1])
    return out


def _backup_h5_keys(src: Path, dst: Path, keys: list[str]) -> None:
    if not src.exists():
        return
    with h5py.File(src, "r") as h5_in, h5py.File(dst, "w") as h5_out:
        for key in keys:
            if key in h5_in:
                h5_out.create_dataset(key, data=h5_in[key][()])


def _known_thu006_keys(data_dir: Path, current_df: pd.DataFrame, dataset_h5: Path) -> list[str]:
    keys = {
        str(int(value))
        for value in current_df[current_df["Name"].astype(str).eq(DATASET_NAME)]["Id"].tolist()
    }
    if dataset_h5.exists():
        with h5py.File(dataset_h5, "r") as h5f:
            keys.update(str(key) for key in h5f.keys())
    for metadata_path in data_dir.glob("metadata*.xlsx"):
        try:
            df = pd.read_excel(metadata_path, usecols=["Id", "Name"])
        except Exception:
            continue
        rows = df[df["Name"].astype(str).eq(DATASET_NAME)]
        keys.update(str(int(value)) for value in rows["Id"].tolist())
    return sorted(keys, key=int)


def _write_dataset_h5(path: Path, arrays: dict[str, object]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with h5py.File(tmp, "w") as h5f:
        for key, array in arrays.items():
            h5f.create_dataset(key, data=array)
    tmp.replace(path)


def _update_cache_h5(path: Path, old_keys: list[str], arrays: dict[str, object]) -> None:
    with h5py.File(path, "a") as h5f:
        for key in old_keys:
            if key in h5f:
                del h5f[key]
        for key, array in arrays.items():
            h5f.create_dataset(key, data=array)


def repair(data_dir: Path, metadata_file: str, apply: bool) -> None:
    metadata_path = data_dir / metadata_file
    dataset_h5 = data_dir / f"{DATASET_NAME}.h5"
    cache_h5 = data_dir / "cache.h5"
    df = pd.read_excel(metadata_path)
    thu_rows = df[df["Name"].astype(str).eq(DATASET_NAME)]
    canonical_rows = thu_rows[thu_rows["File"].astype(str).str.startswith("vibration/")]
    old_keys = _known_thu006_keys(data_dir, df, dataset_h5)

    if thu_rows.empty:
        raise RuntimeError(f"No {DATASET_NAME} rows found in {metadata_path}")
    if canonical_rows.empty:
        raise RuntimeError(
            f"No canonical {DATASET_NAME} vibration/ rows found in {metadata_path}; "
            "refusing to rewrite metadata/cache."
        )

    arrays = _load_thu006_arrays(data_dir, canonical_rows)
    if not arrays or len(arrays) != len(canonical_rows):
        raise RuntimeError(
            f"Loaded {len(arrays)} {DATASET_NAME} array(s) for "
            f"{len(canonical_rows)} canonical metadata row(s); refusing to apply."
        )
    fixed_df = _canonical_metadata(df, arrays)

    print(f"metadata: {metadata_path}")
    print(f"old {DATASET_NAME} rows: {len(thu_rows)}")
    print(f"new {DATASET_NAME} rows: {len(canonical_rows)}")
    print("new shapes:")
    for key, array in arrays.items():
        print(f"  {key}: {array.shape}")

    if not apply:
        print("dry-run only; pass --apply to write metadata and H5 files")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = data_dir / "backups" / f"thu006_fix_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(metadata_path, backup_dir / metadata_path.name)

    _backup_h5_keys(dataset_h5, backup_dir / f"{DATASET_NAME}_old.h5", old_keys)
    _backup_h5_keys(cache_h5, backup_dir / "cache_THU006_old.h5", old_keys)

    fixed_df.to_excel(metadata_path, index=False)
    _write_dataset_h5(dataset_h5, arrays)
    _update_cache_h5(cache_h5, old_keys, arrays)
    print(f"backup: {backup_dir}")
    print("applied")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="/home/user/data/PHMbenchdata/PHM-Vibench",
        type=Path,
    )
    parser.add_argument("--metadata-file", default="metadata.xlsx")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    repair(args.data_dir, args.metadata_file, args.apply)


if __name__ == "__main__":
    main()

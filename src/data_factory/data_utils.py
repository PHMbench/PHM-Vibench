from __future__ import annotations

import os
from pathlib import Path
import subprocess
from typing import Optional

import pandas as pd


def download_data(
    data_file: Optional[str] = "metadata.xlsx",
    save_path: Optional[str] = "./data/",
    source: str = "auto",
) -> bool:
    """Explicitly download one data file from a configured remote provider.

    This helper is retained for explicit provider workflows. The maintained
    ``ExplicitDataFactory`` never calls it during a normal local run.
    """

    os.makedirs(save_path, exist_ok=True)
    target = os.path.join(save_path, data_file)
    if os.path.exists(target):
        print(f"[INFO] 数据文件已存在: {target}")
        return True

    success = False
    if source in {"auto", "modelscope"}:
        print(f"[INFO] 尝试从 ModelScope 下载 {data_file}...")
        try:
            command = (
                "modelscope download --dataset PHMbench/PHM-Vibench "
                f"{data_file} --local_dir {save_path}"
            )
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode == 0:
                print(f"[SUCCESS] 从 ModelScope 成功下载 {data_file}")
                success = True
            else:
                print(f"[WARNING] ModelScope 下载失败: {result.stderr}")
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            print(f"[WARNING] ModelScope 下载失败: {exc}")

    if not success and source in {"auto", "huggingface"}:
        print(f"[INFO] 尝试从 Hugging Face 下载 {data_file}...")
        try:
            from huggingface_hub import hf_hub_download

            hf_hub_download(
                repo_id="PHMbench/PHM-Vibench",
                filename=data_file,
                repo_type="dataset",
                local_dir=save_path,
                local_dir_use_symlinks=False,
            )
            print(f"[SUCCESS] 从 Hugging Face 成功下载 {data_file}")
            success = True
        except ImportError:
            print("[WARNING] huggingface_hub 未安装")
        except Exception as exc:
            print(f"[WARNING] Hugging Face 下载失败: {exc}")

    if not success:
        print(f"[ERROR] 无法下载 {data_file}")
    return success


def _metadata_encoding(value: object | None) -> str:
    if value is None:
        return "utf-8-sig"
    if not isinstance(value, str) or not value.strip():
        raise TypeError(
            "data.metadata_encoding must be a non-empty encoding name when provided"
        )
    return value.strip()


def read_metadata_table(
    file_path: str | os.PathLike[str],
    *,
    encoding: object | None = None,
) -> pd.DataFrame:
    """Read one declared local metadata file with extension-defined semantics.

    ``.csv`` means comma-separated text and ``.tsv`` means tab-separated text.
    Both default to UTF-8 with optional BOM. Excel files use pandas' Excel
    reader. The function never guesses a separator, drops undecodable bytes, or
    tries a second encoding after a parse failure.
    """

    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Metadata file not found: {path}. Set data.data_dir and "
            "data.metadata_file to an existing local file. Normal runs do not "
            "download replacement metadata."
        )

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        if encoding is not None:
            raise ValueError(
                "data.metadata_encoding applies only to .csv and .tsv metadata"
            )
        return pd.read_excel(path)

    if suffix == ".csv":
        separator = ","
    elif suffix == ".tsv":
        separator = "\t"
    else:
        raise ValueError(
            f"Unsupported metadata format {suffix!r} for {path}. "
            "Use .csv, .tsv, .xlsx, or .xls."
        )

    return pd.read_csv(
        path,
        sep=separator,
        encoding=_metadata_encoding(encoding),
    )


def smart_read_csv(
    file_path: str | os.PathLike[str],
    auto_detect: bool = True,
    *,
    encoding: object | None = None,
) -> pd.DataFrame:
    """Compatibility alias for strict extension-defined metadata parsing.

    ``auto_detect`` is retained for existing callers but no longer enables
    encoding or delimiter guessing.
    """

    if not isinstance(auto_detect, bool):
        raise TypeError("auto_detect must be a boolean")
    return read_metadata_table(file_path, encoding=encoding)


class MetadataAccessor:
    """Provide dictionary-like access to a metadata DataFrame."""

    def __init__(self, dataframe: pd.DataFrame, key_column: str = "Id"):
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                f"metadata must be a pandas.DataFrame, got {type(dataframe).__name__}"
            )
        if dataframe.empty:
            raise ValueError("metadata contains no rows")
        if key_column not in dataframe.columns:
            raise ValueError(
                f"metadata is missing required column {key_column!r}; "
                f"observed columns: {list(dataframe.columns)}"
            )
        if dataframe[key_column].isna().any():
            raise ValueError(f"metadata column {key_column!r} contains missing values")
        duplicated = dataframe.loc[
            dataframe[key_column].duplicated(keep=False), key_column
        ].tolist()
        if duplicated:
            raise ValueError(
                f"metadata column {key_column!r} must be unique; duplicate values: "
                f"{duplicated}"
            )

        self.df = dataframe.copy()
        self.key_column = key_column
        self.df.set_index(key_column, inplace=True, drop=False)

    def __getitem__(self, key):
        try:
            return self.df.loc[key].to_dict()
        except KeyError as exc:
            raise KeyError(f"找不到ID为{key}的记录") from exc

    def __contains__(self, key):
        return key in self.df.index

    def keys(self):
        return list(self.df[self.key_column])

    def items(self):
        for key in self.keys():
            yield key, self[key]

    def values(self):
        return [row.to_dict() for _, row in self.df.iterrows()]

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __len__(self):
        return len(self.df)

    def query(self, query_str):
        return self.df.query(query_str)

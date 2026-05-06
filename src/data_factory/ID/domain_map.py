from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = ["domain_id", "load", "rpm", "system_id", "sampling_rate"]


def hash_file(path: str | Path) -> str:
    """Return the SHA256 hash for a domain map file."""
    file_path = Path(path)
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_domain_map(path: str | Path) -> pd.DataFrame:
    """Load a CSV domain map and validate the PHM generative contract."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"domain_map not found: {file_path}")
    df = pd.read_csv(file_path)
    validate_domain_map(df)
    return df


def validate_domain_map(df: pd.DataFrame, required_columns: Iterable[str] = REQUIRED_COLUMNS) -> None:
    """Validate required columns and unique domain IDs."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"domain_map missing required columns: {missing}")
    if df["domain_id"].isna().any():
        raise ValueError("domain_map.domain_id contains missing values")
    duplicated = df["domain_id"][df["domain_id"].duplicated()].tolist()
    if duplicated:
        raise ValueError(f"domain_map.domain_id must be unique; duplicates={duplicated}")


def load_domain_map_with_hash(path: str | Path) -> tuple[pd.DataFrame, str]:
    """Load and validate a domain map, returning the dataframe and SHA256."""
    return load_domain_map(path), hash_file(path)

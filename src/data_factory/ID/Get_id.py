from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd


def _declared_values(value: Any, name: str) -> list[Any]:
    """Return one explicit non-empty configuration value list."""

    if value is None:
        raise ValueError(f"{name} must be declared.")
    if isinstance(value, (str, bytes)):
        values = [value]
    elif isinstance(value, Iterable):
        values = list(value)
    else:
        values = [value]
    if not values:
        raise ValueError(f"{name} must contain at least one value.")
    if any(pd.isna(item) for item in values):
        raise ValueError(f"{name} must not contain null values: {values!r}.")
    return values


def _require_valid_classification_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Reject invalid supervised labels instead of deleting metadata rows."""

    if "Label" not in df.columns:
        raise ValueError("Supervised classification metadata must contain 'Label'.")

    invalid = df[df["Label"].isna() | (df["Label"] == -1)]
    if not invalid.empty:
        invalid_ids = (
            invalid["Id"].tolist() if "Id" in invalid.columns else invalid.index.tolist()
        )
        raise ValueError(
            "Supervised classification metadata contains missing or -1 labels for "
            f"ID(s) {invalid_ids}. Fix the metadata instead of dropping rows."
        )
    return df.copy()


def _available_domains(df: pd.DataFrame, context: str) -> list[Any]:
    """Return sorted, fully declared domains for one selected population."""

    if "Domain_id" not in df.columns:
        raise ValueError(f"{context} metadata must contain 'Domain_id'.")
    invalid = df[df["Domain_id"].isna()]
    if not invalid.empty:
        invalid_ids = (
            invalid["Id"].tolist() if "Id" in invalid.columns else invalid.index.tolist()
        )
        raise ValueError(
            f"{context} metadata contains missing Domain_id for ID(s) {invalid_ids}."
        )
    domains = sorted(df["Domain_id"].unique().tolist())
    if not domains:
        raise ValueError(f"{context} metadata contains no domains.")
    return domains


def _require_nonempty_split(train_df: pd.DataFrame, test_df: pd.DataFrame, context: str) -> None:
    if train_df.empty:
        raise ValueError(f"{context} produced an empty training/validation population.")
    if test_df.empty:
        raise ValueError(f"{context} produced an empty test population.")


def Get_DG_ids(metadata_accessor, args_task):
    """Return IDs for one executable domain-generalization protocol."""

    target_systems = _declared_values(
        getattr(args_task, "target_system_id", None),
        "task.target_system_id",
    )
    if "Dataset_id" not in metadata_accessor.df.columns:
        raise ValueError("DG metadata must contain 'Dataset_id'.")

    system_df = metadata_accessor.df[
        metadata_accessor.df["Dataset_id"].isin(target_systems)
    ].copy()
    if system_df.empty:
        raise ValueError(
            f"DG target_system_id={target_systems!r} matched no metadata rows."
        )
    system_df = _require_valid_classification_labels(system_df)
    all_domains = _available_domains(system_df, "DG")

    raw_target_count = getattr(args_task, "target_domain_num", 0)
    target_count = int(raw_target_count or 0)
    if target_count < 0:
        raise ValueError(
            f"task.target_domain_num must be non-negative, got {raw_target_count!r}."
        )

    if target_count > 0:
        if len(all_domains) <= target_count:
            raise ValueError(
                "DG target_domain_num cannot be satisfied: requested "
                f"{target_count} test domain(s), but available domains are "
                f"{all_domains}. At least one source domain must remain for training."
            )
        train_domains = all_domains[:-target_count]
        test_domains = all_domains[-target_count:]
    else:
        source_domains = _declared_values(
            getattr(args_task, "source_domain_id", None),
            "task.source_domain_id",
        )
        target_domains = _declared_values(
            getattr(args_task, "target_domain_id", None),
            "task.target_domain_id",
        )
        overlap = sorted(set(source_domains) & set(target_domains))
        if overlap:
            raise ValueError(
                f"DG source and target domains must be disjoint; overlap={overlap}."
            )
        missing = sorted(
            (set(source_domains) | set(target_domains)) - set(all_domains),
            key=str,
        )
        if missing:
            raise ValueError(
                f"DG requested domain(s) {missing} are absent; available={all_domains}."
            )
        train_domains = source_domains
        test_domains = target_domains

    train_df = system_df[system_df["Domain_id"].isin(train_domains)]
    test_df = system_df[system_df["Domain_id"].isin(test_domains)]
    _require_nonempty_split(train_df, test_df, "DG")

    train_val_ids = train_df["Id"].tolist()
    test_ids = test_df["Id"].tolist()
    if set(train_val_ids) & set(test_ids):
        raise ValueError("DG produced overlapping train and test IDs.")

    print("DG划分 - 使用显式可执行域协议")
    print(f"  - 训练域: {train_domains}")
    print(f"  - 测试域: {test_domains}")
    print(f"训练/验证样本数: {len(train_val_ids)}")
    print(f"测试样本数: {len(test_ids)}")
    return train_val_ids, test_ids


def Get_CDDG_ids(metadata_accessor, args_task):
    """Return IDs for a per-system cross-domain generalization protocol."""

    target_systems = _declared_values(
        getattr(args_task, "target_system_id", None),
        "task.target_system_id",
    )
    raw_target_count = getattr(args_task, "target_domain_num", None)
    if raw_target_count is None:
        raise ValueError("CDDG requires task.target_domain_num.")
    target_count = int(raw_target_count)
    if target_count <= 0:
        raise ValueError(
            f"CDDG task.target_domain_num must be positive, got {raw_target_count!r}."
        )
    if "Dataset_id" not in metadata_accessor.df.columns:
        raise ValueError("CDDG metadata must contain 'Dataset_id'.")

    filtered_df = metadata_accessor.df[
        metadata_accessor.df["Dataset_id"].isin(target_systems)
    ].copy()
    if filtered_df.empty:
        raise ValueError(
            f"CDDG target_system_id={target_systems!r} matched no metadata rows."
        )
    filtered_df = _require_valid_classification_labels(filtered_df)

    train_domains: dict[Any, list[Any]] = {}
    test_domains: dict[Any, list[Any]] = {}
    train_val_ids: list[Any] = []
    test_ids: list[Any] = []

    for dataset_id in target_systems:
        dataset_df = filtered_df[filtered_df["Dataset_id"] == dataset_id]
        if dataset_df.empty:
            raise ValueError(
                f"CDDG target system {dataset_id!r} matched no metadata rows."
            )
        domains = _available_domains(dataset_df, f"CDDG dataset {dataset_id!r}")
        if len(domains) <= target_count:
            raise ValueError(
                f"CDDG dataset {dataset_id!r} cannot reserve {target_count} test "
                f"domain(s) from available domains {domains}; at least one training "
                "domain must remain."
            )

        train_domains[dataset_id] = domains[:-target_count]
        test_domains[dataset_id] = domains[-target_count:]

        train_rows = dataset_df[
            dataset_df["Domain_id"].isin(train_domains[dataset_id])
        ]
        test_rows = dataset_df[
            dataset_df["Domain_id"].isin(test_domains[dataset_id])
        ]
        _require_nonempty_split(train_rows, test_rows, f"CDDG dataset {dataset_id!r}")
        train_val_ids.extend(train_rows["Id"].tolist())
        test_ids.extend(test_rows["Id"].tolist())

    if set(train_val_ids) & set(test_ids):
        raise ValueError("CDDG produced overlapping train and test IDs.")

    print(
        f"CDDG划分 - 每个数据集保留最后 {target_count} 个 domain 作为测试集"
    )
    for dataset_id in target_systems:
        print(f"数据集 {dataset_id}:")
        print(f"  - 训练域: {train_domains[dataset_id]}")
        print(f"  - 测试域: {test_domains[dataset_id]}")
    print(f"训练/验证样本数: {len(train_val_ids)}")
    print(f"测试样本数: {len(test_ids)}")
    return train_val_ids, test_ids

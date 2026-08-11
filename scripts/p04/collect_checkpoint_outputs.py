"""Collect frozen P04 mechanism-evaluator inputs from one Lightning checkpoint.

The collector performs exactly one intact model forward per batch.  Expert
deletions and fixed-mass output substitutions are then computed algebraically
from that intact forward, so neither operation can rerun or otherwise change the
router.  Every expert-indexed output is deterministically blinded before it is
written.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

# Direct ``python scripts/p04/...`` execution otherwise exposes only the p04
# script directory, not the nested Vibench package root containing ``src``.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.configs.config_utils import load_config
from src.model_factory import build_model
from scripts.p04.evaluate_role_identification import (
    COLLECTION_PHASE_ORDER,
    build_preintervention_assignment_seal,
)


SCHEMA = "p04.mechanism-evaluator-input.v1"
BLINDING_DOMAIN = "P04-BLIND-v1"
SELECTED_PARTITIONS = ("identification", "intervention")
EXPECTED_PARTITIONS = (
    "train",
    "optimization_validation",
    "identification",
    "intervention",
)
DELETION_DENOMINATOR_MINIMUM = 1.0e-6
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_file(path: str | Path, description: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} does not exist: {resolved}")
    return resolved


def _mapping(value: Any, description: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "__dict__"):
        return vars(value)
    raise ValueError(f"{description} must be a mapping")


def _config_value(config: Any, *keys: str) -> Any:
    value = config
    traversed: list[str] = []
    for key in keys:
        traversed.append(key)
        if isinstance(value, Mapping):
            if key not in value:
                raise ValueError(f"resolved config is missing {'.'.join(traversed)}")
            value = value[key]
        elif hasattr(value, key):
            value = getattr(value, key)
        else:
            raise ValueError(f"resolved config is missing {'.'.join(traversed)}")
    return value


def _assert_configured_path(
    configured: Any,
    supplied: Path,
    config_path: Path,
    description: str,
) -> None:
    if configured is None or not str(configured).strip():
        raise ValueError(f"resolved config must bind {description}")
    configured_path = Path(str(configured)).expanduser()
    if configured_path.is_absolute():
        candidates = {configured_path.resolve()}
    else:
        candidates = {
            (Path.cwd() / configured_path).resolve(),
            (config_path.parent / configured_path).resolve(),
            (Path(__file__).resolve().parents[2] / configured_path).resolve(),
        }
    if supplied not in candidates:
        rendered = ", ".join(sorted(str(candidate) for candidate in candidates))
        raise ValueError(
            f"resolved config {description} does not identify supplied file; "
            f"configured candidates=[{rendered}], supplied={supplied}"
        )


def _require_integer(value: Any, description: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{description} must be an integer, got {value!r}")
    return int(value)


def _integer_array(values: Iterable[Any], description: str) -> np.ndarray:
    converted = [_require_integer(value, description) for value in values]
    return np.asarray(converted, dtype=np.int64)


def _string_array(values: Iterable[Any], description: str) -> np.ndarray:
    converted: list[str] = []
    for value in values:
        if value is None or bool(pd.isna(value)):
            raise ValueError(f"{description} contains a missing value")
        text = str(value)
        if not text:
            raise ValueError(f"{description} contains an empty value")
        converted.append(text)
    return np.asarray(converted, dtype=np.str_)


def _partition_ids(entry: Any, partition: str) -> list[Any]:
    if isinstance(entry, Mapping):
        ids = entry.get("ids")
    else:
        ids = entry
    if not isinstance(ids, list) or not ids:
        raise ValueError(
            f"frozen partition {partition!r} must contain a non-empty ids list"
        )
    return ids


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read frozen partition manifest: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("frozen partition manifest must use schema_version 1")
    return payload


def _load_artifact_hash_ledger(root: Path) -> tuple[Path, dict[str, str]]:
    ledger_path = _require_file(root / "artifact_hashes.sha256", "artifact hash ledger")
    records: dict[str, str] = {}
    for line_number, line in enumerate(
        ledger_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if "  " not in line:
            raise ValueError(f"invalid artifact hash ledger line {line_number}")
        digest, relative = line.split("  ", 1)
        if not SHA256_PATTERN.fullmatch(digest) or not relative:
            raise ValueError(f"invalid artifact hash ledger line {line_number}")
        normalized = Path(relative).as_posix()
        if Path(normalized).is_absolute() or ".." in Path(normalized).parts:
            raise ValueError(f"unsafe artifact path on ledger line {line_number}")
        if normalized in records:
            raise ValueError(f"duplicate artifact path on ledger line {line_number}")
        records[normalized] = digest
    if not records:
        raise ValueError("artifact hash ledger must not be empty")
    return ledger_path, records


def _load_generator_provenance(
    metadata_path: Path,
    partition_manifest_sha256: str,
) -> tuple[Path, str, str, Path, dict[str, str]]:
    """Validate the governed sibling generator manifest and its source binding."""

    generator_path = _require_file(
        metadata_path.parent / "generator_manifest.json", "generator manifest"
    )
    try:
        payload = json.loads(generator_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read generator manifest: {exc}") from exc
    if not isinstance(payload, Mapping) or payload.get("schema_id") != (
        "p04.synthetic-generator.v1"
    ):
        raise ValueError("generator manifest must use p04.synthetic-generator.v1")
    content_hashes = _mapping(
        payload.get("content_hashes"), "generator manifest content_hashes"
    )
    expected = {
        "metadata_sha256": _sha256_file(metadata_path),
        "partition_manifest_sha256": partition_manifest_sha256,
    }
    for key, digest in expected.items():
        if content_hashes.get(key) != digest:
            raise ValueError(f"generator manifest {key} does not match frozen input")

    source = _mapping(payload.get("source"), "generator manifest source")
    source_digest = source.get("sha256")
    if not isinstance(source_digest, str) or not SHA256_PATTERN.fullmatch(source_digest):
        raise ValueError("generator manifest source is missing a lowercase SHA-256")
    source_relative = source.get("path")
    if not isinstance(source_relative, str) or not source_relative:
        raise ValueError("generator manifest source.path must be non-empty")
    vibench_root = Path(__file__).resolve().parents[2]
    source_path = (vibench_root / source_relative).resolve()
    try:
        source_path.relative_to(vibench_root)
    except ValueError as exc:
        raise ValueError("generator manifest source.path escapes the Vibench root") from exc
    if _sha256_file(_require_file(source_path, "generator source")) != source_digest:
        raise ValueError("generator manifest source SHA-256 does not match local source")
    ledger_path, ledger = _load_artifact_hash_ledger(metadata_path.parent)
    governed_files = {
        "metadata.csv": _sha256_file(metadata_path),
        "partition_manifest.json": partition_manifest_sha256,
        "generator_manifest.json": _sha256_file(generator_path),
    }
    for relative, digest in governed_files.items():
        if ledger.get(relative) != digest:
            raise ValueError(f"artifact hash ledger mismatch for {relative}")
    return generator_path, _sha256_file(generator_path), source_digest, ledger_path, ledger


def _validate_metadata_and_manifest(
    metadata_path: Path,
    manifest: Mapping[str, Any],
    config: Any,
) -> tuple[pd.DataFrame, dict[str, list[int]]]:
    expected_metadata_hash = manifest.get("metadata_file_sha256")
    if not isinstance(expected_metadata_hash, str) or not SHA256_PATTERN.fullmatch(
        expected_metadata_hash
    ):
        raise ValueError("frozen partition manifest is missing metadata_file_sha256")
    if _sha256_file(metadata_path) != expected_metadata_hash:
        raise ValueError("frozen partition manifest metadata SHA-256 does not match input")

    frame = pd.read_csv(metadata_path)
    required_columns = {
        "Id",
        "Dataset_id",
        "Domain_id",
        "Label",
        "Name",
        "File",
        "Split_group",
        "Split_stratum",
        "Partition",
        "Mechanism",
        "Nuisance_cell",
        "Draw",
    }
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(f"metadata is missing required columns: {missing_columns}")
    if frame.empty:
        raise ValueError("metadata must contain at least one row")

    metadata_ids = _integer_array(frame["Id"].tolist(), "metadata Id")
    if np.any(metadata_ids < 0):
        raise ValueError("metadata Id values must be non-negative")
    if len(set(metadata_ids.tolist())) != len(metadata_ids):
        raise ValueError("metadata Id values must be unique integers")
    frame = frame.copy()
    frame["Id"] = metadata_ids
    frame_by_id = frame.set_index("Id", drop=False)

    split = _config_value(config, "data", "split")
    config_group_key = str(_config_value(split, "group_key"))
    config_stratify_key = str(_config_value(split, "stratify_key"))
    if manifest.get("group_key") != config_group_key:
        raise ValueError("manifest group_key does not match resolved config")
    if manifest.get("stratify_key") != config_stratify_key:
        raise ValueError("manifest stratify_key does not match resolved config")
    for column in (config_group_key, config_stratify_key):
        if column not in frame.columns:
            raise ValueError(f"metadata does not contain configured column {column!r}")
        if frame[column].isna().any():
            raise ValueError(f"metadata column {column!r} contains missing values")

    config_partition_map = {
        str(key): str(value)
        for key, value in _mapping(
            _config_value(split, "partition_map"), "data.split.partition_map"
        ).items()
    }
    expected_partition_map = {
        "train": "train",
        "val": "optimization_validation",
        "test": "intervention",
    }
    if config_partition_map != expected_partition_map:
        raise ValueError(
            "resolved config partition_map must map train/val/test to the frozen "
            "train/optimization_validation/intervention partitions"
        )
    manifest_partition_map = manifest.get("partition_map")
    if manifest_partition_map is not None and {
        str(key): str(value)
        for key, value in _mapping(
            manifest_partition_map, "manifest partition_map"
        ).items()
    } != expected_partition_map:
        raise ValueError("manifest partition_map does not match the frozen mapping")

    partitions_raw = manifest.get("partitions")
    if not isinstance(partitions_raw, Mapping):
        raise ValueError("frozen partition manifest must contain partitions")
    if set(partitions_raw) != set(EXPECTED_PARTITIONS):
        raise ValueError(
            "frozen partition manifest must contain exactly "
            f"{list(EXPECTED_PARTITIONS)}"
        )

    resolved: dict[str, list[int]] = {}
    id_owner: dict[int, str] = {}
    group_owner: dict[str, str] = {}
    metadata_id_set = set(metadata_ids.tolist())
    for partition in EXPECTED_PARTITIONS:
        entry = partitions_raw[partition]
        ids = [
            _require_integer(value, f"manifest {partition} sample ID")
            for value in _partition_ids(entry, partition)
        ]
        if len(ids) != len(set(ids)):
            raise ValueError(f"frozen partition {partition!r} contains duplicate IDs")
        unknown = sorted(set(ids) - metadata_id_set)
        if unknown:
            raise ValueError(
                f"frozen partition {partition!r} references unknown IDs: {unknown[:5]}"
            )
        for sample_id in ids:
            if sample_id in id_owner:
                raise ValueError(
                    f"sample ID {sample_id} occurs in both {id_owner[sample_id]!r} "
                    f"and {partition!r}"
                )
            id_owner[sample_id] = partition

        rows = frame_by_id.loc[ids]
        observed_partition = set(_string_array(rows["Partition"], "Partition").tolist())
        if observed_partition != {partition}:
            raise ValueError(
                f"metadata Partition values disagree with frozen partition {partition!r}"
            )
        observed_groups = {str(value) for value in rows[config_group_key].tolist()}
        if not isinstance(entry, Mapping) or not isinstance(entry.get("groups"), list):
            raise ValueError(f"frozen partition {partition!r} must record groups")
        manifest_groups = [str(value) for value in entry["groups"]]
        if len(manifest_groups) != len(set(manifest_groups)):
            raise ValueError(f"frozen partition {partition!r} contains duplicate groups")
        if set(manifest_groups) != observed_groups:
            raise ValueError(
                f"frozen partition {partition!r} groups do not match metadata"
            )
        for group in observed_groups:
            if group in group_owner:
                raise ValueError(
                    f"split group {group!r} occurs in both {group_owner[group]!r} "
                    f"and {partition!r}"
                )
            group_owner[group] = partition
        resolved[partition] = ids

    if set(id_owner) != metadata_id_set:
        missing = sorted(metadata_id_set - set(id_owner))
        raise ValueError(f"frozen partitions do not cover metadata IDs: {missing[:5]}")

    # A group is indivisible and has one frozen stratification label.
    for group, group_rows in frame.groupby(config_group_key, sort=False):
        group_partitions = group_rows["Partition"].astype(str).unique().tolist()
        if len(group_partitions) != 1:
            raise ValueError(f"split group {group!r} crosses partitions")
        strata = group_rows[config_stratify_key].astype(str).unique().tolist()
        if len(strata) != 1:
            raise ValueError(f"split group {group!r} has multiple strata")

    labels = _integer_array(frame["Label"].tolist(), "metadata Label")
    diagnoses = labels.copy()
    draws = _integer_array(frame["Draw"].tolist(), "metadata Draw")
    nuisance_cells = _integer_array(
        frame["Nuisance_cell"].tolist(), "metadata Nuisance_cell"
    )
    if np.any(labels < 0) or np.any(draws < 0) or np.any(nuisance_cells < 0):
        raise ValueError("Label, Draw, and Nuisance_cell values must be non-negative")
    num_classes = _require_integer(
        _config_value(config, "model", "num_classes"), "model.num_classes"
    )
    if np.any(labels >= num_classes):
        raise ValueError("metadata Label exceeds model.num_classes")
    frame["Label"] = labels
    frame["Diagnosis"] = diagnoses
    frame["Draw"] = draws
    frame["Nuisance_cell"] = nuisance_cells
    _string_array(frame["Mechanism"].tolist(), "metadata Mechanism")
    _string_array(frame["Name"].tolist(), "metadata Name")
    _string_array(frame["File"].tolist(), "metadata File")
    return frame, resolved


def _validate_config_contract(
    config: Any,
    config_path: Path,
    metadata_path: Path,
    manifest_path: Path,
    manifest_sha256: str,
    arm: str,
    seed: int,
) -> None:
    if str(_config_value(config, "model", "type")) != "MoE" or str(
        _config_value(config, "model", "name")
    ) != "M_04_RoleConstrainedMoE":
        raise ValueError("collector requires model MoE/M_04_RoleConstrainedMoE")
    if str(_config_value(config, "protocol", "arm")) != arm:
        raise ValueError("CLI arm does not match resolved config protocol.arm")
    if _require_integer(
        _config_value(config, "environment", "seed"), "environment.seed"
    ) != seed:
        raise ValueError("CLI seed does not match resolved config environment.seed")

    expected_representation = {
        "FULL": "role_constrained",
        "HOMO": "homogeneous_raw",
        "RAND": "role_constrained",
    }[arm]
    if str(_config_value(config, "model", "expert_representation_mode")) != (
        expected_representation
    ):
        raise ValueError(
            f"resolved config uses the wrong expert representation for arm {arm}"
        )
    configured_permutation = _integer_array(
        _config_value(config, "model", "role_prior_permutation"),
        "model.role_prior_permutation",
    ).tolist()
    if arm in {"FULL", "HOMO"}:
        if configured_permutation != [0, 1, 2, 3] or str(
            _config_value(config, "model", "role_prior_assignment")
        ) != "aligned":
            raise ValueError(f"resolved {arm} config must use the aligned role prior")
    else:
        bindings = _config_value(
            config, "protocol", "random_role_prior_permutations"
        )
        if not isinstance(bindings, list):
            raise ValueError("RAND protocol must contain seed/permutation bindings")
        matches = [
            _mapping(binding, "RAND seed/permutation binding")
            for binding in bindings
            if _require_integer(
                _mapping(binding, "RAND seed/permutation binding").get("seed"),
                "RAND binding seed",
            )
            == seed
        ]
        if len(matches) != 1:
            raise ValueError("RAND protocol must contain exactly one binding for CLI seed")
        expected_permutation = _integer_array(
            matches[0].get("permutation"), "RAND binding permutation"
        ).tolist()
        if configured_permutation != expected_permutation:
            raise ValueError("RAND model permutation does not match its frozen seed binding")
        if sorted(configured_permutation) != [0, 1, 2, 3] or any(
            index == value for index, value in enumerate(configured_permutation)
        ):
            raise ValueError("RAND role-prior permutation must be fixed-point-free")
        if str(_config_value(config, "model", "role_prior_assignment")) != (
            "external_deranged"
        ):
            raise ValueError("RAND config must use role_prior_assignment=external_deranged")

    split = _config_value(config, "data", "split")
    if str(_config_value(split, "manifest_mode")) != "read_only":
        raise ValueError("resolved config must use data.split.manifest_mode=read_only")
    _assert_configured_path(
        _config_value(split, "manifest_path"),
        manifest_path,
        config_path,
        "data.split.manifest_path",
    )
    configured_manifest_hash = str(_config_value(split, "manifest_sha256"))
    if not SHA256_PATTERN.fullmatch(configured_manifest_hash):
        raise ValueError("resolved config must bind a lowercase manifest SHA-256")
    if configured_manifest_hash != manifest_sha256:
        raise ValueError("resolved config manifest SHA-256 does not match supplied manifest")

    data_dir = Path(str(_config_value(config, "data", "data_dir"))).expanduser()
    metadata_file = Path(str(_config_value(config, "data", "metadata_file")))
    configured_metadata = metadata_file if metadata_file.is_absolute() else data_dir / metadata_file
    _assert_configured_path(
        configured_metadata,
        metadata_path,
        config_path,
        "data.data_dir/data.metadata_file",
    )

    expected_data_values = {
        "normalization": "none",
        "window_size": 512,
        "stride": 512,
        "num_window": 1,
        "dtype": "float32",
    }
    for key, expected in expected_data_values.items():
        observed = _config_value(config, "data", key)
        if observed != expected:
            raise ValueError(f"resolved config data.{key} must equal {expected!r}")
    if _require_integer(_config_value(config, "model", "input_dim"), "model.input_dim") != 2:
        raise ValueError("resolved config model.input_dim must equal 2")


def _blinding_permutation(arm: str, seed: int, expert_count: int) -> np.ndarray:
    material = f"{BLINDING_DOMAIN}|{arm}|{seed}".encode("utf-8")
    digest = hashlib.sha256(material).digest()
    rng_seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return np.random.default_rng(rng_seed).permutation(expert_count).astype(np.int64)


def _load_checkpoint_strict(model: torch.nn.Module, checkpoint_path: Path) -> None:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("state_dict"), Mapping):
        raise ValueError("checkpoint must be a Lightning mapping with state_dict")
    state_dict = payload["state_dict"]
    network_state: dict[str, torch.Tensor] = {}
    for name, value in state_dict.items():
        if not isinstance(name, str) or not name.startswith("network."):
            continue
        stripped = name[len("network.") :]
        if not stripped or stripped in network_state:
            raise ValueError("checkpoint contains an invalid network.* key mapping")
        network_state[stripped] = value
    if not network_state:
        raise ValueError("checkpoint state_dict contains no network.* parameters")

    expected = set(model.state_dict())
    observed = set(network_state)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if missing or unexpected:
        raise ValueError(
            "strict network.* checkpoint mapping failed: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    model.load_state_dict(network_state, strict=True)


def _reader_for_name(name: str) -> tuple[Any, ModuleType]:
    if name != "P04_Synthetic":
        raise ValueError(f"mechanism collector only accepts Name='P04_Synthetic', got {name!r}")
    module = importlib.import_module(f"src.data_factory.reader.{name}")
    reader = getattr(module, "read", None)
    if not callable(reader):
        raise ValueError(f"reader module for {name!r} has no callable read")
    return reader, module


def _source_path(data_root: Path, name: str, file_name: str) -> Path:
    reader_root = (data_root / "raw" / name).resolve()
    candidate = (reader_root / file_name).resolve()
    try:
        candidate.relative_to(reader_root)
    except ValueError as exc:
        raise ValueError(
            f"metadata File escapes the governed raw directory: {file_name!r}"
        ) from exc
    return _require_file(candidate, "synthetic source sample")


def _module_sha256(module: ModuleType, description: str) -> str:
    source = getattr(module, "__file__", None)
    if not source:
        raise ValueError(f"unable to locate {description} source file")
    return _sha256_file(_require_file(source, f"{description} source"))


def _validate_diagnostics(
    logits: torch.Tensor,
    diagnostics: Mapping[str, Any],
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    required = ("routing_weights", "expert_features", "expert_logits")
    missing = [name for name in required if name not in diagnostics]
    if missing:
        raise ValueError(f"model diagnostics are missing keys: {missing}")
    routing = diagnostics["routing_weights"]
    features = diagnostics["expert_features"]
    expert_logits = diagnostics["expert_logits"]
    if not all(isinstance(value, torch.Tensor) for value in (routing, features, expert_logits)):
        raise ValueError("model diagnostics must be tensors")
    if logits.ndim != 2 or logits.shape[0] != batch_size:
        raise ValueError("model logits must have shape [batch, classes]")
    if routing.shape != (batch_size, 4):
        raise ValueError("routing_weights must have shape [batch, 4]")
    if features.ndim != 3 or features.shape[:2] != (batch_size, 4):
        raise ValueError("expert_features must have shape [batch, 4, feature_dim]")
    if expert_logits.shape != (batch_size, 4, logits.shape[1]):
        raise ValueError("expert_logits must have shape [batch, 4, classes]")
    for name, value in (
        ("logits", logits),
        ("routing_weights", routing),
        ("expert_features", features),
        ("expert_logits", expert_logits),
    ):
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains NaN or Inf")
    if torch.any(routing < 0.0) or not torch.allclose(
        routing.sum(dim=-1),
        torch.ones(batch_size, dtype=routing.dtype, device=routing.device),
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError("routing_weights are not valid probabilities")
    reconstructed = torch.sum(expert_logits * routing.unsqueeze(-1), dim=1)
    if not torch.allclose(logits, reconstructed, rtol=1e-5, atol=1e-6):
        raise ValueError("intact logits do not equal the routed expert-logit mixture")
    return routing, features, expert_logits


def _algebraic_interventions(
    logits: torch.Tensor,
    routing: torch.Tensor,
    expert_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    denominators = 1.0 - routing
    if torch.any(denominators < DELETION_DENOMINATOR_MINIMUM):
        minimum = float(denominators.min().detach().cpu())
        raise ValueError(
            "primary deletion denominator is below 1e-6; "
            f"minimum observed={minimum:.9g}"
        )
    deleted: list[torch.Tensor] = []
    for deleted_index in range(routing.shape[1]):
        effective = routing.clone()
        effective[:, deleted_index] = 0.0
        effective = effective / denominators[:, deleted_index].unsqueeze(-1)
        deleted.append(torch.sum(expert_logits * effective.unsqueeze(-1), dim=1))
    deleted_logits = torch.stack(deleted, dim=1)

    # Axes are [observation, matched canonical slot, replacement canonical slot, class].
    matched_mass = routing[:, :, None, None]
    matched_output = expert_logits[:, :, None, :]
    replacement_output = expert_logits[:, None, :, :]
    fixed_mass = (
        logits[:, None, None, :]
        - matched_mass * matched_output
        + matched_mass * replacement_output
    )
    diagonal = torch.arange(routing.shape[1], device=routing.device)
    fixed_mass[:, diagonal, diagonal, :] = logits[:, None, :]
    return deleted_logits, fixed_mass


def _blind_expert_axes(
    permutation: np.ndarray,
    routing: torch.Tensor,
    features: torch.Tensor,
    expert_logits: torch.Tensor,
    deleted_logits: torch.Tensor,
    fixed_mass: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    index = torch.as_tensor(permutation, dtype=torch.long, device=routing.device)
    return (
        routing.index_select(1, index),
        features.index_select(1, index),
        expert_logits.index_select(1, index),
        deleted_logits.index_select(1, index),
        fixed_mass.index_select(1, index).index_select(2, index),
    )


def _write_npz_exclusive(output_path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output_path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def collect_checkpoint_outputs(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    metadata_path: str | Path,
    partition_manifest_path: str | Path,
    output_path: str | Path,
    arm: str,
    seed: int,
    device: str = "cpu",
) -> dict[str, Any]:
    """Collect, blind, validate, and atomically write one evaluator-input NPZ."""

    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if output.suffix != ".npz":
        raise ValueError("output path must end in .npz")
    config_file = _require_file(config_path, "resolved config")
    checkpoint_file = _require_file(checkpoint_path, "checkpoint")
    metadata_file = _require_file(metadata_path, "metadata CSV")
    manifest_file = _require_file(partition_manifest_path, "frozen partition manifest")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if arm not in {"FULL", "HOMO", "RAND"}:
        raise ValueError("arm must be one of FULL, HOMO, or RAND")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")

    config = load_config(config_file)
    manifest_sha256 = _sha256_file(manifest_file)
    _validate_config_contract(
        config,
        config_file,
        metadata_file,
        manifest_file,
        manifest_sha256,
        arm,
        seed,
    )
    manifest = _load_manifest(manifest_file)
    frame, partitions = _validate_metadata_and_manifest(metadata_file, manifest, config)
    (
        _generator_manifest_file,
        generator_manifest_sha256,
        generator_source_sha256,
        artifact_hash_ledger_file,
        artifact_hash_ledger,
    ) = _load_generator_provenance(metadata_file, manifest_sha256)
    frame_by_id = frame.set_index("Id", drop=False)

    model = build_model(config.model, metadata=frame)
    _load_checkpoint_strict(model, checkpoint_file)
    torch_device = torch.device(device)
    model.to(torch_device)
    model.eval()

    expert_count = 4
    permutation = _blinding_permutation(arm, seed, expert_count)
    inverse_permutation = np.empty_like(permutation)
    inverse_permutation[permutation] = np.arange(expert_count, dtype=np.int64)

    reader_hashes: dict[str, str] = {}
    data_root = metadata_file.parent

    def read_partition(partition: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "sample_ids": [],
            "partition_labels": [],
            "labels": [],
            "mechanisms": [],
            "diagnoses": [],
            "nuisance_cells": [],
            "draws": [],
            "source_files": [],
            "source_hashes": [],
            "input_samples": [],
        }
        for sample_id in partitions[partition]:
            row = frame_by_id.loc[sample_id]
            name = str(row["Name"])
            reader, reader_module = _reader_for_name(name)
            if name not in reader_hashes:
                reader_hashes[name] = _module_sha256(reader_module, f"{name} reader")
            source = _source_path(data_root, name, str(row["File"]))
            sample = reader(str(source), config.data)
            if sample.dtype != np.float32 or sample.shape != (512, 2):
                raise ValueError(
                    f"reader returned invalid sample for ID {sample_id}: "
                    f"shape={sample.shape}, dtype={sample.dtype}"
                )
            if not np.isfinite(sample).all():
                raise ValueError(f"reader returned non-finite sample for ID {sample_id}")
            payload["sample_ids"].append(sample_id)
            payload["partition_labels"].append(partition)
            payload["labels"].append(int(row["Label"]))
            payload["mechanisms"].append(str(row["Mechanism"]))
            payload["diagnoses"].append(int(row["Diagnosis"]))
            payload["nuisance_cells"].append(int(row["Nuisance_cell"]))
            payload["draws"].append(int(row["Draw"]))
            payload["source_files"].append(str(source.relative_to(data_root)))
            source_relative = source.relative_to(data_root).as_posix()
            source_sha256 = _sha256_file(source)
            if artifact_hash_ledger.get(source_relative) != source_sha256:
                raise ValueError(
                    f"artifact hash ledger mismatch for source sample {source_relative}"
                )
            payload["source_hashes"].append(source_sha256)
            payload["input_samples"].append(sample)
        return payload

    batch_size = _require_integer(_config_value(config, "data", "batch_size"), "data.batch_size")
    if batch_size <= 0:
        raise ValueError("data.batch_size must be positive")

    def forward_partition(input_samples: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
        parts: dict[str, list[np.ndarray]] = {
            "logits": [],
            "routing": [],
            "features": [],
            "expert_logits": [],
            "deleted": [],
            "fixed_mass": [],
        }
        with torch.no_grad():
            for start in range(0, len(input_samples), batch_size):
                batch_array = np.stack(input_samples[start : start + batch_size], axis=0)
                batch = torch.from_numpy(batch_array).to(torch_device)
                logits, diagnostics = model(batch, return_diagnostics=True)
                routing, features, expert_logits = _validate_diagnostics(
                    logits, diagnostics, batch.shape[0]
                )
                deleted_logits, fixed_mass = _algebraic_interventions(
                    logits, routing, expert_logits
                )
                (
                    routing,
                    features,
                    expert_logits,
                    deleted_logits,
                    fixed_mass,
                ) = _blind_expert_axes(
                    permutation,
                    routing,
                    features,
                    expert_logits,
                    deleted_logits,
                    fixed_mass,
                )
                parts["logits"].append(logits.detach().cpu().numpy())
                parts["routing"].append(routing.detach().cpu().numpy())
                parts["features"].append(features.detach().cpu().numpy())
                parts["expert_logits"].append(expert_logits.detach().cpu().numpy())
                parts["deleted"].append(deleted_logits.detach().cpu().numpy())
                parts["fixed_mass"].append(fixed_mass.detach().cpu().numpy())
        return {name: np.concatenate(values, axis=0) for name, values in parts.items()}

    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    previous_cudnn_benchmark = torch.backends.cudnn.benchmark
    previous_cpu_rng_state = torch.random.get_rng_state()
    previous_cuda_rng_states = (
        torch.cuda.get_rng_state_all() if torch_device.type == "cuda" else None
    )
    phase_events: list[str] = []
    try:
        torch.manual_seed(seed)
        if torch_device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        identification_payload = read_partition("identification")
        identification_outputs = forward_partition(
            identification_payload["input_samples"]
        )
        phase_events.append(COLLECTION_PHASE_ORDER[0])
        assignment_seal, assignment_seal_sha256 = (
            build_preintervention_assignment_seal(
                identification_outputs["features"],
                identification_payload["mechanisms"],
                identification_payload["diagnoses"],
                identification_payload["nuisance_cells"],
                identification_payload["draws"],
                identification_payload["sample_ids"],
                arm=arm,
                seed=seed,
                require_frozen_design=True,
            )
        )
        assignment_seal_json = json.dumps(
            assignment_seal,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        phase_events.append(COLLECTION_PHASE_ORDER[1])

        # No intervention source file is read and no intervention forward occurs
        # until the anonymous identification assignment has been sealed above.
        intervention_payload = read_partition("intervention")
        intervention_outputs = forward_partition(intervention_payload["input_samples"])
        phase_events.append(COLLECTION_PHASE_ORDER[2])
    finally:
        torch.random.set_rng_state(previous_cpu_rng_state)
        if previous_cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(previous_cuda_rng_states)
        torch.use_deterministic_algorithms(previous_deterministic)
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic
        torch.backends.cudnn.benchmark = previous_cudnn_benchmark

    if tuple(phase_events) != COLLECTION_PHASE_ORDER:
        raise RuntimeError("collector phase ordering invariant was not satisfied")
    payloads = (identification_payload, intervention_payload)
    outputs_by_partition = (identification_outputs, intervention_outputs)
    sample_ids = [value for payload in payloads for value in payload["sample_ids"]]
    partition_labels = [
        value for payload in payloads for value in payload["partition_labels"]
    ]
    labels = [value for payload in payloads for value in payload["labels"]]
    mechanisms = [value for payload in payloads for value in payload["mechanisms"]]
    diagnoses = [value for payload in payloads for value in payload["diagnoses"]]
    nuisance_cells = [
        value for payload in payloads for value in payload["nuisance_cells"]
    ]
    draws = [value for payload in payloads for value in payload["draws"]]
    source_files = [value for payload in payloads for value in payload["source_files"]]
    source_hashes = [
        value for payload in payloads for value in payload["source_hashes"]
    ]

    model_module = importlib.import_module(model.__class__.__module__)
    provenance = {
        "schema": SCHEMA,
        "arm": arm,
        "seed": seed,
        "selected_partitions": list(SELECTED_PARTITIONS),
        "selected_partition_counts": {
            partition: len(partitions[partition]) for partition in SELECTED_PARTITIONS
        },
        "hashes": {
            "config_sha256": _sha256_file(config_file),
            "checkpoint_sha256": _sha256_file(checkpoint_file),
            "manifest_sha256": manifest_sha256,
            "partition_manifest_sha256": manifest_sha256,
            "generator_manifest_sha256": generator_manifest_sha256,
            "generator_source_sha256": generator_source_sha256,
            "artifact_hash_ledger_sha256": _sha256_file(artifact_hash_ledger_file),
            "metadata_sha256": _sha256_file(metadata_file),
            "collector_source_sha256": _sha256_file(Path(__file__).resolve()),
            "model_source_sha256": _module_sha256(model_module, "model"),
            "reader_source_sha256": reader_hashes,
        },
        "blinding": {
            "domain": BLINDING_DOMAIN,
            "derivation": "SHA-256(domain|arm|seed), first 8 bytes as big-endian RNG seed",
            "permutation_direction": "canonical_expert_index_at_each_blinded_column",
            "blinding_permutation": permutation.tolist(),
            "designated_role_to_expert_direction": "canonical_constrained_slot_to_blinded_column",
            "designated_role_to_expert": inverse_permutation.tolist(),
            "rand_target_rule": "canonical_constrained_representation_slots_not_deranged_prior",
        },
        "assignment_seal": {
            "content": assignment_seal,
            "sha256": assignment_seal_sha256,
            "canonical_json": assignment_seal_json,
        },
        "ordering": {
            "phases": list(COLLECTION_PHASE_ORDER),
            "observed_phase_events": phase_events,
            "assignment_sealed_before_intervention_read": True,
            "intervention_signal_files_read_before_seal": 0,
        },
        "intervention": {
            "router_forward_count_per_batch": 1,
            "deleted_logits": "renormalized algebra from intact routing and expert logits",
            "denominator_invalid_if_below": DELETION_DENOMINATOR_MINIMUM,
            "fixed_mass_swap": "f - w_matched*z_matched + w_matched*z_nonmatching",
            "fixed_mass_swap_diagonal": "intact_logits",
        },
    }
    arrays: dict[str, np.ndarray] = {
        "schema": np.asarray(SCHEMA),
        "schema_id": np.asarray(SCHEMA),
        "arm": np.asarray(arm),
        "seed": np.asarray(seed, dtype=np.int64),
        "sample_id": np.asarray(sample_ids, dtype=np.int64),
        "partition": np.asarray(partition_labels, dtype=np.str_),
        "label": np.asarray(labels, dtype=np.int64),
        "mechanism": np.asarray(mechanisms, dtype=np.str_),
        "diagnosis": np.asarray(diagnoses, dtype=np.int64),
        "nuisance_cell": np.asarray(nuisance_cells, dtype=np.int64),
        "draw": np.asarray(draws, dtype=np.int64),
        "logits": np.concatenate(
            [partition_output["logits"] for partition_output in outputs_by_partition],
            axis=0,
        ),
        "routing_weights": np.concatenate(
            [partition_output["routing"] for partition_output in outputs_by_partition],
            axis=0,
        ),
        "expert_features": np.concatenate(
            [partition_output["features"] for partition_output in outputs_by_partition],
            axis=0,
        ),
        "expert_logits": np.concatenate(
            [
                partition_output["expert_logits"]
                for partition_output in outputs_by_partition
            ],
            axis=0,
        ),
        "deleted_logits": np.concatenate(
            [partition_output["deleted"] for partition_output in outputs_by_partition],
            axis=0,
        ),
        "fixed_mass_swap_logits": np.concatenate(
            [
                partition_output["fixed_mass"]
                for partition_output in outputs_by_partition
            ],
            axis=0,
        ),
        "fixed_mass_swap_diagonal_policy": np.asarray("intact_logits"),
        "blinding_domain": np.asarray(BLINDING_DOMAIN),
        "blinding_permutation": permutation,
        "blinding_permutation_direction": np.asarray(
            "canonical_expert_index_at_each_blinded_column"
        ),
        "designated_role_to_expert": inverse_permutation,
        "designated_role_to_expert_direction": np.asarray(
            "canonical_constrained_slot_to_blinded_column"
        ),
        "assignment_seal_json": np.asarray(assignment_seal_json),
        "assignment_seal_sha256": np.asarray(assignment_seal_sha256),
        "collection_phase_order_json": np.asarray(
            json.dumps(list(COLLECTION_PHASE_ORDER), separators=(",", ":"))
        ),
        "assignment_sealed_before_intervention_read": np.asarray(True),
        "source_file": np.asarray(source_files, dtype=np.str_),
        "source_sha256": np.asarray(source_hashes, dtype=np.str_),
        "config_sha256": np.asarray(provenance["hashes"]["config_sha256"]),
        "checkpoint_sha256": np.asarray(provenance["hashes"]["checkpoint_sha256"]),
        "manifest_sha256": np.asarray(provenance["hashes"]["manifest_sha256"]),
        "partition_manifest_sha256": np.asarray(
            provenance["hashes"]["partition_manifest_sha256"]
        ),
        "generator_manifest_sha256": np.asarray(
            provenance["hashes"]["generator_manifest_sha256"]
        ),
        "generator_source_sha256": np.asarray(
            provenance["hashes"]["generator_source_sha256"]
        ),
        "artifact_hash_ledger_sha256": np.asarray(
            provenance["hashes"]["artifact_hash_ledger_sha256"]
        ),
        "metadata_sha256": np.asarray(provenance["hashes"]["metadata_sha256"]),
        "collector_source_sha256": np.asarray(
            provenance["hashes"]["collector_source_sha256"]
        ),
        "model_source_sha256": np.asarray(provenance["hashes"]["model_source_sha256"]),
        "reader_source_sha256_json": np.asarray(
            json.dumps(reader_hashes, sort_keys=True, separators=(",", ":"))
        ),
        "provenance_json": np.asarray(
            json.dumps(provenance, sort_keys=True, separators=(",", ":"))
        ),
    }
    _write_npz_exclusive(output, arrays)
    return {
        "schema": SCHEMA,
        "output": str(output),
        "samples": len(sample_ids),
        "partition_counts": provenance["selected_partition_counts"],
        "arm": arm,
        "seed": seed,
        "assignment_seal_sha256": assignment_seal_sha256,
        "output_sha256": _sha256_file(output),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect blinded P04 mechanism-evaluator inputs from one checkpoint."
    )
    parser.add_argument("--config", required=True, help="Resolved experiment YAML")
    parser.add_argument("--checkpoint", required=True, help="Exact Lightning checkpoint")
    parser.add_argument("--metadata", required=True, help="Exact synthetic metadata CSV")
    parser.add_argument(
        "--partition-manifest", required=True, help="Exact frozen partition manifest"
    )
    parser.add_argument("--output", required=True, help="New output .npz path")
    parser.add_argument("--arm", required=True, help="Frozen uppercase arm identifier")
    parser.add_argument("--seed", required=True, type=int, help="Frozen training seed")
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), default="cpu", help="Inference device"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = collect_checkpoint_outputs(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        metadata_path=args.metadata,
        partition_manifest_path=args.partition_manifest,
        output_path=args.output,
        arm=args.arm,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

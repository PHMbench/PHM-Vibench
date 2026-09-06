"""Deterministic P07 synthetic path universe and independent signal oracle.

This module is deliberately independent of the executable operator-path model.
It defines the preregisterable objects needed to generate and audit E7/E8
software artifacts without importing the model's private operator functions or
executor.  Nothing in this module declares a paper claim evidence-eligible.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Final, Literal, Mapping, Sequence, TypeAlias, cast

import torch
import torch.nn.functional as F


OperatorName = Literal["I", "D1", "ABS", "SQUARE", "MA3", "HT"]
RawPath: TypeAlias = tuple[OperatorName, OperatorName, OperatorName]
CanonicalPath: TypeAlias = tuple[OperatorName, ...]
JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


SCHEMA_VERSION: Final[int] = 1
PROTOCOL_ID: Final[str] = "P07-E7-E8-PATH-UNIVERSE-v2"
K_STAGES: Final[int] = 3
OPERATORS: Final[tuple[OperatorName, ...]] = (
    "I",
    "D1",
    "ABS",
    "SQUARE",
    "MA3",
    "HT",
)
NON_IDENTITY_OPERATORS: Final[tuple[OperatorName, ...]] = OPERATORS[1:]

PRIMARY_SELECTION_SALT: Final[str] = "P07-E7-v2-primary-selection"
COMPOSITION_SPLIT_SALT: Final[str] = "P07-E7-v2-composition-split"
PRIMARY_SINGLETON_COUNT: Final[int] = 46
PRIMARY_AMBIGUOUS_COUNT: Final[int] = 26

COMPOSITION_SPLIT_COUNTS: Final[dict[str, dict[str, int]]] = {
    "train": {"ambiguous": 13, "singleton": 23},
    "validation": {"ambiguous": 7, "singleton": 11},
    "test": {"ambiguous": 6, "singleton": 12},
}
SINGLETON_POSITION_MINIMUMS: Final[dict[str, int]] = {
    "train": 2,
    "validation": 1,
    "test": 1,
}

GENERATOR_SEED_NAMESPACES: Final[dict[str, tuple[int, ...]]] = {
    "fit": (1103, 1109),
    "validation": (2203, 2207),
    "test": (3301, 3307),
}
SAMPLES_PER_GENERATOR_SEED: Final[dict[str, int]] = {
    "fit": 256,
    "validation": 128,
    "test": 256,
}
OPTIMIZATION_SEEDS: Final[tuple[int, ...]] = (
    7,
    20,
    31,
    42,
    100,
    113,
    127,
    139,
    151,
    163,
    179,
    193,
    211,
    227,
    241,
    257,
    271,
    283,
    307,
    331,
    347,
    367,
    389,
    409,
    449,
)
CORRUPTION_SEED_DOMAIN: Final[str] = "P07-E8-corruption-v2"


def _corruption_seed(optimization_seed: int) -> int:
    token = f"{CORRUPTION_SEED_DOMAIN}|{optimization_seed}".encode("ascii")
    return int(hashlib.sha256(token).hexdigest()[:16], 16) % (2**63)


CORRUPTION_SEED_BY_OPTIMIZATION_SEED: Final[dict[int, int]] = {
    seed: _corruption_seed(seed) for seed in OPTIMIZATION_SEEDS
}
_ALL_GENERATOR_SEEDS: Final[set[int]] = {
    seed for namespace in GENERATOR_SEED_NAMESPACES.values() for seed in namespace
}
if len(set(CORRUPTION_SEED_BY_OPTIMIZATION_SEED.values())) != len(
    OPTIMIZATION_SEEDS
):
    raise RuntimeError("Derived corruption seeds are not unique.")
if not _ALL_GENERATOR_SEEDS.isdisjoint(
    CORRUPTION_SEED_BY_OPTIMIZATION_SEED.values()
):
    raise RuntimeError("Derived corruption seeds overlap a generator namespace.")

# Tokens are in execution order: the first token is applied to x first.
_PAIR_REWRITES: Final[
    tuple[tuple[str, tuple[OperatorName, OperatorName], tuple[OperatorName, ...]], ...]
] = (
    ("abs_idempotence", ("ABS", "ABS"), ("ABS",)),
    ("abs_of_square", ("SQUARE", "ABS"), ("SQUARE",)),
    ("square_of_abs", ("ABS", "SQUARE"), ("SQUARE",)),
    ("abs_of_hilbert_envelope", ("HT", "ABS"), ("HT",)),
)

_TRIPLE_REWRITES: Final[
    tuple[
        tuple[
            str,
            tuple[OperatorName, OperatorName, OperatorName],
            tuple[OperatorName, ...],
        ],
        ...,
    ]
] = (
    ("abs_after_ma3_abs", ("ABS", "MA3", "ABS"), ("ABS", "MA3")),
    (
        "abs_after_ma3_square",
        ("SQUARE", "MA3", "ABS"),
        ("SQUARE", "MA3"),
    ),
    (
        "abs_after_ma3_hilbert_envelope",
        ("HT", "MA3", "ABS"),
        ("HT", "MA3"),
    ),
)

EQUIVALENCE_REWRITE_WHITELIST: Final[tuple[dict[str, JsonValue], ...]] = (
    {
        "rewrite_id": "identity_elision",
        "expression": "I(a)->a",
        "lhs_execution_order": ["I"],
        "rhs_execution_order": [],
    },
    {
        "rewrite_id": "abs_idempotence",
        "expression": "ABS(ABS(a))->ABS(a)",
        "lhs_execution_order": ["ABS", "ABS"],
        "rhs_execution_order": ["ABS"],
    },
    {
        "rewrite_id": "abs_of_square",
        "expression": "ABS(SQUARE(a))->SQUARE(a)",
        "lhs_execution_order": ["SQUARE", "ABS"],
        "rhs_execution_order": ["SQUARE"],
    },
    {
        "rewrite_id": "square_of_abs",
        "expression": "SQUARE(ABS(a))->SQUARE(a)",
        "lhs_execution_order": ["ABS", "SQUARE"],
        "rhs_execution_order": ["SQUARE"],
    },
    {
        "rewrite_id": "abs_of_hilbert_envelope",
        "expression": "ABS(HT(a))->HT(a)",
        "lhs_execution_order": ["HT", "ABS"],
        "rhs_execution_order": ["HT"],
    },
    {
        "rewrite_id": "abs_after_ma3_abs",
        "expression": "ABS(MA3(ABS(a)))->MA3(ABS(a))",
        "lhs_execution_order": ["ABS", "MA3", "ABS"],
        "rhs_execution_order": ["ABS", "MA3"],
    },
    {
        "rewrite_id": "abs_after_ma3_square",
        "expression": "ABS(MA3(SQUARE(a)))->MA3(SQUARE(a))",
        "lhs_execution_order": ["SQUARE", "MA3", "ABS"],
        "rhs_execution_order": ["SQUARE", "MA3"],
    },
    {
        "rewrite_id": "abs_after_ma3_hilbert_envelope",
        "expression": "ABS(MA3(HT(a)))->MA3(HT(a))",
        "lhs_execution_order": ["HT", "MA3", "ABS"],
        "rhs_execution_order": ["HT", "MA3"],
    },
)

_RAW_PATH_ID_PREFIX: Final[str] = "P07-RAW-"
_CLASS_ID_PREFIX: Final[str] = "P07-SEM-"
_SAMPLE_ID_PREFIX: Final[str] = "P07-SAMPLE-"


@dataclass(frozen=True, slots=True)
class PathRecord:
    """One raw path and its preregistered semantic class."""

    raw_path: RawPath
    raw_path_id: str
    raw_path_sha256: str
    raw_expression: str
    canonical_path: CanonicalPath
    canonical_expression: str
    class_id: str
    class_sha256: str


@dataclass(frozen=True, slots=True)
class EquivalenceClass:
    """All raw paths accepted as one semantic composition."""

    class_id: str
    class_sha256: str
    canonical_path: CanonicalPath
    canonical_expression: str
    members: tuple[PathRecord, ...]

    @property
    def multiplicity(self) -> int:
        return len(self.members)

    @property
    def ambiguous(self) -> bool:
        return self.multiplicity > 1


def _validate_json_value(value: Any, *, location: str = "$") -> None:
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite JSON number at {location}.")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, location=f"{location}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"JSON object key at {location} must be a string.")
            _validate_json_value(item, location=f"{location}.{key}")
        return
    raise TypeError(
        f"Unsupported canonical JSON value at {location}: {type(value).__name__}."
    )


def canonical_json_bytes(value: JsonValue) -> bytes:
    """Serialize an exact JSON-domain value with one canonical byte encoding."""

    _validate_json_value(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: JsonValue) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def strict_canonical_json_loads(serialized: str | bytes) -> JsonValue:
    """Load canonical JSON, rejecting duplicates, NaN and noncanonical bytes."""

    if isinstance(serialized, bytes):
        try:
            raw = serialized.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("Canonical JSON must be valid UTF-8.") from error
    elif isinstance(serialized, str):
        raw = serialized
    else:
        raise TypeError("serialized must be str or bytes.")

    def reject_duplicate_keys(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key!r}.")
            result[key] = value
        return result

    def reject_constant(value: str) -> JsonValue:
        raise ValueError(f"Non-finite JSON constant {value!r} is forbidden.")

    try:
        parsed = json.loads(
            raw,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError("Invalid JSON.") from error
    _validate_json_value(parsed)
    normalized = cast(JsonValue, parsed)
    if canonical_json_bytes(normalized) != raw.encode("utf-8"):
        raise ValueError("JSON is valid but not in canonical byte form.")
    return normalized


def _require_operator(value: Any) -> OperatorName:
    if not isinstance(value, str) or value not in OPERATORS:
        raise ValueError(f"Operator must be one of {OPERATORS}, got {value!r}.")
    return cast(OperatorName, value)


def validate_raw_path(path: Sequence[str]) -> RawPath:
    if isinstance(path, (str, bytes)):
        raise TypeError("A raw path must be a sequence of operator names, not text.")
    values = tuple(path)
    if len(values) != K_STAGES:
        raise ValueError(f"A P07 raw path must have exactly {K_STAGES} stages.")
    normalized = tuple(_require_operator(value) for value in values)
    return cast(RawPath, normalized)


def canonicalize_path(path: Sequence[str]) -> CanonicalPath:
    """Apply only the frozen equivalence whitelist to one K-stage raw path."""

    raw_path = validate_raw_path(path)
    tokens = [operator for operator in raw_path if operator != "I"]
    maximum_rewrites = K_STAGES * (len(_PAIR_REWRITES) + len(_TRIPLE_REWRITES))
    rewrites = 0
    while True:
        changed = False
        for index in range(max(0, len(tokens) - 2)):
            triple = cast(
                tuple[OperatorName, OperatorName, OperatorName],
                tuple(tokens[index : index + 3]),
            )
            for _rewrite_id, lhs, rhs in _TRIPLE_REWRITES:
                if triple == lhs:
                    tokens[index : index + 3] = list(rhs)
                    rewrites += 1
                    if rewrites > maximum_rewrites:
                        raise RuntimeError("Equivalence rewrite failed to terminate.")
                    changed = True
                    break
            if changed:
                break
        if changed:
            continue
        for index in range(max(0, len(tokens) - 1)):
            pair = cast(tuple[OperatorName, OperatorName], tuple(tokens[index : index + 2]))
            for _rewrite_id, lhs, rhs in _PAIR_REWRITES:
                if pair == lhs:
                    tokens[index : index + 2] = list(rhs)
                    rewrites += 1
                    if rewrites > maximum_rewrites:
                        raise RuntimeError("Equivalence rewrite failed to terminate.")
                    changed = True
                    break
            if changed:
                break
        if not changed:
            break
    return tuple(tokens)


def expression_for_path(path: Sequence[str], *, canonical: bool = False) -> str:
    operators: Sequence[str]
    operators = canonicalize_path(path) if canonical else validate_raw_path(path)
    expression = "x"
    for operator in operators:
        expression = f"{operator}({expression})"
    return expression


def _raw_path_hash(path: RawPath) -> str:
    return canonical_json_sha256(
        {
            "kind": "p07_raw_path",
            "operators": list(path),
            "schema_version": SCHEMA_VERSION,
        }
    )


def _class_hash(canonical_path: CanonicalPath) -> str:
    return canonical_json_sha256(
        {
            "canonical_execution_order": list(canonical_path),
            "kind": "p07_semantic_class",
            "rewrite_whitelist_id": "p07-equivalence-whitelist-v2",
            "schema_version": SCHEMA_VERSION,
        }
    )


@lru_cache(maxsize=1)
def enumerate_path_records() -> tuple[PathRecord, ...]:
    records: list[PathRecord] = []
    seen_raw_ids: set[str] = set()
    for raw_values in itertools.product(OPERATORS, repeat=K_STAGES):
        raw_path = cast(RawPath, raw_values)
        canonical_path = canonicalize_path(raw_path)
        raw_hash = _raw_path_hash(raw_path)
        class_hash = _class_hash(canonical_path)
        raw_path_id = f"{_RAW_PATH_ID_PREFIX}{raw_hash}"
        class_id = f"{_CLASS_ID_PREFIX}{class_hash}"
        if raw_path_id in seen_raw_ids:
            raise RuntimeError("Raw path SHA-256 collision detected.")
        seen_raw_ids.add(raw_path_id)
        records.append(
            PathRecord(
                raw_path=raw_path,
                raw_path_id=raw_path_id,
                raw_path_sha256=raw_hash,
                raw_expression=expression_for_path(raw_path),
                canonical_path=canonical_path,
                canonical_expression=expression_for_path(raw_path, canonical=True),
                class_id=class_id,
                class_sha256=class_hash,
            )
        )
    if len(records) != len(OPERATORS) ** K_STAGES:
        raise RuntimeError("Raw path enumeration count is inconsistent with the contract.")
    return tuple(records)


@lru_cache(maxsize=1)
def enumerate_equivalence_classes() -> tuple[EquivalenceClass, ...]:
    grouped: dict[str, list[PathRecord]] = {}
    for record in enumerate_path_records():
        grouped.setdefault(record.class_id, []).append(record)
    classes: list[EquivalenceClass] = []
    for class_id in sorted(grouped):
        members = tuple(grouped[class_id])
        first = members[0]
        if any(
            member.class_sha256 != first.class_sha256
            or member.canonical_path != first.canonical_path
            or member.canonical_expression != first.canonical_expression
            for member in members[1:]
        ):
            raise RuntimeError("A semantic class contains inconsistent canonical records.")
        classes.append(
            EquivalenceClass(
                class_id=class_id,
                class_sha256=first.class_sha256,
                canonical_path=first.canonical_path,
                canonical_expression=first.canonical_expression,
                members=members,
            )
        )
    if len(classes) != 116:
        raise RuntimeError(f"Expected 116 semantic classes, found {len(classes)}.")
    if sum(item.ambiguous for item in classes) != PRIMARY_AMBIGUOUS_COUNT:
        raise RuntimeError("Equivalence-class ambiguity count is inconsistent with the contract.")
    return tuple(classes)


def _record_payload(record: PathRecord) -> dict[str, JsonValue]:
    return {
        "canonical_execution_order": list(record.canonical_path),
        "canonical_expression": record.canonical_expression,
        "class_id": record.class_id,
        "class_sha256": record.class_sha256,
        "raw_execution_order": list(record.raw_path),
        "raw_expression": record.raw_expression,
        "raw_path_id": record.raw_path_id,
        "raw_path_sha256": record.raw_path_sha256,
    }


def _class_payload(item: EquivalenceClass) -> dict[str, JsonValue]:
    return {
        "ambiguous": item.ambiguous,
        "canonical_execution_order": list(item.canonical_path),
        "canonical_expression": item.canonical_expression,
        "class_id": item.class_id,
        "class_sha256": item.class_sha256,
        "member_raw_path_ids": [member.raw_path_id for member in item.members],
        "multiplicity": item.multiplicity,
    }


def _path_universe_payload() -> dict[str, JsonValue]:
    records = enumerate_path_records()
    classes = enumerate_equivalence_classes()
    singleton_count = sum(item.multiplicity == 1 for item in classes)
    ambiguous_count = sum(item.multiplicity > 1 for item in classes)
    return {
        "class_count": len(classes),
        "classes": [_class_payload(item) for item in classes],
        "equivalence_rewrite_whitelist": [
            dict(item) for item in EQUIVALENCE_REWRITE_WHITELIST
        ],
        "k_stages": K_STAGES,
        "operator_execution_order": list(OPERATORS),
        "oracle_algorithm_id": "p07-independent-torch-float-oracle-v1",
        "path_count": len(records),
        "paths": [_record_payload(record) for record in records],
        "protocol_id": PROTOCOL_ID,
        "schema_version": SCHEMA_VERSION,
        "semantic_class_counts": {
            "ambiguous": ambiguous_count,
            "singleton": singleton_count,
        },
    }


def build_path_universe_manifest() -> dict[str, JsonValue]:
    payload = _path_universe_payload()
    return {**payload, "manifest_sha256": canonical_json_sha256(payload)}


def validate_path_universe_manifest(manifest: Any) -> dict[str, JsonValue]:
    if not isinstance(manifest, dict):
        raise TypeError("Path-universe manifest must be a JSON object.")
    expected = build_path_universe_manifest()
    if set(manifest) != set(expected):
        raise ValueError("Path-universe manifest has an invalid key set.")
    _validate_json_value(manifest)
    digest = manifest.get("manifest_sha256")
    if not _is_sha256(digest):
        raise ValueError("Path-universe manifest_sha256 is invalid.")
    payload = dict(manifest)
    payload.pop("manifest_sha256")
    if canonical_json_sha256(cast(dict[str, JsonValue], payload)) != digest:
        raise ValueError("Path-universe manifest hash mismatch.")
    if canonical_json_bytes(cast(JsonValue, manifest)) != canonical_json_bytes(expected):
        raise ValueError("Path-universe manifest does not match the frozen universe.")
    return cast(dict[str, JsonValue], manifest)


def path_universe_manifest_json() -> str:
    return canonical_json_bytes(build_path_universe_manifest()).decode("utf-8")


def load_path_universe_manifest(serialized: str | bytes) -> dict[str, JsonValue]:
    parsed = strict_canonical_json_loads(serialized)
    return validate_path_universe_manifest(parsed)


def _rank(class_id: str, *, salt: str, purpose: str) -> str:
    return canonical_json_sha256(
        {
            "class_id": class_id,
            "purpose": purpose,
            "salt": salt,
            "schema_version": SCHEMA_VERSION,
        }
    )


def _ranked_classes(
    classes: Sequence[EquivalenceClass], *, salt: str, purpose: str
) -> list[EquivalenceClass]:
    return sorted(
        classes,
        key=lambda item: (
            _rank(item.class_id, salt=salt, purpose=purpose),
            item.class_id,
        ),
    )


def _balanced_singleton_blocks() -> tuple[tuple[EquivalenceClass, ...], ...]:
    """Enumerate five-path blocks covering every operator once per position."""

    singleton_by_path = {
        item.members[0].raw_path: item
        for item in enumerate_equivalence_classes()
        if not item.ambiguous and item.canonical_expression != "x"
    }
    blocks: dict[tuple[str, ...], tuple[EquivalenceClass, ...]] = {}
    for second_column in itertools.permutations(NON_IDENTITY_OPERATORS):
        for third_column in itertools.permutations(NON_IDENTITY_OPERATORS):
            raw_paths = tuple(
                cast(RawPath, (first, second, third))
                for first, second, third in zip(
                    NON_IDENTITY_OPERATORS,
                    second_column,
                    third_column,
                )
            )
            if not all(path in singleton_by_path for path in raw_paths):
                continue
            block = tuple(singleton_by_path[path] for path in raw_paths)
            key = tuple(sorted(item.class_id for item in block))
            blocks[key] = block
    ranked = sorted(
        blocks.values(),
        key=lambda block: (
            canonical_json_sha256(
                {
                    "class_ids": sorted(item.class_id for item in block),
                    "purpose": "balanced-singleton-block",
                    "salt": PRIMARY_SELECTION_SALT,
                }
            ),
            tuple(sorted(item.class_id for item in block)),
        ),
    )
    if not ranked:
        raise RuntimeError("No balanced singleton coverage blocks are available.")
    return tuple(ranked)


@lru_cache(maxsize=1)
def _choose_disjoint_balanced_blocks() -> tuple[
    tuple[tuple[EquivalenceClass, ...], ...], int
]:
    required = sum(SINGLETON_POSITION_MINIMUMS.values())
    blocks = _balanced_singleton_blocks()
    for attempt in range(1024):
        ranked = sorted(
            blocks,
            key=lambda block: (
                canonical_json_sha256(
                    {
                        "attempt": attempt,
                        "class_ids": sorted(item.class_id for item in block),
                        "purpose": "disjoint-balanced-block-selection",
                        "salt": PRIMARY_SELECTION_SALT,
                    }
                ),
                tuple(sorted(item.class_id for item in block)),
            ),
        )
        chosen: list[tuple[EquivalenceClass, ...]] = []
        used_ids: set[str] = set()
        for block in ranked:
            block_ids = {item.class_id for item in block}
            if used_ids.isdisjoint(block_ids):
                chosen.append(block)
                used_ids.update(block_ids)
            if len(chosen) == required:
                return tuple(chosen), attempt
    raise RuntimeError("Could not choose enough disjoint balanced singleton blocks.")


@lru_cache(maxsize=1)
def _singleton_composition_splits() -> tuple[
    tuple[EquivalenceClass, ...],
    tuple[EquivalenceClass, ...],
    tuple[EquivalenceClass, ...],
]:
    candidates = [
        item
        for item in enumerate_equivalence_classes()
        if not item.ambiguous and item.canonical_expression != "x"
    ]
    if len(candidates) != 89:
        raise RuntimeError("Expected 89 nonempty singleton classes.")
    chosen_blocks, _attempt = _choose_disjoint_balanced_blocks()
    used_ids = {
        item.class_id for block in chosen_blocks for item in block
    }

    selected_by_split: dict[str, list[EquivalenceClass]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    block_offset = 0
    for split in ("train", "validation", "test"):
        block_count = SINGLETON_POSITION_MINIMUMS[split]
        for block in chosen_blocks[block_offset : block_offset + block_count]:
            selected_by_split[split].extend(block)
        block_offset += block_count

    available = [item for item in candidates if item.class_id not in used_ids]
    for split in ("train", "validation", "test"):
        filler_count = (
            COMPOSITION_SPLIT_COUNTS[split]["singleton"]
            - len(selected_by_split[split])
        )
        ranked_fillers = _ranked_classes(
            available,
            salt=PRIMARY_SELECTION_SALT,
            purpose=f"singleton-{split}-filler",
        )
        fillers = ranked_fillers[:filler_count]
        selected_by_split[split].extend(fillers)
        filler_ids = {item.class_id for item in fillers}
        used_ids.update(filler_ids)
        available = [item for item in available if item.class_id not in filler_ids]

    total = sum(len(items) for items in selected_by_split.values())
    if total != PRIMARY_SINGLETON_COUNT:
        raise RuntimeError("Constrained singleton selection count is inconsistent.")
    return (
        tuple(selected_by_split["train"]),
        tuple(selected_by_split["validation"]),
        tuple(selected_by_split["test"]),
    )


@lru_cache(maxsize=1)
def primary_classes() -> tuple[EquivalenceClass, ...]:
    classes = enumerate_equivalence_classes()
    ambiguous = [item for item in classes if item.ambiguous]
    singleton_splits = _singleton_composition_splits()
    selected_singletons = tuple(item for split in singleton_splits for item in split)
    if len(ambiguous) != PRIMARY_AMBIGUOUS_COUNT:
        raise RuntimeError("Primary path strata do not match the frozen universe.")
    selected = tuple(sorted((*ambiguous, *selected_singletons), key=lambda item: item.class_id))
    if len(selected) != PRIMARY_AMBIGUOUS_COUNT + PRIMARY_SINGLETON_COUNT:
        raise RuntimeError("Primary class selection count is inconsistent.")
    return selected


def _take_split(
    classes: Sequence[EquivalenceClass],
    *,
    stratum: str,
    counts: Mapping[str, int],
) -> dict[str, list[EquivalenceClass]]:
    ranked = _ranked_classes(
        classes,
        salt=COMPOSITION_SPLIT_SALT,
        purpose=f"composition-split-{stratum}",
    )
    if sum(counts.values()) != len(ranked):
        raise RuntimeError(f"Split counts do not cover the {stratum} stratum.")
    result: dict[str, list[EquivalenceClass]] = {}
    offset = 0
    for split in ("train", "validation", "test"):
        count = counts[split]
        result[split] = ranked[offset : offset + count]
        offset += count
    return result


def _seed_namespace_payload() -> dict[str, JsonValue]:
    return {
        "corruption_seed_derivation": {
            "algorithm": "sha256_first_16_hex_mod_2^63",
            "domain": CORRUPTION_SEED_DOMAIN,
        },
        "corruption_seed_by_optimization_seed": {
            str(key): value
            for key, value in sorted(CORRUPTION_SEED_BY_OPTIMIZATION_SEED.items())
        },
        "generator_seed_namespaces": {
            key: list(value) for key, value in sorted(GENERATOR_SEED_NAMESPACES.items())
        },
        "optimization_seed_policy": {
            "approximate_paired_mde_dz": 0.69,
            "maximum_directional_family_items": 6,
            "paired_seed_count": len(OPTIMIZATION_SEEDS),
            "power_not_claimed_at_dz": 0.5,
            "practical_target_dz": 0.8,
            "status": "practical_floor_not_formal_power_guarantee",
        },
        "optimization_seeds": list(OPTIMIZATION_SEEDS),
        "protocol_id": PROTOCOL_ID,
        "samples_per_generator_seed": dict(sorted(SAMPLES_PER_GENERATOR_SEED.items())),
        "schema_version": SCHEMA_VERSION,
    }


def build_seed_namespace_manifest() -> dict[str, JsonValue]:
    payload = _seed_namespace_payload()
    return {**payload, "manifest_sha256": canonical_json_sha256(payload)}


def validate_seed_namespace_manifest(manifest: Any) -> dict[str, JsonValue]:
    if not isinstance(manifest, dict):
        raise TypeError("Seed namespace manifest must be a JSON object.")
    expected = build_seed_namespace_manifest()
    if set(manifest) != set(expected):
        raise ValueError("Seed namespace manifest has an invalid key set.")
    _validate_json_value(manifest)
    digest = manifest.get("manifest_sha256")
    if not _is_sha256(digest):
        raise ValueError("Seed namespace manifest_sha256 is invalid.")
    payload = dict(manifest)
    payload.pop("manifest_sha256")
    if canonical_json_sha256(cast(dict[str, JsonValue], payload)) != digest:
        raise ValueError("Seed namespace manifest hash mismatch.")
    if canonical_json_bytes(cast(JsonValue, manifest)) != canonical_json_bytes(expected):
        raise ValueError("Seed namespace manifest does not match the frozen seed contract.")
    return cast(dict[str, JsonValue], manifest)


def _composition_split_payload() -> dict[str, JsonValue]:
    selected = primary_classes()
    ambiguous = [item for item in selected if item.ambiguous]
    singleton_by_split = dict(
        zip(("train", "validation", "test"), _singleton_composition_splits())
    )
    singleton = [item for items in singleton_by_split.values() for item in items]
    ambiguous_counts = {
        split: values["ambiguous"] for split, values in COMPOSITION_SPLIT_COUNTS.items()
    }
    ambiguous_split = _take_split(
        ambiguous,
        stratum="ambiguous",
        counts=ambiguous_counts,
    )
    splits: dict[str, JsonValue] = {}
    for split in ("train", "validation", "test"):
        members = sorted(
            (*ambiguous_split[split], *singleton_by_split[split]),
            key=lambda item: item.class_id,
        )
        splits[split] = {
            "ambiguous_count": sum(item.ambiguous for item in members),
            "class_ids": [item.class_id for item in members],
            "count": len(members),
            "singleton_count": sum(not item.ambiguous for item in members),
        }
    universe_manifest = build_path_universe_manifest()
    seed_manifest = build_seed_namespace_manifest()
    return {
        "composition_splits": splits,
        "path_universe_sha256": cast(str, universe_manifest["manifest_sha256"]),
        "primary_class_count": len(selected),
        "primary_class_ids": [item.class_id for item in selected],
        "primary_selection": {
            "ambiguous_policy": "all_nonempty_ambiguous_classes",
            "ambiguous_selected": len(ambiguous),
            "balanced_block_search_attempt": _choose_disjoint_balanced_blocks()[1],
            "selection_salt": PRIMARY_SELECTION_SALT,
            "singleton_policy": "disjoint_balanced_blocks_then_sha256_fill",
            "singleton_position_minimums": dict(SINGLETON_POSITION_MINIMUMS),
            "singleton_selected": len(singleton),
        },
        "protocol_id": PROTOCOL_ID,
        "schema_version": SCHEMA_VERSION,
        "seed_namespace_manifest": seed_manifest,
        "seed_namespace_sha256": cast(str, seed_manifest["manifest_sha256"]),
        "split_algorithm": "ambiguous_sha256_rank_and_singleton_coverage_constrained_sha256_v1",
        "split_salt": COMPOSITION_SPLIT_SALT,
    }


def build_composition_split_manifest() -> dict[str, JsonValue]:
    payload = _composition_split_payload()
    return {**payload, "manifest_sha256": canonical_json_sha256(payload)}


def validate_composition_split_manifest(manifest: Any) -> dict[str, JsonValue]:
    if not isinstance(manifest, dict):
        raise TypeError("Composition split manifest must be a JSON object.")
    expected = build_composition_split_manifest()
    if set(manifest) != set(expected):
        raise ValueError("Composition split manifest has an invalid key set.")
    _validate_json_value(manifest)
    digest = manifest.get("manifest_sha256")
    if not _is_sha256(digest):
        raise ValueError("Composition split manifest_sha256 is invalid.")
    payload = dict(manifest)
    payload.pop("manifest_sha256")
    if canonical_json_sha256(cast(dict[str, JsonValue], payload)) != digest:
        raise ValueError("Composition split manifest hash mismatch.")
    if canonical_json_bytes(cast(JsonValue, manifest)) != canonical_json_bytes(expected):
        raise ValueError("Composition split manifest does not match the frozen split.")
    return cast(dict[str, JsonValue], manifest)


def composition_split_manifest_json() -> str:
    return canonical_json_bytes(build_composition_split_manifest()).decode("utf-8")


def load_composition_split_manifest(serialized: str | bytes) -> dict[str, JsonValue]:
    parsed = strict_canonical_json_loads(serialized)
    return validate_composition_split_manifest(parsed)


def make_sample_id(split: str, generator_seed: int, sample_index: int) -> str:
    if split not in GENERATOR_SEED_NAMESPACES:
        raise ValueError(f"Unknown generator split {split!r}.")
    if isinstance(generator_seed, bool) or not isinstance(generator_seed, int):
        raise TypeError("generator_seed must be an integer.")
    if generator_seed not in GENERATOR_SEED_NAMESPACES[split]:
        raise ValueError(f"Seed {generator_seed} is not registered for split {split!r}.")
    if isinstance(sample_index, bool) or not isinstance(sample_index, int):
        raise TypeError("sample_index must be an integer.")
    if not 0 <= sample_index < SAMPLES_PER_GENERATOR_SEED[split]:
        raise ValueError(
            f"sample_index must be in [0, {SAMPLES_PER_GENERATOR_SEED[split]}) "
            f"for split {split!r}."
        )
    digest = canonical_json_sha256(
        {
            "generator_seed": generator_seed,
            "kind": "p07_generator_sample",
            "protocol_id": PROTOCOL_ID,
            "sample_index": sample_index,
            "split": split,
        }
    )
    return f"{_SAMPLE_ID_PREFIX}{digest}"


@lru_cache(maxsize=1)
def registered_sample_ids() -> frozenset[str]:
    identifiers = {
        make_sample_id(split, seed, index)
        for split, seeds in GENERATOR_SEED_NAMESPACES.items()
        for seed in seeds
        for index in range(SAMPLES_PER_GENERATOR_SEED[split])
    }
    expected_count = sum(
        len(seeds) * SAMPLES_PER_GENERATOR_SEED[split]
        for split, seeds in GENERATOR_SEED_NAMESPACES.items()
    )
    if len(identifiers) != expected_count:
        raise RuntimeError("Registered sample ID collision detected.")
    return frozenset(identifiers)


def validate_sample_id(sample_id: Any) -> str:
    if not isinstance(sample_id, str) or not sample_id.startswith(_SAMPLE_ID_PREFIX):
        raise ValueError("sample_id is not a registered P07 sample identifier.")
    digest_text = sample_id.removeprefix(_SAMPLE_ID_PREFIX)
    if not _is_sha256(digest_text):
        raise ValueError("sample_id does not contain a valid SHA-256 digest.")
    if sample_id not in registered_sample_ids():
        raise ValueError("sample_id is hash-shaped but absent from the registered namespace.")
    return sample_id


def derive_sample_seed(sample_id: str, purpose: str, *components: str | int) -> int:
    """Derive a stable nonnegative 63-bit seed without Python's process hash."""

    sample_id = validate_sample_id(sample_id)
    if not isinstance(purpose, str) or not purpose.strip():
        raise ValueError("purpose must be nonempty text.")
    normalized_components: list[JsonValue] = []
    for component in components:
        if isinstance(component, bool) or not isinstance(component, (str, int)):
            raise TypeError("Seed derivation components must be strings or integers.")
        normalized_components.append(component)
    digest = canonical_json_sha256(
        {
            "components": normalized_components,
            "purpose": purpose,
            "sample_id": sample_id,
            "schema_version": SCHEMA_VERSION,
        }
    )
    return int(digest[:16], 16) % (2**63)


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _validate_oracle_input(x: torch.Tensor) -> None:
    if not isinstance(x, torch.Tensor):
        raise TypeError("Oracle input must be a torch.Tensor.")
    if x.ndim != 3:
        raise ValueError(f"Oracle input must use (batch,length,channels), got {tuple(x.shape)}.")
    if any(int(size) <= 0 for size in x.shape):
        raise ValueError("Oracle input dimensions must be nonempty.")
    if int(x.shape[1]) < 2:
        raise ValueError("Oracle input length must be at least two.")
    if x.dtype not in {torch.float32, torch.float64}:
        raise TypeError("Oracle input must use torch.float32 or torch.float64.")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("Oracle input contains non-finite values.")


def oracle_apply_operator(operator: str, x: torch.Tensor) -> torch.Tensor:
    """Apply one independently implemented frozen operator to a BLC tensor."""

    name = _require_operator(operator)
    _validate_oracle_input(x)
    if name == "I":
        output = x
    elif name == "D1":
        output = torch.cat((torch.zeros_like(x[:, :1]), x[:, 1:] - x[:, :-1]), dim=1)
    elif name == "ABS":
        output = x.abs()
    elif name == "SQUARE":
        output = x.square()
    elif name == "MA3":
        x_bcl = x.permute(0, 2, 1)
        padded = F.pad(x_bcl, (1, 1), mode="replicate")
        output = F.avg_pool1d(padded, kernel_size=3, stride=1).permute(0, 2, 1)
    elif name == "HT":
        x_bcl = x.permute(0, 2, 1)
        length = int(x_bcl.shape[-1])
        spectrum = torch.fft.fft(x_bcl, dim=-1)
        multiplier = torch.zeros(length, dtype=x.dtype, device=x.device)
        multiplier[0] = 1.0
        if length % 2 == 0:
            multiplier[1 : length // 2] = 2.0
            multiplier[length // 2] = 1.0
        else:
            multiplier[1 : (length + 1) // 2] = 2.0
        output = torch.fft.ifft(spectrum * multiplier, dim=-1).abs().permute(0, 2, 1)
    else:  # pragma: no cover - exhaustive Literal guard
        raise AssertionError(f"Oracle operator implementation missing for {name}.")
    if output.shape != x.shape or output.dtype != x.dtype or output.device != x.device:
        raise RuntimeError(f"Oracle operator {name} violated shape/dtype/device preservation.")
    if not bool(torch.isfinite(output).all()):
        raise ValueError(f"Oracle operator {name} produced non-finite values.")
    return output


def oracle_execute_path(x: torch.Tensor, path: Sequence[str]) -> torch.Tensor:
    """Execute exactly one validated K-stage raw path in stage order."""

    raw_path = validate_raw_path(path)
    _validate_oracle_input(x)
    output = x
    for operator in raw_path:
        output = oracle_apply_operator(operator, output)
    return output


__all__ = [
    "COMPOSITION_SPLIT_COUNTS",
    "CORRUPTION_SEED_DOMAIN",
    "CORRUPTION_SEED_BY_OPTIMIZATION_SEED",
    "EQUIVALENCE_REWRITE_WHITELIST",
    "EquivalenceClass",
    "GENERATOR_SEED_NAMESPACES",
    "K_STAGES",
    "NON_IDENTITY_OPERATORS",
    "OPERATORS",
    "OPTIMIZATION_SEEDS",
    "OperatorName",
    "PRIMARY_AMBIGUOUS_COUNT",
    "PRIMARY_SINGLETON_COUNT",
    "PROTOCOL_ID",
    "PathRecord",
    "RawPath",
    "SAMPLES_PER_GENERATOR_SEED",
    "SINGLETON_POSITION_MINIMUMS",
    "build_composition_split_manifest",
    "build_path_universe_manifest",
    "build_seed_namespace_manifest",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "canonicalize_path",
    "composition_split_manifest_json",
    "derive_sample_seed",
    "enumerate_equivalence_classes",
    "enumerate_path_records",
    "expression_for_path",
    "load_composition_split_manifest",
    "load_path_universe_manifest",
    "make_sample_id",
    "oracle_apply_operator",
    "oracle_execute_path",
    "path_universe_manifest_json",
    "primary_classes",
    "registered_sample_ids",
    "strict_canonical_json_loads",
    "validate_composition_split_manifest",
    "validate_path_universe_manifest",
    "validate_raw_path",
    "validate_sample_id",
    "validate_seed_namespace_manifest",
]

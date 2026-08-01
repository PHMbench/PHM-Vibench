from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace

import pytest

from src.data_factory.sample_cohort_manifest import build_sample_cohort_manifest
from src.data_factory.split_consumer import (
    BoundSplit,
    ContractPins,
    SplitConsumerError,
    authoritative_unit_derivation_version,
    bind_split_contract,
    compute_authoritative_unit_mapping_sha256,
    compute_metadata_rows_sha256,
)
from src.data_factory.split_manifest import build_split_manifest


_RELEASE_SHA = "a" * 64
_METADATA_SHA = "b" * 64
_TRANSFORM_SHA = "c" * 64
_UNIT_MAPPING_ID = "fixture-owner-map"
_UNIT_MAPPING_VERSION = "v1"


def _assigned_specs() -> list[tuple[int, str, str, str, int, object]]:
    return [
        (1, "r1.bin", "unit-train-1", "source", 0, None),
        (2, "r2.bin", "unit-train-2", "source", 1, 2),
        (3, "r3.bin", "unit-validation", "source", 0, None),
        (4, "nested/r4.bin", "unit-test-1", "target", 0, "terminal"),
        (5, "nested/r5.bin", "unit-test-2", "target", 1, "terminal"),
        (6, "nested/r6.bin", "unit-test-3", "target", 0, "terminal"),
        (7, "nested/r7.bin", "unit-test-4", "target", 1, "terminal"),
        (8, "nested/r8.bin", "unit-test-5", "target", 0, "terminal"),
    ]


def _rows() -> list[dict[str, object]]:
    rows = [
        {
            "Id": row_id,
            "Dataset_id": "fixture",
            "Name": "FixtureSource",
            "File": file_name,
            "Record_id": f"fixture:raw/FixtureSource/{file_name}",
            "Physical_unit_id": unit,
            "Domain_id": domain,
            "Target": target,
            "Temporal_index": temporal,
        }
        for row_id, file_name, unit, domain, target, temporal in _assigned_specs()
    ]
    rows.append(
        {
            "Id": 9,
            "Dataset_id": "fixture",
            "Name": "FixtureSource",
            "File": "excluded.bin",
            "Record_id": "fixture:raw/FixtureSource/excluded.bin",
            "Physical_unit_id": "unit-excluded",
            "Domain_id": "source",
            "Target": -1,
            "Temporal_index": None,
        }
    )
    return rows


def _unit_mapping() -> dict[str, str]:
    return {
        str(row_id): unit
        for row_id, _, unit, _, _, _ in _assigned_specs()
    }


def _assignments() -> list[dict[str, object]]:
    assignments = []
    for row_id, file_name, unit, domain, target, temporal in _assigned_specs():
        split = "train" if row_id <= 2 else "validation" if row_id == 3 else "test"
        assignments.append(
            {
                "dataset_id": "fixture",
                "metadata_row_id": str(row_id),
                "source_record_path": f"raw/FixtureSource/{file_name}",
                "record_id": f"fixture:raw/FixtureSource/{file_name}",
                "physical_unit_id": unit,
                "domain_id": domain,
                "split": split,
                "target": target,
                "temporal_index": temporal,
            }
        )
    return assignments


def _split_manifest(
    *,
    assignments: list[dict[str, object]] | None = None,
    provenance_overrides: dict[str, object] | None = None,
    physical_unit_version: str | None = None,
) -> dict[str, object]:
    provenance: dict[str, object] = {
        "dataset_id": "fixture",
        "dataset_release_id": "fixture-release-v1",
        "dataset_release_sha256": _RELEASE_SHA,
        "metadata_path": "metadata.csv",
        "metadata_sha256": _METADATA_SHA,
        "task_transform_id": "fixture-transform-v1",
        "task_transform_sha256": _TRANSFORM_SHA,
    }
    provenance.update(provenance_overrides or {})
    return build_split_manifest(
        provenance=provenance,
        split={
            "seed": 17,
            "algorithm": "fixture-grouped-v1",
            "source_domains": ["source"],
            "target_domains": ["target"],
        },
        identity_derivation_versions={
            "metadata_row_id": "metadata-integer-decimal.v1",
            "source_record_path": "raw-name-file-posix.v1",
            "record_id": "dataset-source-path.v1",
            "physical_unit_id": physical_unit_version
            or authoritative_unit_derivation_version(
                _UNIT_MAPPING_ID, _UNIT_MAPPING_VERSION
            ),
        },
        assignments=assignments if assignments is not None else _assignments(),
        exclusions=[{"metadata_row_id": "9", "reason": "fixture invalid label"}],
    )


def _entries() -> list[dict[str, object]]:
    # Deliberately not split-sorted: global cohort order must be retained.
    assignments = {item["metadata_row_id"]: item for item in _assignments()}

    def entry(row_id: str, suffix: str, start: int) -> dict[str, object]:
        assignment = assignments[row_id]
        return {
            "sample_id": f"sample-{row_id}-{suffix}",
            "physical_unit_id": assignment["physical_unit_id"],
            "record_id": assignment["record_id"],
            "start": start,
            "end": start + 4,
            "split": assignment["split"],
        }

    return [
        entry("4", "test", 4),
        entry("1", "train-a", 0),
        entry("3", "validation", 0),
        entry("1", "train-b", 4),
        entry("2", "train", 0),
        *(entry(str(row_id), "test", 0) for row_id in range(5, 9)),
    ]


def _cohort_manifest(
    split_manifest: dict[str, object],
    *,
    entries: list[dict[str, object]] | None = None,
    provenance_overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    provenance: dict[str, object] = {
        "dataset_id": "fixture",
        "dataset_release_id": "fixture-release-v1",
        "dataset_release_sha256": _RELEASE_SHA,
        "task_transform_id": "fixture-transform-v1",
        "task_transform_sha256": _TRANSFORM_SHA,
        "split_manifest_sha256": split_manifest["manifest_sha256"],
    }
    provenance.update(provenance_overrides or {})
    return build_sample_cohort_manifest(
        provenance=provenance,
        window_sampling_rule={
            "window": {
                "rule_id": "fixture-window",
                "version": "v1",
                "coordinate_system": "raw_sample_index",
                "parameters": {"length": 4},
            },
            "sampling": {
                "rule_id": "fixture-census",
                "version": "v1",
                "parameters": {},
            },
        },
        entries=entries if entries is not None else _entries(),
    )


def _fixture() -> tuple[
    list[dict[str, object]], dict[str, object], dict[str, object], ContractPins
]:
    rows = _rows()
    split = _split_manifest()
    cohort = _cohort_manifest(split)
    unit_mapping = _unit_mapping()
    pins = ContractPins(
        split_manifest_sha256=split["manifest_sha256"],
        cohort_manifest_sha256=cohort["manifest_sha256"],
        dataset_id="fixture",
        dataset_release_id="fixture-release-v1",
        dataset_release_sha256=_RELEASE_SHA,
        metadata_sha256=_METADATA_SHA,
        metadata_rows_sha256=compute_metadata_rows_sha256(rows),
        task_transform_id="fixture-transform-v1",
        task_transform_sha256=_TRANSFORM_SHA,
        physical_unit_mapping_id=_UNIT_MAPPING_ID,
        physical_unit_mapping_version=_UNIT_MAPPING_VERSION,
        physical_unit_mapping_sha256=compute_authoritative_unit_mapping_sha256(
            unit_mapping,
            mapping_id=_UNIT_MAPPING_ID,
            version=_UNIT_MAPPING_VERSION,
        ),
    )
    return rows, split, cohort, pins


def _bind(
    rows: list[dict[str, object]],
    split: dict[str, object],
    cohort: dict[str, object],
    pins: ContractPins,
    *,
    unit_mapping: dict[str, str] | None = None,
    repin_metadata_rows: bool = True,
    repin_unit_mapping: bool = True,
) -> BoundSplit:
    unit_mapping = _unit_mapping() if unit_mapping is None else unit_mapping
    if repin_metadata_rows:
        pins = replace(pins, metadata_rows_sha256=compute_metadata_rows_sha256(rows))
    if repin_unit_mapping:
        pins = replace(
            pins,
            physical_unit_mapping_sha256=compute_authoritative_unit_mapping_sha256(
                unit_mapping,
                mapping_id=pins.physical_unit_mapping_id,
                version=pins.physical_unit_mapping_version,
            ),
        )
    return bind_split_contract(
        metadata_rows=rows,
        split_manifest=split,
        cohort_manifest=cohort,
        pins=pins,
        authoritative_unit_by_metadata_id=unit_mapping,
    )


def test_success_returns_frozen_three_way_ids_and_ordered_windows() -> None:
    rows, split, cohort, pins = _fixture()

    bound = _bind(list(reversed(rows)), split, cohort, pins)

    assert bound.ids_for("train") == ("1", "2")
    assert bound.ids_for("validation") == ("3",)
    assert bound.ids_for("test") == ("4", "5", "6", "7", "8")
    assert [window.sample_id for window in bound.ordered_windows] == [
        "sample-4-test",
        "sample-1-train-a",
        "sample-3-validation",
        "sample-1-train-b",
        "sample-2-train",
        "sample-5-test",
        "sample-6-test",
        "sample-7-test",
        "sample-8-test",
    ]
    assert [window.sample_id for window in bound.train[0].windows] == [
        "sample-1-train-a",
        "sample-1-train-b",
    ]
    assert bound.excluded_metadata_row_ids == ("9",)
    with pytest.raises(FrozenInstanceError):
        bound.train[0].split = "test"  # type: ignore[misc]


@pytest.mark.parametrize("pin_field", ["split_manifest_sha256", "cohort_manifest_sha256"])
def test_external_manifest_pin_mismatch_is_rejected(pin_field: str) -> None:
    rows, split, cohort, pins = _fixture()
    pins = replace(pins, **{pin_field: "0" * 64})

    with pytest.raises(SplitConsumerError, match="external pin"):
        _bind(rows, split, cohort, pins)


@pytest.mark.parametrize(
    ("pin_field", "value"),
    [
        ("dataset_id", "other"),
        ("dataset_release_id", "other-release"),
        ("dataset_release_sha256", "d" * 64),
        ("metadata_sha256", "f" * 64),
        ("task_transform_id", "other-transform"),
        ("task_transform_sha256", "e" * 64),
    ],
)
def test_split_provenance_must_match_external_pins(
    pin_field: str, value: str
) -> None:
    rows, split, cohort, pins = _fixture()
    pins = replace(pins, **{pin_field: value})

    with pytest.raises(SplitConsumerError, match="split provenance"):
        _bind(rows, split, cohort, pins)


def test_cohort_must_reference_same_split_and_provenance() -> None:
    rows, split, _, pins = _fixture()
    cohort = _cohort_manifest(
        split,
        provenance_overrides={"split_manifest_sha256": "d" * 64},
    )
    pins = replace(pins, cohort_manifest_sha256=cohort["manifest_sha256"])

    with pytest.raises(SplitConsumerError, match="different split"):
        _bind(rows, split, cohort, pins)


def test_internal_manifest_tamper_is_rejected() -> None:
    rows, split, cohort, pins = _fixture()
    tampered = deepcopy(split)
    tampered["assignments"][0]["record_id"] = "tampered"

    with pytest.raises(SplitConsumerError, match="manifest validation failed"):
        _bind(rows, tampered, cohort, pins)


@pytest.mark.parametrize("drop_or_add", ["drop", "add"])
def test_assignments_and_exclusions_must_exactly_cover_metadata(
    drop_or_add: str,
) -> None:
    rows, split, cohort, pins = _fixture()
    if drop_or_add == "drop":
        rows.pop()
    else:
        extra = deepcopy(rows[0])
        extra["Id"] = 99
        extra["File"] = "extra.bin"
        extra["Record_id"] = "fixture:raw/FixtureSource/extra.bin"
        rows.append(extra)

    with pytest.raises(SplitConsumerError, match="exactly cover"):
        _bind(rows, split, cohort, pins)


def test_metadata_id_canonicalization_and_collision_fail_closed() -> None:
    rows, split, cohort, pins = _fixture()
    rows[0]["Id"] = "1"
    with pytest.raises(SplitConsumerError, match="metadata-integer-decimal"):
        _bind(rows, split, cohort, pins)

    rows = _rows()
    rows[1]["Id"] = 1
    with pytest.raises(SplitConsumerError, match="duplicate canonical"):
        _bind(rows, split, cohort, pins)


@pytest.mark.parametrize("mutation", ["order", "required_field"])
def test_ordered_metadata_rows_have_an_independent_external_pin(
    mutation: str,
) -> None:
    rows, split, cohort, pins = _fixture()
    if mutation == "order":
        rows.reverse()
    else:
        rows[-1]["Target"] = "changed-exclusion"

    with pytest.raises(SplitConsumerError, match="metadata rows.*external pin"):
        _bind(
            rows,
            split,
            cohort,
            pins,
            repin_metadata_rows=False,
        )


def test_zero_integer_and_zero_string_cannot_fake_two_target_classes() -> None:
    rows, _, _, pins = _fixture()
    assignments = _assignments()
    assignments[1]["target"] = "0"
    rows[1]["Target"] = "0"
    for offset, assignment in enumerate(assignments[3:]):
        target: object = 0 if offset % 2 == 0 else "0"
        assignment["target"] = target
        rows[3 + offset]["Target"] = target
    split = _split_manifest(assignments=assignments)
    cohort = _cohort_manifest(split)
    pins = replace(
        pins,
        split_manifest_sha256=split["manifest_sha256"],
        cohort_manifest_sha256=cohort["manifest_sha256"],
    )

    with pytest.raises(SplitConsumerError, match="mixed string/integer"):
        _bind(rows, split, cohort, pins)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("Dataset_id", "other", "pinned dataset"),
        ("File", "other.bin", "source_record_path"),
        ("Record_id", "other-record", "dataset-source-path"),
        ("Physical_unit_id", "other-unit", "physical_unit_id"),
        ("Domain_id", "target", "domain_id"),
        ("Target", 9, "target"),
        ("Temporal_index", 3, "temporal_index"),
    ],
)
def test_metadata_identity_and_transformed_task_values_must_match(
    field: str, value: object, message: str
) -> None:
    rows, split, cohort, pins = _fixture()
    rows[0][field] = value

    with pytest.raises(SplitConsumerError, match=message):
        _bind(rows, split, cohort, pins)


def test_non_normalized_metadata_file_path_is_rejected() -> None:
    rows, split, cohort, pins = _fixture()
    rows[0]["File"] = "../escape.bin"

    with pytest.raises(SplitConsumerError, match="normalized relative POSIX"):
        _bind(rows, split, cohort, pins)


@pytest.mark.parametrize("control", ["\x00", "\x1f", "\x7f", "\x85"])
def test_c0_c1_path_controls_are_rejected(control: str) -> None:
    rows, split, cohort, pins = _fixture()
    rows[0]["File"] = f"bad{control}.bin"

    with pytest.raises(SplitConsumerError, match="control-free"):
        _bind(rows, split, cohort, pins)


def test_excluded_rows_still_require_complete_safe_metadata() -> None:
    rows, split, cohort, pins = _fixture()
    rows[-1]["File"] = "../excluded-escape.bin"
    with pytest.raises(SplitConsumerError, match="normalized relative POSIX"):
        _bind(rows, split, cohort, pins)

    rows = _rows()
    del rows[-1]["File"]
    with pytest.raises(SplitConsumerError, match="missing required keys"):
        _bind(rows, split, cohort, pins)


@pytest.mark.parametrize("kind", ["unknown_record", "unit", "split"])
def test_cohort_entries_must_match_split_assignments(kind: str) -> None:
    rows, split, _, pins = _fixture()
    entries = _entries()
    if kind == "unknown_record":
        entries[0]["record_id"] = "fixture:raw/FixtureSource/unknown.bin"
    elif kind == "unit":
        entries[0]["physical_unit_id"] = "other-unit"
    else:
        entries[0]["split"] = "train"
    cohort = _cohort_manifest(split, entries=entries)
    pins = replace(pins, cohort_manifest_sha256=cohort["manifest_sha256"])

    message = "no split assignment" if kind == "unknown_record" else kind
    with pytest.raises(SplitConsumerError, match=message):
        _bind(rows, split, cohort, pins)


def test_every_assigned_record_requires_a_cohort_entry() -> None:
    rows, split, _, pins = _fixture()
    entries = [
        entry
        for entry in _entries()
        if entry["record_id"] != _assignments()[1]["record_id"]
    ]
    cohort = _cohort_manifest(split, entries=entries)
    pins = replace(pins, cohort_manifest_sha256=cohort["manifest_sha256"])

    with pytest.raises(SplitConsumerError, match="every assigned record"):
        _bind(rows, split, cohort, pins)


def test_failure_occurs_before_any_raw_io_callback() -> None:
    rows, split, cohort, pins = _fixture()
    rows[0]["Physical_unit_id"] = "wrong-unit"
    raw_io_calls: list[str] = []

    def bind_then_raw() -> None:
        _bind(rows, split, cohort, pins)
        raw_io_calls.append("raw")  # pragma: no cover - must remain unreachable

    with pytest.raises(SplitConsumerError, match="physical_unit_id"):
        bind_then_raw()
    assert raw_io_calls == []


def test_duplicate_derived_record_id_is_rejected() -> None:
    rows, split, cohort, pins = _fixture()
    rows[1]["File"] = rows[0]["File"]
    rows[1]["Record_id"] = rows[0]["Record_id"]

    with pytest.raises(SplitConsumerError, match="duplicate derived record_id"):
        _bind(rows, split, cohort, pins)


def test_authoritative_unit_mapping_hash_and_coverage_are_external() -> None:
    rows, split, cohort, pins = _fixture()
    changed_mapping = _unit_mapping()
    changed_mapping["4"] = "silently-changed-unit"
    with pytest.raises(SplitConsumerError, match="unit mapping.*external pin"):
        _bind(
            rows,
            split,
            cohort,
            pins,
            unit_mapping=changed_mapping,
            repin_unit_mapping=False,
        )

    missing_mapping = _unit_mapping()
    del missing_mapping["4"]
    with pytest.raises(SplitConsumerError, match="missing row '4'"):
        _bind(rows, split, cohort, pins, unit_mapping=missing_mapping)


def test_split_explicitly_binds_authoritative_unit_map_identity() -> None:
    rows, _, _, pins = _fixture()
    split = _split_manifest(
        physical_unit_version=authoritative_unit_derivation_version(
            "different-map", "v2"
        )
    )
    cohort = _cohort_manifest(split)
    pins = replace(
        pins,
        split_manifest_sha256=split["manifest_sha256"],
        cohort_manifest_sha256=cohort["manifest_sha256"],
    )

    with pytest.raises(SplitConsumerError, match="not supported"):
        _bind(rows, split, cohort, pins)


@pytest.mark.parametrize("profile_failure", ["units", "classes"])
def test_p02_test_profile_is_enforced(profile_failure: str) -> None:
    rows, _, _, pins = _fixture()
    assignments = _assignments()
    unit_mapping = _unit_mapping()
    if profile_failure == "units":
        assignments[4]["physical_unit_id"] = assignments[3]["physical_unit_id"]
        rows[4]["Physical_unit_id"] = rows[3]["Physical_unit_id"]
        unit_mapping["5"] = unit_mapping["4"]
    else:
        for assignment in assignments[3:]:
            assignment["target"] = 0
        for row in rows[3:8]:
            row["Target"] = 0

    split = _split_manifest(assignments=assignments)
    entries = _entries()
    if profile_failure == "units":
        entries[5]["physical_unit_id"] = assignments[4]["physical_unit_id"]
    cohort = _cohort_manifest(split, entries=entries)
    pins = replace(
        pins,
        split_manifest_sha256=split["manifest_sha256"],
        cohort_manifest_sha256=cohort["manifest_sha256"],
    )

    message = "five unique" if profile_failure == "units" else "two target classes"
    with pytest.raises(SplitConsumerError, match=message):
        _bind(rows, split, cohort, pins, unit_mapping=unit_mapping)


def test_unsupported_identity_derivation_is_rejected() -> None:
    rows, _, _, _ = _fixture()
    split = build_split_manifest(
        provenance={
            "dataset_id": "fixture",
            "dataset_release_id": "fixture-release-v1",
            "dataset_release_sha256": _RELEASE_SHA,
            "metadata_path": "metadata.csv",
            "metadata_sha256": _METADATA_SHA,
            "task_transform_id": "fixture-transform-v1",
            "task_transform_sha256": _TRANSFORM_SHA,
        },
        split={
            "seed": 17,
            "algorithm": "fixture-grouped-v1",
            "source_domains": ["source"],
            "target_domains": ["target"],
        },
        identity_derivation_versions={
            "metadata_row_id": "unguarded-fallback-v0",
            "source_record_path": "raw-name-file-posix.v1",
            "record_id": "dataset-source-path.v1",
            "physical_unit_id": authoritative_unit_derivation_version(
                _UNIT_MAPPING_ID, _UNIT_MAPPING_VERSION
            ),
        },
        assignments=_assignments(),
        exclusions=[{"metadata_row_id": "9", "reason": "fixture invalid label"}],
    )
    cohort = _cohort_manifest(split)
    pins = ContractPins(
        split_manifest_sha256=split["manifest_sha256"],
        cohort_manifest_sha256=cohort["manifest_sha256"],
        dataset_id="fixture",
        dataset_release_id="fixture-release-v1",
        dataset_release_sha256=_RELEASE_SHA,
        metadata_sha256=_METADATA_SHA,
        metadata_rows_sha256=compute_metadata_rows_sha256(rows),
        task_transform_id="fixture-transform-v1",
        task_transform_sha256=_TRANSFORM_SHA,
        physical_unit_mapping_id=_UNIT_MAPPING_ID,
        physical_unit_mapping_version=_UNIT_MAPPING_VERSION,
        physical_unit_mapping_sha256=compute_authoritative_unit_mapping_sha256(
            _unit_mapping(),
            mapping_id=_UNIT_MAPPING_ID,
            version=_UNIT_MAPPING_VERSION,
        ),
    )

    with pytest.raises(SplitConsumerError, match="not supported"):
        _bind(rows, split, cohort, pins)

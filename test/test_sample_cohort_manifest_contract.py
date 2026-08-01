from __future__ import annotations

from copy import deepcopy

import pytest

from src.data_factory.sample_cohort_manifest import (
    SAMPLE_COHORT_MANIFEST_SCHEMA,
    SampleCohortManifestError,
    build_sample_cohort_manifest,
    compute_sample_cohort_manifest_sha256,
    dumps_sample_cohort_manifest,
    loads_sample_cohort_manifest,
    validate_sample_cohort_manifest,
)


def _provenance() -> dict[str, str]:
    return {
        "dataset_id": "CWRU",
        "dataset_release_id": "CWRU-test-release",
        "dataset_release_sha256": "a" * 64,
        "task_transform_id": "cwru-4-class-test",
        "task_transform_sha256": "b" * 64,
        "split_manifest_sha256": "c" * 64,
    }


def _rule() -> dict[str, object]:
    return {
        "window": {
            "rule_id": "fixed-raw-window",
            "version": "1",
            "coordinate_system": "raw_sample_index",
            "parameters": {"length": 10, "stride": 5},
        },
        "sampling": {
            "rule_id": "ordered-fixture-sampling",
            "version": "1",
            "parameters": {"seed": 7, "limit_per_record": None},
        },
    }


def _entries() -> list[dict[str, object]]:
    return [
        {
            "sample_id": "sample-001",
            "physical_unit_id": "unit-001",
            "record_id": "record-001",
            "start": 0,
            "end": 10,
            "split": "train",
        },
        {
            "sample_id": "sample-002",
            "physical_unit_id": "unit-001",
            "record_id": "record-001",
            "start": 5,
            "end": 15,
            "split": "train",
        },
        {
            "sample_id": "sample-003",
            "physical_unit_id": "unit-002",
            "record_id": "record-002",
            "start": 0,
            "end": 10,
            "split": "validation",
        },
        {
            "sample_id": "sample-004",
            "physical_unit_id": "unit-003",
            "record_id": "record-003",
            "start": 0,
            "end": 10,
            "split": "test",
        },
    ]


def _build(
    *,
    provenance: dict[str, str] | None = None,
    rule: dict[str, object] | None = None,
    entries: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return build_sample_cohort_manifest(
        provenance=provenance if provenance is not None else _provenance(),
        window_sampling_rule=rule if rule is not None else _rule(),
        entries=entries if entries is not None else _entries(),
    )


def test_sample_cohort_round_trip_and_ordered_canonical_hash() -> None:
    manifest = _build()
    payload = dumps_sample_cohort_manifest(manifest)
    declared_hash = manifest["manifest_sha256"]
    reordered = _build(entries=list(reversed(_entries())))

    assert loads_sample_cohort_manifest(payload) == manifest
    assert loads_sample_cohort_manifest(payload.encode("utf-8")) == manifest
    assert manifest["schema"] == SAMPLE_COHORT_MANIFEST_SCHEMA
    assert declared_hash == compute_sample_cohort_manifest_sha256(manifest)
    assert reordered["manifest_sha256"] != declared_hash
    manifest["manifest_sha256"] = "0" * 64
    assert compute_sample_cohort_manifest_sha256(manifest) == declared_hash
    assert '": ' not in payload
    assert ", " not in payload
    assert "\n" not in payload


def test_dataset_task_split_and_rule_identity_are_bound() -> None:
    manifest = _build()

    assert manifest["provenance"] == _provenance()
    assert manifest["window_sampling_rule"] == _rule()
    assert manifest["window_sampling_rule"]["window"]["coordinate_system"] == (
        "raw_sample_index"
    )


def test_membership_is_derived_from_ordered_entries() -> None:
    manifest = _build()

    assert manifest["membership"]["ordered_sample_ids"] == [
        "sample-001",
        "sample-002",
        "sample-003",
        "sample-004",
    ]
    assert manifest["membership"]["by_split"]["train"] == {
        "entry_count": 2,
        "sample_ids": ["sample-001", "sample-002"],
        "record_ids": ["record-001"],
        "physical_unit_ids": ["unit-001"],
    }


def test_same_split_overlap_is_allowed_and_audited() -> None:
    manifest = _build()
    audit = manifest["audits"]["within_split_raw_interval_overlap"]

    assert audit["allowed"] is True
    assert audit["overlap_count"] == 1
    assert audit["overlaps"] == [
        {
            "record_id": "record-001",
            "left_sample_id": "sample-001",
            "right_sample_id": "sample-002",
            "overlap_start": 5,
            "overlap_end": 10,
            "split": "train",
        }
    ]
    assert manifest["audits"]["cross_split_raw_interval_overlap"] == {
        "passed": True,
        "overlap_count": 0,
        "overlaps": [],
    }


def test_duplicate_sample_is_rejected() -> None:
    entries = _entries()
    entries[1]["sample_id"] = entries[0]["sample_id"]

    with pytest.raises(SampleCohortManifestError, match="duplicate sample_id"):
        _build(entries=entries)


def test_duplicate_raw_coordinate_window_is_rejected() -> None:
    entries = _entries()
    duplicate = dict(entries[0])
    duplicate["sample_id"] = "sample-duplicate-coordinate"
    entries.append(duplicate)

    with pytest.raises(SampleCohortManifestError, match="duplicate raw-coordinate window"):
        _build(entries=entries)


def test_record_maps_to_exactly_one_physical_unit() -> None:
    entries = _entries()
    entries[1]["physical_unit_id"] = "unit-conflict"

    with pytest.raises(SampleCohortManifestError, match="maps to multiple"):
        _build(entries=entries)


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [(-1, 5, "start must be a non-negative"), (5, 5, "non-empty interval")],
)
def test_negative_or_empty_interval_is_rejected(
    start: int, end: int, message: str
) -> None:
    entries = _entries()
    entries[0]["start"] = start
    entries[0]["end"] = end

    with pytest.raises(SampleCohortManifestError, match=message):
        _build(entries=entries)


@pytest.mark.parametrize("split", ["val", "holdout", "TRAIN"])
def test_unknown_split_is_rejected(split: str) -> None:
    entries = _entries()
    entries[0]["split"] = split

    with pytest.raises(SampleCohortManifestError, match="is not one of"):
        _build(entries=entries)


@pytest.mark.parametrize(
    ("identity", "message"),
    [
        ("record_id", "record_id .* appears in both"),
        ("physical_unit_id", "physical_unit_id .* appears in both"),
    ],
)
def test_record_or_unit_cannot_cross_splits(identity: str, message: str) -> None:
    entries = _entries()
    entries[2][identity] = entries[0][identity]
    if identity == "record_id":
        entries[2]["physical_unit_id"] = entries[0]["physical_unit_id"]
    entries[2]["start"] = 20
    entries[2]["end"] = 30

    with pytest.raises(SampleCohortManifestError, match=message):
        _build(entries=entries)


def test_cross_split_raw_coordinate_overlap_is_rejected() -> None:
    entries = _entries()
    entries[2]["record_id"] = entries[0]["record_id"]
    entries[2]["start"] = 8
    entries[2]["end"] = 12

    with pytest.raises(
        SampleCohortManifestError, match="cross-split raw-coordinate interval overlap"
    ):
        _build(entries=entries)


def test_membership_tamper_is_rejected() -> None:
    tampered = _build()
    tampered["membership"]["by_split"]["train"]["entry_count"] = 99

    with pytest.raises(SampleCohortManifestError, match="derived from entries"):
        validate_sample_cohort_manifest(tampered)


def test_content_and_hash_tamper_are_rejected() -> None:
    content_tamper = deepcopy(_build())
    content_tamper["window_sampling_rule"]["sampling"]["version"] = "2"
    with pytest.raises(SampleCohortManifestError, match="does not match"):
        validate_sample_cohort_manifest(content_tamper)

    hash_tamper = _build()
    hash_tamper["manifest_sha256"] = "0" * 64
    with pytest.raises(SampleCohortManifestError, match="does not match"):
        validate_sample_cohort_manifest(hash_tamper)


def test_missing_required_entry_key_is_rejected() -> None:
    entries = _entries()
    del entries[0]["physical_unit_id"]

    with pytest.raises(SampleCohortManifestError, match="missing required keys"):
        _build(entries=entries)


def test_duplicate_json_object_key_is_rejected() -> None:
    payload = dumps_sample_cohort_manifest(_build())
    duplicated = payload.replace(
        '"schema":"p02.sample-cohort-manifest.v1"',
        '"schema":"p02.sample-cohort-manifest.v1",'
        '"schema":"p02.sample-cohort-manifest.v1"',
    )

    with pytest.raises(SampleCohortManifestError, match="duplicate JSON object key"):
        loads_sample_cohort_manifest(duplicated)

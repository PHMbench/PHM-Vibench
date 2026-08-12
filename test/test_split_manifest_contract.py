from __future__ import annotations

from copy import deepcopy

import pytest

from src.data_factory.split_manifest import (
    SPLIT_MANIFEST_SCHEMA,
    SplitManifestError,
    build_split_manifest,
    compute_split_manifest_sha256,
    dumps_split_manifest,
    loads_split_manifest,
    validate_split_manifest,
)


_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64
_DIGEST_C = "c" * 64


def _provenance() -> dict[str, str]:
    return {
        "dataset_id": "CWRU",
        "dataset_release_id": "CWRU-test-release",
        "dataset_release_sha256": _DIGEST_A,
        "metadata_path": "data/metadata.csv",
        "metadata_sha256": _DIGEST_B,
        "task_transform_id": "cwru-4-class-test",
        "task_transform_sha256": _DIGEST_C,
    }


def _split() -> dict[str, object]:
    return {
        "seed": 7,
        "algorithm": "fixture-grouped-v1",
        "source_domains": ["source"],
        "target_domains": ["target"],
    }


def _identity_versions() -> dict[str, str]:
    return {
        "metadata_row_id": "fixture-row-id-v1",
        "source_record_path": "relative-posix-v1",
        "record_id": "fixture-record-id-v1",
        "physical_unit_id": "fixture-unit-id-v1",
    }


def _assignments() -> list[dict[str, object]]:
    return [
        {
            "dataset_id": "CWRU",
            "metadata_row_id": "row-001",
            "source_record_path": "raw/source/record-001.mat",
            "record_id": "record-001",
            "physical_unit_id": "unit-001",
            "domain_id": "source",
            "split": "train",
            "target": 0,
            "temporal_index": None,
        },
        {
            "dataset_id": "CWRU",
            "metadata_row_id": "row-002",
            "source_record_path": "raw/source/record-002.mat",
            "record_id": "record-002",
            "physical_unit_id": "unit-002",
            "domain_id": "source",
            "split": "validation",
            "target": 0,
            "temporal_index": None,
        },
        {
            "dataset_id": "CWRU",
            "metadata_row_id": "row-003",
            "source_record_path": "raw/target/record-003.mat",
            "record_id": "record-003",
            "physical_unit_id": "unit-003",
            "domain_id": "target",
            "split": "test",
            "target": 0,
            "temporal_index": 3,
        },
    ]


def _build(
    *,
    provenance: dict[str, str] | None = None,
    split: dict[str, object] | None = None,
    identity_versions: dict[str, str] | None = None,
    assignments: list[dict[str, object]] | None = None,
    exclusions: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    return build_split_manifest(
        provenance=provenance if provenance is not None else _provenance(),
        split=split if split is not None else _split(),
        identity_derivation_versions=(
            identity_versions if identity_versions is not None else _identity_versions()
        ),
        assignments=assignments if assignments is not None else _assignments(),
        exclusions=(
            exclusions
            if exclusions is not None
            else [{"metadata_row_id": "row-004", "reason": "missing label"}]
        ),
    )


def test_split_manifest_round_trip_and_canonical_hash() -> None:
    manifest = _build()
    payload = dumps_split_manifest(manifest)
    declared_hash = manifest["manifest_sha256"]
    reordered = _build(assignments=list(reversed(_assignments())))

    assert loads_split_manifest(payload) == manifest
    assert loads_split_manifest(payload.encode("utf-8")) == manifest
    assert manifest["schema"] == SPLIT_MANIFEST_SCHEMA
    assert declared_hash == compute_split_manifest_sha256(manifest)
    assert reordered == manifest
    manifest["manifest_sha256"] = "0" * 64
    assert compute_split_manifest_sha256(manifest) == declared_hash
    assert '": ' not in payload
    assert ", " not in payload
    assert "\n" not in payload


def test_histograms_membership_and_audits_are_derived() -> None:
    manifest = _build()

    assert manifest["histograms"]["label"]["train"] == [
        {"target": 0, "count": 1}
    ]
    assert manifest["histograms"]["domain"]["test"] == [
        {"domain_id": "target", "count": 1}
    ]
    assert manifest["histograms"]["unit"]["validation"] == [
        {"physical_unit_id": "unit-002", "record_count": 1}
    ]
    assert manifest["split_membership"]["test"]["record_ids"] == ["record-003"]
    assert manifest["split_membership"]["train"]["physical_unit_ids"] == [
        "unit-001"
    ]
    assert manifest["audits"]["pairwise_overlap"]["passed"] is True
    assert manifest["audits"]["label_support"] == {
        "passed": True,
        "train_targets": [0],
        "validation_missing_from_train": [],
        "test_missing_from_train": [],
    }


@pytest.mark.parametrize(
    ("container", "missing_key"),
    [
        ("provenance", "dataset_id"),
        ("provenance", "dataset_release_id"),
        ("provenance", "dataset_release_sha256"),
        ("provenance", "metadata_path"),
        ("provenance", "metadata_sha256"),
        ("provenance", "task_transform_id"),
        ("provenance", "task_transform_sha256"),
        ("split", "seed"),
        ("split", "algorithm"),
        ("split", "source_domains"),
        ("split", "target_domains"),
        ("identity", "metadata_row_id"),
        ("identity", "source_record_path"),
        ("identity", "record_id"),
        ("identity", "physical_unit_id"),
        ("assignment", "dataset_id"),
        ("assignment", "metadata_row_id"),
        ("assignment", "source_record_path"),
        ("assignment", "record_id"),
        ("assignment", "physical_unit_id"),
        ("assignment", "domain_id"),
        ("assignment", "split"),
        ("assignment", "target"),
        ("assignment", "temporal_index"),
        ("exclusion", "metadata_row_id"),
        ("exclusion", "reason"),
    ],
)
def test_missing_required_keys_fail_closed(container: str, missing_key: str) -> None:
    provenance = _provenance()
    split = _split()
    identity_versions = _identity_versions()
    assignments = _assignments()
    exclusions = [{"metadata_row_id": "row-004", "reason": "missing label"}]
    if container == "provenance":
        del provenance[missing_key]
    elif container == "split":
        del split[missing_key]
    elif container == "identity":
        del identity_versions[missing_key]
    elif container == "assignment":
        del assignments[0][missing_key]
    else:
        del exclusions[0][missing_key]

    with pytest.raises(SplitManifestError, match="missing required keys"):
        _build(
            provenance=provenance,
            split=split,
            identity_versions=identity_versions,
            assignments=assignments,
            exclusions=exclusions,
        )


@pytest.mark.parametrize(
    ("duplicate_field", "message"),
    [
        ("metadata_row_id", "duplicate assignment metadata_row_id"),
        ("record_id", "duplicate assignment record_id"),
    ],
)
def test_duplicate_assignment_identity_is_rejected(
    duplicate_field: str, message: str
) -> None:
    assignments = _assignments()
    assignments[1][duplicate_field] = assignments[0][duplicate_field]

    with pytest.raises(SplitManifestError, match=message):
        _build(assignments=assignments)


def test_physical_unit_cannot_cross_splits() -> None:
    assignments = _assignments()
    assignments[1]["physical_unit_id"] = assignments[0]["physical_unit_id"]

    with pytest.raises(SplitManifestError, match="appears in both"):
        _build(assignments=assignments)


def test_physical_unit_may_have_multiple_records_within_one_split() -> None:
    assignments = _assignments()
    assignments.append(
        {
            **assignments[0],
            "metadata_row_id": "row-005",
            "source_record_path": "raw/source/record-005.mat",
            "record_id": "record-005",
        }
    )

    manifest = _build(assignments=assignments)

    assert manifest["histograms"]["unit"]["train"] == [
        {"physical_unit_id": "unit-001", "record_count": 2}
    ]
    assert validate_split_manifest(manifest) == manifest


@pytest.mark.parametrize("split_name", ["val", "holdout", "TRAIN"])
def test_unknown_split_is_rejected(split_name: str) -> None:
    assignments = _assignments()
    assignments[0]["split"] = split_name

    with pytest.raises(SplitManifestError, match="is not one of"):
        _build(assignments=assignments)


def test_empty_split_is_rejected() -> None:
    assignments = [
        assignment
        for assignment in _assignments()
        if assignment["split"] != "validation"
    ]

    with pytest.raises(SplitManifestError, match="required splits empty"):
        _build(assignments=assignments)


def test_label_support_failure_is_rejected() -> None:
    assignments = _assignments()
    assignments[2]["target"] = 9

    with pytest.raises(SplitManifestError, match="not represented in train"):
        _build(assignments=assignments)


def test_undeclared_domain_and_non_normalized_path_are_rejected() -> None:
    assignments = _assignments()
    assignments[0]["domain_id"] = "unknown"
    with pytest.raises(SplitManifestError, match="not a declared"):
        _build(assignments=assignments)

    assignments = _assignments()
    assignments[0]["source_record_path"] = "raw/../record.mat"
    with pytest.raises(SplitManifestError, match="normalized relative POSIX path"):
        _build(assignments=assignments)


def test_assigned_row_cannot_also_be_excluded() -> None:
    with pytest.raises(SplitManifestError, match="both assigned and excluded"):
        _build(exclusions=[{"metadata_row_id": "row-001", "reason": "bad record"}])


def test_derived_count_tamper_is_rejected() -> None:
    tampered = _build()
    tampered["histograms"]["label"]["train"][0]["count"] = 99

    with pytest.raises(SplitManifestError, match="derived from assignments"):
        validate_split_manifest(tampered)


def test_content_tamper_is_rejected() -> None:
    manifest = _build()
    tampered = deepcopy(manifest)
    tampered["assignments"][0]["record_id"] = "record-tampered"

    with pytest.raises(SplitManifestError, match="does not match"):
        validate_split_manifest(tampered)


def test_declared_hash_tamper_is_rejected() -> None:
    tampered = _build()
    tampered["manifest_sha256"] = "0" * 64

    with pytest.raises(SplitManifestError, match="does not match"):
        validate_split_manifest(tampered)


def test_duplicate_json_object_key_is_rejected() -> None:
    payload = dumps_split_manifest(_build())
    duplicated = payload.replace(
        '"schema":"p02.split-manifest.v1"',
        '"schema":"p02.split-manifest.v1","schema":"p02.split-manifest.v1"',
    )

    with pytest.raises(SplitManifestError, match="duplicate JSON object key"):
        loads_split_manifest(duplicated)

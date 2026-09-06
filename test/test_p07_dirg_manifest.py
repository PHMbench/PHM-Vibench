from __future__ import annotations

import csv
import hashlib
import inspect
import io
import json
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.data_factory.reader.RM_020_DIRG import read as read_dirg
from src.utils.p07_protocol import dirg_manifest as manifest_module
from src.utils.p07_protocol import dirg_preprocessing as preprocessing
from src.utils.p07_protocol.cwru_manifest import deterministic_window_coordinates
from src.utils.p07_protocol.dirg_manifest import (
    ACCESS_RIGHT,
    CLASS_BY_CONDITION,
    CONDITION_IDS,
    DATASET_DOI,
    DATASET_NAME,
    DIRGManifestError,
    DOMAIN_IDS,
    EXPECTED_CHANNELS,
    EXPECTED_FILE_COUNT,
    EXPECTED_SAMPLE_LENGTH,
    EXPECTED_SAMPLE_RATE_HZ,
    FILES_PER_SPLIT,
    LICENSE_ID,
    OFFICIAL_RECORD_URL,
    RELATED_ARTICLE_DOI,
    ROTATIONS,
    WINDOWS_PER_SPLIT,
    build_dirg_manifest,
    strict_canonical_json_loads,
    validate_dirg_manifest,
    validate_serialized_dirg_manifest,
    verify_dirg_source_bindings,
)


REAL_ROOT = Path("/home/user/data/PHMbenchdata/PHM-Vibench")
REAL_METADATA = REAL_ROOT / "metadata.xlsx"
REAL_RAW = REAL_ROOT / "raw" / DATASET_NAME
REAL_READER = Path("src/data_factory/reader/RM_020_DIRG.py")
REAL_PREPROCESSING = Path("src/utils/p07_protocol/dirg_preprocessing.py")

FIELDS = (
    "Id",
    "Dataset_id",
    "Name",
    "TYPE",
    "File",
    "Label",
    "Label_Description",
    "Fault_level",
    "Domain_id",
    "Domain_description",
    "Sample_rate",
    "Sample_lenth",
    "Channel",
)


def _metadata_row(
    *,
    metadata_id: int,
    file_name: str,
    label: int,
    label_description: str,
    severity: int,
    domain_id: int,
    sample_rate: int = EXPECTED_SAMPLE_RATE_HZ,
    sample_length: int = EXPECTED_SAMPLE_LENGTH,
    dataset_id: int = 916,
) -> dict[str, object]:
    return {
        "Id": metadata_id,
        "Dataset_id": dataset_id,
        "Name": DATASET_NAME,
        "TYPE": "Vibration",
        "File": file_name,
        "Label": label,
        "Label_Description": label_description,
        "Fault_level": severity,
        "Domain_id": domain_id,
        "Domain_description": f"operating condition {domain_id}",
        "Sample_rate": sample_rate,
        "Sample_lenth": sample_length,
        "Channel": EXPECTED_CHANNELS,
    }


def _synthetic_rows() -> list[dict[str, object]]:
    rows = []
    metadata_id = 1_000
    for domain_id in range(1, 18):
        rows.append(
            _metadata_row(
                metadata_id=metadata_id,
                file_name=f"C0A_D{domain_id:02d}.mat",
                label=0,
                label_description="Healthy bearing (0A)",
                severity=0,
                domain_id=domain_id,
            )
        )
        metadata_id += 1
    for condition_id in CONDITION_IDS:
        domains = DOMAIN_IDS if condition_id == "C3" else tuple(range(1, 18))
        _, class_name, observed_label = CLASS_BY_CONDITION[condition_id]
        description = (
            "Inner ring defect, synthetic indentation"
            if class_name == "inner_ring"
            else "Roller defect, synthetic indentation"
        )
        severity = manifest_module.SEVERITY_BY_CONDITION[condition_id]
        for domain_id in domains:
            rows.append(
                _metadata_row(
                    metadata_id=metadata_id,
                    file_name=f"{condition_id}A_D{domain_id:02d}.mat",
                    label=observed_label,
                    label_description=description,
                    severity=severity,
                    domain_id=domain_id,
                )
            )
            metadata_id += 1
    for index in range(65):
        rows.append(
            _metadata_row(
                metadata_id=metadata_id,
                file_name=f"E4A{index:03d}.mat",
                label=2,
                label_description="Roller defect endurance evolution",
                severity=3,
                domain_id=12,
                sample_rate=102_400,
                sample_length=819_600,
            )
        )
        metadata_id += 1
    assert len(rows) == 180
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _fixture(root: Path, rows: list[dict[str, object]] | None = None):
    rows = _synthetic_rows() if rows is None else rows
    raw_dir = root / "raw" / DATASET_NAME
    raw_dir.mkdir(parents=True)
    for row in rows:
        file_name = str(row["File"])
        (raw_dir / file_name).write_bytes(
            b"synthetic-dirg-mat\x00" + file_name.encode("ascii")
        )
    (raw_dir / "readme.md").write_text("source pointer", encoding="utf-8")
    (raw_dir / "FileNames.mat").write_bytes(b"inventory")
    (raw_dir / "FileNamesEndurance.mat").write_bytes(b"endurance inventory")
    (raw_dir / "preview.png").write_bytes(b"png")
    metadata_path = root / "metadata.csv"
    _write_csv(metadata_path, rows)
    reader_path = root / "RM_020_DIRG.py"
    reader_path.write_text(
        '"""Incorrect RM_017_Ottawa19 docstring."""\ndef read(path): return path\n',
        encoding="utf-8",
    )
    preprocessing_path = root / "dirg_preprocessing.py"
    preprocessing_path.write_text(
        "WINDOW_ALGORITHM_ID = 'p07-evenly-distributed-nonoverlap-v1'\n",
        encoding="utf-8",
    )
    return metadata_path, raw_dir, reader_path, preprocessing_path


def _build(paths):
    metadata_path, raw_dir, reader_path, preprocessing_path = paths
    return build_dirg_manifest(
        metadata_path=metadata_path,
        raw_dir=raw_dir,
        reader_source_path=reader_path,
        preprocessing_source_path=preprocessing_path,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_manifest_selects_verified_balanced_grid_and_binds_official_source(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    manifest = _build(paths)
    payload = manifest.payload()

    assert len(manifest.specimens) == EXPECTED_FILE_COUNT
    assert manifest.observed_dataset_ids == (916,)
    assert {
        (item.condition_id, item.domain_id) for item in manifest.specimens
    } == {
        (condition_id, domain_id)
        for condition_id in CONDITION_IDS
        for domain_id in DOMAIN_IDS
    }
    assert {item.class_name for item in manifest.specimens} == {
        "inner_ring",
        "roller",
    }
    assert all(item.class_name != "outer" for item in manifest.specimens)
    assert sum(item.class_name == "inner_ring" for item in manifest.specimens) == 39
    assert sum(item.class_name == "roller" for item in manifest.specimens) == 39
    assert all(item.file_weight == 1.0 for item in manifest.specimens)
    assert payload["official_source"] == {
        "record_id": 3_559_553,
        "official_record_url": OFFICIAL_RECORD_URL,
        "official_api_url": "https://zenodo.org/api/records/3559553",
        "access_right": ACCESS_RIGHT,
        "license_id": LICENSE_ID,
        "dataset_doi": DATASET_DOI,
        "related_article_doi": RELATED_ARTICLE_DOI,
        "sensor_description": "two_triaxial_accelerometers",
        "expected_sensor_channel_count": EXPECTED_CHANNELS,
        "metadata_verification_basis": "official_zenodo_api_record",
        "metadata_verified_on": "2026-08-01",
    }
    contract = payload["dataset_contract"]
    assert contract["authoritative_local_name"] == DATASET_NAME
    assert contract["dataset_selection_key"] == "Name"
    assert contract["dataset_id_semantics"] == (
        "observed_only_not_selection_key_not_assumed_stable"
    )
    assert contract["physical_bearing_identity"] == "unauthenticated"
    assert contract["independent_replicate_unit"] == "unauthenticated"
    assert contract["file_observation_independence_claimed"] is False
    assert contract["cross_split_physical_bearing_disjointness"] == "unauditable"
    assert contract["domain_id_semantics"] == (
        "matched_operating_condition_only_not_independent_bearing"
    )
    assert contract["requested_outer_nomenclature_status"] == (
        "rejected_metadata_authenticates_roller"
    )
    assert payload["p0_blockers"] == [
        {
            "code": "physical_bearing_identity_unauthenticated",
            "blocks": [
                "independent_bearing_replication_claim",
                "cross_split_physical_bearing_disjointness_claim",
                "independent_bearing_generalization_inference",
            ],
            "does_not_block": [
                "read_only_manifest_validation",
                "file_level_descriptive_benchmark",
            ],
        }
    ]
    assert payload["claim_boundary"]["evidence_eligible"] is False
    assert manifest.reader_source_caveats == (
        "docstring_misidentifies_dataset_as_RM_017_Ottawa19",
    )
    assert manifest.raw_unmapped_entries == (
        "FileNames.mat",
        "FileNamesEndurance.mat",
        "preview.png",
        "readme.md",
    )
    for item in manifest.specimens:
        assert item.raw_sha256 == _sha256(paths[1] / item.file_name)


def test_window_grid_is_exact_cwru_equivalent_full_span_and_source_bound(
    tmp_path: Path,
) -> None:
    coordinates = preprocessing.uniform_window_coordinates()
    cwru_equivalent = deterministic_window_coordinates(EXPECTED_SAMPLE_LENGTH)

    assert preprocessing.WINDOW_ALGORITHM_ID == (
        "p07-evenly-distributed-nonoverlap-v1"
    )
    assert len(coordinates) == preprocessing.WINDOW_COUNT == 24
    assert [(item.start, item.stop) for item in coordinates] == [
        (item.start, item.stop) for item in cwru_equivalent
    ]
    assert coordinates[0].start == 0
    assert coordinates[-1].stop == EXPECTED_SAMPLE_LENGTH
    assert all(item.length == preprocessing.WINDOW_LENGTH for item in coordinates)
    assert all(
        left.stop <= right.start
        for left, right in zip(coordinates, coordinates[1:])
    )
    assert coordinates[-1].start > (
        preprocessing.WINDOW_COUNT * preprocessing.WINDOW_LENGTH
    )

    manifest = _build(_fixture(tmp_path))
    protocol = manifest.payload()["window_protocol"]
    assert protocol["window_algorithm_id"] == preprocessing.WINDOW_ALGORITHM_ID
    assert protocol["coordinate_set_sha256"] == (
        preprocessing.coordinate_set_sha256(coordinates)
    )
    assert protocol["coordinates"] == [item.to_payload() for item in coordinates]
    assert protocol["full_record_span_bound"] is True
    assert protocol["source_sha256"] == manifest.preprocessing_source_sha256
    assert all(item.windows == coordinates for item in manifest.specimens)


def test_three_rotations_are_condition_disjoint_and_balance_26_files_per_split(
    tmp_path: Path,
) -> None:
    manifest = _build(_fixture(tmp_path))
    by_key = {item.specimen_key: item for item in manifest.specimens}

    assert tuple(
        (fold.train_severity, fold.validation_severity, fold.test_severity)
        for fold in manifest.folds
    ) == ROTATIONS
    for fold in manifest.folds:
        condition_sets = tuple(
            set(values)
            for values in (
                fold.train_condition_ids,
                fold.validation_condition_ids,
                fold.test_condition_ids,
            )
        )
        assert not condition_sets[0] & condition_sets[1]
        assert not condition_sets[0] & condition_sets[2]
        assert not condition_sets[1] & condition_sets[2]
        key_sets = tuple(
            set(values)
            for values in (
                fold.train_specimen_keys,
                fold.validation_specimen_keys,
                fold.test_specimen_keys,
            )
        )
        assert tuple(map(len, key_sets)) == (
            FILES_PER_SPLIT,
            FILES_PER_SPLIT,
            FILES_PER_SPLIT,
        )
        assert not key_sets[0] & key_sets[1]
        assert not key_sets[0] & key_sets[2]
        assert not key_sets[1] & key_sets[2]
        assert set.union(*key_sets) == set(by_key)
        for keys, severity, conditions in (
            (key_sets[0], fold.train_severity, condition_sets[0]),
            (key_sets[1], fold.validation_severity, condition_sets[1]),
            (key_sets[2], fold.test_severity, condition_sets[2]),
        ):
            assert {by_key[key].severity for key in keys} == {severity}
            assert {by_key[key].condition_id for key in keys} == conditions
            assert {by_key[key].domain_id for key in keys} == set(DOMAIN_IDS)
            assert {by_key[key].class_name for key in keys} == {
                "inner_ring",
                "roller",
            }
        fold_payload = fold.to_dict()
        assert fold_payload["files_per_split"] == FILES_PER_SPLIT
        assert fold_payload["windows_per_split"] == WINDOWS_PER_SPLIT
    for role in (
        "train_specimen_keys",
        "validation_specimen_keys",
        "test_specimen_keys",
    ):
        assigned = [key for fold in manifest.folds for key in getattr(fold, role)]
        assert len(assigned) == EXPECTED_FILE_COUNT
        assert set(assigned) == set(by_key)


def test_preprocessing_is_stateless_population_standardized_and_fail_closed() -> None:
    generator = np.random.default_rng(20260801)
    recording = generator.normal(
        size=(EXPECTED_SAMPLE_LENGTH, EXPECTED_CHANNELS)
    ).astype(np.float64)
    input_sha256 = hashlib.sha256(recording.tobytes()).hexdigest()

    windows = preprocessing.materialize_dirg_windows(recording)

    assert windows.shape == (
        preprocessing.WINDOW_COUNT,
        preprocessing.WINDOW_LENGTH,
        EXPECTED_CHANNELS,
    )
    assert windows.dtype == np.float64
    assert np.isfinite(windows).all()
    assert np.allclose(windows.mean(axis=1), 0.0, atol=3e-15, rtol=0)
    assert np.allclose(windows.std(axis=1, ddof=0), 1.0, atol=3e-15, rtol=0)
    assert hashlib.sha256(recording.tobytes()).hexdigest() == input_sha256
    preprocessing.validate_materialized_windows(windows)

    affine = preprocessing.materialize_dirg_windows(recording * 2.0 + 7.0)
    assert np.allclose(affine, windows, atol=2e-14, rtol=0)
    with pytest.raises(ValueError, match="exact shape"):
        preprocessing.materialize_dirg_windows(recording[:-1])
    nonfinite = recording.copy()
    nonfinite[0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        preprocessing.materialize_dirg_windows(nonfinite)
    with pytest.raises(ValueError, match="zero-variance"):
        preprocessing.materialize_dirg_windows(
            np.zeros((EXPECTED_SAMPLE_LENGTH, EXPECTED_CHANNELS), dtype=np.float64)
        )
    bad_coordinates = list(preprocessing.uniform_window_coordinates())
    bad_coordinates[1] = replace(bad_coordinates[1], start=1, stop=4097)
    with pytest.raises(ValueError, match="overlap|uniform"):
        preprocessing.materialize_dirg_windows(
            recording, coordinates=bad_coordinates
        )


def test_manifest_canonical_self_hash_duplicate_and_fold_drift_fail_closed(
    tmp_path: Path,
) -> None:
    manifest = _build(_fixture(tmp_path))
    serialized = manifest.canonical_json()
    parsed = validate_serialized_dirg_manifest(
        serialized, expected_manifest=manifest
    )

    assert parsed["root_sha256"] == manifest.root_sha256
    assert strict_canonical_json_loads(serialized) == json.loads(serialized)
    payload = manifest.payload()
    expected_root = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert manifest.root_sha256 == expected_root
    with pytest.raises(DIRGManifestError, match="canonical"):
        strict_canonical_json_loads(serialized + "\n")
    duplicate_root = serialized[:-1] + ',"root_sha256":"' + manifest.root_sha256 + '"}'
    with pytest.raises(DIRGManifestError, match="Duplicate"):
        strict_canonical_json_loads(duplicate_root)
    tampered = json.loads(serialized)
    tampered["dataset_contract"]["channels"] = 3
    tampered_serialized = json.dumps(
        tampered,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    with pytest.raises(DIRGManifestError, match="self-hash"):
        validate_serialized_dirg_manifest(tampered_serialized)
    with pytest.raises(DIRGManifestError, match="root SHA-256"):
        validate_dirg_manifest(replace(manifest, root_sha256="0" * 64))

    duplicate_specimens = (
        manifest.specimens[0],
        manifest.specimens[0],
        *manifest.specimens[2:],
    )
    with pytest.raises(DIRGManifestError, match="duplicate"):
        validate_dirg_manifest(
            replace(manifest, specimens=duplicate_specimens)
        )
    first_fold = manifest.folds[0]
    leaked_fold = replace(
        first_fold,
        validation_specimen_keys=first_fold.train_specimen_keys,
    )
    with pytest.raises(DIRGManifestError, match="fold layout|condition isolation"):
        validate_dirg_manifest(
            replace(manifest, folds=(leaked_fold, *manifest.folds[1:]))
        )


def test_builder_rejects_duplicate_metadata_wrong_roller_label_and_duplicate_bytes(
    tmp_path: Path,
) -> None:
    duplicate_rows = _synthetic_rows()
    duplicate_rows[-1] = dict(duplicate_rows[0])
    with pytest.raises(DIRGManifestError, match="duplicate"):
        _build(_fixture(tmp_path / "duplicate-row", duplicate_rows))

    wrong_label_rows = _synthetic_rows()
    target = next(
        row for row in wrong_label_rows if str(row["File"]).startswith("C4A_")
    )
    target["Label_Description"] = "Outer ring defect"
    with pytest.raises(DIRGManifestError, match="does not authenticate class 'roller'"):
        _build(_fixture(tmp_path / "outer-label", wrong_label_rows))

    paths = _fixture(tmp_path / "duplicate-bytes")
    first = paths[1] / "C1A_D01.mat"
    second = paths[1] / "C2A_D01.mat"
    second.write_bytes(first.read_bytes())
    with pytest.raises(DIRGManifestError, match="duplicate raw bytes"):
        _build(paths)


def test_dataset_id_is_observed_not_filtered_and_source_drift_is_detected(
    tmp_path: Path,
) -> None:
    rows = _synthetic_rows()
    target = next(row for row in rows if row["File"] == "C1A_D01.mat")
    target["Dataset_id"] = 999
    paths = _fixture(tmp_path, rows)
    manifest = _build(paths)

    assert manifest.observed_dataset_ids == (916, 999)
    verify_dirg_source_bindings(
        manifest,
        metadata_path=paths[0],
        raw_dir=paths[1],
        reader_source_path=paths[2],
        preprocessing_source_path=paths[3],
    )
    paths[2].write_text("changed reader", encoding="utf-8")
    with pytest.raises(DIRGManifestError, match="reader_source_sha256"):
        verify_dirg_source_bindings(
            manifest,
            metadata_path=paths[0],
            raw_dir=paths[1],
            reader_source_path=paths[2],
            preprocessing_source_path=paths[3],
        )


def test_builder_source_is_read_only() -> None:
    source = inspect.getsource(manifest_module.build_dirg_manifest)
    assert ".write" not in source
    assert ".unlink" not in source
    assert ".replace" not in source


def test_real_78_files_reader_and_window_materialization_are_exact_and_read_only() -> None:
    assert REAL_METADATA.is_file()
    assert REAL_RAW.is_dir()
    before_metadata = (REAL_METADATA.stat().st_size, REAL_METADATA.stat().st_mtime_ns)
    before_raw_tree = {
        path.name: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in REAL_RAW.iterdir()
        if path.is_file()
    }

    manifest = build_dirg_manifest(
        metadata_path=REAL_METADATA,
        raw_dir=REAL_RAW,
        reader_source_path=REAL_READER,
        preprocessing_source_path=REAL_PREPROCESSING,
    )
    aggregate = hashlib.sha256()
    for item in manifest.specimens:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            recording = read_dirg(str(REAL_RAW / item.file_name))
        assert isinstance(recording, np.ndarray)
        assert recording.shape == (EXPECTED_SAMPLE_LENGTH, EXPECTED_CHANNELS)
        assert recording.dtype == np.float64
        assert np.isfinite(recording).all()
        windows = preprocessing.materialize_dirg_windows(
            recording, coordinates=item.windows
        )
        preprocessing.validate_materialized_windows(windows)
        aggregate.update(item.specimen_key.encode("ascii"))
        aggregate.update(item.raw_sha256.encode("ascii"))
        aggregate.update(windows.tobytes(order="C"))

    after_raw_tree = {
        path.name: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in REAL_RAW.iterdir()
        if path.is_file()
    }
    assert len(manifest.specimens) == EXPECTED_FILE_COUNT
    assert before_raw_tree == after_raw_tree
    assert before_metadata == (
        REAL_METADATA.stat().st_size,
        REAL_METADATA.stat().st_mtime_ns,
    )
    assert manifest.observed_dataset_ids == (16,)
    expected_reader_caveats = (
        ("docstring_misidentifies_dataset_as_RM_017_Ottawa19",)
        if b"RM_017_Ottawa19" in REAL_READER.read_bytes()
        else ()
    )
    assert manifest.reader_source_caveats == expected_reader_caveats
    assert preprocessing.preprocessing_source_sha256() == (
        manifest.preprocessing_source_sha256
    )
    print(f"DIRG_MANIFEST_ROOT_SHA256={manifest.root_sha256}")
    print(f"DIRG_READER_SOURCE_SHA256={manifest.reader_source_sha256}")
    print(
        "DIRG_PREPROCESSING_SOURCE_SHA256="
        f"{manifest.preprocessing_source_sha256}"
    )
    print(f"DIRG_MATERIALIZATION_SHA256={aggregate.hexdigest()}")

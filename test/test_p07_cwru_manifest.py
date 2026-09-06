from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, MutableMapping, Tuple

import pytest

from src.utils.p07_protocol.cwru_manifest import (
    CWRUManifestError,
    OFFICIAL_12K_DRIVE_END_SPECIMENS,
    OFFICIAL_SOURCE_URL,
    WINDOW_COUNT,
    WINDOW_SIZE,
    OfficialSpecimen,
    build_cwru_manifest,
    deterministic_window_coordinates,
)


FIELDS = (
    "Id",
    "Dataset_id",
    "Name",
    "File",
    "Label",
    "Fault_level",
    "Domain_id",
    "Load_hp",
    "Sample_rate",
    "Sample_lenth",
    "Channel",
)


def _row(specimen: OfficialSpecimen, metadata_id: int) -> Dict[str, object]:
    return {
        "Id": metadata_id,
        "Dataset_id": 1,
        "Name": "RM_001_CWRU",
        "File": specimen.file_name,
        "Label": specimen.label,
        "Fault_level": specimen.fault_level,
        "Domain_id": specimen.domain_id,
        "Load_hp": specimen.load_hp,
        "Sample_rate": 12000,
        "Sample_lenth": WINDOW_COUNT * WINDOW_SIZE + metadata_id * 31,
        "Channel": 2,
    }


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _fixture(
    root: Path,
    *,
    reverse_rows: bool = False,
    mutate_row: Callable[[MutableMapping[str, object], OfficialSpecimen], None]
    | None = None,
    missing_raw: str | None = None,
) -> Tuple[Path, Path, Path, Path]:
    raw_dir = root / "raw" / "RM_001_CWRU"
    raw_dir.mkdir(parents=True)
    rows = []
    for metadata_id, specimen in enumerate(OFFICIAL_12K_DRIVE_END_SPECIMENS, start=1001):
        row = _row(specimen, metadata_id)
        if mutate_row is not None:
            mutate_row(row, specimen)
        rows.append(row)
        if specimen.file_name != missing_raw:
            (raw_dir / specimen.file_name).write_bytes(
                b"fake-cwru-mat\x00" + specimen.file_name.encode("ascii")
            )
    if reverse_rows:
        rows.reverse()
    metadata_path = root / "metadata.csv"
    _write_csv(metadata_path, rows)
    reader_path = root / "reader.py"
    preprocessing_path = root / "preprocessing.py"
    reader_path.write_bytes(b"def read(path): return path\n")
    preprocessing_path.write_bytes(b"WINDOW_SIZE = 4096\n")
    return metadata_path, raw_dir, reader_path, preprocessing_path


def _build(paths: Tuple[Path, Path, Path, Path]):
    metadata_path, raw_dir, reader_path, preprocessing_path = paths
    return build_cwru_manifest(
        metadata_path=metadata_path,
        raw_dir=raw_dir,
        reader_source_path=reader_path,
        preprocessing_source_path=preprocessing_path,
    )


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_official_registry_is_the_fixed_36_file_factorial() -> None:
    registry = OFFICIAL_12K_DRIVE_END_SPECIMENS

    assert len(registry) == 36
    assert len({item.file_name for item in registry}) == 36
    assert len({item.specimen_key for item in registry}) == 36
    assert {
        (item.fault_type, item.diameter_code, item.load_hp) for item in registry
    } == {
        (fault, diameter, load)
        for fault in ("IR", "B", "OR@6")
        for diameter in ("007", "014", "021")
        for load in range(4)
    }


def test_builder_is_read_only_and_binds_sources_raw_files_folds_and_windows(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    _, raw_dir, reader_path, preprocessing_path = paths
    before = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in raw_dir.iterdir()
    }

    manifest = _build(paths)

    after = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in raw_dir.iterdir()
    }
    assert after == before
    assert manifest.official_source_url == OFFICIAL_SOURCE_URL
    assert manifest.reader_source_sha256 == _sha256(reader_path.read_bytes())
    assert manifest.preprocessing_source_sha256 == _sha256(
        preprocessing_path.read_bytes()
    )
    assert len(manifest.specimens) == 36
    assert len(manifest.folds) == 3
    assert all(item.file_weight == 1.0 for item in manifest.specimens)
    assert all(len(item.windows) == WINDOW_COUNT for item in manifest.specimens)

    by_key = {item.specimen_key: item for item in manifest.specimens}
    for item in manifest.specimens:
        assert item.raw_sha256 == _sha256((raw_dir / item.file_name).read_bytes())
        assert item.windows[0].start == 0
        assert item.windows[-1].stop == item.sample_length
        assert all(window.stop - window.start == WINDOW_SIZE for window in item.windows)
        assert all(
            left.stop <= right.start
            for left, right in zip(item.windows, item.windows[1:])
        )

    for fold in manifest.folds:
        train = set(fold.train_specimen_keys)
        validation = set(fold.validation_specimen_keys)
        test = set(fold.test_specimen_keys)
        excluded = set(fold.excluded_specimen_keys)
        assert (len(train), len(validation), len(test), len(excluded)) == (12, 12, 12, 0)
        assert not train & validation
        assert not train & test
        assert not validation & test
        assert train | validation | test | excluded == set(by_key)
        expected_cells = {
            (fault_type, load_hp)
            for fault_type in ("IR", "B", "OR@6")
            for load_hp in range(4)
        }
        for keys, diameter_code in (
            (train, fold.train_diameter_code),
            (validation, fold.validation_diameter_code),
            (test, fold.test_diameter_code),
        ):
            assert {by_key[key].diameter_code for key in keys} == {diameter_code}
            assert {
                (by_key[key].fault_type, by_key[key].load_hp) for key in keys
            } == expected_cells
        assert fold.evaluation_unit == "file"
        assert fold.weighting == "equal_file"

    expected_rotations = {
        ("007", "014", "021"),
        ("014", "021", "007"),
        ("021", "007", "014"),
    }
    assert {
        (
            fold.train_diameter_code,
            fold.validation_diameter_code,
            fold.test_diameter_code,
        )
        for fold in manifest.folds
    } == expected_rotations
    for role in (
        "train_specimen_keys",
        "validation_specimen_keys",
        "test_specimen_keys",
    ):
        role_keys = [key for fold in manifest.folds for key in getattr(fold, role)]
        assert len(role_keys) == 36
        assert set(role_keys) == set(by_key)

    fold_protocol = manifest.payload()["fold_protocol"]
    assert fold_protocol == {
        "fold_count": 3,
        "rotation_axis": "diameter_code",
        "diameter_count_per_split": 1,
        "files_per_split": 12,
        "loads_hp_per_split": [0, 1, 2, 3],
        "fault_types_per_split": ["IR", "B", "OR@6"],
        "all_files_used_per_fold": True,
        "each_file_tested_once_across_folds": True,
        "evaluation_unit": "file",
        "weighting": "equal_file",
    }

    expected_root = _sha256(
        json.dumps(
            manifest.payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    )
    assert manifest.root_sha256 == expected_root
    assert json.loads(manifest.canonical_json())["root_sha256"] == expected_root


def test_manifest_is_portable_across_metadata_row_and_directory_order(tmp_path: Path) -> None:
    first = _build(_fixture(tmp_path / "first"))
    second = _build(_fixture(tmp_path / "second", reverse_rows=True))

    assert first.metadata_subset_sha256 == second.metadata_subset_sha256
    assert first.root_sha256 == second.root_sha256
    assert first.canonical_json() == second.canonical_json()


def test_builder_ignores_blank_unrelated_metadata_rows(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    metadata_path = paths[0]
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.append({field: "" for field in FIELDS})
    _write_csv(metadata_path, rows)

    manifest = _build(paths)

    assert len(manifest.specimens) == 36


def test_raw_content_change_changes_specimen_and_root_hash(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    first = _build(paths)
    target = paths[1] / OFFICIAL_12K_DRIVE_END_SPECIMENS[0].file_name
    target.write_bytes(target.read_bytes() + b"changed")

    second = _build(paths)

    assert first.specimens[0].raw_sha256 != second.specimens[0].raw_sha256
    assert first.root_sha256 != second.root_sha256


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("Dataset_id", 2, "Dataset_id must be 1"),
        ("Name", "OTHER", "Name must be"),
        ("Domain_id", 9, "Domain_id must be"),
        ("Load_hp", 9, "Load_hp must be"),
        ("Label", 9, "Label must be"),
        ("Fault_level", 9, "Fault_level must be"),
        ("Sample_rate", 48000, "Sample_rate must be 12000"),
        ("Channel", 1, "Channel must be 2"),
        ("Sample_lenth", float("nan"), "Sample_lenth must be a finite integer"),
    ],
)
def test_builder_fails_closed_on_invalid_selected_metadata(
    tmp_path: Path, field: str, bad_value: object, message: str
) -> None:
    target_file = OFFICIAL_12K_DRIVE_END_SPECIMENS[0].file_name

    def mutate(row: MutableMapping[str, object], specimen: OfficialSpecimen) -> None:
        if specimen.file_name == target_file:
            row[field] = bad_value

    with pytest.raises(CWRUManifestError, match=message):
        _build(_fixture(tmp_path, mutate_row=mutate))


def test_builder_rejects_missing_raw_file_and_duplicate_metadata(tmp_path: Path) -> None:
    missing = OFFICIAL_12K_DRIVE_END_SPECIMENS[-1].file_name
    with pytest.raises(CWRUManifestError, match="does not exist"):
        _build(_fixture(tmp_path / "missing", missing_raw=missing))

    paths = _fixture(tmp_path / "duplicate")
    metadata_path = paths[0]
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.append(dict(rows[0]))
    _write_csv(metadata_path, rows)
    with pytest.raises(CWRUManifestError, match="exactly one metadata row"):
        _build(paths)


def test_window_builder_rejects_short_or_non_integer_recordings() -> None:
    with pytest.raises(CWRUManifestError, match="cannot provide"):
        deterministic_window_coordinates(WINDOW_COUNT * WINDOW_SIZE - 1)
    with pytest.raises(CWRUManifestError, match="positive integer"):
        deterministic_window_coordinates(float(WINDOW_COUNT * WINDOW_SIZE))  # type: ignore[arg-type]
    with pytest.raises(CWRUManifestError, match="at least 2"):
        deterministic_window_coordinates(WINDOW_SIZE, window_count=1)

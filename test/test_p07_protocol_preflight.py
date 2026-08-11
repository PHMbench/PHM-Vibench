from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import pytest
import yaml

from scripts.p07_protocol_preflight import (
    DEFAULT_CONFIG_PATH,
    PreflightError,
    canonical_json,
    main,
    run_preflight,
)
from src.utils.p07_protocol.cwru_manifest import (
    OFFICIAL_12K_DRIVE_END_SPECIMENS,
    WINDOW_COUNT,
    WINDOW_SIZE,
    CWRUManifest,
    OfficialSpecimen,
    build_cwru_manifest,
)
from src.utils.p07_protocol.dirg_manifest import (
    ACCESS_RIGHT as DIRG_ACCESS_RIGHT,
    CLASS_BY_CONDITION as DIRG_CLASS_BY_CONDITION,
    CONDITION_IDS as DIRG_CONDITION_IDS,
    DATASET_DOI as DIRG_DATASET_DOI,
    DATASET_NAME as DIRG_DATASET_NAME,
    DOMAIN_IDS as DIRG_DOMAIN_IDS,
    EXPECTED_CHANNELS as DIRG_EXPECTED_CHANNELS,
    EXPECTED_SAMPLE_LENGTH as DIRG_EXPECTED_SAMPLE_LENGTH,
    EXPECTED_SAMPLE_RATE_HZ as DIRG_EXPECTED_SAMPLE_RATE_HZ,
    FILES_PER_SPLIT as DIRG_FILES_PER_SPLIT,
    LICENSE_ID as DIRG_LICENSE_ID,
    OFFICIAL_RECORD_ID as DIRG_OFFICIAL_RECORD_ID,
    OFFICIAL_RECORD_URL as DIRG_OFFICIAL_RECORD_URL,
    RELATED_ARTICLE_DOI as DIRG_RELATED_ARTICLE_DOI,
    SEVERITY_BY_CONDITION as DIRG_SEVERITY_BY_CONDITION,
    WINDOWS_PER_SPLIT as DIRG_WINDOWS_PER_SPLIT,
    DIRGManifest,
    build_dirg_manifest,
)
from src.utils.p07_protocol.path_universe import OPTIMIZATION_SEEDS


_CWRU_METADATA_FIELDS = (
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

_DIRG_METADATA_FIELDS = (
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

_REAL_MANIFEST_HASHES = {
    "path_universe_sha256": "60b4907005403eaad082b50c169170f5433eb3bc9ec33fea999d6450af9ad338",
    "composition_split_sha256": "ebe91496c8a50d39f9ae072337dec2dec1ae435b328315b8bd08980ed0f569ce",
    "seed_namespace_sha256": "0ecdb2747616732e8246298b51684b219dd277c9d15f517168c750e55ec765d1",
    "synthetic_generator_sha256": "26884d3cdff9437ff804988eb2212695736ded5eb80d178a90a1eabe49551b82",
    "nuisance_sha256": "9b2aa503f168594c5ba9588c510b1cd9d8c11dc7f094adbd05b9e14ddfb9044b",
}

_REAL_CWRU_HASHES = {
    "root_sha256": "cfae807edeea0124f936e277516ca70c642eb1c2ac60b081e998016d84705da5",
    "metadata_subset_sha256": "79471db84cb7a1eb390ad9bbfd4fa2cf0c4ac20ea46d5be1d04d452175379e8b",
    "reader_source_sha256": "11e48d87f4d85566c4a5851157f3e22211a2ca58c29758e8ddbba9e770c8047b",
    "preprocessing_source_sha256": "19d7a6c21d13b9afbaa40bf0b76f8ffca5c77bbaacdf603149ebce9ee8c4dcb2",
}

_REAL_DIRG_HASHES = {
    "root_sha256": "188165997156a5d066d93e9877c7fe4b0bf0a010e1935a4353d725e6681f0ae9",
    "metadata_file_sha256": "0b61d1f8b1f74811309a6bec9827b0c2f2956940e877a8070275e108ce2c9c30",
    "metadata_name_subset_sha256": "4b80b5ca29101858e770cdc6ecfa6c979df075385610367f18fb317d73cf9a85",
    "metadata_selected_subset_sha256": "2011e9d447125032f215242deda8d7a8883431ee8f759acf83b4ae0230ca030d",
    "raw_inventory_name_size_sha256": "4f2c5359a8fb1a555afef342537df19e2d24c109873d1cf5b9e669e8761969f0",
    "reader_source_sha256": "c1eae7f69608ab6aeca9898b25a3aab119a40f85986ea13efacd6a5747a09856",
    "preprocessing_source_sha256": "774d6b9323b2cc8e8e2e8643fdc0af90b8b5bf209989fc1562554fae7ce1c3a6",
}


def _metadata_row(specimen: OfficialSpecimen, metadata_id: int) -> dict[str, object]:
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


def _write_metadata(
    path: Path,
    rows: Iterable[Mapping[str, object]],
    *,
    fieldnames: tuple[str, ...],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _raw_fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    root.mkdir(parents=True)
    raw_dir = root / "raw" / "RM_001_CWRU"
    raw_dir.mkdir(parents=True)
    rows = []
    for metadata_id, specimen in enumerate(OFFICIAL_12K_DRIVE_END_SPECIMENS, start=1001):
        rows.append(_metadata_row(specimen, metadata_id))
        (raw_dir / specimen.file_name).write_bytes(
            b"fake-cwru-mat\x00" + specimen.file_name.encode("ascii")
        )
    metadata_path = root / "metadata.csv"
    _write_metadata(metadata_path, rows, fieldnames=_CWRU_METADATA_FIELDS)
    reader_source = root / "reader.py"
    preprocessing_source = root / "preprocessing.py"
    reader_source.write_text("def read(path): return path\n", encoding="utf-8")
    preprocessing_source.write_text("WINDOW_SIZE = 4096\n", encoding="utf-8")
    return metadata_path, raw_dir, reader_source, preprocessing_source


def _build_cwru(paths: tuple[Path, Path, Path, Path]) -> CWRUManifest:
    metadata, raw_dir, reader, preprocessing = paths
    return build_cwru_manifest(
        metadata_path=metadata,
        raw_dir=raw_dir,
        reader_source_path=reader,
        preprocessing_source_path=preprocessing,
    )


def _dirg_metadata_row(
    *,
    metadata_id: int,
    file_name: str,
    label: int,
    label_description: str,
    severity: int,
    domain_id: int,
    sample_rate: int = DIRG_EXPECTED_SAMPLE_RATE_HZ,
    sample_length: int = DIRG_EXPECTED_SAMPLE_LENGTH,
) -> dict[str, object]:
    return {
        "Id": metadata_id,
        "Dataset_id": 916,
        "Name": DIRG_DATASET_NAME,
        "TYPE": "Vibration",
        "File": file_name,
        "Label": label,
        "Label_Description": label_description,
        "Fault_level": severity,
        "Domain_id": domain_id,
        "Domain_description": f"operating condition {domain_id}",
        "Sample_rate": sample_rate,
        "Sample_lenth": sample_length,
        "Channel": DIRG_EXPECTED_CHANNELS,
    }


def _dirg_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    metadata_id = 1_000
    for domain_id in range(1, 18):
        rows.append(
            _dirg_metadata_row(
                metadata_id=metadata_id,
                file_name=f"C0A_D{domain_id:02d}.mat",
                label=0,
                label_description="Healthy bearing (0A)",
                severity=0,
                domain_id=domain_id,
            )
        )
        metadata_id += 1
    for condition_id in DIRG_CONDITION_IDS:
        domains = (
            DIRG_DOMAIN_IDS if condition_id == "C3" else tuple(range(1, 18))
        )
        _, class_name, observed_label = DIRG_CLASS_BY_CONDITION[condition_id]
        description = (
            "Inner ring defect, synthetic indentation"
            if class_name == "inner_ring"
            else "Roller defect, synthetic indentation"
        )
        for domain_id in domains:
            rows.append(
                _dirg_metadata_row(
                    metadata_id=metadata_id,
                    file_name=f"{condition_id}A_D{domain_id:02d}.mat",
                    label=observed_label,
                    label_description=description,
                    severity=DIRG_SEVERITY_BY_CONDITION[condition_id],
                    domain_id=domain_id,
                )
            )
            metadata_id += 1
    for index in range(65):
        rows.append(
            _dirg_metadata_row(
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


def _dirg_fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    root.mkdir(parents=True)
    raw_dir = root / "raw" / DIRG_DATASET_NAME
    raw_dir.mkdir(parents=True)
    rows = _dirg_rows()
    for row in rows:
        file_name = str(row["File"])
        (raw_dir / file_name).write_bytes(
            b"fake-dirg-mat\x00" + file_name.encode("ascii")
        )
    (raw_dir / "readme.md").write_text("source pointer", encoding="utf-8")
    (raw_dir / "FileNames.mat").write_bytes(b"inventory")
    (raw_dir / "FileNamesEndurance.mat").write_bytes(b"endurance inventory")
    (raw_dir / "preview.png").write_bytes(b"png")
    metadata_path = root / "metadata.csv"
    _write_metadata(metadata_path, rows, fieldnames=_DIRG_METADATA_FIELDS)
    reader_source = root / "RM_020_DIRG.py"
    preprocessing_source = root / "dirg_preprocessing.py"
    reader_source.write_text(
        '"""RM_020_DIRG reader."""\ndef read(path): return path\n',
        encoding="utf-8",
    )
    preprocessing_source.write_text(
        "WINDOW_ALGORITHM_ID = 'p07-evenly-distributed-nonoverlap-v1'\n",
        encoding="utf-8",
    )
    return metadata_path, raw_dir, reader_source, preprocessing_source


def _build_dirg(paths: tuple[Path, Path, Path, Path]) -> DIRGManifest:
    metadata, raw_dir, reader, preprocessing = paths
    return build_dirg_manifest(
        metadata_path=metadata,
        raw_dir=raw_dir,
        reader_source_path=reader,
        preprocessing_source_path=preprocessing,
    )


def _fixture_config(
    root: Path,
) -> tuple[
    Path,
    dict[str, tuple[Path, Path, Path, Path]],
    dict[str, CWRUManifest | DIRGManifest],
]:
    paths = {
        "cwru": _raw_fixture(root / "inputs" / "cwru"),
        "dirg": _dirg_fixture(root / "inputs" / "dirg"),
    }
    manifests: dict[str, CWRUManifest | DIRGManifest] = {
        "cwru": _build_cwru(paths["cwru"]),
        "dirg": _build_dirg(paths["dirg"]),
    }
    config = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    cwru_manifest = manifests["cwru"]
    assert isinstance(cwru_manifest, CWRUManifest)
    config["cwru"].update(
        {
            "root_sha256": cwru_manifest.root_sha256,
            "metadata_subset_sha256": cwru_manifest.metadata_subset_sha256,
            "reader_source_sha256": cwru_manifest.reader_source_sha256,
            "preprocessing_source_sha256": (
                cwru_manifest.preprocessing_source_sha256
            ),
        }
    )
    dirg_manifest = manifests["dirg"]
    assert isinstance(dirg_manifest, DIRGManifest)
    config["dirg"].update(
        {
            "root_sha256": dirg_manifest.root_sha256,
            "metadata_file_sha256": dirg_manifest.metadata_file_sha256,
            "metadata_name_subset_sha256": (
                dirg_manifest.metadata_name_subset_sha256
            ),
            "metadata_selected_subset_sha256": (
                dirg_manifest.metadata_selected_subset_sha256
            ),
            "raw_inventory_name_size_sha256": (
                dirg_manifest.raw_inventory_name_size_sha256
            ),
            "reader_source_sha256": dirg_manifest.reader_source_sha256,
            "preprocessing_source_sha256": (
                dirg_manifest.preprocessing_source_sha256
            ),
        }
    )
    config_path = root / "fixture_protocol.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    return config_path, paths, manifests


def _snapshot_files(root: Path) -> dict[str, tuple[bytes, int]]:
    return {
        path.relative_to(root).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }


def _call_kwargs(
    config_path: Path,
    paths: Mapping[str, tuple[Path, Path, Path, Path]],
) -> dict[str, Any]:
    cwru_metadata, cwru_raw, cwru_reader, cwru_preprocessing = paths["cwru"]
    dirg_metadata, dirg_raw, dirg_reader, dirg_preprocessing = paths["dirg"]
    return {
        "config_path": config_path,
        "protocol_sha256": "a" * 64,
        "cwru_metadata_path": cwru_metadata,
        "cwru_raw_dir": cwru_raw,
        "cwru_reader_source_path": cwru_reader,
        "cwru_preprocessing_source_path": cwru_preprocessing,
        "dirg_metadata_path": dirg_metadata,
        "dirg_raw_dir": dirg_raw,
        "dirg_reader_source_path": dirg_reader,
        "dirg_preprocessing_source_path": dirg_preprocessing,
    }


def test_production_config_freezes_real_bindings_and_false_approval() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    registry_path = DEFAULT_CONFIG_PATH.parents[3] / "configs" / "config_registry.csv"
    with registry_path.open("r", encoding="utf-8", newline="") as handle:
        registry_rows = [
            row
            for row in csv.DictReader(handle)
            if row["id"] == "p07_g040_protocol_preflight"
        ]

    assert len(registry_rows) == 1
    assert registry_rows[0]["category"] == "protocol"
    assert registry_rows[0]["status"] == "/"
    assert registry_rows[0]["path"] == DEFAULT_CONFIG_PATH.relative_to(
        DEFAULT_CONFIG_PATH.parents[3]
    ).as_posix()
    assert config["mode"] == "check_only"
    assert config["claim_evidence"] is False
    assert config["expected_evidence_state"] == "not_evidence"
    assert config["protocol_sha256"] == {"source": "cli_required", "value": None}
    assert config["approval"] == {
        "experiment_protocol_approved": False,
        "thresholds_approved": False,
        "evidence_execution_allowed": False,
    }
    assert config["manifests"] == _REAL_MANIFEST_HASHES
    assert {
        key: config["cwru"][key] for key in _REAL_CWRU_HASHES
    } == _REAL_CWRU_HASHES
    assert {
        key: config["dirg"][key] for key in _REAL_DIRG_HASHES
    } == _REAL_DIRG_HASHES
    assert config["runtime"]["input_paths"] == {
        "cwru_metadata_path": "cli_required",
        "cwru_raw_dir": "cli_required",
        "cwru_reader_source_path": "cli_required",
        "cwru_preprocessing_source_path": "cli_required",
        "dirg_metadata_path": "cli_required",
        "dirg_raw_dir": "cli_required",
        "dirg_reader_source_path": "cli_required",
        "dirg_preprocessing_source_path": "cli_required",
    }
    assert {
        key: config["dirg"][key]
        for key in (
            "official_record_id",
            "official_record_url",
            "dataset_doi",
            "related_article_doi",
            "access_right",
            "license_id",
            "selected_file_count",
            "fold_count",
            "files_per_split",
            "windows_per_split",
            "physical_bearing_identity",
            "independent_replicate_unit",
            "file_observation_independence_claimed",
        )
    } == {
        "official_record_id": DIRG_OFFICIAL_RECORD_ID,
        "official_record_url": DIRG_OFFICIAL_RECORD_URL,
        "dataset_doi": DIRG_DATASET_DOI,
        "related_article_doi": DIRG_RELATED_ARTICLE_DOI,
        "access_right": DIRG_ACCESS_RIGHT,
        "license_id": DIRG_LICENSE_ID,
        "selected_file_count": 78,
        "fold_count": 3,
        "files_per_split": DIRG_FILES_PER_SPLIT,
        "windows_per_split": DIRG_WINDOWS_PER_SPLIT,
        "physical_bearing_identity": "unauthenticated",
        "independent_replicate_unit": "unauthenticated",
        "file_observation_independence_claimed": False,
    }
    assert tuple(config["seeds"]["optimization"]) == OPTIMIZATION_SEEDS
    assert config["seeds"]["optimization_count"] == 25
    assert config["seeds"]["generator_roles"] == {
        "fit": [1103, 1109],
        "checkpoint_selection": [2203],
        "threshold_calibration": [2207],
        "confirmatory_test": [3301, 3307],
    }
    assert config["seeds"]["generator_role_overlap_allowed"] is False
    assert len(config["thresholds"]) == 11
    assert all(record["approved"] is False for record in config["thresholds"].values())
    assert all(
        "cwru_and_dirg_each" in config["thresholds"][threshold_id]["rule"]
        for threshold_id in ("T-C9-ACC-NI", "T-C9-FID-MAX", "T-C9-LATENCY-MAX")
    )
    assert {
        arm_id: record["trainable_parameter_count"]
        for arm_id, record in config["parameter_contract"]["cwru"].items()
    } == {
        "proposed": 2864,
        "dense_operator_mixture": 2864,
        "random_dictionary": 2864,
        "attention_cnn": 2917,
        "explainable_cnn": 3006,
        "discrete_search": None,
    }
    assert {
        arm_id: record["trainable_parameter_count"]
        for arm_id, record in config["parameter_contract"]["dirg"].items()
    } == {
        "proposed": 4892,
        "dense_operator_mixture": 4892,
        "random_dictionary": 4892,
        "attention_cnn": 4720,
        "explainable_cnn": 5123,
        "discrete_search": None,
    }
    assert config["runtime"]["hardware"] == {
        "allowed_single_gpu_indices": [0, 1],
        "forbidden_physical_gpu_indices": [2],
        "multi_gpu_allowed": False,
    }


def test_cli_defaults_to_read_only_check_and_false_gate_is_not_evidence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path, paths, manifests = _fixture_config(tmp_path)
    before = _snapshot_files(tmp_path)
    cwru_metadata, cwru_raw, cwru_reader, cwru_preprocessing = paths["cwru"]
    dirg_metadata, dirg_raw, dirg_reader, dirg_preprocessing = paths["dirg"]

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--protocol-sha256",
            "a" * 64,
            "--cwru-metadata-path",
            str(cwru_metadata),
            "--cwru-raw-dir",
            str(cwru_raw),
            "--cwru-reader-source-path",
            str(cwru_reader),
            "--cwru-preprocessing-source-path",
            str(cwru_preprocessing),
            "--dirg-metadata-path",
            str(dirg_metadata),
            "--dirg-raw-dir",
            str(dirg_raw),
            "--dirg-reader-source-path",
            str(dirg_reader),
            "--dirg-preprocessing-source-path",
            str(dirg_preprocessing),
        ]
    )

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert captured.out == canonical_json(summary) + "\n"
    assert _snapshot_files(tmp_path) == before
    assert summary["mode"] == "check_only"
    assert summary["status"] == "passed"
    assert summary["claim_evidence"] is False
    assert summary["evidence_state"] == "not_evidence"
    assert summary["gate_reason_codes"] == [
        "human_gate_not_approved",
        "threshold_unapproved_or_null",
    ]
    assert summary["training_started"] is False
    assert summary["raw_write_performed"] is False
    assert summary["emitted_files"] == []
    cwru_manifest = manifests["cwru"]
    dirg_manifest = manifests["dirg"]
    assert summary["cwru"]["root_sha256"] == cwru_manifest.root_sha256
    assert summary["dirg"]["root_sha256"] == dirg_manifest.root_sha256
    assert summary["dirg"]["selected_file_count"] == 78
    assert summary["dirg"]["fold_count"] == 3
    assert summary["dirg"]["files_per_split"] == 26
    assert summary["dirg"]["windows_per_split"] == 624
    assert summary["dirg"]["physical_bearing_identity"] == "unauthenticated"
    assert summary["dirg"]["physical_bearing_independence_claimed"] is False
    assert summary["c9_dataset_conjunction"] == ["cwru", "dirg"]
    assert summary["parameter_counts"] == {
        "cwru": {
            "attention_cnn": 2917,
            "dense_operator_mixture": 2864,
            "discrete_search": None,
            "explainable_cnn": 3006,
            "proposed": 2864,
            "random_dictionary": 2864,
        },
        "dirg": {
            "attention_cnn": 4720,
            "dense_operator_mixture": 4892,
            "discrete_search": None,
            "explainable_cnn": 5123,
            "proposed": 4892,
            "random_dictionary": 4892,
        },
    }


def test_manifest_hash_drift_fails_before_dataset_access(tmp_path: Path) -> None:
    config = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    config["manifests"]["path_universe_sha256"] = "0" * 64
    config_path = tmp_path / "drift.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    missing = tmp_path / "must-not-be-read"

    with pytest.raises(PreflightError, match="manifests.path_universe_sha256 drift"):
        run_preflight(
            config_path=config_path,
            protocol_sha256="b" * 64,
            cwru_metadata_path=missing,
            cwru_raw_dir=missing,
            cwru_reader_source_path=missing,
            cwru_preprocessing_source_path=missing,
            dirg_metadata_path=missing,
            dirg_raw_dir=missing,
            dirg_reader_source_path=missing,
            dirg_preprocessing_source_path=missing,
        )


@pytest.mark.parametrize(
    ("device", "indices", "multi_gpu", "message"),
    [
        ("cuda", (2,), False, "GPU 2 is forbidden"),
        ("cuda", (0, 1), False, "multi-GPU is forbidden"),
        ("cuda", (0,), True, "multi-GPU is forbidden"),
    ],
)
def test_gpu2_and_multigpu_requests_fail_before_dataset_access(
    tmp_path: Path,
    device: str,
    indices: tuple[int, ...],
    multi_gpu: bool,
    message: str,
) -> None:
    missing = tmp_path / "must-not-be-read"

    with pytest.raises(PreflightError, match=message):
        run_preflight(
            config_path=DEFAULT_CONFIG_PATH,
            protocol_sha256="c" * 64,
            cwru_metadata_path=missing,
            cwru_raw_dir=missing,
            cwru_reader_source_path=missing,
            cwru_preprocessing_source_path=missing,
            dirg_metadata_path=missing,
            dirg_raw_dir=missing,
            dirg_reader_source_path=missing,
            dirg_preprocessing_source_path=missing,
            device=device,
            physical_gpu_indices=indices,
            multi_gpu=multi_gpu,
        )


def test_emit_is_atomic_derived_only_and_cannot_target_raw_tree(tmp_path: Path) -> None:
    config_path, paths, manifests = _fixture_config(tmp_path)
    kwargs = _call_kwargs(config_path, paths)
    raw_dirs = (paths["cwru"][1], paths["dirg"][1])
    raw_before = {str(path): _snapshot_files(path) for path in raw_dirs}
    for raw_dir in raw_dirs:
        forbidden_target = raw_dir / "derived"
        with pytest.raises(PreflightError, match="must not be a raw dataset"):
            run_preflight(**kwargs, emit_dir=forbidden_target)
        assert not forbidden_target.exists()
    assert {
        str(path): _snapshot_files(path) for path in raw_dirs
    } == raw_before

    emit_dir = tmp_path / "derived"
    summary = run_preflight(**kwargs, emit_dir=emit_dir)

    expected_names = set(summary["emitted_files"])
    assert expected_names == {
        "composition_split_manifest.json",
        "cwru_manifest.json",
        "dirg_manifest.json",
        "nuisance_manifest.json",
        "p07_protocol_preflight_summary.json",
        "path_universe_manifest.json",
        "seed_namespace_manifest.json",
        "synthetic_generator_manifest.json",
    }
    assert {path.name for path in emit_dir.iterdir()} == expected_names
    for path in emit_dir.iterdir():
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert path.read_text(encoding="utf-8") == canonical_json(payload)
    emitted_summary = json.loads(
        (emit_dir / "p07_protocol_preflight_summary.json").read_text(encoding="utf-8")
    )
    assert emitted_summary == summary
    assert summary["mode"] == "emit_derived"
    assert summary["claim_evidence"] is False
    assert summary["training_started"] is False
    emitted_dirg = json.loads(
        (emit_dir / "dirg_manifest.json").read_text(encoding="utf-8")
    )
    assert emitted_dirg == manifests["dirg"].to_dict()
    assert {
        str(path): _snapshot_files(path) for path in raw_dirs
    } == raw_before
    assert not any(path.name.startswith(".derived.stage-") for path in tmp_path.iterdir())


def test_dirg_source_binding_drift_fails_closed(tmp_path: Path) -> None:
    config_path, paths, _manifests = _fixture_config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["dirg"]["reader_source_sha256"] = "0" * 64
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(PreflightError, match="dirg.reader_source_sha256 drift"):
        run_preflight(**_call_kwargs(config_path, paths))


def test_corrected_dirg_reader_declares_float64_contract() -> None:
    reader_path = DEFAULT_CONFIG_PATH.parents[3] / (
        "src/data_factory/reader/RM_020_DIRG.py"
    )
    source = reader_path.read_text(encoding="utf-8")

    assert "RM_017_Ottawa19" not in source
    assert "RM_020_DIRG MATLAB recording" in source
    assert "retaining the reader's float64 contract" in source
    assert "data.astype(np.float64)" in source

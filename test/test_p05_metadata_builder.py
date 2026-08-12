from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

import scripts.build_p05_metadata as builder


def _synthetic_source() -> pd.DataFrame:
    rows: list[dict] = []
    next_id = 1

    # Frozen CWRU structure: 75 source-domain and 23 domain-3 rows.  The
    # per-class totals make sklearn's stratified 0.25 split produce the same
    # class-count shape as the approved production protocol.
    source_counts = {0: 3, 1: 18, 2: 21, 3: 33}
    for label, count in source_counts.items():
        for index in range(count):
            rows.append(
                {
                    "Id": next_id,
                    "Dataset_id": 1,
                    "Name": "RM_001_CWRU",
                    "File": f"cwru-source-{label}-{index:02d}.mat",
                    "Label": label,
                    "Domain_id": index % 3,
                    "Sample_rate": 12000,
                }
            )
            next_id += 1
    test_counts = {0: 1, 1: 6, 2: 7, 3: 9}
    for label, count in test_counts.items():
        for index in range(count):
            rows.append(
                {
                    "Id": next_id,
                    "Dataset_id": 1,
                    "Name": "RM_001_CWRU",
                    "File": f"cwru-test-{label}-{index:02d}.mat",
                    "Label": label,
                    "Domain_id": 3,
                    "Sample_rate": 12000,
                }
            )
            next_id += 1
    for label in range(4):
        rows.append(
            {
                "Id": next_id,
                "Dataset_id": 1,
                "Name": "RM_001_CWRU",
                "File": f"cwru-48k-{label}.mat",
                "Label": label,
                "Domain_id": label % 3,
                "Sample_rate": 48000,
            }
        )
        next_id += 1
    for label in (-1, None):
        rows.append(
            {
                "Id": next_id,
                "Dataset_id": 1,
                "Name": "RM_001_CWRU",
                "File": f"cwru-ignored-{next_id}.mat",
                "Label": label,
                "Domain_id": 0,
                "Sample_rate": 12000,
            }
        )
        next_id += 1

    conditions = {0: "35Hz12kN", 1: "37.5Hz11kN", 2: "40Hz10kN"}
    positive_label = 1
    for domain_id, condition in conditions.items():
        for bearing_index in range(1, 6):
            bearing = f"Bearing{domain_id + 1}_{bearing_index}"
            for file_index, label in enumerate((-1, 0, positive_label)):
                rows.append(
                    {
                        "Id": next_id,
                        "Dataset_id": 2,
                        "Name": "RM_002_XJTU",
                        "File": f"{condition}/{bearing}/{file_index}.csv",
                        "Label": label,
                        "Domain_id": domain_id,
                        "Sample_rate": 25600,
                    }
                )
                next_id += 1
            positive_label += 1

    # An unrelated dataset is ignored but still participates in global Id
    # uniqueness validation.
    rows.append(
        {
            "Id": next_id,
            "Dataset_id": 99,
            "Name": "UNRELATED",
            "File": "ignored.csv",
            "Label": 0,
            "Domain_id": 0,
            "Sample_rate": 1,
        }
    )
    return pd.DataFrame(rows)


def _write_workbook(path: Path, frame: pd.DataFrame) -> str:
    frame.to_excel(path, index=False, engine="openpyxl")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_contract(frame: pd.DataFrame) -> builder.ExpectedContract:
    result = builder.derive_protocol_metadata(frame)
    payload = builder.semantic_metadata_bytes(result.metadata)
    return builder.ExpectedContract(
        row_count=len(result.metadata),
        payload_bytes=len(payload),
        semantic_sha256=hashlib.sha256(payload).hexdigest(),
        summary=result.summary,
    )


def _paths(root: Path) -> dict[str, Path]:
    return {
        "output_csv_path": root / "metadata_p05_v2.csv",
        "metadata_manifest_path": root / "metadata_p05_v2.manifest.json",
        "cwru_split_manifest_path": root / "split_cwru_p05_v2.manifest.json",
        "xjtu_split_manifest_path": root / "split_xjtu_p05_v2.manifest.json",
    }


def test_builder_is_deterministic_read_only_and_idempotent(tmp_path: Path) -> None:
    source = _synthetic_source()
    workbook = tmp_path / "source.xlsx"
    workbook_sha = _write_workbook(workbook, source)
    workbook_before = workbook.read_bytes()
    contract = _synthetic_contract(source)
    outputs = _paths(tmp_path / "protocol")

    first = builder.build_p05_metadata_package(
        workbook_path=workbook,
        expected_workbook_sha256=workbook_sha,
        expected_contract=contract,
        **outputs,
    )
    first_bytes = {name: path.read_bytes() for name, path in outputs.items()}
    first_mtimes = {name: path.stat().st_mtime_ns for name, path in outputs.items()}
    second = builder.build_p05_metadata_package(
        workbook_path=workbook,
        expected_workbook_sha256=workbook_sha,
        expected_contract=contract,
        **outputs,
    )

    assert workbook.read_bytes() == workbook_before
    assert hashlib.sha256(workbook.read_bytes()).hexdigest() == workbook_sha
    assert first["metadata_semantic_sha256"] == contract.semantic_sha256
    assert set(first["outputs"].values()) == {"created"}
    assert set(second["outputs"].values()) == {"reused_identical"}
    assert {name: path.read_bytes() for name, path in outputs.items()} == first_bytes
    assert {name: path.stat().st_mtime_ns for name, path in outputs.items()} == first_mtimes

    metadata = pd.read_csv(outputs["output_csv_path"])
    assert tuple(metadata.columns) == builder.PROTOCOL_COLUMNS
    assert (metadata["Label"] == metadata["Protocol_Label"]).all()
    assert set(metadata["Protocol_Fold"]) == {-1}
    xjtu = metadata.loc[metadata["Dataset_id"] == 2]
    assert set(xjtu.loc[xjtu["Domain_id"] == 0, "Protocol_Split"]) == {"train"}
    assert set(xjtu.loc[xjtu["Domain_id"] == 1, "Protocol_Split"]) == {"validation"}
    assert set(xjtu.loc[xjtu["Domain_id"] == 2, "Protocol_Split"]) == {"test"}
    assert xjtu["Protocol_Group"].nunique() == 15

    manifest = json.loads(outputs["metadata_manifest_path"].read_text(encoding="utf-8"))
    assert manifest["source_workbook"]["sha256"] == workbook_sha
    assert manifest["derived_metadata"]["semantic_serialization"]["sha256"] == contract.semantic_sha256
    for split_key in ("cwru_split_manifest_path", "xjtu_split_manifest_path"):
        split = json.loads(outputs[split_key].read_text(encoding="utf-8"))
        assert split["metadata_semantic_sha256"] == contract.semantic_sha256
        assert not (
            set(split["roles"]["train"]["groups"])
            & set(split["roles"]["validation"]["groups"])
        )


def test_builder_rejects_wrong_source_hash_and_contract_without_outputs(tmp_path: Path) -> None:
    source = _synthetic_source()
    workbook = tmp_path / "source.xlsx"
    workbook_sha = _write_workbook(workbook, source)
    contract = _synthetic_contract(source)

    wrong_hash_outputs = _paths(tmp_path / "wrong-hash")
    with pytest.raises(ValueError, match="source workbook SHA-256 mismatch"):
        builder.build_p05_metadata_package(
            workbook_path=workbook,
            expected_workbook_sha256="0" * 64,
            expected_contract=contract,
            **wrong_hash_outputs,
        )
    assert not any(path.exists() for path in wrong_hash_outputs.values())

    wrong_count_outputs = _paths(tmp_path / "wrong-count")
    with pytest.raises(ValueError, match="derived row count mismatch"):
        builder.build_p05_metadata_package(
            workbook_path=workbook,
            expected_workbook_sha256=workbook_sha,
            expected_contract=replace(contract, row_count=contract.row_count + 1),
            **wrong_count_outputs,
        )
    assert not any(path.exists() for path in wrong_count_outputs.values())

    wrong_digest_outputs = _paths(tmp_path / "wrong-digest")
    with pytest.raises(ValueError, match="semantic SHA-256 mismatch"):
        builder.build_p05_metadata_package(
            workbook_path=workbook,
            expected_workbook_sha256=workbook_sha,
            expected_contract=replace(contract, semantic_sha256="f" * 64),
            **wrong_digest_outputs,
        )
    assert not any(path.exists() for path in wrong_digest_outputs.values())


def test_builder_rejects_conflicting_existing_output_before_other_writes(tmp_path: Path) -> None:
    source = _synthetic_source()
    workbook = tmp_path / "source.xlsx"
    workbook_sha = _write_workbook(workbook, source)
    contract = _synthetic_contract(source)
    outputs = _paths(tmp_path / "protocol")
    outputs["xjtu_split_manifest_path"].parent.mkdir(parents=True)
    outputs["xjtu_split_manifest_path"].write_text("tampered\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        builder.build_p05_metadata_package(
            workbook_path=workbook,
            expected_workbook_sha256=workbook_sha,
            expected_contract=contract,
            **outputs,
        )
    assert outputs["xjtu_split_manifest_path"].read_text(encoding="utf-8") == "tampered\n"
    assert not outputs["output_csv_path"].exists()
    assert not outputs["metadata_manifest_path"].exists()
    assert not outputs["cwru_split_manifest_path"].exists()


@pytest.mark.parametrize("mutation, message", [
    ("missing_column", "missing required columns"),
    ("duplicate_id", "duplicate Id records"),
    ("duplicate_record", "duplicate target source records"),
    ("unknown_cwru_label", "unknown CWRU label"),
    ("unknown_xjtu_label", "unknown XJTU label"),
])
def test_builder_fails_closed_on_invalid_source(
    tmp_path: Path, mutation: str, message: str
) -> None:
    source = _synthetic_source()
    if mutation == "missing_column":
        source = source.drop(columns=["Sample_rate"])
    elif mutation == "duplicate_id":
        source.loc[source.index[1], "Id"] = source.loc[source.index[0], "Id"]
    elif mutation == "duplicate_record":
        duplicate = source.loc[[source.index[0]]].copy()
        duplicate["Id"] = int(source["Id"].max()) + 1
        source = pd.concat([source, duplicate], ignore_index=True)
    elif mutation == "unknown_cwru_label":
        cwru_index = source.index[source["Dataset_id"] == 1][0]
        source.loc[cwru_index, "Label"] = 9
    elif mutation == "unknown_xjtu_label":
        xjtu_index = source.index[(source["Dataset_id"] == 2) & (source["Label"] == 0)][0]
        source.loc[xjtu_index, "Label"] = -2
    else:  # pragma: no cover
        raise AssertionError(mutation)

    workbook = tmp_path / f"{mutation}.xlsx"
    workbook_sha = _write_workbook(workbook, source)
    outputs = _paths(tmp_path / f"outputs-{mutation}")
    with pytest.raises(ValueError, match=message):
        builder.build_p05_metadata_package(
            workbook_path=workbook,
            expected_workbook_sha256=workbook_sha,
            expected_contract=builder.PRODUCTION_CONTRACT,
            **outputs,
        )
    assert not any(path.exists() for path in outputs.values())


def test_cli_uses_explicit_paths_and_caller_hash(tmp_path: Path, monkeypatch) -> None:
    source = _synthetic_source()
    workbook = tmp_path / "source.xlsx"
    workbook_sha = _write_workbook(workbook, source)
    contract = _synthetic_contract(source)
    monkeypatch.setattr(builder, "PRODUCTION_CONTRACT", contract)
    outputs = _paths(tmp_path / "cli-protocol")

    assert builder.main(
        [
            "--workbook",
            str(workbook),
            "--workbook-sha256",
            workbook_sha,
            "--output-csv",
            str(outputs["output_csv_path"]),
            "--metadata-manifest",
            str(outputs["metadata_manifest_path"]),
            "--cwru-split-manifest",
            str(outputs["cwru_split_manifest_path"]),
            "--xjtu-split-manifest",
            str(outputs["xjtu_split_manifest_path"]),
        ]
    ) == 0
    assert all(path.is_file() for path in outputs.values())


def test_production_contract_is_frozen() -> None:
    assert builder.PRODUCTION_CONTRACT.row_count == 8471
    assert builder.PRODUCTION_CONTRACT.payload_bytes == 2163841
    assert (
        builder.PRODUCTION_CONTRACT.semantic_sha256
        == "87392b6517b6bde753c63a982d998ee5b090ab9ed106f36b294b3ddfdcb3e381"
    )

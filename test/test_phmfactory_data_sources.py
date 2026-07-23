from __future__ import annotations

from pathlib import Path
import shutil
from types import SimpleNamespace

import h5py
from openpyxl import Workbook
import pytest

from phmfactory import cli
from phmfactory.data_sources import (
    BundleValidationError,
    compare_bundle_hashes,
    download_bundle,
    load_bundle_spec,
    validate_bundle,
)
from phmfactory.data_sources import bundle as bundle_module


def _write_workbook(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(headers)
    for row in rows:
        sheet.append(row)
    workbook.save(path)
    workbook.close()


def _write_bundle(
    root: Path,
    *,
    include_corpus: bool = False,
    missing_signal_id: bool = False,
    signal_shape: tuple[int, ...] = (4, 2),
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _write_workbook(
        root / "metadata.xlsx",
        ["Id", "Name", "Sample_lenth", "Channel", "Dataset_id"],
        [
            [1, "RM_001_CWRU", 4, 2, 1],
            [2, "RM_001_CWRU", 4, 2, 1],
            [99, "OTHER", 4, 2, 2],
        ],
    )
    with h5py.File(root / "RM_001_CWRU.h5", "w") as handle:
        handle.create_dataset("1", shape=signal_shape, dtype="float32")
        if not missing_signal_id:
            handle.create_dataset("2", shape=(4, 2), dtype="float32")
    if include_corpus:
        _write_workbook(
            root / "corpus.xlsx",
            ["Id", "Text"],
            [[1, "normal bearing"], [2, "fault bearing"]],
        )
    return root


def test_manifest_declares_two_explicit_public_providers() -> None:
    spec = load_bundle_spec()
    assert spec.bundle_id == "cwru-demo-v1"
    assert spec.metadata_file == "metadata.xlsx"
    assert spec.signal_file == "RM_001_CWRU.h5"
    assert spec.corpus_file == "corpus.xlsx"
    assert set(spec.providers) == {"huggingface", "modelscope"}
    assert spec.providers["huggingface"]["repo_id"] == "PHMbench/PHM-Vibench"
    assert spec.providers["modelscope"]["repo_id"] == "PHMbench/PHM-Vibench"


def test_validate_bundle_accepts_missing_optional_corpus(tmp_path: Path) -> None:
    validation = validate_bundle(_write_bundle(tmp_path / "bundle"))
    assert validation.metadata_rows == 3
    assert validation.selected_rows == 2
    assert validation.signal_keys == 2
    assert validation.corpus_present is False
    assert {item.name for item in validation.files} == {
        "metadata.xlsx",
        "RM_001_CWRU.h5",
    }


def test_validate_bundle_checks_optional_corpus_foreign_keys(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path / "bundle", include_corpus=True)
    validation = validate_bundle(root)
    assert validation.corpus_present is True

    _write_workbook(root / "corpus.xlsx", ["Id", "Text"], [[777, "unknown"]])
    with pytest.raises(BundleValidationError, match="unknown Id"):
        validate_bundle(root)


def test_validate_bundle_rejects_missing_signal_id(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path / "bundle", missing_signal_id=True)
    with pytest.raises(BundleValidationError, match="absent"):
        validate_bundle(root)


def test_validate_bundle_requires_two_dimensional_lc_shape(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path / "bundle", signal_shape=(4, 2, 1))
    with pytest.raises(BundleValidationError, match=r"shape \(L, C\)"):
        validate_bundle(root)


def test_compare_bundle_hashes_requires_provider_parity(tmp_path: Path) -> None:
    left = _write_bundle(tmp_path / "left")
    right = tmp_path / "right"
    shutil.copytree(left, right)
    hashes = compare_bundle_hashes(left, right)
    assert set(hashes) == {"metadata.xlsx", "RM_001_CWRU.h5"}

    with h5py.File(right / "RM_001_CWRU.h5", "a") as handle:
        handle["1"][0, 0] = 3.0
    with pytest.raises(BundleValidationError, match="hashes differ"):
        compare_bundle_hashes(left, right)


def test_huggingface_download_materializes_required_files_and_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = _write_bundle(tmp_path / "remote")

    def fake_hf_hub_download(**kwargs: object) -> str:
        filename = str(kwargs["filename"])
        if filename == "corpus.xlsx":
            raise FileNotFoundError(filename)
        destination = Path(str(kwargs["local_dir"])) / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(remote / filename, destination)
        return str(destination)

    monkeypatch.setitem(
        __import__("sys").modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=fake_hf_hub_download),
    )
    result = download_bundle(
        source="huggingface",
        destination=tmp_path / "downloaded",
        revision="test-revision",
    )
    assert result.provider == "huggingface"
    assert result.validation.corpus_present is False
    assert (result.directory / ".phmfactory-bundle.yaml").is_file()


def test_modelscope_provider_uses_argument_list_without_shell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = load_bundle_spec()
    remote = _write_bundle(tmp_path / "remote")
    destination = tmp_path / "downloaded"
    commands: list[list[str]] = []

    monkeypatch.setattr(bundle_module.shutil, "which", lambda name: "/usr/bin/modelscope")

    def fake_runner(command: list[str], **kwargs: object) -> SimpleNamespace:
        commands.append(command)
        filename = command[command.index("PHMbench/PHM-Vibench") + 1]
        if filename == "corpus.xlsx":
            return SimpleNamespace(returncode=1, stderr="not published")
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(remote / filename, destination / filename)
        assert "shell" not in kwargs
        return SimpleNamespace(returncode=0, stderr="")

    bundle_module._download_modelscope(
        spec,
        spec.providers["modelscope"],
        destination,
        "test-revision",
        runner=fake_runner,
    )
    assert commands
    assert all(isinstance(command, list) for command in commands)
    assert validate_bundle(destination).selected_rows == 2


def test_data_validate_cli_uses_same_public_contract(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _write_bundle(tmp_path / "bundle")
    result = cli.main(["data", "validate", "--path", str(root)])
    assert result.selected_rows == 2
    output = capsys.readouterr().out
    assert "bundle=cwru-demo-v1" in output
    assert "corpus_present=false" in output

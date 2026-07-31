from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import shutil
from types import SimpleNamespace

import h5py
from openpyxl import Workbook
import yaml
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
        [
            "Id",
            "Name",
            "Sample_lenth",
            "Channel",
            "Dataset_id",
            "Label",
            "Domain_id",
        ],
        [
            [1, "RM_001_CWRU", 4, 2, 1, 0, 0],
            [2, "RM_001_CWRU", 4, 2, 1, 1, 1],
            [99, "OTHER", 4, 2, 2, 0, 0],
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



@pytest.mark.parametrize("missing_column", ["Dataset_id", "Label", "Domain_id"])
def test_validate_bundle_requires_runtime_metadata_columns(
    tmp_path: Path,
    missing_column: str,
) -> None:
    root = _write_bundle(tmp_path / "bundle")
    headers = [
        "Id",
        "Name",
        "Sample_lenth",
        "Channel",
        "Dataset_id",
        "Label",
        "Domain_id",
    ]
    rows = [
        [1, "RM_001_CWRU", 4, 2, 1, 0, 0],
        [2, "RM_001_CWRU", 4, 2, 1, 1, 1],
        [99, "OTHER", 4, 2, 2, 0, 0],
    ]
    index = headers.index(missing_column)
    _write_workbook(
        root / "metadata.xlsx",
        [value for offset, value in enumerate(headers) if offset != index],
        [
            [value for offset, value in enumerate(row) if offset != index]
            for row in rows
        ],
    )
    with pytest.raises(BundleValidationError, match=missing_column):
        validate_bundle(root)


def test_validate_bundle_rejects_duplicate_selected_metadata_ids(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path / "bundle")
    _write_workbook(
        root / "metadata.xlsx",
        [
            "Id",
            "Name",
            "Sample_lenth",
            "Channel",
            "Dataset_id",
            "Label",
            "Domain_id",
        ],
        [
            [1, "RM_001_CWRU", 4, 2, 1, 0, 0],
            [1, "RM_001_CWRU", 4, 2, 1, 1, 1],
        ],
    )
    with pytest.raises(BundleValidationError, match="duplicate selected Id"):
        validate_bundle(root)

def test_validate_bundle_enforces_logical_hash_keys(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path / "bundle")
    metadata_digest = sha256((root / "metadata.xlsx").read_bytes()).hexdigest()
    signal_digest = sha256((root / "RM_001_CWRU.h5").read_bytes()).hexdigest()
    spec = replace(
        load_bundle_spec(),
        expected_sha256={
  "metadata": metadata_digest,
  "signals": signal_digest,
        },
    )
    assert validate_bundle(root, spec=spec).selected_rows == 2

    bad_metadata = replace(
        spec,
        expected_sha256={"metadata": "0" * 64, "signals": signal_digest},
    )
    with pytest.raises(BundleValidationError, match="metadata.xlsx"):
        validate_bundle(root, spec=bad_metadata)

    bad_signals = replace(
        spec,
        expected_sha256={"metadata": metadata_digest, "signals": "1" * 64},
    )
    with pytest.raises(BundleValidationError, match="RM_001_CWRU.h5"):
        validate_bundle(root, spec=bad_signals)

    conflicting = replace(
        spec,
        expected_sha256={
  "metadata": metadata_digest,
  "metadata.xlsx": "0" * 64,
  "signals": signal_digest,
        },
    )
    with pytest.raises(BundleValidationError, match="Conflicting SHA-256 pins"):
        validate_bundle(root, spec=conflicting)


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



def test_download_reuses_only_matching_recorded_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote_a = _write_bundle(tmp_path / "remote-a")
    remote_b = _write_bundle(tmp_path / "remote-b")
    with h5py.File(remote_b / "RM_001_CWRU.h5", "a") as handle:
        handle["1"][0, 0] = 7.0
    calls: list[str] = []

    def fake_download(spec, provider, root: Path, revision: str) -> None:
        calls.append(revision)
        remote = remote_a if revision == "revision-a" else remote_b
        for _, local_name, required in bundle_module._bundle_files(spec):
            source = remote / local_name
            if source.is_file():
                shutil.copy2(source, root / local_name)
            elif required:
                raise FileNotFoundError(source)

    monkeypatch.setattr(bundle_module, "_download_huggingface", fake_download)
    destination = tmp_path / "downloaded"
    first = download_bundle(
        source="huggingface",
        destination=destination,
        revision="revision-a",
    )
    first_hash = next(
        item.sha256
        for item in first.validation.files
        if item.name == "RM_001_CWRU.h5"
    )

    reused = download_bundle(
        source="huggingface",
        destination=destination,
        revision="revision-a",
    )
    assert calls == ["revision-a"]
    assert reused.validation.files == first.validation.files

    refreshed = download_bundle(
        source="huggingface",
        destination=destination,
        revision="revision-b",
    )
    assert calls == ["revision-a", "revision-b"]
    refreshed_hash = next(
        item.sha256
        for item in refreshed.validation.files
        if item.name == "RM_001_CWRU.h5"
    )
    assert refreshed_hash != first_hash
    provenance = yaml.safe_load(
        (destination / ".phmfactory-bundle.yaml").read_text(encoding="utf-8")
    )
    assert provenance["provider"] == "huggingface"
    assert provenance["requested_revision"] == "revision-b"


def test_download_rejects_tampered_cache_provenance_and_refreshes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = _write_bundle(tmp_path / "remote")
    calls: list[str] = []

    def fake_download(spec, provider, root: Path, revision: str) -> None:
        calls.append(revision)
        for _, local_name, required in bundle_module._bundle_files(spec):
            source = remote / local_name
            if source.is_file():
                shutil.copy2(source, root / local_name)
            elif required:
                raise FileNotFoundError(source)

    monkeypatch.setattr(bundle_module, "_download_huggingface", fake_download)
    destination = tmp_path / "downloaded"
    download_bundle(
        source="huggingface",
        destination=destination,
        revision="revision-a",
    )
    provenance_path = destination / ".phmfactory-bundle.yaml"
    payload = yaml.safe_load(provenance_path.read_text(encoding="utf-8"))
    payload["provider"] = "modelscope"
    provenance_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    download_bundle(
        source="huggingface",
        destination=destination,
        revision="revision-a",
    )
    assert calls == ["revision-a", "revision-a"]


def test_forced_refresh_removes_stale_optional_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with_corpus = _write_bundle(tmp_path / "with-corpus", include_corpus=True)
    without_corpus = _write_bundle(tmp_path / "without-corpus")
    selected = {"revision-a": with_corpus, "revision-b": without_corpus}

    def fake_download(spec, provider, root: Path, revision: str) -> None:
        remote = selected[revision]
        for _, local_name, required in bundle_module._bundle_files(spec):
            source = remote / local_name
            if source.is_file():
                shutil.copy2(source, root / local_name)
            elif required:
                raise FileNotFoundError(source)

    monkeypatch.setattr(bundle_module, "_download_huggingface", fake_download)
    destination = tmp_path / "downloaded"
    first = download_bundle(
        source="huggingface",
        destination=destination,
        revision="revision-a",
    )
    assert first.validation.corpus_present is True
    assert (destination / "corpus.xlsx").is_file()

    refreshed = download_bundle(
        source="huggingface",
        destination=destination,
        revision="revision-b",
        force=True,
    )
    assert refreshed.validation.corpus_present is False
    assert not (destination / "corpus.xlsx").exists()

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

@pytest.mark.parametrize(
    ("sample_length", "channel_count"),
    [(4.5, 2), (4, 2.5)],
)
def test_validate_bundle_rejects_fractional_shape_metadata(
    tmp_path: Path,
    sample_length: float,
    channel_count: float,
) -> None:
    root = _write_bundle(tmp_path / "bundle")
    _write_workbook(
        root / "metadata.xlsx",
        [
            "Id",
            "Name",
            "Sample_lenth",
            "Channel",
            "Dataset_id",
            "Label",
            "Domain_id",
        ],
        [
            [1, "RM_001_CWRU", sample_length, channel_count, 1, 0, 0],
            [2, "RM_001_CWRU", 4, 2, 1, 1, 1],
        ],
    )
    with pytest.raises(BundleValidationError, match="positive integer"):
        validate_bundle(root)

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.utils.p07_protocol.artifact_store import (
    ARTIFACT_INDEX_NAME,
    COMPLETION_MARKER_NAME,
    ArtifactStoreError,
    DerivedArtifactStore,
    audit_finalized_store,
)


def _store(tmp_path: Path, *, name: str = "run") -> DerivedArtifactStore:
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    return DerivedArtifactStore(
        (tmp_path / name).resolve(),
        run_id="P07-test-run",
        protocol_id="P07-G040-v3",
        immutable_source_roots=(raw.resolve(),),
        bindings={"protocol_sha256": "a" * 64, "human_gate": False},
    )


def test_write_finalize_and_audit_round_trip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = store.write_canonical_json(
        "metrics.json", {"z": 2, "a": 1}, role="derived_metrics"
    )
    second = store.write_bytes("nested/atoms.bin", b"abc", role="analysis_atoms")

    inventory = store.finalize(required_artifacts=("metrics.json", "nested/atoms.bin"))
    audited = audit_finalized_store(store.output_root)

    assert first.byte_count == len(b'{"a":1,"z":2}\n')
    assert second.sha256 == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert inventory.artifact_index_sha256 == audited.artifact_index_sha256
    assert inventory.completion_marker_sha256 == audited.completion_marker_sha256
    assert [item.relative_path for item in audited.artifacts] == [
        "metrics.json",
        "nested/atoms.bin",
    ]
    marker = json.loads(
        (store.output_root / COMPLETION_MARKER_NAME).read_text(encoding="utf-8")
    )
    assert marker["state"] == "complete"
    assert marker["artifact_index_sha256"] == inventory.artifact_index_sha256


def test_output_root_must_be_new_absolute_and_outside_raw(tmp_path: Path) -> None:
    raw = (tmp_path / "raw").resolve()
    raw.mkdir()
    with pytest.raises(ValueError, match="absolute"):
        DerivedArtifactStore(
            "relative/run",
            run_id="x",
            protocol_id="p",
            immutable_source_roots=(raw,),
            bindings={},
        )
    with pytest.raises(ArtifactStoreError, match="immutable source"):
        DerivedArtifactStore(
            raw / "derived",
            run_id="x",
            protocol_id="p",
            immutable_source_roots=(raw,),
            bindings={},
        )
    existing = (tmp_path / "existing").resolve()
    existing.mkdir()
    with pytest.raises(ArtifactStoreError, match="new path"):
        DerivedArtifactStore(
            existing,
            run_id="x",
            protocol_id="p",
            immutable_source_roots=(raw,),
            bindings={},
        )


@pytest.mark.parametrize(
    "path",
    ("", "/absolute", "../escape", "a/../escape", "a//b", "a\\b", "."),
)
def test_artifact_paths_reject_noncanonical_or_traversal(
    tmp_path: Path, path: str
) -> None:
    store = _store(tmp_path)
    with pytest.raises(ValueError):
        store.write_bytes(path, b"x", role="test")


def test_store_refuses_overwrite_reserved_and_post_finalize_writes(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.write_bytes("metrics.json", b"one", role="test")
    with pytest.raises(ArtifactStoreError, match="already registered"):
        store.write_bytes("metrics.json", b"two", role="test")
    with pytest.raises(ArtifactStoreError, match="reserved"):
        store.write_bytes(ARTIFACT_INDEX_NAME, b"x", role="test")
    store.finalize(required_artifacts=("metrics.json",))
    with pytest.raises(ArtifactStoreError, match="already finalized"):
        store.write_bytes("late.bin", b"late", role="test")


def test_finalize_fails_closed_when_required_artifact_is_missing(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.write_bytes("present.bin", b"x", role="test")
    with pytest.raises(ArtifactStoreError, match="missing.bin"):
        store.finalize(required_artifacts=("present.bin", "missing.bin"))
    assert not (store.output_root / COMPLETION_MARKER_NAME).exists()


def test_materialize_is_hashed_and_empty_output_is_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    record = store.materialize(
        "checkpoint.pt",
        role="checkpoint",
        writer=lambda path: path.write_bytes(b"checkpoint"),
    )
    assert record.byte_count == 10
    with pytest.raises(ArtifactStoreError, match="nonempty"):
        store.materialize("empty.bin", role="bad", writer=lambda path: None)
    assert not (store.output_root / "empty.bin").exists()


def test_parent_symlink_is_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (store.output_root / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ArtifactStoreError, match="regular directory"):
        store.write_bytes("linked/escape.bin", b"x", role="test")
    assert not (outside / "escape.bin").exists()


def test_audit_detects_tampering_and_unindexed_files(tmp_path: Path) -> None:
    store = _store(tmp_path, name="tampered")
    store.write_bytes("metrics.json", b"truth", role="test")
    store.finalize(required_artifacts=("metrics.json",))
    (store.output_root / "metrics.json").write_bytes(b"altered")
    with pytest.raises(ArtifactStoreError, match="byte identity"):
        audit_finalized_store(store.output_root)

    clean = _store(tmp_path, name="extra")
    clean.write_bytes("metrics.json", b"truth", role="test")
    clean.finalize(required_artifacts=("metrics.json",))
    (clean.output_root / "unindexed.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(ArtifactStoreError, match="unindexed"):
        audit_finalized_store(clean.output_root)


def test_bindings_forbid_nonfinite_json(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    with pytest.raises(ValueError, match="canonical-JSON"):
        DerivedArtifactStore(
            (tmp_path / "run").resolve(),
            run_id="x",
            protocol_id="p",
            immutable_source_roots=(raw.resolve(),),
            bindings={"bad": float("nan")},
        )

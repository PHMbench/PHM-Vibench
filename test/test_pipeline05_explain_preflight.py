from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src import Pipeline_05_Explainable_Fault_Diagnosis as pipeline05


class _DataFactory:
    def __init__(self, batch):
        self.batch = batch
        self.calls = 0

    def get_dataloader(self, split: str):
        assert split == "test"
        self.calls += 1
        return [self.batch]


def _trainer(*, enabled: bool, explainer: str = "timefreq") -> SimpleNamespace:
    return SimpleNamespace(
        extensions=SimpleNamespace(
            explain=SimpleNamespace(enable=enabled, explainer=explainer)
        )
    )


def test_disabled_explain_path_does_not_inspect_a_batch(tmp_path) -> None:
    factory = _DataFactory((object(), object()))
    pipeline05._write_explain_preflight(factory, _trainer(enabled=False), tmp_path)
    assert factory.calls == 0


def test_explain_path_rejects_a_batch_without_metadata(tmp_path) -> None:
    factory = _DataFactory((object(), object()))
    with pytest.raises(ValueError, match="exactly \\(x, y, meta\\)"):
        pipeline05._write_explain_preflight(factory, _trainer(enabled=True), tmp_path)


def test_explain_path_records_then_rejects_degraded_metadata(tmp_path) -> None:
    factory = _DataFactory((object(), object(), {"machine_id": "M1"}))

    with pytest.raises(ValueError, match="MISSING_META, DEGRADED_METADATA"):
        pipeline05._write_explain_preflight(factory, _trainer(enabled=True), tmp_path)

    snapshot = json.loads((tmp_path / "data_metadata_snapshot.json").read_text())
    eligibility = json.loads((tmp_path / "explain" / "eligibility.json").read_text())
    assert snapshot["degraded"] is True
    assert eligibility["ok"] is False


def test_explain_path_requires_successful_artifact_writes(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    factory = _DataFactory((object(), object(), {"sampling_rate": 12000}))

    def fail_write(*args, **kwargs) -> None:
        raise OSError("read-only")

    monkeypatch.setattr(
        pipeline05,
        "write_metadata_snapshot",
        fail_write,
    )

    with pytest.raises(OSError, match="read-only"):
        pipeline05._write_explain_preflight(factory, _trainer(enabled=True), tmp_path)


def test_explain_path_accepts_complete_batch_metadata(tmp_path) -> None:
    factory = _DataFactory((object(), object(), {"sampling_rate": 12000}))
    pipeline05._write_explain_preflight(factory, _trainer(enabled=True), tmp_path)
    eligibility = json.loads((tmp_path / "explain" / "eligibility.json").read_text())
    assert eligibility["ok"] is True

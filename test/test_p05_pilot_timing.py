from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

import src.trainer_factory.p05_pilot_timing as pilot_timing
from src.trainer_factory.Default_trainer import call_backs
from src.trainer_factory.p05_pilot_timing import (
    P05PilotTimingCallback,
    P05PilotTimingError,
    build_p05_pilot_timing_callback,
)


class FakeCuda:
    def __init__(
        self,
        *,
        available: bool = True,
        allocated: int = 3_000,
        reserved: int = 5_000,
    ) -> None:
        self.available = available
        self.allocated = allocated
        self.reserved = reserved
        self.events: list[tuple[str, str]] = []

    def is_available(self) -> bool:
        return self.available

    def synchronize(self, device: torch.device) -> None:
        self.events.append(("synchronize", str(device)))

    def reset_peak_memory_stats(self, device: torch.device) -> None:
        self.events.append(("reset", str(device)))

    def max_memory_allocated(self, device: torch.device) -> int:
        self.events.append(("allocated", str(device)))
        return self.allocated

    def max_memory_reserved(self, device: torch.device) -> int:
        self.events.append(("reserved", str(device)))
        return self.reserved


class FakeClock:
    def __init__(self, values: list[float]) -> None:
        self.values = iter(values)

    def __call__(self) -> float:
        return next(self.values)


def _clock_values(
    *,
    startup: float = 1.0,
    durations: tuple[float, ...] = (2.0, 3.0, 4.0, 5.0, 6.0),
) -> list[float]:
    values = [0.0, startup]
    cursor = startup
    for index, duration in enumerate(durations):
        if index > 0:
            cursor += 0.5
            values.append(cursor)
        cursor += duration
        values.append(cursor)
    return values


def _run_complete_callback(
    package,
    *,
    cuda: FakeCuda | None = None,
    clock_values: list[float] | None = None,
):
    cuda_api = cuda or FakeCuda()
    callback = P05PilotTimingCallback(
        package,
        cuda_api=cuda_api,
        clock=FakeClock(clock_values or _clock_values()),
    )
    trainer = SimpleNamespace(max_epochs=5, current_epoch=0, world_size=1)
    module = SimpleNamespace(device=torch.device("cuda:0"))
    callback.on_fit_start(trainer, module)
    for epoch in range(5):
        trainer.current_epoch = epoch
        callback.on_train_epoch_start(trainer, module)
        callback.on_train_epoch_end(trainer, module)
    callback.on_fit_end(trainer, module)
    return callback, trainer, cuda_api


def _trainer_args(**overrides):
    values = {
        "p05_pilot_mode": True,
        "num_epochs": 5,
        "device": "cuda",
        "early_stopping": False,
        "monitor": "val_loss",
        "save_top_k": 1,
        "pruning": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_pilot_timing_requires_explicit_literal_activation(tmp_path) -> None:
    assert build_p05_pilot_timing_callback(SimpleNamespace(), tmp_path) is None
    assert (
        build_p05_pilot_timing_callback(
            SimpleNamespace(p05_pilot_mode=False),
            tmp_path,
        )
        is None
    )
    with pytest.raises(P05PilotTimingError, match="literal boolean"):
        build_p05_pilot_timing_callback(
            SimpleNamespace(p05_pilot_mode="true"),
            tmp_path,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"num_epochs": 4}, "exactly 5"),
        ({"num_epochs": None}, "exactly 5"),
        ({"device": "cpu"}, "exactly 'cuda'"),
        ({"early_stopping": True}, "early_stopping must be false"),
    ],
)
def test_pilot_timing_factory_rejects_non_frozen_runtime(
    tmp_path,
    overrides,
    message,
) -> None:
    with pytest.raises(P05PilotTimingError, match=message):
        build_p05_pilot_timing_callback(_trainer_args(**overrides), tmp_path)


def test_default_callback_factory_binds_only_explicit_pilot(tmp_path) -> None:
    pilot_callbacks = call_backs(_trainer_args(), str(tmp_path))
    legacy_callbacks = call_backs(
        _trainer_args(p05_pilot_mode=False),
        str(tmp_path / "legacy"),
    )

    selected = [item for item in pilot_callbacks if isinstance(item, P05PilotTimingCallback)]
    assert len(selected) == 1
    assert selected[0].package_dir == (
        tmp_path / "artifacts" / "p05_pilot_timing"
    ).resolve()
    assert not any(isinstance(item, P05PilotTimingCallback) for item in legacy_callbacks)


def test_pilot_timing_records_synchronized_complete_epochs_and_memory(tmp_path) -> None:
    package = tmp_path / "pilot-timing"
    callback, trainer, cuda = _run_complete_callback(package)

    assert callback.result is not None
    assert callback.result.status == "created"
    assert trainer.p05_pilot_timing_result == callback.result
    manifest_bytes = callback.result.manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    assert manifest["schema_name"] == "p05.non_evidence_pilot_timing"
    assert manifest["evidence_eligible"] is False
    assert manifest["claim_support"] == "forbidden"
    assert manifest["measurement_status"] == "complete"
    assert manifest["measurements"] == {
        "startup_seconds": 1.0,
        "epoch_seconds_1_through_5": [2.0, 3.0, 4.0, 5.0, 6.0],
        "median_epoch_seconds_2_through_5": 4.5,
        "peak_allocated_memory": 3_000,
        "peak_reserved_memory": 5_000,
    }
    assert manifest["timing_contract"]["median_epoch_numbers"] == [2, 3, 4, 5]
    assert manifest["timing_contract"]["memory_unit"] == "bytes"
    assert cuda.events == [
        ("reset", "cuda:0"),
        *(("synchronize", "cuda:0"),) * 10,
        ("allocated", "cuda:0"),
        ("reserved", "cuda:0"),
    ]
    semantic = {key: value for key, value in manifest.items() if key != "content"}
    canonical = json.dumps(
        semantic,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    assert manifest["content"]["semantic_sha256"] == hashlib.sha256(
        canonical
    ).hexdigest()
    assert callback.result.manifest_sha256 == hashlib.sha256(manifest_bytes).hexdigest()
    assert not list(tmp_path.glob(".pilot-timing.*.tmp"))


def test_pilot_timing_missing_epoch_fails_without_artifact(tmp_path) -> None:
    package = tmp_path / "incomplete"
    callback = P05PilotTimingCallback(
        package,
        cuda_api=FakeCuda(),
        clock=FakeClock(_clock_values(durations=(2.0, 3.0, 4.0, 5.0))),
    )
    trainer = SimpleNamespace(max_epochs=5, current_epoch=0, world_size=1)
    module = SimpleNamespace(device=torch.device("cuda:0"))
    callback.on_fit_start(trainer, module)
    for epoch in range(4):
        trainer.current_epoch = epoch
        callback.on_train_epoch_start(trainer, module)
        callback.on_train_epoch_end(trainer, module)

    with pytest.raises(P05PilotTimingError, match="exactly five complete"):
        callback.on_fit_end(trainer, module)

    assert not package.exists()


@pytest.mark.parametrize(
    ("device", "cuda_available", "message"),
    [
        (torch.device("cpu"), True, "forbids non-CUDA"),
        (torch.device("cuda:0"), False, "requires CUDA availability"),
    ],
)
def test_pilot_timing_rejects_non_cuda_execution(
    tmp_path,
    device,
    cuda_available,
    message,
) -> None:
    package = tmp_path / "non-cuda"
    callback = P05PilotTimingCallback(
        package,
        cuda_api=FakeCuda(available=cuda_available),
        clock=FakeClock([0.0]),
    )
    trainer = SimpleNamespace(max_epochs=5, current_epoch=0, world_size=1)

    with pytest.raises(P05PilotTimingError, match=message):
        callback.on_fit_start(trainer, SimpleNamespace(device=device))

    assert not package.exists()


def test_pilot_timing_create_only_conflict_preserves_existing_bytes(tmp_path) -> None:
    package = tmp_path / "conflict"
    first, _, _ = _run_complete_callback(package)
    assert first.result is not None
    before = first.result.manifest_path.read_bytes()
    second = P05PilotTimingCallback(
        package,
        cuda_api=FakeCuda(),
        clock=FakeClock(_clock_values()),
    )

    with pytest.raises(FileExistsError, match="conflicts"):
        second.on_fit_start(
            SimpleNamespace(max_epochs=5, current_epoch=0, world_size=1),
            SimpleNamespace(device=torch.device("cuda:0")),
        )

    assert first.result.manifest_path.read_bytes() == before


def test_pilot_timing_write_failure_leaves_no_partial_package(
    tmp_path,
    monkeypatch,
) -> None:
    package = tmp_path / "failed"

    def fail_write(path, content):
        del path, content
        raise RuntimeError("synthetic timing write failure")

    monkeypatch.setattr(pilot_timing, "_write_manifest_file", fail_write)
    with pytest.raises(RuntimeError, match="synthetic timing write failure"):
        _run_complete_callback(package)

    assert not package.exists()
    assert not list(tmp_path.glob(".failed.*.tmp"))


def test_pilot_timing_rejects_invalid_peak_memory(tmp_path) -> None:
    package = tmp_path / "memory"
    with pytest.raises(P05PilotTimingError, match="at least allocated"):
        _run_complete_callback(
            package,
            cuda=FakeCuda(allocated=5_000, reserved=4_000),
        )

    assert not package.exists()

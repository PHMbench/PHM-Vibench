"""Fail-closed CUDA timing for the frozen five-epoch P05 pilot."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import shutil
import statistics
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch


SCHEMA_NAME = "p05.non_evidence_pilot_timing"
SCHEMA_VERSION = 1
EXPECTED_EPOCHS = 5
PACKAGE_NAME = "p05_pilot_timing"
MANIFEST_NAME = "manifest.json"


class P05PilotTimingError(RuntimeError):
    """Raised when a pilot timing boundary or runtime contract is incomplete."""


@dataclass(frozen=True)
class P05PilotTimingResult:
    """Location and hashes for one completed non-evidence timing artifact."""

    package_dir: Path
    manifest_path: Path
    semantic_sha256: str
    manifest_sha256: str
    status: str


def p05_pilot_mode_enabled(args_trainer: Any) -> bool:
    """Return the exact pilot flag without inferring it from evidence mode."""

    value = getattr(args_trainer, "p05_pilot_mode", False)
    if type(value) is not bool:
        raise P05PilotTimingError("trainer.p05_pilot_mode must be a literal boolean")
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_noreplace(source: Path, target: Path) -> None:
    """Atomically install a directory without an overwrite-capable fallback."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic create-only pilot export requires Linux renameat2")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(target),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            "P05 pilot timing artifact conflicts with an existing target",
            str(target),
        )
    raise OSError(error_number, os.strerror(error_number), str(target))


def _write_manifest_file(path: Path, content: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _assert_create_only_target(target: Path) -> None:
    if target.is_symlink():
        raise FileExistsError(
            f"refusing create-only P05 pilot timing export through symlink: {target}"
        )
    if target.exists():
        raise FileExistsError(
            f"P05 pilot timing artifact conflicts with an existing target: {target}"
        )


def _write_create_only_artifact(
    target: Path,
    semantic_manifest: Mapping[str, Any],
) -> P05PilotTimingResult:
    _assert_create_only_target(target)
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError(f"P05 pilot timing parent must be a real directory: {parent}")
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=str(parent),
        )
    )
    try:
        semantic_sha256 = _sha256_bytes(_canonical_json_bytes(semantic_manifest))
        manifest = {
            **semantic_manifest,
            "content": {"semantic_sha256": semantic_sha256},
        }
        manifest_bytes = (
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        _write_manifest_file(temporary / MANIFEST_NAME, manifest_bytes)
        _fsync_directory(temporary)
        _rename_directory_noreplace(temporary, target)
        _fsync_directory(parent)
        manifest_path = target / MANIFEST_NAME
        return P05PilotTimingResult(
            package_dir=target,
            manifest_path=manifest_path,
            semantic_sha256=semantic_sha256,
            manifest_sha256=_sha256_file(manifest_path),
            status="created",
        )
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


class P05PilotTimingCallback(pl.Callback):
    """Collect synchronized full-epoch CUDA timing for one frozen pilot.

    In Lightning 2.3, ``on_train_epoch_end`` follows the epoch's scheduled
    validation loop. The synchronized start/end hooks therefore bound one
    complete fit epoch rather than only its training batches.
    """

    def __init__(
        self,
        package_dir: str | Path,
        *,
        cuda_api: Any = torch.cuda,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        super().__init__()
        self.package_dir = Path(os.path.abspath(os.fspath(package_dir)))
        self._cuda = cuda_api
        self._clock = clock
        self._device: torch.device | None = None
        self._fit_started_at: float | None = None
        self._active_epoch: int | None = None
        self._epoch_started_at: float | None = None
        self._startup_seconds: float | None = None
        self._epoch_seconds: list[float] = []
        self._finished = False
        self.result: P05PilotTimingResult | None = None

    def _now(self, *, boundary: str) -> float:
        try:
            raw_value = self._clock()
            if isinstance(raw_value, bool):
                raise TypeError("boolean clocks are invalid")
            value = float(raw_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise P05PilotTimingError(
                f"pilot clock returned an invalid value at {boundary}"
            ) from exc
        if not math.isfinite(value):
            raise P05PilotTimingError(
                f"pilot clock returned a non-finite value at {boundary}"
            )
        return value

    def _synchronize(self, *, boundary: str) -> None:
        if self._device is None:
            raise P05PilotTimingError("pilot CUDA device was not initialized")
        try:
            self._cuda.synchronize(self._device)
        except Exception as exc:
            raise P05PilotTimingError(
                f"CUDA synchronization failed at {boundary}"
            ) from exc

    @staticmethod
    def _current_epoch(trainer: "pl.Trainer") -> int:
        epoch = getattr(trainer, "current_epoch", None)
        if type(epoch) is not int or epoch < 0:
            raise P05PilotTimingError("trainer.current_epoch must be a non-negative integer")
        return epoch

    def on_fit_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
    ) -> None:
        if self._fit_started_at is not None or self._finished:
            raise P05PilotTimingError("P05 pilot timing callback cannot be reused")
        _assert_create_only_target(self.package_dir)
        max_epochs = getattr(trainer, "max_epochs", None)
        if type(max_epochs) is not int or max_epochs != EXPECTED_EPOCHS:
            raise P05PilotTimingError(
                f"P05 pilot timing requires exactly {EXPECTED_EPOCHS} configured epochs"
            )
        world_size = getattr(trainer, "world_size", 1)
        if type(world_size) is not int or world_size != 1:
            raise P05PilotTimingError("P05 pilot timing requires one training process")
        try:
            device = torch.device(getattr(pl_module, "device", None))
        except (TypeError, RuntimeError) as exc:
            raise P05PilotTimingError("P05 pilot timing requires a concrete CUDA device") from exc
        if device.type != "cuda":
            raise P05PilotTimingError("P05 pilot timing forbids non-CUDA execution")
        try:
            cuda_available = self._cuda.is_available()
        except Exception as exc:
            raise P05PilotTimingError("P05 pilot CUDA availability query failed") from exc
        if cuda_available is not True:
            raise P05PilotTimingError("P05 pilot timing requires CUDA availability")
        self._device = device
        try:
            self._cuda.reset_peak_memory_stats(device)
        except Exception as exc:
            raise P05PilotTimingError("P05 pilot peak-memory reset failed") from exc
        self._fit_started_at = self._now(boundary="fit_start")

    def on_train_epoch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
    ) -> None:
        del pl_module
        if self._fit_started_at is None:
            raise P05PilotTimingError("pilot epoch started before fit timing initialization")
        if self._active_epoch is not None or self._epoch_started_at is not None:
            raise P05PilotTimingError("pilot epoch start occurred before the prior epoch ended")
        epoch = self._current_epoch(trainer)
        expected_epoch = len(self._epoch_seconds)
        if epoch != expected_epoch or epoch >= EXPECTED_EPOCHS:
            raise P05PilotTimingError(
                f"pilot epoch order mismatch: expected {expected_epoch}, got {epoch}"
            )
        self._synchronize(boundary=f"epoch_{epoch + 1}_start")
        started_at = self._now(boundary=f"epoch_{epoch + 1}_start")
        if epoch == 0:
            startup_seconds = started_at - self._fit_started_at
            if not math.isfinite(startup_seconds) or startup_seconds < 0.0:
                raise P05PilotTimingError("pilot startup_seconds must be finite and non-negative")
            self._startup_seconds = startup_seconds
        self._active_epoch = epoch
        self._epoch_started_at = started_at

    def on_train_epoch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
    ) -> None:
        del pl_module
        epoch = self._current_epoch(trainer)
        if self._active_epoch != epoch or self._epoch_started_at is None:
            raise P05PilotTimingError(
                f"pilot epoch {epoch + 1} ended without its synchronized start"
            )
        self._synchronize(boundary=f"epoch_{epoch + 1}_end")
        ended_at = self._now(boundary=f"epoch_{epoch + 1}_end")
        duration = ended_at - self._epoch_started_at
        if not math.isfinite(duration) or duration <= 0.0:
            raise P05PilotTimingError("pilot epoch_seconds must be finite and positive")
        self._epoch_seconds.append(duration)
        self._active_epoch = None
        self._epoch_started_at = None

    def _peak_memory(self) -> tuple[int, int]:
        if self._device is None:
            raise P05PilotTimingError("pilot CUDA device was not initialized")
        try:
            allocated = self._cuda.max_memory_allocated(self._device)
            reserved = self._cuda.max_memory_reserved(self._device)
        except Exception as exc:
            raise P05PilotTimingError("P05 pilot peak-memory query failed") from exc
        if type(allocated) is not int or allocated < 0:
            raise P05PilotTimingError("peak allocated CUDA memory must be non-negative bytes")
        if type(reserved) is not int or reserved < allocated:
            raise P05PilotTimingError(
                "peak reserved CUDA memory must be integer bytes and at least allocated"
            )
        return allocated, reserved

    def _semantic_manifest(self, *, allocated: int, reserved: int) -> dict[str, Any]:
        if self._device is None or self._startup_seconds is None:
            raise P05PilotTimingError("P05 pilot timing state is incomplete")
        epoch_seconds = list(self._epoch_seconds)
        median_seconds = float(statistics.median(epoch_seconds[1:]))
        return {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "paper_id": "P05",
            "artifact_class": "engineering_pilot_timing",
            "measurement_status": "complete",
            "evidence_eligible": False,
            "claim_support": "forbidden",
            "cuda_device": str(self._device),
            "timing_contract": {
                "clock": "time.perf_counter_monotonic_seconds",
                "startup_boundary": (
                    "on_fit_start_to_cuda_synchronized_on_train_epoch_start_epoch_1"
                ),
                "epoch_boundary": (
                    "cuda_synchronized_on_train_epoch_start_to_"
                    "cuda_synchronized_on_train_epoch_end"
                ),
                "epoch_end_includes_scheduled_validation": True,
                "expected_complete_epochs": EXPECTED_EPOCHS,
                "median_epoch_numbers": [2, 3, 4, 5],
                "memory_reset_boundary": "on_fit_start_before_epoch_1",
                "memory_source": {
                    "allocated": "torch.cuda.max_memory_allocated",
                    "reserved": "torch.cuda.max_memory_reserved",
                },
                "memory_unit": "bytes",
            },
            "measurements": {
                "startup_seconds": self._startup_seconds,
                "epoch_seconds_1_through_5": epoch_seconds,
                "median_epoch_seconds_2_through_5": median_seconds,
                "peak_allocated_memory": allocated,
                "peak_reserved_memory": reserved,
            },
        }

    def on_fit_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
    ) -> None:
        del pl_module
        if self._finished:
            raise P05PilotTimingError("P05 pilot timing artifact was already finalized")
        if self._active_epoch is not None or self._epoch_started_at is not None:
            raise P05PilotTimingError("P05 pilot ended with an incomplete active epoch")
        if self._startup_seconds is None or len(self._epoch_seconds) != EXPECTED_EPOCHS:
            raise P05PilotTimingError(
                "P05 pilot timing requires exactly five complete epoch measurements"
            )
        allocated, reserved = self._peak_memory()
        semantic_manifest = self._semantic_manifest(
            allocated=allocated,
            reserved=reserved,
        )
        result = _write_create_only_artifact(self.package_dir, semantic_manifest)
        self.result = result
        self._finished = True
        setattr(trainer, "p05_pilot_timing_result", result)


def build_p05_pilot_timing_callback(
    args_trainer: Any,
    run_dir: str | Path,
) -> P05PilotTimingCallback | None:
    """Build the callback only for a literal ``p05_pilot_mode=true`` flag."""

    if not p05_pilot_mode_enabled(args_trainer):
        return None
    num_epochs = getattr(args_trainer, "num_epochs", None)
    if type(num_epochs) is not int or num_epochs != EXPECTED_EPOCHS:
        raise P05PilotTimingError(
            f"trainer.num_epochs must be exactly {EXPECTED_EPOCHS} in P05 pilot mode"
        )
    if getattr(args_trainer, "device", None) != "cuda":
        raise P05PilotTimingError("trainer.device must be exactly 'cuda' in P05 pilot mode")
    if getattr(args_trainer, "early_stopping", False) is not False:
        raise P05PilotTimingError("trainer.early_stopping must be false in P05 pilot mode")
    package_dir = Path(run_dir) / "artifacts" / PACKAGE_NAME
    return P05PilotTimingCallback(package_dir)


__all__ = [
    "EXPECTED_EPOCHS",
    "P05PilotTimingCallback",
    "P05PilotTimingError",
    "P05PilotTimingResult",
    "build_p05_pilot_timing_callback",
    "p05_pilot_mode_enabled",
]

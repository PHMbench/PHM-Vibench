"""Fail-closed runtime and artifact helpers for the P08 evidence pipeline.

These helpers deliberately avoid importing torch at module import time.  The
GPU preflight is intended to run before a trainer initializes CUDA so that the
visible-to-physical device map remains inspectable and unambiguous.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


FORBIDDEN_PHYSICAL_GPU_INDICES = frozenset({2})
ALLOWED_PHYSICAL_GPU_INDICES = frozenset({0, 1, 3, 4, 5, 6, 7})
_DISTRIBUTED_STRATEGY_MARKERS = ("ddp", "fsdp", "deepspeed")


@dataclass(frozen=True)
class DevicePreflightRecord:
    """Machine-readable result of the P08 single-device guard."""

    status: str
    mode: str
    physical_gpu_indices: tuple[int, ...]
    visible_to_physical_gpu_map: dict[str, int]
    cuda_visible_devices: str
    cuda_device_count: int
    cuda_device_names: tuple[str, ...]
    world_size: int
    local_world_size: int
    trainer_strategy: str
    multi_gpu: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _positive_int_from_environment(
    environment: Mapping[str, str],
    name: str,
    *,
    default: int,
) -> int:
    raw = str(environment.get(name, default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 1:
        raise RuntimeError(f"{name} must be at least 1, got {value}")
    return value


def _probe_cuda() -> tuple[int, tuple[str, ...]]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - the evidence environment has torch
        raise RuntimeError("torch is required to probe the CUDA runtime") from exc

    count = int(torch.cuda.device_count())
    names = tuple(str(torch.cuda.get_device_name(index)) for index in range(count))
    return count, names


def strict_single_gpu_preflight(
    *,
    trainer_strategy: str = "auto",
    require_gpu: bool = True,
    environment: Mapping[str, str] | None = None,
    cuda_device_count: int | None = None,
    cuda_device_names: Sequence[str] | None = None,
) -> DevicePreflightRecord:
    """Validate and record the frozen P08 single-device execution policy.

    A CUDA run must expose exactly one numeric physical index through
    ``CUDA_VISIBLE_DEVICES``.  An absent mask, UUID/MIG selector, multiple
    selectors, physical GPU 2, distributed world size, or distributed trainer
    strategy fails before CUDA work starts.  CPU-only utility jobs are allowed
    only when ``require_gpu=False`` and the mask is explicitly empty or ``-1``.

    ``cuda_device_count`` and ``cuda_device_names`` are injectable so the guard
    can be tested without a GPU.  In production they should be omitted.
    """

    env = dict(os.environ if environment is None else environment)
    world_size = _positive_int_from_environment(env, "WORLD_SIZE", default=1)
    local_world_size = _positive_int_from_environment(
        env, "LOCAL_WORLD_SIZE", default=1
    )
    if world_size > 1 or local_world_size > 1:
        raise RuntimeError(
            "P08 forbids multi-process/multi-GPU execution: "
            f"WORLD_SIZE={world_size}, LOCAL_WORLD_SIZE={local_world_size}"
        )

    strategy = str(trainer_strategy).strip() or "auto"
    strategy_lower = strategy.lower()
    if any(marker in strategy_lower for marker in _DISTRIBUTED_STRATEGY_MARKERS):
        raise RuntimeError(f"P08 forbids distributed trainer strategy {strategy!r}")

    if "CUDA_VISIBLE_DEVICES" not in env:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must be set explicitly so the physical GPU map "
            "can be audited"
        )
    visible_spec = str(env["CUDA_VISIBLE_DEVICES"]).strip()
    cpu_declared = visible_spec in {"", "-1"}

    if cpu_declared:
        physical_indices: tuple[int, ...] = ()
        if require_gpu:
            raise RuntimeError("P08 evidence run requires exactly one visible GPU")
    else:
        tokens = [token.strip() for token in visible_spec.split(",")]
        if len(tokens) != 1 or not tokens[0]:
            raise RuntimeError(
                "P08 requires exactly one CUDA_VISIBLE_DEVICES entry, got "
                f"{visible_spec!r}"
            )
        if not tokens[0].isascii() or not tokens[0].isdigit():
            raise RuntimeError(
                "P08 requires a numeric physical GPU index; UUID/MIG selectors "
                "cannot establish the frozen physical-index exclusion"
            )
        physical_indices = (int(tokens[0]),)
        if physical_indices[0] in FORBIDDEN_PHYSICAL_GPU_INDICES:
            raise RuntimeError("physical GPU index 2 is forbidden for P08")
        if physical_indices[0] not in ALLOWED_PHYSICAL_GPU_INDICES:
            raise RuntimeError(
                f"physical GPU index {physical_indices[0]} is not in the frozen "
                f"allowed set {sorted(ALLOWED_PHYSICAL_GPU_INDICES)}"
            )

    if cuda_device_count is None:
        observed_count, observed_names = _probe_cuda()
    else:
        observed_count = int(cuda_device_count)
        if observed_count < 0:
            raise ValueError("cuda_device_count cannot be negative")
        observed_names = tuple(str(name) for name in (cuda_device_names or ()))

    expected_count = 0 if cpu_declared else 1
    if observed_count != expected_count:
        raise RuntimeError(
            "CUDA runtime visibility disagrees with the declared single-device "
            f"mask: expected {expected_count}, observed {observed_count}"
        )
    if observed_names and len(observed_names) != observed_count:
        raise ValueError(
            "cuda_device_names length must equal cuda_device_count when supplied"
        )

    mapping = (
        {} if cpu_declared else {"0": int(physical_indices[0])}
    )
    return DevicePreflightRecord(
        status="pass",
        mode="cpu" if cpu_declared else "cuda",
        physical_gpu_indices=physical_indices,
        visible_to_physical_gpu_map=mapping,
        cuda_visible_devices=visible_spec,
        cuda_device_count=observed_count,
        cuda_device_names=observed_names,
        world_size=world_size,
        local_world_size=local_world_size,
        trainer_strategy=strategy,
        multi_gpu=False,
    )


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.is_file():
        raise FileNotFoundError(f"artifact does not exist: {candidate}")
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except ImportError:  # pragma: no cover - numpy is required by the metrics module
        pass
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    ).encode("utf-8")


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(
    path: str | Path,
    content: bytes,
    *,
    replace: bool = False,
) -> Path:
    """Durably replace one file using a same-directory temporary file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not replace:
        raise FileExistsError(f"refusing to overwrite existing artifact: {target}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if target.exists() and not replace:
            raise FileExistsError(f"refusing to overwrite existing artifact: {target}")
        os.replace(temporary_name, target)
        _fsync_directory(target.parent)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return target


def atomic_write_text(
    path: str | Path,
    content: str,
    *,
    replace: bool = False,
) -> Path:
    return atomic_write_bytes(path, content.encode("utf-8"), replace=replace)


def atomic_write_json(
    path: str | Path,
    value: Any,
    *,
    replace: bool = False,
) -> Path:
    payload = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    )
    return atomic_write_text(path, payload + "\n", replace=replace)


class EvidenceWriter:
    """Constrain atomic evidence writes to one run artifact directory."""

    def __init__(self, run_root: str | Path) -> None:
        root = Path(run_root)
        root.mkdir(parents=True, exist_ok=True)
        self.run_root = root.resolve()

    def _target(self, relative_path: str | Path) -> Path:
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"artifact path must stay below run root: {relative}")
        target = self.run_root / relative
        resolved_parent = target.parent.resolve()
        if not resolved_parent.is_relative_to(self.run_root):
            raise ValueError(f"artifact path escapes run root: {relative}")
        return target

    def write_bytes(
        self, relative_path: str | Path, content: bytes, *, replace: bool = False
    ) -> tuple[Path, str]:
        target = atomic_write_bytes(
            self._target(relative_path), content, replace=replace
        )
        return target, sha256_file(target)

    def write_text(
        self, relative_path: str | Path, content: str, *, replace: bool = False
    ) -> tuple[Path, str]:
        return self.write_bytes(relative_path, content.encode("utf-8"), replace=replace)

    def write_json(
        self, relative_path: str | Path, value: Any, *, replace: bool = False
    ) -> tuple[Path, str]:
        target = atomic_write_json(
            self._target(relative_path), value, replace=replace
        )
        return target, sha256_file(target)

    def write_jsonl(
        self,
        relative_path: str | Path,
        rows: Iterable[Mapping[str, Any]],
        *,
        replace: bool = False,
    ) -> tuple[Path, str]:
        lines = [canonical_json_bytes(dict(row)).decode("utf-8") for row in rows]
        content = "\n".join(lines) + ("\n" if lines else "")
        return self.write_text(relative_path, content, replace=replace)

    def write_provenance(
        self,
        provenance: Mapping[str, Any],
        *,
        relative_path: str | Path = "provenance.json",
        replace: bool = False,
    ) -> tuple[Path, str]:
        """Validate minimum execution provenance before writing it."""

        payload = dict(provenance)
        required = {
            "command",
            "conda_environment",
            "git_commit",
            "config_sha256",
            "data_sha256",
            "gpu_preflight",
        }
        missing = sorted(required.difference(payload))
        if missing:
            raise ValueError(f"provenance is missing required fields: {missing}")
        if payload["conda_environment"] != "LQ_signal":
            raise ValueError("P08 evidence provenance requires conda environment LQ_signal")
        command_value = payload["command"]
        if not isinstance(command_value, str) or not command_value.strip():
            raise ValueError("provenance command must be a non-empty string")
        command = command_value
        if "conda run" not in command or "-n LQ_signal" not in command:
            raise ValueError(
                "P08 evidence command must include 'conda run -n LQ_signal'"
            )
        for digest_name in ("config_sha256", "data_sha256"):
            digest = str(payload[digest_name]).lower()
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"{digest_name} must be a complete SHA-256 digest")
        commit = str(payload["git_commit"]).lower()
        if not 7 <= len(commit) <= 64 or any(
            character not in "0123456789abcdef" for character in commit
        ):
            raise ValueError("git_commit must be a hexadecimal Git object ID")
        preflight = payload["gpu_preflight"]
        if isinstance(preflight, DevicePreflightRecord):
            preflight = preflight.to_dict()
            payload["gpu_preflight"] = preflight
        if not isinstance(preflight, Mapping):
            raise ValueError("gpu_preflight provenance must be a mapping")
        if preflight.get("status") != "pass" or preflight.get("multi_gpu") is not False:
            raise ValueError("gpu_preflight must record a passing non-multi-GPU guard")
        if int(preflight.get("world_size", 0)) != 1 or int(
            preflight.get("local_world_size", 0)
        ) != 1:
            raise ValueError("gpu_preflight provenance must record unit world sizes")
        strategy = str(preflight.get("trainer_strategy", "")).lower()
        if any(marker in strategy for marker in _DISTRIBUTED_STRATEGY_MARKERS):
            raise ValueError("gpu_preflight provenance contains a distributed strategy")
        physical = tuple(int(index) for index in preflight.get("physical_gpu_indices", ()))
        if len(physical) > 1:
            raise ValueError("gpu_preflight provenance contains multiple physical GPUs")
        if any(index in FORBIDDEN_PHYSICAL_GPU_INDICES for index in physical):
            raise ValueError("provenance contains forbidden physical GPU index 2")
        if any(index not in ALLOWED_PHYSICAL_GPU_INDICES for index in physical):
            raise ValueError("gpu_preflight provenance contains an unapproved physical GPU")
        mode = preflight.get("mode")
        if (mode == "cuda" and len(physical) != 1) or (
            mode == "cpu" and physical
        ):
            raise ValueError("gpu_preflight mode and physical GPU list disagree")
        if mode not in {"cpu", "cuda"}:
            raise ValueError("gpu_preflight mode must be 'cpu' or 'cuda'")
        expected_map = {} if not physical else {"0": physical[0]}
        if dict(preflight.get("visible_to_physical_gpu_map", {})) != expected_map:
            raise ValueError("gpu_preflight visible-to-physical map is inconsistent")
        return self.write_json(relative_path, payload, replace=replace)

    def write_sha256_manifest(
        self,
        *,
        relative_path: str | Path = "artifact_manifest.sha256",
        exclude: Iterable[str | Path] = (),
        replace: bool = False,
    ) -> tuple[Path, str]:
        """Hash every regular artifact except the manifest itself."""

        manifest_target = self._target(relative_path)
        excluded = {Path(item).as_posix() for item in exclude}
        excluded.add(manifest_target.relative_to(self.run_root).as_posix())
        entries: list[tuple[str, str]] = []
        for candidate in sorted(self.run_root.rglob("*")):
            if candidate.is_symlink():
                raise ValueError(f"symlinks are not valid evidence artifacts: {candidate}")
            if not candidate.is_file():
                continue
            rel = candidate.relative_to(self.run_root).as_posix()
            if rel in excluded or candidate.name.endswith(".tmp"):
                continue
            if "\n" in rel or "\r" in rel:
                raise ValueError(f"artifact name contains a newline: {rel!r}")
            entries.append((rel, sha256_file(candidate)))
        content = "".join(f"{digest}  {rel}\n" for rel, digest in entries)
        return self.write_text(relative_path, content, replace=replace)

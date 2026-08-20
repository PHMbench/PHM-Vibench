"""Shared device-selection contract for preflight and Trainer construction.

The module has no training-framework import at module load time. CPU-only preflight can
therefore validate a configuration without importing PyTorch Lightning or the Trainer
Factory package. Torch is loaded only when hardware inspection is actually required.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any


DEVICE_MODES = ("cpu", "cuda", "auto")


def _load_torch():
    """Import torch only when CUDA/MPS inspection is required."""

    import torch

    return torch


def _available_auto_accelerator() -> tuple[str, int | None]:
    """Return the accelerator selected by an explicit ``device=auto`` request."""

    torch = _load_torch()
    if torch.cuda.is_available():
        return "gpu", int(torch.cuda.device_count())

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and callable(getattr(mps, "is_available", None)):
        if bool(mps.is_available()):
            return "mps", 1

    return "cpu", None


def resolve_device_request(args_trainer: Any) -> tuple[str, int]:
    """Resolve one explicit trainer device request without silent fallback.

    ``cpu`` and ``cuda`` are exact requests. ``auto`` performs hardware inspection only
    because the user explicitly selected it. A CPU request is resolved without importing
    torch, which keeps config-only inspection independent of the training stack.
    """

    if not hasattr(args_trainer, "device"):
        raise ValueError(
            "trainer.device is required and must be one of: cpu, cuda, auto"
        )
    requested = str(args_trainer.device).strip().lower()
    if requested not in DEVICE_MODES:
        raise ValueError(
            f"unsupported trainer.device {args_trainer.device!r}; expected one of: "
            + ", ".join(DEVICE_MODES)
        )

    devices = getattr(
        args_trainer,
        "devices",
        getattr(args_trainer, "gpus", 1),
    )
    if isinstance(devices, bool) or not isinstance(devices, Integral) or devices < 1:
        raise ValueError(
            "trainer.devices/trainer.gpus must be a positive integer, "
            f"got {devices!r}"
        )
    devices = int(devices)

    if requested == "cpu":
        return "cpu", devices

    if requested == "auto":
        accelerator, available = _available_auto_accelerator()
        if available is not None and devices > available:
            raise RuntimeError(
                "trainer.device='auto' selected an accelerator with fewer devices "
                f"than requested: accelerator={accelerator}, requested={devices}, "
                f"available={available}."
            )
        return accelerator, devices

    torch = _load_torch()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "trainer.device='cuda' was requested, but CUDA is unavailable. "
            "Set trainer.device=cpu or repair the CUDA runtime; no CPU fallback "
            "was applied."
        )
    available_devices = int(torch.cuda.device_count())
    if devices > available_devices:
        raise RuntimeError(
            "trainer.device='cuda' requested more devices than are available: "
            f"requested={devices}, available={available_devices}."
        )
    return "gpu", devices


__all__ = ["DEVICE_MODES", "resolve_device_request"]

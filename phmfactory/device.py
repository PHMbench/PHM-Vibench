"""Resolve the one public device request used by preflight and Trainer creation.

CPU resolution does not import the training stack. CUDA and ``auto`` inspect hardware
only because the user explicitly requested them. The resolver never changes the requested
mode, falls back to another device, or reconciles legacy aliases.
"""

from __future__ import annotations

from typing import Any


DEVICE_MODES = ("cpu", "cuda", "auto")


def _load_torch():
    """Import torch only when hardware inspection is required."""

    import torch

    return torch


def _available_auto_accelerator() -> tuple[str, int | None]:
    """Return the accelerator selected by an explicit ``device=auto`` request."""

    torch = _load_torch()
    if torch.cuda.is_available():
        return "gpu", int(torch.cuda.device_count())

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and callable(getattr(mps, "is_available", None)):
        if mps.is_available():
            return "mps", 1

    return "cpu", None


def resolve_device_request(args_trainer: Any) -> tuple[str, int]:
    """Resolve one exact device mode and one exact positive device count.

    Maintained configurations use only ``trainer.device`` and ``trainer.devices``.
    ``trainer.gpus`` is rejected rather than translated, because translation would leave
    two authorities for the same runtime decision.
    """

    if hasattr(args_trainer, "gpus"):
        raise ValueError(
            "trainer.gpus is unsupported; replace it with trainer.devices"
        )

    if not hasattr(args_trainer, "device"):
        raise ValueError(
            "trainer.device is required and must be one of: cpu, cuda, auto"
        )
    requested = args_trainer.device
    if not isinstance(requested, str):
        raise TypeError(
            "trainer.device must be a string chosen from: cpu, cuda, auto; "
            f"got {type(requested).__name__}"
        )
    if requested not in DEVICE_MODES:
        raise ValueError(
            f"unsupported trainer.device {requested!r}; expected one of: "
            + ", ".join(DEVICE_MODES)
        )

    if not hasattr(args_trainer, "devices"):
        raise ValueError("trainer.devices is required and must be a positive integer")
    devices = args_trainer.devices
    if isinstance(devices, bool) or not isinstance(devices, int):
        raise TypeError(
            "trainer.devices must be an integer, "
            f"got {type(devices).__name__}"
        )
    if devices < 1:
        raise ValueError(
            f"trainer.devices must be positive, got {devices}"
        )

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

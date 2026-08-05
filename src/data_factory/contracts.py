"""Small runtime contracts for user-visible PHMFactory data loaders."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def require_nonempty_dataloaders(
    data_factory: Any,
    args_task: Any,
    args_data: Any,
    splits: Iterable[str] = ("train", "val", "test"),
) -> dict[str, int]:
    """Return loader batch counts or fail before model construction.

    The check uses ``len(loader)`` only. It does not consume a batch, change
    sampler order, inspect tensors, or impose a universal batch schema.
    """

    task_type = str(getattr(args_task, "type", "unknown"))
    task_name = str(getattr(args_task, "name", "unknown"))
    batch_size = getattr(args_data, "batch_size", "unknown")
    counts: dict[str, int] = {}

    for split in splits:
        loader = data_factory.get_dataloader(split)
        if loader is None:
            raise RuntimeError(
                f"Data contract failed for {task_type}/{task_name}: "
                f"{split} loader is None. Check the dataset adapter and sampler."
            )
        try:
            batch_count = int(len(loader))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Data contract failed for {task_type}/{task_name}: cannot "
                f"determine the number of {split} batches from "
                f"{type(loader).__name__}."
            ) from exc

        if batch_count <= 0:
            raise RuntimeError(
                f"Data contract failed for {task_type}/{task_name}: {split} "
                f"loader has 0 batches. Check selected IDs, split ratios, "
                f"data.window_size, data.num_window, and "
                f"data.batch_size={batch_size}."
            )
        counts[str(split)] = batch_count

    return counts


def format_loader_summary(counts: dict[str, int]) -> str:
    """Return a compact user-readable loader summary."""

    return ", ".join(
        f"{split}={count} batch{'es' if count != 1 else ''}"
        for split, count in counts.items()
    )


__all__ = ["format_loader_summary", "require_nonempty_dataloaders"]

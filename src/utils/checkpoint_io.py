from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Iterable, Optional, Union

import torch


def _path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _trusted_checkpoint_roots(
    extra_roots: Optional[Iterable[Union[str, Path]]] = None,
) -> list[Path]:
    roots = [Path.cwd()]
    env_roots = os.environ.get("PHM_TRUSTED_CHECKPOINT_ROOTS", "")
    if env_roots:
        roots.extend(Path(item) for item in env_roots.split(os.pathsep) if item)
    if extra_roots:
        roots.extend(Path(item) for item in extra_roots)
    return [root.expanduser().resolve() for root in roots]


def resolve_trusted_checkpoint_path(
    checkpoint_path: Union[str, Path],
    trusted_roots: Optional[Iterable[Union[str, Path]]] = None,
) -> Path:
    """Resolve a checkpoint path and ensure it is under an allowed root."""
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: {path}")

    roots = _trusted_checkpoint_roots(trusted_roots)
    if not any(path == root or _path_is_relative_to(path, root) for root in roots):
        allowed = ", ".join(str(root) for root in roots)
        raise ValueError(
            f"Refusing to load checkpoint outside trusted roots: {path}. "
            f"Set PHM_TRUSTED_CHECKPOINT_ROOTS to opt in. Trusted roots: {allowed}"
        )
    return path


def safe_torch_load(
    checkpoint_path: Union[str, Path],
    *,
    map_location="cpu",
    trusted_roots: Optional[Iterable[Union[str, Path]]] = None,
):
    """Load a trusted local torch file, preferring weights-only deserialization."""
    path = resolve_trusted_checkpoint_path(checkpoint_path, trusted_roots)
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # Older PyTorch versions do not support weights_only.
        return torch.load(path, map_location=map_location)
    except pickle.UnpicklingError:
        # Lightning checkpoints can contain metadata unsupported by weights_only.
        # The fallback is allowed only after trusted-root validation above.
        return torch.load(path, map_location=map_location, weights_only=False)

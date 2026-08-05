"""Utilities for instantiating models from configuration."""

from __future__ import annotations

import importlib
import os
from typing import Any, Mapping

import torch

from ..utils.utils import get_num_classes


def resolve_model_module(args_model: Any) -> str:
    """Return the Python import path for the model module."""
    return f"src.model_factory.{args_model.type}.{args_model.name}"


def model_factory(args_model: Any, metadata: Any):
    """Instantiate a model by name and load an explicitly configured checkpoint.

    A configured checkpoint is part of the requested experiment. If it cannot be
    loaded, model construction fails instead of continuing with random or partial
    initialization.
    """
    if not getattr(args_model, "num_classes", None):
        inferred = get_num_classes(metadata)
        if isinstance(inferred, dict):
            args_model.num_classes = (
                next(iter(inferred.values())) if len(inferred) == 1 else inferred
            )
        else:
            args_model.num_classes = inferred

    module_path = resolve_model_module(args_model)
    model_module = importlib.import_module(module_path)
    model_cls = model_module.Model

    try:
        model = model_cls(args_model, metadata)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to create model '{args_model.type}.{args_model.name}': {exc}"
        ) from exc

    weights_path = getattr(args_model, "weights_path", None)
    if weights_path:
        strict = bool(getattr(args_model, "weights_strict", True))
        try:
            load_ckpt(model, weights_path, strict=strict)
        except FileNotFoundError:
            raise
        except Exception as exc:
            suggestion = (
                "Check that the checkpoint belongs to this model. "
                "For intentional transfer learning with a compatible subset of "
                "parameters, set model.weights_strict=false."
            )
            raise RuntimeError(
                f"Failed to load checkpoint '{weights_path}' for model "
                f"'{args_model.type}.{args_model.name}': {exc}. {suggestion}"
            ) from exc

    return model


def _extract_state_dict(checkpoint: Any, ckpt_path: str) -> Mapping[str, Any]:
    """Return a state dict from a plain or Lightning-style checkpoint."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            f"Checkpoint '{ckpt_path}' must contain a mapping, "
            f"got {type(checkpoint).__name__}."
        )

    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, Mapping):
        raise TypeError(
            f"Checkpoint '{ckpt_path}' contains a non-mapping state_dict."
        )
    return state_dict


def load_ckpt(model: Any, ckpt_path: str, *, strict: bool = True) -> None:
    """Load ``ckpt_path`` into ``model``.

    Strict loading is the default. Non-strict loading is intended only for
    explicit transfer-learning use and still requires at least one compatible
    parameter.
    """
    checkpoint_path = os.fspath(ckpt_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Configured checkpoint does not exist or is not a file: "
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = _extract_state_dict(checkpoint, checkpoint_path)

    if strict:
        model.load_state_dict(state_dict, strict=True)
        return

    model_state = model.state_dict()
    matched = {}
    skipped = []

    for name, parameter in state_dict.items():
        if name not in model_state:
            skipped.append((name, "not present in model"))
            continue

        checkpoint_shape = getattr(parameter, "shape", None)
        model_shape = getattr(model_state[name], "shape", None)
        if checkpoint_shape != model_shape:
            skipped.append(
                (name, f"shape {checkpoint_shape} does not match {model_shape}")
            )
            continue

        matched[name] = parameter

    if not matched:
        raise RuntimeError(
            f"Checkpoint '{checkpoint_path}' matched zero model parameters."
        )

    model.load_state_dict(matched, strict=False)

    if skipped:
        print(
            f"Loaded {len(matched)} compatible parameters from "
            f"'{checkpoint_path}'; skipped {len(skipped)} incompatible entries."
        )

"""Utilities for instantiating models from configuration."""

from __future__ import annotations

import importlib
import os
from typing import Any, Mapping

import torch

from ..utils.label_ontology import validate_metadata_label_ontology
from ..utils.utils import get_num_classes


def resolve_model_module(args_model: Any) -> str:
    """Return the Python import path for the model module."""
    return f"src.model_factory.{args_model.type}.{args_model.name}"


def model_factory(args_model: Any, metadata: Any):
    """Instantiate a model by name and load an explicitly configured checkpoint.

    Model import, construction, and checkpoint failures retain their original
    exception type and traceback. A configured checkpoint is part of the requested
    experiment. If it cannot be loaded, model construction fails instead of continuing
    with random or partial initialization.
    """
    # Validate every label ontology that is actually supplied, even when
    # num_classes was configured manually. Some isolated model/checkpoint uses
    # intentionally provide no metadata and an explicit output width; in that
    # case there is no ontology to validate or silently reinterpret.
    if metadata is not None:
        validate_metadata_label_ontology(
            metadata,
            group_field="Dataset_id",
            require_labels=False,
        )

    if not getattr(args_model, "num_classes", None):
        if metadata is None:
            raise ValueError(
                "model.num_classes is required when model construction receives "
                "no metadata"
            )
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
    model = model_cls(args_model, metadata)

    weights_path = getattr(args_model, "weights_path", None)
    if weights_path:
        strict = getattr(args_model, "weights_strict", True)
        if not isinstance(strict, bool):
            raise TypeError(
                "model.weights_strict must be a boolean; use true for exact "
                "checkpoint loading or false for an explicitly compatible subset"
            )
        load_ckpt(model, weights_path, strict=strict)

    return model


def _extract_state_dict(checkpoint: Any, ckpt_path: str) -> Mapping[str, Any]:
    """Return bare-model weights from a plain or PHMFactory Lightning checkpoint."""
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

    # PHMFactory task modules register the bare model as ``self.network``.
    # Support that one canonical Lightning key space without guessing unrelated
    # prefixes such as module., model., backbone., or encoder.
    network_state = {
        key.removeprefix("network."): value
        for key, value in state_dict.items()
        if key.startswith("network.")
    }
    return network_state or state_dict


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

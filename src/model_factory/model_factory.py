"""Utilities for instantiating models from configuration."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from ..utils.utils import get_num_classes
from ..utils.registry import Registry




def resolve_model_module(args_model: Any) -> str:
    """Return the Python import path for the model module."""
    return f"src.model_factory.{args_model.type}.{args_model.name}"


def model_factory(args_model: Any, metadata: Any):
    """Instantiate a model by name.

    Parameters
    ----------
    args_model : Namespace
        Configuration namespace with at least ``name`` and ``type``
        fields. Other attributes are passed to the model's ``Model``
        constructor.
    metadata : Any
        Dataset metadata, used here only to compute ``num_classes``.

    Returns
    -------
    nn.Module
        Instantiated model ready for training.
    """
    # Respect an explicit `num_classes` from config (common for classification models).
    # Otherwise infer from metadata; if only one dataset_id is present, collapse to an int.
    if not getattr(args_model, "num_classes", None):
        inferred = get_num_classes(metadata)
        if isinstance(inferred, dict):
            args_model.num_classes = next(iter(inferred.values())) if len(inferred) == 1 else inferred
        else:
            args_model.num_classes = inferred
    # key = f"{args_model.type}.{args_model.name}"


    module_path = resolve_model_module(args_model)
    model_module = importlib.import_module(module_path)
    model_cls = model_module.Model

    try:
        model = model_cls(args_model, metadata)
        
        weights_path = getattr(args_model, "weights_path", None)
        if weights_path:
            load_ckpt(model, weights_path)
        
        return model
    
    except Exception as exc:
        raise RuntimeError(
            f"Failed to create model {args_model.type}.{args_model.name}"
        ) from exc
    

def load_ckpt(model, ckpt_path):
    """Load weights from ``ckpt_path`` into ``model``.

    Parameters
    ----------
    model : nn.Module
        Model instance to be updated.
    ckpt_path : str
        Path to a PyTorch checkpoint file.
    """
    path = Path(ckpt_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file {path} does not exist.")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Checkpoint payload must be a mapping: {path}")
    state_dict = payload.get("state_dict", payload)
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"Checkpoint state_dict must be a mapping: {path}")
    model.load_state_dict(dict(state_dict), strict=True)

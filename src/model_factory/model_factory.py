"""Utilities for instantiating models from configuration."""

from __future__ import annotations

import importlib
import os
from typing import Any

import torch
from ..utils.utils import get_num_classes, get_num_channels
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
    args_model.num_classes = get_num_classes(metadata)

    # TSPN family prefers inferring in_channels from metadata (Spec 12.14).
    if getattr(args_model, "name", None) in {"TSPN", "TSPN_CL"}:
        channels_map = get_num_channels(metadata)
        if isinstance(channels_map, dict) and channels_map:
            uniq = sorted({int(v) for v in channels_map.values()})
            if len(uniq) == 1:
                inferred = int(uniq[0])
                if getattr(args_model, "in_channels", None) is not None and int(getattr(args_model, "in_channels")) != inferred:
                    print(
                        f"[WARN] model.in_channels={getattr(args_model, 'in_channels')} will be overridden by "
                        f"inferred in_channels={inferred} from metadata."
                    )
                args_model.in_channels = inferred
            elif len(uniq) > 1:
                raise ValueError(f"Inconsistent in_channels across dataset_ids: {channels_map}")
    # key = f"{args_model.type}.{args_model.name}"


    module_path = resolve_model_module(args_model)
    model_module = importlib.import_module(module_path)
    model_cls = model_module.Model

    try:
        model = model_cls(args_model, metadata)
        
        if hasattr(args_model, "weights_path") and args_model.weights_path:
            weights_path = args_model.weights_path
            if os.path.exists(weights_path):
                try:
                    load_ckpt(model, weights_path)
                except Exception as e:  # pragma: no cover - runtime safeguard
                    print(f"加载权重时出错: {e}")
        
        return model
    
    except Exception as e:
        raise RuntimeError(f"创建模型实例时出错: {str(e)}")
    

def load_ckpt(model, ckpt_path):
    """Load weights from ``ckpt_path`` into ``model``.

    Parameters
    ----------
    model : nn.Module
        Model instance to be updated.
    ckpt_path : str
        Path to a PyTorch checkpoint file.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file {ckpt_path} does not exist.")
    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model_dict = model.state_dict()
    matched_dict = {}
    skipped = []
    for name, param in state_dict.items():
        if name in model_dict:
            matched_dict[name] = param
        else:
            skipped.append((name, "not in model"))
    # 加载匹配的权重
    model.load_state_dict(matched_dict, strict=False)
    # 打印跳过的参数
    if skipped:
        print("跳过以下不匹配的参数：")
        for name, model_sz in skipped:
            print(f"  {name}: checkpoint vs model {model_sz}")
    print(f"已加载匹配的权重: {ckpt_path}")

"""B01: one ordered, frozen, x-only classification pass; no adaptation method."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler

from phmfactory.adaptation_protocol import build_adaptation_view, build_evaluation_view
from src.config_schema import AdaptationProtocolConfig
from src.model_factory.model_factory import load_ckpt


def _check_loader(loader: DataLoader) -> int:
    """Accept only existing synchronous, complete, deterministic batch orders."""
    if type(loader) is not DataLoader or loader.num_workers != 0:
        raise ValueError("B01 requires a DataLoader with num_workers=0")
    if not getattr(loader, "in_order", True) or loader.drop_last:
        raise ValueError("B01 requires in-order delivery without drop_last")
    sampler = loader.batch_sampler
    if type(sampler) is BatchSampler:
        if type(sampler.sampler) is not SequentialSampler:
            raise ValueError("B01 requires a sequential, not randomized, sampler")
        if sampler.sampler.data_source is not loader.dataset or sampler.drop_last:
            raise ValueError("B01 sampler must cover the target dataset without drop_last")
    else:
        # Reuse the native evaluation order; never silently replace its batching.
        from src.data_factory.samplers.Sampler import Same_system_Sampler

        if type(sampler) is not Same_system_Sampler:
            raise ValueError("B01 does not support this batch sampler")
        if (sampler.dataset is not loader.dataset or sampler.shuffle
                or sampler.drop_last or sampler.system_metadata_key != "Dataset_id"):
            raise ValueError("B01 requires the unshuffled complete native evaluation sampler")
        indices = [index for batch in sampler for index in batch]
        if sorted(indices) != list(range(len(loader.dataset))):
            raise ValueError("B01 native sampler must visit every target sample exactly once")
    size = len(loader.dataset)
    if size == 0:
        raise ValueError("B01 requires a non-empty target population")
    return size


def _tensors(model: nn.Module) -> dict[str, torch.Tensor]:
    # named_buffers includes non-persistent buffers, unlike state_dict().
    return {**{f"parameter:{k}": v for k, v in model.named_parameters()},
            **{f"buffer:{k}": v for k, v in model.named_buffers()}}


def _check_frozen(model: nn.Module, originals: dict, values: dict, flags: dict,
                  modules: dict) -> None:
    current_modules = dict(model.named_modules())
    if (current_modules != modules
            or any(module.training for module in current_modules.values())):
        raise RuntimeError("source-only inference changed model modules or eval mode")
    current = _tensors(model)
    if current.keys() != originals.keys():
        raise RuntimeError("source-only inference changed registered model state")
    for name, value in current.items():
        saved = values[name]
        if (value is not originals[name] or value.dtype != saved.dtype
                or value.device != saved.device or value.shape != saved.shape
                or value.requires_grad != flags[name] or not torch.equal(value, saved)):
            raise RuntimeError(f"source-only inference mutated {name}")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("source-only inference created model gradients")


def run_source_only_stream(
    model: nn.Module,
    loader: DataLoader,
    protocol: AdaptationProtocolConfig,
    *,
    checkpoint_path: str | Path,
    evaluate: Callable[[torch.Tensor, Mapping[str, Any]], None],
) -> int:
    """Strictly load source weights, predict once in loader order, then evaluate.

    The caller owns a target-only loader, model architecture/device/eval mode, and
    the existing Task metric accumulator. Only ``x`` reaches ``model(x)``; labels,
    file IDs and domain IDs are evaluator-only. No preprocessing, device transfer,
    training, optimizer, adaptation, checkpoint selection or result writer is added.

    Returns the complete sample count, not a metric or a benchmark-readiness flag.
    The caller computes population metrics and writes its existing results only after
    this function returns. On failure, discard partial metrics and the model; no
    automatic rollback or success result is produced. Registered tensors are checked
    exactly each batch (one state copy, O(model state) comparison cost per batch).

    B01 supports synchronous sequential/native evaluation loaders, fully observed
    [B,L,C] inputs and a single [B,K] closed-set classification head. Reference parity
    requires identical batch partitions as well as checkpoint, inputs and order.
    Arbitrary Python state, stochastic preprocessing and hostile callbacks are not a
    sandboxed execution contract. Use trusted modules and a frozen preprocessing path.
    """
    if not isinstance(protocol, AdaptationProtocolConfig):
        raise TypeError("protocol must be AdaptationProtocolConfig")
    # Revalidate to reject a mutated or model_construct-created schema instance.
    protocol = AdaptationProtocolConfig.model_validate(protocol.model_dump())
    expected = {"regime": "source_only", "source_access": "checkpoint_only",
                "target_label_access": "none", "timing": "predict_then_update",
                "state_persistence": "persistent", "label_space": "closed_set"}
    for key, value in expected.items():
        if getattr(protocol, key) != value:
            raise ValueError(f"B01 requires protocol.{key}={value!r}")
    if not isinstance(model, nn.Module) or not callable(evaluate):
        raise TypeError("B01 requires a torch module and an evaluator callback")
    if any(module.training for module in model.modules()):
        raise ValueError("call model.eval() explicitly before source-only inference")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError("source-only inference requires no existing model gradients")
    size = _check_loader(loader)
    load_ckpt(model, checkpoint_path, strict=True)
    originals = _tensors(model)
    values = {name: value.detach().clone() for name, value in originals.items()}
    flags = {name: value.requires_grad for name, value in originals.items()}
    modules = dict(model.named_modules())
    if any(not torch.isfinite(value).all() for value in values.values()):
        raise ValueError("source checkpoint contains non-finite registered state")
    count = 0
    for batch in loader:
        _check_frozen(model, originals, values, flags, modules)
        view = build_adaptation_view(batch, protocol)
        if view.get("mask") is not None:
            raise ValueError("B01 does not support masked classification inputs")
        x = view["x"]
        if not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[0] == 0:
            raise ValueError("B01 requires non-empty [B,L,C] tensor inputs")
        if not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError("B01 requires finite floating-point inputs without casting")
        with torch.no_grad():
            prediction = model(x)
        _check_frozen(model, originals, values, flags, modules)
        if (not isinstance(prediction, torch.Tensor) or prediction.ndim != 2
                or prediction.shape[0] != x.shape[0] or prediction.shape[1] < 2
                or not prediction.is_floating_point()
                or not torch.isfinite(prediction).all()):
            raise ValueError("B01 requires finite floating-point [B,K>=2] logits")
        # Copy outputs so an evaluator cannot mutate an aliased model buffer.
        evaluate(prediction.detach().clone(), build_evaluation_view(batch))
        _check_frozen(model, originals, values, flags, modules)
        count += x.shape[0]
    if count != size:
        raise RuntimeError(f"source-only stream evaluated {count} samples; expected {size}")
    return count


__all__ = ["run_source_only_stream"]

"""Qualify/export a native source for the paper-owned physical-prior GFS method.

This module does not train a source, infer acquisition histories, construct a
prompt mechanism, or implement adaptation. Supply the actual native
``forward(x, p)`` model and its explicit TorchScript export. Qualification uses
representative source inputs only; it cannot certify the declared provenance.
"""
from __future__ import annotations

import copy
import io
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


def _private(model: nn.Module) -> nn.Module:
    result = copy.deepcopy(model).eval()
    for parameter in result.parameters():
        # ScriptModule deepcopy can create non-leaf clones.
        parameter.detach_().requires_grad_(True)
        parameter.grad = None
    return result


def _state(model: nn.Module) -> tuple[Any, ...]:
    return (
        {k: v.detach().clone() for k, v in model.state_dict().items()},
        {k: (v.requires_grad, None if v.grad is None else v.grad.detach().clone())
         for k, v in model.named_parameters()},
        {k: v.training for k, v in model.named_modules()},
    )


def _unchanged(before: tuple[Any, ...], model: nn.Module) -> None:
    after = _state(model)
    if before[0].keys() != after[0].keys() or before[1].keys() != after[1].keys():
        raise AssertionError("Source state structure changed during qualification.")
    for name, value in before[0].items():
        if not torch.equal(value, after[0][name]):
            raise AssertionError(f"Source state changed during qualification: {name}")
    for name, (flag, grad) in before[1].items():
        now_flag, now_grad = after[1][name]
        if flag != now_flag or (grad is None) != (now_grad is None):
            raise AssertionError(f"Source gradient state changed: {name}")
        if grad is not None and not torch.equal(grad, now_grad):
            raise AssertionError(f"Source gradient changed: {name}")
    if before[2] != after[2]:
        raise AssertionError("Source training flags changed during qualification.")


def _features(model: nn.Module, x: Tensor, prompt: Tensor) -> tuple[Tensor, Tensor]:
    raw = model(x, prompt)
    if not isinstance(raw, Tensor) or raw.ndim != 2 or raw.shape[0] != x.shape[0]:
        raise ValueError("Source forward(x, p) must return [batch, feature_dim].")
    norms = raw.norm(dim=1, keepdim=True)
    if not torch.isfinite(raw).all() or not torch.isfinite(norms).all() or torch.any(norms <= 1e-10):
        raise ValueError("Source produced zero or non-finite representations.")
    return raw, raw / norms


def _check_spec(spec: Mapping[str, Any], base: Tensor, injection: Tensor,
                feature_dim: int, parameters: Mapping[str, Tensor]) -> None:
    required = (
        "base_classes", "prompt_dim", "logit_scale", "a3_parameter_names",
        "excluded_target_folds", "fitted_or_selected_group_ids",
        "excluded_class_ids", "source_training_seed", "fitted_or_selected_folds",
        "source_fit_description", "group_definition", "physical_fields", "adaptation",
    )
    for key in required:
        if key not in spec or spec[key] is None or spec[key] == "" or spec[key] == []:
            raise ValueError(f"Source provenance/setting is missing: {key}")
    classes = spec["base_classes"]
    if len(classes) != len(set(classes)) or not all(type(c) is int for c in classes):
        raise ValueError("Base classes must be distinct integer labels.")
    for key in ("excluded_class_ids", "fitted_or_selected_group_ids"):
        values = spec[key]
        if not isinstance(values, list) or any(type(v) is not int for v in values) or len(values) != len(set(values)):
            raise ValueError(f"{key} must contain distinct integer IDs.")
    for key in ("excluded_target_folds", "fitted_or_selected_folds"):
        values = spec[key]
        if (not isinstance(values, list) or any(not isinstance(v, str) or not v for v in values)
                or len(values) != len(set(values))):
            raise ValueError(f"{key} must contain distinct nonempty fold names.")
    if type(spec["source_training_seed"]) is not int or spec["source_training_seed"] < 0:
        raise ValueError("source_training_seed must be a nonnegative integer.")
    if set(classes) & set(spec["excluded_class_ids"]):
        raise ValueError("Source base classes overlap declared excluded novel classes.")
    if set(spec["fitted_or_selected_folds"]) & set(spec["excluded_target_folds"]):
        raise ValueError("Source-fitted/selected folds overlap declared held-out folds.")
    if type(spec["prompt_dim"]) is not int or spec["prompt_dim"] <= 0:
        raise ValueError("prompt_dim must be a positive integer.")
    if not np.isfinite(spec["logit_scale"]) or spec["logit_scale"] <= 0:
        raise ValueError("logit_scale must be finite and positive.")
    if base.shape != (len(classes), feature_dim) or not torch.isfinite(base).all():
        raise ValueError("Base anchors must match declared classes and representation width.")
    if not torch.allclose(base.norm(dim=1), torch.ones_like(base[:, 0]), atol=1e-5, rtol=1e-5):
        raise ValueError("Base anchors must already be unit normalized; export does not repair them.")
    if (injection.ndim != 2 or injection.shape[0] != spec["prompt_dim"]
            or injection.shape[1] == 0 or injection.shape[1] % feature_dim
            or not torch.isfinite(injection).all()):
        raise ValueError("Source physical injection must have finite shape [P, D*M].")
    names = spec["a3_parameter_names"]
    if len(names) != len(set(names)) or any(name not in parameters for name in names):
        raise ValueError("A3 must name distinct, existing exported nonprompt parameters.")
    size = sum(parameters[name].numel() for name in names)
    if abs(size - spec["prompt_dim"]) > .05 * spec["prompt_dim"]:
        raise ValueError("A3 parameter count must match the prompt within five percent.")
    for field in spec["physical_fields"]:
        if not isinstance(field, dict) or any(not field.get(k) for k in (
            "column", "unit", "granularity", "availability",
            "source_only_preprocessing", "missingness_rule",
        )):
            raise ValueError("Each physical field needs units, availability and source-only preprocessing provenance.")
    settings = spec["adaptation"]
    if not isinstance(settings, dict) or any(k not in settings for k in (
            "offset_norm", "prompt_radius", "steps", "learning_rate", "regularization", "trajectory_steps")):
        raise ValueError("Source adaptation must contain explicit bounds, budgets and schedules.")
    for key in ("offset_norm", "prompt_radius"):
        value = settings[key]
        if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"adaptation.{key} must be finite and positive.")
    if settings["offset_norm"] > settings["prompt_radius"]:
        raise ValueError("The physical-prior norm exceeds the total-prompt radius.")
    for arm in ("A2", "A3", "A4", "A6", "A7", "A8"):
        for key in ("steps", "learning_rate", "regularization"):
            if not isinstance(settings[key], dict) or arm not in settings[key]:
                raise ValueError(f"Missing source-selected adaptation.{key}.{arm}.")
            value = settings[key][arm]
            if key == "steps":
                valid = type(value) is int and value >= 0
            else:
                valid = type(value) in (int, float) and np.isfinite(value) and (
                    value > 0 if key == "learning_rate" else value >= 0)
            if not valid:
                raise ValueError(f"Invalid source-selected adaptation.{key}.{arm}.")
    for arm in ("A2", "A3"):
        schedule = settings["trajectory_steps"].get(arm) if isinstance(settings["trajectory_steps"], dict) else None
        if (not isinstance(schedule, list) or not schedule or any(type(s) is not int for s in schedule)
                or schedule != sorted(set(schedule)) or schedule[0] < 0 or schedule[-1] > settings["steps"][arm]):
            raise ValueError(f"{arm} trajectory must be distinct ordered steps within its adaptation budget.")
    # No declaration here proves source-only fitting. Keep the original account,
    # including its limitations, rather than deriving it from an encoder shape.
    json.dumps(dict(spec), allow_nan=False)


def qualify_source(
    native: nn.Module, exported: nn.Module, x: Tensor, prompts: Sequence[Tensor],
    *, base_anchors: Tensor, physical_injection: Tensor, source_spec: Mapping[str, Any],
    atol: float = 1e-6, rtol: float = 1e-5,
) -> dict[str, Any]:
    """Check actual/native equivalence and differentiability on supplied source probes.

    ``prompts`` starts with zero and includes a nonzero source-selected probe.
    Both models must already include the admitted preprocessing/injection. The
    comparison checks raw outputs and normalized-feature gradients; a prompt
    that changes only feature magnitude is not an adaptation mechanism here.
    This is a finite probe test, not a global equivalence proof.
    """
    if (x.ndim < 2 or x.shape[0] < 2 or not x.is_floating_point()
            or not torch.isfinite(x).all()):
        raise ValueError("Supply at least two finite representative source inputs for batch-size checks.")
    if atol < 0 or rtol < 0 or not np.isfinite([atol, rtol]).all():
        raise ValueError("Equivalence tolerances must be finite and nonnegative.")
    if len(prompts) < 2 or torch.count_nonzero(prompts[0]) != 0:
        raise ValueError("Qualification requires zero and nonzero prompt probes.")
    if not any(torch.count_nonzero(p) for p in prompts[1:]):
        raise ValueError("Qualification requires a nonzero prompt probe.")
    for p in prompts:
        if (p.shape != (source_spec["prompt_dim"],) or p.device != x.device
                or p.dtype != x.dtype or not torch.isfinite(p).all()):
            raise ValueError("Prompt probes must match source dimension, device and dtype.")
    originals = ((native, _state(native)), (exported, _state(exported)))
    try:
        reference, candidate = _private(native), _private(exported)
        reference_parameters = dict(reference.named_parameters())
        candidate_parameters = dict(candidate.named_parameters())
        if reference_parameters.keys() != candidate_parameters.keys():
            raise ValueError("Native/exported parameter identities differ; A2/A3 cannot be compared.")
        initial_raw, _ = _features(candidate, x.detach().clone(), prompts[0].detach().clone())
        _check_spec(source_spec, base_anchors, physical_injection,
                    initial_raw.shape[1], candidate_parameters)
        if any(float(p.norm()) > source_spec["adaptation"]["prompt_radius"] + atol for p in prompts):
            raise ValueError("Qualification prompts exceed the declared total-prompt radius.")
        # Tracing can freeze the example batch size; real support/query batches
        # have different sizes. Check the serialized model on a singleton too.
        with torch.no_grad():
            for p in prompts:
                expected, _ = _features(reference, x[:1].detach().clone(), p.detach().clone())
                got, _ = _features(candidate, x[:1].detach().clone(), p.detach().clone())
                if expected.shape != got.shape or not torch.allclose(expected, got, atol=atol, rtol=rtol):
                    raise ValueError("Native/exported singleton-batch forward equivalence failed.")
        # Two fixed nonuniform projections avoid the constant squared norm of
        # normalized features. No labels, target data or global RNG are used.
        sensitive: set[str] = set()
        prompt_sensitivity = []
        observed_features = []
        for supplied in prompts:
            prompt_sensitive = False
            grads_by_model = []
            values = []
            for model, parameters in ((reference, reference_parameters), (candidate, candidate_parameters)):
                p = supplied.detach().clone().requires_grad_(True)
                raw, normalized = _features(model, x.detach().clone(), p)
                values.append((raw, normalized))
                index = torch.arange(normalized.numel(), device=x.device, dtype=x.dtype).reshape_as(normalized)
                directions = (torch.sin(index + .37), torch.cos(index * .73 + .19))
                local = []
                for direction in directions:
                    objective = (normalized * direction).sum()
                    if not objective.requires_grad:
                        raise ValueError("The export disconnects prompt and nonprompt gradients.")
                    local.append(torch.autograd.grad(objective, (p, *parameters.values()),
                                                     allow_unused=True, retain_graph=True))
                grads_by_model.append(local)
            if values[0][0].shape != values[1][0].shape or not torch.allclose(
                    values[0][0], values[1][0], atol=atol, rtol=rtol):
                raise ValueError("Native/exported forward equivalence failed on a supplied prompt probe.")
            observed_features.append(values[1][1].detach())
            for ref_grads, got_grads in zip(*grads_by_model):
                for name, expected, got in zip(("prompt", *candidate_parameters), ref_grads, got_grads):
                    if (expected is None) != (got is None):
                        raise ValueError(f"Native/exported gradient connectivity differs: {name}")
                    if got is None:
                        continue
                    if not torch.isfinite(got).all() or not torch.isfinite(expected).all():
                        raise ValueError(f"Non-finite normalized-feature gradient: {name}")
                    if not torch.allclose(expected, got, atol=atol, rtol=rtol):
                        raise ValueError(f"Native/exported gradient equivalence failed: {name}")
                    if torch.count_nonzero(got):
                        if name == "prompt":
                            prompt_sensitive = True
                        else:
                            sensitive.add(name)
            prompt_sensitivity.append(prompt_sensitive)
        if not all(prompt_sensitivity) or not any(not torch.allclose(observed_features[0], z,
                atol=atol, rtol=rtol) for z in observed_features[1:]):
            raise ValueError("Prompt must change normalized features and have a nonzero gradient.")
        if not sensitive:
            raise ValueError("A2 requires differentiable nonprompt source parameters.")
        if not set(source_spec["a3_parameter_names"]).issubset(sensitive):
            raise ValueError("Every declared A3 parameter tensor must affect normalized features.")
    finally:
        for model, before in originals:
            _unchanged(before, model)
    return dict(forward_equivalent=True, prompt_gradient_nonzero=True,
                nonprompt_gradient_nonzero=True, a3_gradient_nonzero=True,
                native_state_preserved=True, probe_count=len(prompts), atol=atol, rtol=rtol,
                batch_sizes=[1, x.shape[0]],
                scope="Supplied representative source inputs and zero/nonzero prompts only; "
                      "declared source provenance is not independently verified.")


def export_source(
    destination: str | Path, native: nn.Module, exported: torch.jit.ScriptModule,
    x: Tensor, prompts: Sequence[Tensor], *, base_anchors: Tensor,
    physical_injection: Tensor, source_spec: Mapping[str, Any],
    atol: float = 1e-6, rtol: float = 1e-5,
) -> dict[str, Any]:
    """Save a qualified explicit TorchScript source to a new directory.

    Do not provide target/query inputs to qualify or choose this export. The
    caller owns native HSE construction, source-only fitting and its record.
    Existing directories are never overwritten. Failed writes remain visible.
    """
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Source destination already exists: {destination}")
    if not isinstance(exported, torch.jit.ScriptModule):
        raise TypeError("Supply an explicit scripted/traced encoder; export does not infer one.")
    archive = io.BytesIO()
    torch.jit.save(exported, archive)
    serialized = archive.getvalue()
    reloaded = torch.jit.load(io.BytesIO(serialized), map_location=x.device)
    report = qualify_source(native, reloaded, x, prompts, base_anchors=base_anchors,
                            physical_injection=physical_injection, source_spec=source_spec,
                            atol=atol, rtol=rtol)
    spec = copy.deepcopy(dict(source_spec))
    spec["source_qualification"] = report
    encoded = json.dumps(spec, indent=2, allow_nan=False) + "\n"
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "encoder.pt").write_bytes(serialized)
    np.savez(destination / "constants.npz",
             base_anchors=base_anchors.detach().cpu().numpy(),
             physical_injection=physical_injection.detach().cpu().numpy())
    (destination / "source.json").write_text(encoded)
    return report


def adapt_support(
    implementation: Callable[..., Any], model: nn.Module, x: Tensor, y: Tensor,
    groups: Tensor, views: Tensor, base: Tensor, base_classes: Sequence[int],
    novel_classes: Sequence[int], *, arm: str, prior: Tensor, scale: float,
    lr: float, steps: int, lam: float, radius: float, checkpoints: Sequence[int],
    subset: Sequence[str] = (), initial_prompt: Tensor | None = None,
) -> Any:
    """Delegate support-only adaptation to the explicitly supplied paper core.

    Bind ``implementation=P4.core.adapt`` in the paper caller. There is no query
    argument, historical runner fallback, module search, or second objective.
    """
    return implementation(model, x, y, groups, views, base, base_classes, novel_classes,
                          arm=arm, prior=prior, scale=scale, lr=lr, steps=steps, lam=lam,
                          radius=radius, checkpoints=checkpoints, subset=subset,
                          initial_prompt=initial_prompt)

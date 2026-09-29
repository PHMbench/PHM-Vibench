"""Bounded SAR objective/optimizer state for label-free ordered-stream adaptation.

Based on mr-eggplant/SAR @ 20f6e24b17525f34503510afccedc0629b67b7c4.
The production path preserves the upstream reliable-entropy + SAM update and adds the
same operator-level BatchNorm1d extension used for PHM time-series models.

BSD 3-Clause License
Copyright (c) niushuaicheng 2023,
All rights reserved.
Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this list
   of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this
   list of conditions and the following disclaimer in the documentation and/or other
   materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be
   used to endorse or promote products derived from this software without specific
   prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

import torch
from torch import nn


_NORMS = (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)


def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


def _skip_upstream_top(name: str) -> bool:
    """Preserve the official SAR parameter-selection exclusions."""
    return ("layer4" in name or "blocks.9" in name or "blocks.10" in name
            or "blocks.11" in name or "norm." in name or name == "norm")


def _collect_params(model: nn.Module) -> tuple[list[nn.Parameter], tuple[str, ...], list[nn.Module]]:
    params: list[nn.Parameter] = []
    names: list[str] = []
    norms: list[nn.Module] = []
    for module_name, module in model.named_modules():
        if not isinstance(module, _NORMS) or _skip_upstream_top(module_name):
            continue
        norms.append(module)
        for parameter_name, parameter in module.named_parameters(recurse=False):
            if parameter_name in {"weight", "bias"}:
                params.append(parameter)
                names.append(f"{module_name}.{parameter_name}" if module_name else parameter_name)
    return params, tuple(names), norms


class _SAM:
    """Two-step SAM with an explicit SGD owner and no stale perturbation state."""

    def __init__(self, params: list[nn.Parameter], *, lr: float, rho: float):
        self.params = list(params)
        self.rho = float(rho)
        self.base_optimizer = torch.optim.SGD(
            self.params, lr=float(lr), momentum=0.9, weight_decay=0.0,
            dampening=0.0, nesterov=False,
        )
        self._old: dict[nn.Parameter, torch.Tensor] | None = None

    @torch.no_grad()
    def first_step(self) -> None:
        if self._old is not None:
            raise RuntimeError("SAR SAM perturbation is already active")
        grads = [p.grad for p in self.params if p.grad is not None]
        if not grads or any(not torch.isfinite(g).all() for g in grads):
            raise ValueError("SAR first SAM step requires finite gradients")
        device = grads[0].device
        norm = torch.norm(torch.stack([g.norm(p=2).to(device) for g in grads]), p=2)
        if not torch.isfinite(norm):
            raise ValueError("SAR gradient norm is non-finite")
        scale = self.rho / (norm + 1e-12)
        self._old = {}
        for parameter in self.params:
            if parameter.grad is None:
                continue
            self._old[parameter] = parameter.detach().clone()
            parameter.add_(parameter.grad * scale.to(parameter))
        self.zero_grad()

    @torch.no_grad()
    def restore(self) -> None:
        if self._old is None:
            raise RuntimeError("SAR SAM restore requires an active perturbation")
        for parameter, old in self._old.items():
            parameter.copy_(old)
        self._old = None
        self.zero_grad()

    @torch.no_grad()
    def second_step(self) -> None:
        if self._old is None:
            raise RuntimeError("SAR second SAM step requires an active perturbation")
        for parameter, old in self._old.items():
            parameter.copy_(old)
        self._old = None
        self.base_optimizer.step()
        self.zero_grad()

    def zero_grad(self) -> None:
        self.base_optimizer.zero_grad(set_to_none=True)


class SAR:
    """Persistent SAR with pre-update prediction and one two-forward update per batch."""

    def __init__(self, model: nn.Module, *, learning_rate: float, margin_e0: float,
                 rho: float = 0.05, reset_threshold: float = 0.2):
        for name, value in {"learning_rate": learning_rate, "margin_e0": margin_e0,
                            "rho": rho, "reset_threshold": reset_threshold}.items():
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"SAR {name} must be explicit, finite and positive")
        if float(reset_threshold) != 0.2:
            raise ValueError("B03 fixes reset_threshold=0.2 to the official SAR implementation")
        if not isinstance(model, nn.Module) or any(module.training for module in model.modules()):
            raise ValueError("SAR requires an explicitly eval-mode source model")
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise ValueError("SAR requires a source model without existing gradients")
        tensors = list(model.parameters()) + list(model.buffers())
        devices = {tensor.device for tensor in tensors}
        if len(devices) != 1 or next(iter(devices)).type not in {"cpu", "cuda"}:
            raise ValueError("SAR requires an explicitly placed single CPU/CUDA model")
        if any(not torch.isfinite(tensor).all() for tensor in tensors):
            raise ValueError("SAR source model contains non-finite state")

        self.model = model
        self.device = next(iter(devices))
        self.learning_rate = float(learning_rate)
        self.margin_e0 = float(margin_e0)
        self.rho = float(rho)
        self.reset_threshold = float(reset_threshold)

        model.train()
        model.requires_grad_(False)
        self._all_norms = [module for module in model.modules() if isinstance(module, _NORMS)]
        # Upstream configures every BN to use current-batch statistics even when its
        # affine parameters are excluded from optimization by the top-layer rule.
        for module in self._all_norms:
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                if module.affine is False:
                    # Non-affine BN can still provide batch statistics, but it cannot
                    # be the only adaptation parameterization.
                    pass
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
        params, self.parameter_names, self._norms = _collect_params(model)
        if not params:
            raise ValueError("SAR requires selected affine BN/LN/GN parameters")
        for module in self._norms:
            module.requires_grad_(True)
        # collect again after requires_grad_ to ensure skipped top modules stay frozen.
        params = [parameter for name, parameter in model.named_parameters()
                  if name in self.parameter_names]
        if not params or any(not parameter.requires_grad for parameter in params):
            raise RuntimeError("SAR parameter configuration failed")

        self.optimizer = _SAM(params, lr=self.learning_rate, rho=self.rho)
        self.ema: float | None = None
        self.num_batches = 0
        self.num_updates = 0
        self.num_samples = 0
        self.num_skipped_batches = 0
        self.num_recoveries = 0
        self.last_loss: float | None = None
        self.last_reliable_first = 0
        self.last_reliable_second = 0
        self._pending: tuple[torch.Tensor, torch.Tensor] | None = None
        self._pending_rng: tuple[torch.Tensor, torch.Tensor | None] | None = None
        self._trainable_keys = {f"parameter:{name}" for name in self.parameter_names}
        self._modules = dict(model.named_modules())
        self._originals = self._tensors()
        self._flags = {name: tensor.requires_grad for name, tensor in self._originals.items()}
        self._values = {name: tensor.detach().clone() for name, tensor in self._originals.items()}
        self._source_model, self._source_extra = self._capture_model()
        self._source_optimizer = deepcopy(self.optimizer.base_optimizer.state_dict())
        self._momentum_names: set[str] = set()

    def _tensors(self) -> dict[str, torch.Tensor]:
        return {**{f"parameter:{name}": p for name, p in self.model.named_parameters()},
                **{f"buffer:{name}": b for name, b in self.model.named_buffers()}}

    def _capture_model(self) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        state = deepcopy(self.model.state_dict())
        extra = {name: buffer.detach().clone() for name, buffer in self.model.named_buffers()
                 if name not in state}
        return state, extra

    def _restore_model(self, state: dict[str, Any], extra: dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(state, strict=True)
        present = {name: buffer for name, buffer in self.model.named_buffers()
                   if name not in self.model.state_dict()}
        if present.keys() != extra.keys():
            raise ValueError("SAR non-persistent buffer contract differs")
        with torch.no_grad():
            for name, buffer in present.items():
                saved = extra[name]
                if saved.shape != buffer.shape or saved.dtype != buffer.dtype:
                    raise ValueError(f"SAR buffer shape/dtype differs: {name}")
                buffer.copy_(saved)

    def _rng(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        cpu = torch.get_rng_state().clone()
        cuda = torch.cuda.get_rng_state(self.device).clone() if self.device.type == "cuda" else None
        return cpu, cuda

    def _check_rng(self, expected: tuple[torch.Tensor, torch.Tensor | None]) -> None:
        current = self._rng()
        if not torch.equal(current[0], expected[0]) or ((current[1] is None) != (expected[1] is None)):
            raise RuntimeError("SAR evaluator changed torch RNG between the two method forwards")
        if current[1] is not None and not torch.equal(current[1], expected[1]):
            raise RuntimeError("SAR evaluator changed CUDA RNG between the two method forwards")

    def _check_state(self, *, after_update: bool = False) -> None:
        current = self._tensors()
        if (dict(self.model.named_modules()) != self._modules
                or any(not module.training for module in self._modules.values())
                or current.keys() != self._originals.keys()
                or self.optimizer._old is not None):
            raise RuntimeError("SAR model structure/mode/SAM boundary changed")
        for module in self._all_norms:
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)) and (
                    module.track_running_stats or module.running_mean is not None
                    or module.running_var is not None):
                raise RuntimeError("SAR BatchNorm configuration changed")
        for name, tensor in current.items():
            saved = self._values[name]
            allowed = after_update and name in self._trainable_keys
            if (tensor is not self._originals[name] or tensor.shape != saved.shape
                    or tensor.dtype != saved.dtype or tensor.device != saved.device
                    or tensor.requires_grad != self._flags[name]
                    or not torch.isfinite(tensor).all()
                    or (not allowed and not torch.equal(tensor, saved))):
                raise RuntimeError(f"SAR unexpected state change: {name}")
        if any(parameter.grad is not None for parameter in self.model.parameters()):
            raise RuntimeError("SAR unexpected model gradients")

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if self._pending is not None:
            raise RuntimeError("adapt the pending SAR batch before another prediction")
        if torch.is_inference_mode_enabled():
            raise RuntimeError("SAR needs gradients; inference_mode is unsupported")
        self._check_state()
        if (not isinstance(x, torch.Tensor) or x.ndim != 3 or min(x.shape) == 0
                or not x.is_floating_point() or x.device != self.device
                or not torch.isfinite(x).all()):
            raise ValueError("SAR requires finite [B,L,C] floating inputs on the model device")
        with torch.enable_grad():
            logits = self.model(x)
        self._check_state()
        if (not isinstance(logits, torch.Tensor) or logits.ndim != 2
                or logits.shape[0] != x.shape[0] or logits.shape[1] < 2
                or not logits.is_floating_point() or not torch.isfinite(logits).all()
                or not logits.requires_grad):
            raise ValueError("SAR requires finite differentiable [B,K>=2] logits")
        self._pending = (x, logits)
        self._pending_rng = self._rng()
        return logits.detach().clone()

    def _refresh_values(self) -> None:
        for name in self.parameter_names:
            key = f"parameter:{name}"
            self._values[key] = self._originals[key].detach().clone()

    def _recover_source(self) -> None:
        self._restore_model(self._source_model, self._source_extra)
        self.optimizer.base_optimizer.load_state_dict(deepcopy(self._source_optimizer))
        self.optimizer._old = None
        self._momentum_names.clear()
        self.optimizer.zero_grad()
        self._values = {name: tensor.detach().clone() for name, tensor in self._originals.items()}
        self.num_recoveries += 1
        self._check_state()

    def adapt(self) -> float | None:
        if self._pending is None or self._pending_rng is None:
            raise RuntimeError("SAR adapt requires a pending prediction")
        self._check_state()
        self._check_rng(self._pending_rng)
        x, logits = self._pending
        entropies = softmax_entropy(logits)
        first_mask = entropies < self.margin_e0
        self.last_reliable_first = int(first_mask.sum().item())
        self.last_reliable_second = 0
        self.num_batches += 1
        self.num_samples += int(x.shape[0])

        # The upstream empty-set mean is NaN and can still interact with SGD momentum.
        # B03 makes the reliable-sample contract explicit: no selected sample => no update.
        if self.last_reliable_first == 0:
            self.num_skipped_batches += 1
            self.last_loss = None
            self._pending = self._pending_rng = None
            return None

        with torch.enable_grad():
            first_loss = entropies[first_mask].mean()
            if not torch.isfinite(first_loss):
                raise ValueError("SAR first reliable entropy is non-finite")
            first_loss.backward()
        self.optimizer.first_step()
        try:
            with torch.enable_grad():
                second_logits = self.model(x)
            if (not isinstance(second_logits, torch.Tensor) or second_logits.shape != logits.shape
                    or not torch.isfinite(second_logits).all() or not second_logits.requires_grad):
                raise ValueError("SAR perturbed forward returned invalid logits")
            second_entropy = softmax_entropy(second_logits)[first_mask]
            second_mask = second_entropy < self.margin_e0
            self.last_reliable_second = int(second_mask.sum().item())
            if self.last_reliable_second == 0:
                self.optimizer.restore()
                self.num_skipped_batches += 1
                self.last_loss = None
                self._pending = self._pending_rng = None
                self._check_state()
                return None
            second_loss = second_entropy[second_mask].mean()
            if not torch.isfinite(second_loss):
                raise ValueError("SAR second reliable entropy is non-finite")
            second_loss.backward()
            for name, parameter in self.model.named_parameters():
                if parameter.requires_grad and parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise ValueError(f"SAR non-finite second gradient: {name}")
            active_names = {name for name, parameter in self.model.named_parameters()
                            if parameter.requires_grad and parameter.grad is not None}
            self.optimizer.second_step()
            self._momentum_names.update(active_names)
        except Exception:
            if self.optimizer._old is not None:
                self.optimizer.restore()
            raise

        self._refresh_values()
        loss_value = float(second_loss.detach())
        self.ema = loss_value if self.ema is None else 0.9 * self.ema + 0.1 * loss_value
        if not math.isfinite(self.ema):
            raise ValueError("SAR entropy EMA is non-finite")
        self.num_updates += 1
        self.last_loss = loss_value
        self._pending = self._pending_rng = None
        self._check_state()
        if self.ema < self.reset_threshold:
            # Match the upstream forward order: reset model/optimizer, then retain the
            # just-computed EMA value as self.ema for the next recovery decision.
            retained_ema = self.ema
            self._recover_source()
            self.ema = retained_ema
        return self.last_loss

    def state_dict(self) -> dict[str, Any]:
        """Algorithm-state payload at a completed batch boundary, not a stream writer."""
        if self._pending is not None or self.optimizer._old is not None:
            raise RuntimeError("save SAR state only at a completed batch boundary")
        self._check_state()
        model_state, extra = self._capture_model()
        return deepcopy(dict(
            model=model_state, extra_buffers=extra,
            optimizer=self.optimizer.base_optimizer.state_dict(),
            source_model=self._source_model, source_extra_buffers=self._source_extra,
            source_optimizer=self._source_optimizer,
            learning_rate=self.learning_rate, margin_e0=self.margin_e0,
            rho=self.rho, reset_threshold=self.reset_threshold,
            parameter_names=self.parameter_names, momentum_names=tuple(sorted(self._momentum_names)),
            ema=self.ema,
            num_batches=self.num_batches, num_updates=self.num_updates,
            num_samples=self.num_samples, num_skipped_batches=self.num_skipped_batches,
            num_recoveries=self.num_recoveries, last_loss=self.last_loss,
            last_reliable_first=self.last_reliable_first,
            last_reliable_second=self.last_reliable_second,
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None,
        ))

    def _validate_optimizer_state(self) -> None:
        allowed = set(self.optimizer.params)
        if not set(self.optimizer.base_optimizer.state).issubset(allowed):
            raise ValueError("SAR checkpoint SGD state references unknown parameters")
        for parameter, values in self.optimizer.base_optimizer.state.items():
            if set(values) != {"momentum_buffer"}:
                raise ValueError("SAR checkpoint SGD state fields differ")
            momentum = values["momentum_buffer"]
            if (not isinstance(momentum, torch.Tensor) or momentum.shape != parameter.shape
                    or momentum.dtype != parameter.dtype or not torch.isfinite(momentum).all()):
                raise ValueError("SAR checkpoint momentum buffer is invalid")

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if self._pending is not None or self.optimizer._old is not None:
            raise RuntimeError("cannot restore SAR state during a pending batch")
        if state.keys() != self.state_dict().keys():
            raise ValueError("SAR checkpoint fields do not match")
        expected = (self.learning_rate, self.margin_e0, self.rho, self.reset_threshold,
                    self.parameter_names, self.device.type == "cuda")
        observed = (state["learning_rate"], state["margin_e0"], state["rho"],
                    state["reset_threshold"], tuple(state["parameter_names"]),
                    state["cuda_rng"] is not None)
        if observed != expected:
            raise ValueError("SAR checkpoint hyperparameter/parameter/device contract differs")
        momentum_names = tuple(state["momentum_names"])
        if (tuple(sorted(momentum_names)) != momentum_names
                or not set(momentum_names).issubset(set(self.parameter_names))):
            raise ValueError("SAR checkpoint momentum parameter names are invalid")
        for name in ("num_batches", "num_updates", "num_samples", "num_skipped_batches",
                     "num_recoveries", "last_reliable_first", "last_reliable_second"):
            if type(state[name]) is not int or state[name] < 0:
                raise ValueError(f"SAR checkpoint {name} must be a non-negative integer")
        if (state["num_updates"] > state["num_batches"]
                or state["num_skipped_batches"] > state["num_batches"]
                or state["num_updates"] + state["num_skipped_batches"] != state["num_batches"]
                or state["num_batches"] > state["num_samples"]):
            raise ValueError("SAR checkpoint counters are inconsistent")
        for name in ("ema", "last_loss"):
            value = state[name]
            if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise ValueError(f"SAR checkpoint {name} is invalid")

        self._restore_model(state["model"], state["extra_buffers"])
        options = [{key: value for key, value in group.items() if key != "params"}
                   for group in self.optimizer.base_optimizer.param_groups]
        self.optimizer.base_optimizer.load_state_dict(state["optimizer"])
        if options != [{key: value for key, value in group.items() if key != "params"}
                       for group in self.optimizer.base_optimizer.param_groups]:
            raise ValueError("SAR checkpoint SGD configuration differs")
        self._validate_optimizer_state()
        parameter_to_name = {parameter: name for name, parameter in self.model.named_parameters()}
        actual_momentum_names = tuple(sorted(parameter_to_name[p]
            for p in self.optimizer.base_optimizer.state))
        if actual_momentum_names != momentum_names:
            raise ValueError("SAR checkpoint SGD momentum entries are incomplete")
        if state["source_optimizer"] != self._source_optimizer:
            raise ValueError("SAR recovery optimizer anchor differs from the explicit source state")
        if state["source_model"].keys() != self._source_model.keys():
            raise ValueError("SAR recovery model anchor keys differ from the explicit source state")
        for name, expected_tensor in self._source_model.items():
            observed_tensor = state["source_model"][name]
            if (not isinstance(observed_tensor, torch.Tensor)
                    or observed_tensor.shape != expected_tensor.shape
                    or observed_tensor.dtype != expected_tensor.dtype
                    or observed_tensor.device.type != expected_tensor.device.type
                    or not torch.equal(observed_tensor.to(expected_tensor.device), expected_tensor)):
                raise ValueError(f"SAR recovery model anchor differs: {name}")
        if state["source_extra_buffers"].keys() != self._source_extra.keys():
            raise ValueError("SAR recovery non-persistent buffer anchor keys differ")
        for name, expected_tensor in self._source_extra.items():
            observed_tensor = state["source_extra_buffers"][name]
            if (not isinstance(observed_tensor, torch.Tensor)
                    or observed_tensor.shape != expected_tensor.shape
                    or observed_tensor.dtype != expected_tensor.dtype
                    or observed_tensor.device.type != expected_tensor.device.type
                    or not torch.equal(observed_tensor.to(expected_tensor.device), expected_tensor)):
                raise ValueError(f"SAR recovery buffer anchor differs: {name}")

        self.ema = None if state["ema"] is None else float(state["ema"])
        self.num_batches = state["num_batches"]
        self.num_updates = state["num_updates"]
        self.num_samples = state["num_samples"]
        self.num_skipped_batches = state["num_skipped_batches"]
        self.num_recoveries = state["num_recoveries"]
        self.last_loss = state["last_loss"]
        self.last_reliable_first = state["last_reliable_first"]
        self.last_reliable_second = state["last_reliable_second"]
        self._momentum_names = set(momentum_names)
        self._values = {name: tensor.detach().clone() for name, tensor in self._originals.items()}
        self._check_state()
        torch.set_rng_state(state["torch_rng"].cpu())
        if self.device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"].cpu(), self.device)


__all__ = ["SAR", "softmax_entropy"]

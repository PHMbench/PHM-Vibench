"""One-step Tent objective and optimizer; data/evaluation belong to the runtime.

Based on DequanWang/tent @ e9e926a668d85244c66a6d5c006efbd2b82e83e8.
The BN1d extension uses the same channel-wise operator as upstream BN2d.

MIT License
Copyright (c) 2021 Dequan Wang and Evan Shelhamer
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations

from copy import deepcopy
import math

import torch
from torch import nn


class Tent:
    """Persistent Tent with one forward and one Adam update per batch.

    ``predict(x)`` keeps its graph privately and returns an isolated prediction.
    After evaluation, ``adapt()`` consumes that graph without accepting labels,
    inputs or an externally supplied loss. No second stochastic forward is used.
    Call on a fresh, explicitly eval-mode model with source weights already loaded.
    Mode/BN changes below are the declared algorithm, not an inference fallback.
    """

    def __init__(self, model: nn.Module, *, learning_rate: float):
        if (isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float))
                or not math.isfinite(learning_rate) or learning_rate <= 0):
            raise ValueError("Tent learning_rate must be explicit, finite and positive")
        if not isinstance(model, nn.Module) or any(m.training for m in model.modules()):
            raise ValueError("Tent requires an explicitly eval-mode source model")
        if any(p.grad is not None for p in model.parameters()):
            raise ValueError("Tent requires a source model without existing gradients")
        norms = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
        if not norms or any(type(m) not in (nn.BatchNorm1d, nn.BatchNorm2d)
                            or not m.affine for m in norms):
            raise ValueError("Tent requires affine BatchNorm1d/2d; no normalization replacement")
        tensors = list(model.parameters()) + list(model.buffers())
        devices = {t.device for t in tensors}
        if len(devices) != 1 or next(iter(devices)).type not in {"cpu", "cuda"}:
            raise ValueError("Tent requires an explicitly placed, single CPU/CUDA model")
        if any(not torch.isfinite(t).all() for t in tensors):
            raise ValueError("Tent source model contains non-finite state")
        self.model = model
        self.device = next(iter(devices))
        self.learning_rate = float(learning_rate)
        # Match upstream, including training-mode dropout and current-batch BN.
        model.train()
        model.requires_grad_(False)
        for m in norms:
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        self.parameter_names = tuple(n for n, p in model.named_parameters() if p.requires_grad)
        params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(params, lr=self.learning_rate, betas=(0.9, 0.999),
                                          eps=1e-8, weight_decay=0.0)
        self.num_updates = 0
        self.num_samples = 0
        self.last_loss = None
        self._pending = None
        self._trainable_keys = {f"parameter:{n}" for n in self.parameter_names}
        self._norms = norms
        self._modules = dict(model.named_modules())
        self._originals = self._tensors()
        self._flags = {n: t.requires_grad for n, t in self._originals.items()}
        self._values = {n: t.detach().clone() for n, t in self._originals.items()}

    def _tensors(self):
        return {**{f"parameter:{n}": p for n, p in self.model.named_parameters()},
                **{f"buffer:{n}": b for n, b in self.model.named_buffers()}}

    def _check_state(self, *, after_update=False):
        current = self._tensors()
        if (dict(self.model.named_modules()) != self._modules
                or any(not m.training for m in self._modules.values())
                or any(m.track_running_stats or m.running_mean is not None
                       or m.running_var is not None for m in self._norms)
                or current.keys() != self._originals.keys()):
            raise RuntimeError("Tent model structure/mode/normalization changed")
        for name, tensor in current.items():
            saved = self._values[name]
            allowed = after_update and name in self._trainable_keys
            if (tensor is not self._originals[name] or tensor.shape != saved.shape
                    or tensor.dtype != saved.dtype or tensor.device != saved.device
                    or tensor.requires_grad != self._flags[name]
                    or not torch.isfinite(tensor).all()
                    or (not allowed and not torch.equal(tensor, saved))):
                raise RuntimeError(f"Tent unexpected state change: {name}")
        if any(p.grad is not None for p in self.model.parameters()):
            raise RuntimeError("Tent unexpected model gradients")

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if self._pending is not None:
            raise RuntimeError("adapt the pending Tent batch before another prediction")
        if torch.is_inference_mode_enabled():
            raise RuntimeError("Tent needs gradients; inference_mode is unsupported")
        self._check_state()
        if (not isinstance(x, torch.Tensor) or x.ndim != 3 or min(x.shape) == 0
                or not x.is_floating_point() or x.device != self.device
                or not torch.isfinite(x).all()):
            raise ValueError("Tent requires finite [B,L,C] floating inputs on the model device")
        with torch.enable_grad():
            logits = self.model(x)
        self._check_state()
        if (not isinstance(logits, torch.Tensor) or logits.ndim != 2
                or logits.shape[0] != x.shape[0] or logits.shape[1] < 2
                or not logits.is_floating_point() or not torch.isfinite(logits).all()
                or not logits.requires_grad):
            raise ValueError("Tent requires finite differentiable [B,K>=2] logits")
        self._pending = logits
        return logits.detach().clone()

    def adapt(self) -> float:
        if self._pending is None:
            raise RuntimeError("Tent adapt requires a pending prediction")
        self._check_state()
        with torch.enable_grad():
            logits = self._pending
            loss = -(logits.softmax(1) * logits.log_softmax(1)).sum(1).mean(0)
            if not torch.isfinite(loss):
                raise ValueError("Tent entropy is non-finite")
            loss.backward()
        for name, p in self.model.named_parameters():
            if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all()):
                raise ValueError(f"Tent missing or non-finite gradient: {name}")
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self._check_state(after_update=True)
        if any(not torch.isfinite(v).all() for s in self.optimizer.state.values()
               for v in s.values() if isinstance(v, torch.Tensor)):
            raise ValueError("Tent optimizer state is non-finite")
        for name in self.parameter_names:
            key = f"parameter:{name}"
            self._values[key] = self._originals[key].detach().clone()
        self.num_updates += 1
        self.num_samples += logits.shape[0]
        self.last_loss = float(loss.detach())
        self._pending = None
        return self.last_loss

    def state_dict(self) -> dict:
        """Ordinary checkpoint payload at a completed batch boundary, not a writer.

        Stores algorithm/model state and torch RNG; the enclosing runtime still owns
        the stream cursor, preprocessing RNG and evaluation accumulator on resume.
        """
        if self._pending is not None:
            raise RuntimeError("save Tent state only after adapting the pending batch")
        self._check_state()
        model_state = self.model.state_dict()
        return deepcopy(dict(model=model_state, optimizer=self.optimizer.state_dict(),
            extra_buffers={n: b for n, b in self.model.named_buffers() if n not in model_state},
            learning_rate=self.learning_rate, parameter_names=self.parameter_names,
            num_updates=self.num_updates, num_samples=self.num_samples, last_loss=self.last_loss,
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None))

    def load_state_dict(self, state: dict) -> None:
        """Restore a same-architecture, same-device-kind algorithm checkpoint."""
        if self._pending is not None:
            raise RuntimeError("cannot restore Tent state with a pending batch")
        if state.keys() != self.state_dict().keys():
            raise ValueError("Tent checkpoint fields do not match")
        if (state["learning_rate"] != self.learning_rate
                or tuple(state["parameter_names"]) != self.parameter_names
                or (state["cuda_rng"] is None) != (self.device.type == "cpu")):
            raise ValueError("Tent checkpoint optimizer/parameter/device contract differs")
        for name in ("num_updates", "num_samples"):
            if type(state[name]) is not int or state[name] < 0:
                raise ValueError(f"Tent checkpoint {name} must be a non-negative integer")
        self.model.load_state_dict(state["model"], strict=True)
        extra = {n: b for n, b in self.model.named_buffers() if n not in self.model.state_dict()}
        if extra.keys() != state["extra_buffers"].keys():
            raise ValueError("Tent checkpoint non-persistent buffers differ")
        with torch.no_grad():
            for name, buffer in extra.items():
                saved = state["extra_buffers"][name]
                if saved.shape != buffer.shape or saved.dtype != buffer.dtype:
                    raise ValueError(f"Tent checkpoint buffer shape/dtype differs: {name}")
                buffer.copy_(saved)
        options = [{k: v for k, v in g.items() if k != "params"}
                   for g in self.optimizer.param_groups]
        self.optimizer.load_state_dict(state["optimizer"])
        if options != [{k: v for k, v in g.items() if k != "params"}
                       for g in self.optimizer.param_groups]:
            raise ValueError("Tent checkpoint Adam configuration differs")
        if any(not torch.isfinite(v).all() for s in self.optimizer.state.values()
               for v in s.values() if isinstance(v, torch.Tensor)):
            raise ValueError("Tent checkpoint optimizer state is non-finite")
        if (state["num_updates"] > state["num_samples"]
                or (state["last_loss"] is not None and not math.isfinite(state["last_loss"]))):
            raise ValueError("Tent checkpoint counters/loss are invalid")
        self.num_updates, self.num_samples = state["num_updates"], state["num_samples"]
        self.last_loss = state["last_loss"]
        self._values = {n: t.detach().clone() for n, t in self._originals.items()}
        self._check_state()
        torch.set_rng_state(state["torch_rng"].cpu())
        if self.device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"].cpu(), self.device)

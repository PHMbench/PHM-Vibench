from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM1D:
    """Minimal 1D Grad-CAM (ported from model_collection/GradCAM_XFD.py)."""

    def __init__(self, model: nn.Module, target_layer: str):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(_module, _inp, out):
            self.activations = out

        def backward_hook(_module, _grad_input, grad_output):
            self.gradients = grad_output[0]

        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)
                return
        raise ValueError(f"target_layer not found: {self.target_layer}")

    @torch.no_grad()
    def _normalize(self, cam: torch.Tensor) -> torch.Tensor:
        cam = cam - cam.min(dim=1, keepdim=True).values
        cam = cam / (cam.max(dim=1, keepdim=True).values + 1e-8)
        return cam

    def generate_cam(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(output.argmax(dim=1)[0].item())

        one_hot = torch.zeros_like(output)
        one_hot[:, class_idx] = 1.0
        output.backward(gradient=one_hot)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("GradCAM hooks did not capture gradients/activations.")

        gradients = self.gradients
        activations = self.activations
        weights = torch.mean(gradients, dim=2)  # (B, C)

        cam = torch.zeros((input_tensor.size(0), activations.size(2)), device=input_tensor.device)
        for b in range(weights.size(0)):
            cam[b] = torch.sum(weights[b].view(-1, 1) * activations[b], dim=0)

        cam = F.relu(cam)
        cam = self._normalize(cam)
        return cam.detach().cpu().numpy()


@dataclass(frozen=True)
class GradCAMResult:
    target_layer: str
    class_idx: int
    cam_path: str


class GradCAM1DExplainer:
    """File-output wrapper to integrate GradCAM into explain_factory."""

    def __init__(self, target_layer: str = ""):
        self.target_layer = target_layer

    def explain(self, model: nn.Module, x: torch.Tensor, out_dir: Path, class_idx: Optional[int] = None) -> GradCAMResult:
        out_dir.mkdir(parents=True, exist_ok=True)
        target_layer = self.target_layer or self._guess_target_layer(model)
        cam_engine = GradCAM1D(model=model, target_layer=target_layer)
        cam = cam_engine.generate_cam(x, class_idx=class_idx)
        cam_path = out_dir / "gradcam_1d.npy"
        np.save(cam_path, cam)
        result = GradCAMResult(target_layer=target_layer, class_idx=int(class_idx or 0), cam_path=str(cam_path))
        (out_dir / "gradcam_1d.json").write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
        return result

    def _guess_target_layer(self, model: nn.Module) -> str:
        last_conv = ""
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv1d,)):
                last_conv = name
        if not last_conv:
            raise ValueError("Could not auto-detect a Conv1d layer for GradCAM; set target_layer explicitly.")
        return last_conv


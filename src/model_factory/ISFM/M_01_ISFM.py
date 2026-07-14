from __future__ import annotations

from importlib import import_module
from typing import Any

import torch
import torch.nn as nn

from src.model_factory.ISFM.system_utils import resolve_batch_metadata
from src.utils.utils import get_num_classes


# Component IDs remain config-facing. Values are module and symbol names so the
# selected implementation is imported only when the model is instantiated.
Embedding_dict = {
    "E_01_HSE": ("src.model_factory.ISFM.embedding.E_01_HSE", "E_01_HSE"),
    "E_02_HSE_v2": ("src.model_factory.ISFM.embedding.E_02_HSE_rec", "E_02_HSE_v2"),
    "E_03_Patch": ("src.model_factory.ISFM.embedding.E_03_Patch", "E_03_Patch"),
}

Backbone_dict = {
    "B_01_basic_transformer": (
        "src.model_factory.ISFM.backbone.B_01_basic_transformer",
        "B_01_basic_transformer",
    ),
    "B_03_FITS": ("src.model_factory.ISFM.backbone.B_03_FITS", "B_03_FITS"),
    "B_04_Dlinear": ("src.model_factory.ISFM.backbone.B_04_Dlinear", "B_04_Dlinear"),
    "B_05_Mamba": ("src.model_factory.ISFM.backbone.B_05_Mamba", "B_05_Mamba"),
    "B_06_TimesNet": ("src.model_factory.ISFM.backbone.B_06_TimesNet", "B_06_TimesNet"),
    "B_07_TSMixer": ("src.model_factory.ISFM.backbone.B_07_TSMixer", "B_07_TSMixer"),
    "B_08_PatchTST": ("src.model_factory.ISFM.backbone.B_08_PatchTST", "B_08_PatchTST"),
    "B_09_FNO": ("src.model_factory.ISFM.backbone.B_09_FNO", "B_09_FNO"),
}

TaskHead_dict = {
    "H_01_Linear_cla": (
        "src.model_factory.ISFM.task_head.H_01_Linear_cla",
        "H_01_Linear_cla",
    ),
    "H_02_distance_cla": (
        "src.model_factory.ISFM.task_head.H_02_distance_cla",
        "H_02_distance_cla",
    ),
    "H_03_Linear_pred": (
        "src.model_factory.ISFM.task_head.H_03_Linear_pred",
        "H_03_Linear_pred",
    ),
    "H_09_multiple_task": (
        "src.model_factory.ISFM.task_head.H_09_multiple_task",
        "H_09_multiple_task",
    ),
    "MultiTaskHead": (
        "src.model_factory.ISFM.task_head.multi_task_head",
        "MultiTaskHead",
    ),
}


def _load_component(registry: dict[str, tuple[str, str]], component_id: str, kind: str) -> Any:
    """Load one configured component with a component-specific failure message."""
    try:
        module_path, symbol = registry[component_id]
    except KeyError as exc:
        available = ", ".join(sorted(registry))
        raise ValueError(
            f"Unknown ISFM {kind} {component_id!r}. Available values: {available}"
        ) from exc

    try:
        module = import_module(module_path)
    except ModuleNotFoundError as exc:
        missing = exc.name or "unknown dependency"
        raise RuntimeError(
            f"Unable to load ISFM {kind} {component_id!r}: missing dependency {missing!r}."
        ) from exc

    try:
        return getattr(module, symbol)
    except AttributeError as exc:
        raise RuntimeError(
            f"ISFM {kind} {component_id!r} expects symbol {symbol!r} in {module_path!r}."
        ) from exc


class Model(nn.Module):
    """ISFM architecture with configurable embedding, backbone, and task head.

    Only the selected components are imported. This keeps the maintained
    ``M_01_ISFM + B_04_Dlinear`` path independent of optional dependencies used
    by non-selected research backbones.
    """

    def __init__(self, args_m, metadata):
        super().__init__()
        self.metadata = metadata
        self.args_m = args_m

        embedding_cls = _load_component(Embedding_dict, args_m.embedding, "embedding")
        backbone_cls = _load_component(Backbone_dict, args_m.backbone, "backbone")
        self.embedding = embedding_cls(args_m)
        self.backbone = backbone_cls(args_m)

        self.num_classes = self.get_num_classes()
        args_m.num_classes = self.num_classes
        task_head_cls = _load_component(TaskHead_dict, args_m.task_head, "task head")
        self.task_head = task_head_cls(args_m)

    def get_num_classes(self):
        """Return the metadata-derived dataset-to-class-count mapping."""
        return get_num_classes(self.metadata)

    def _embed(self, x, file_id):
        """Apply the configured embedding."""
        if self.args_m.embedding in ("E_01_HSE", "E_02_HSE_v2"):
            _, fs_tensor = resolve_batch_metadata(self.metadata, file_id, device=x.device)
            return self.embedding(x, fs_tensor)
        return self.embedding(x)

    def _encode(self, x):
        """Apply the configured backbone."""
        return self.backbone(x)

    def _head(self, x, file_id=False, task_id=False, return_feature=False):
        """Apply the configured task head."""
        system_ids_tensor, _ = resolve_batch_metadata(self.metadata, file_id, device=x.device)
        system_ids = [int(value) for value in system_ids_tensor.view(-1).tolist()]

        if task_id == "classification":
            return self.task_head(
                x,
                system_id=system_ids,
                return_feature=return_feature,
                task_id=task_id,
            )
        if task_id == "prediction":
            shape = (self.shape[1], self.shape[2]) if len(self.shape) > 2 else (self.shape[1],)
            return self.task_head(
                x,
                return_feature=return_feature,
                task_id=task_id,
                shape=shape,
            )
        return None

    def forward(self, x: torch.Tensor, file_id=False, task_id=False, return_feature=False):
        """Forward pass through embedding, backbone, and task head."""
        self.shape = x.shape
        x = self._embed(x, file_id)
        if return_feature:
            feature = self._encode(x)
            output = self._head(feature, file_id, task_id)
            return output, feature

        x = self._encode(x)
        return self._head(x, file_id, task_id)

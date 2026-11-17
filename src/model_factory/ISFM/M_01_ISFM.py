from __future__ import annotations

from importlib import import_module
import math
from typing import Any

import torch
import torch.nn as nn

from src.model_factory.ISFM.system_utils import resolve_batch_metadata
from src.utils.utils import get_num_classes


# Component IDs remain config-facing. Values are module and symbol names so the
# selected implementation is imported only when the model is instantiated.
Embedding_dict = {
    "SupportConditionedTokenizer": (
        "src.model_factory.ISFM.embedding.SupportConditionedTokenizer",
        "SupportConditionedTokenizer",
    ),
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

    def _embed(self, x, file_id, *, sample_rates=None, incremental=None,
               availability=None, start_indices_L=None, start_indices_C=None):
        """Apply the configured embedding."""
        if self.args_m.embedding == "SupportConditionedTokenizer":
            if incremental is None or availability is None:
                raise ValueError('projected incremental coordinates and availability are required')
            if sample_rates is not None or start_indices_L is not None or start_indices_C is not None:
                raise ValueError('projected physical cells must not be repatched')
            return self.embedding(x, incremental, availability)
        if incremental is not None or availability is not None:
            raise ValueError('incremental/availability require SupportConditionedTokenizer')
        if self.args_m.embedding in ("E_01_HSE", "E_02_HSE_v2"):
            if sample_rates is None:
                _, fs_tensor = resolve_batch_metadata(self.metadata, file_id, device=x.device)
            else:
                fs_tensor = sample_rates
            if start_indices_L is not None or start_indices_C is not None:
                if self.args_m.embedding != "E_01_HSE":
                    raise ValueError('explicit patch starts currently require E_01_HSE')
                return self.embedding(x, fs_tensor, start_indices_L=start_indices_L,
                                      start_indices_C=start_indices_C)
            return self.embedding(x, fs_tensor)
        if sample_rates is not None or start_indices_L is not None or start_indices_C is not None:
            raise ValueError('sampling metadata requires a physical HSE embedding')
        return self.embedding(x)

    def projected_inputs(self, common, incremental, availability, *, grid_rate_hz,
                         duration_s, effective_rate_hz=None):
        """Prepare the same admitted physical union for supported embeddings.

        HSE uses its actual acquisition grid for positional time; its input is
        converted from the common grid only if the round trip retains the union.
        The two-branch tokenizer performs its own stored source-RMS division.
        """
        if (common.ndim != 3 or incremental.shape != common.shape
                or availability.shape != (len(common),)
                or not ((availability == 0) | (availability == 1)).all()):
            raise ValueError('projected inputs need matching [B,K,P] blocks and binary availability')
        admitted = torch.where(availability[:, None, None].bool(), incremental,
                               torch.zeros_like(incremental))
        if not torch.isfinite(common).all() or not torch.isfinite(admitted).all():
            raise ValueError('admitted projected coordinates must be finite')
        if self.args_m.embedding == 'SupportConditionedTokenizer':
            return common, dict(incremental=incremental, availability=availability)
        if self.args_m.embedding not in ('E_01_HSE', 'E_03_Patch'):
            raise ValueError('embedding does not declare projected physical input semantics')
        rms = float(self.args_m.source_rms)
        if not math.isfinite(rms) or rms <= 0:
            raise ValueError('source RMS must be finite and positive')
        union = (common + admitted).reshape(len(common), -1) / rms
        grid = float(grid_rate_hz)
        duration = float(duration_s)
        if (not math.isfinite(grid) or not math.isfinite(duration) or grid <= 0 or duration <= 0
                or not math.isclose(union.shape[1], grid * duration, abs_tol=1e-6, rel_tol=0)):
            raise ValueError('projected grid must match physical rate and duration')
        if self.args_m.embedding == 'E_01_HSE':
            rate = float(effective_rate_hz)
            if not math.isfinite(rate) or rate <= 0 or not math.isclose(rate*duration, round(rate*duration), abs_tol=1e-6):
                raise ValueError('HSE requires an explicit effective sampling rate and integer window length')
            count = round(rate * duration)
            if count < 2:
                raise ValueError('HSE requires at least two physical samples')
            if count != union.shape[1]:
                def resample(values, size):
                    source_size = values.shape[-1]
                    spectrum = torch.fft.rfft(values, dim=-1, norm='forward')
                    keep = min(source_size, size) // 2 + 1
                    result = spectrum.new_zeros((len(values), size // 2 + 1))
                    result[:, :keep] = spectrum[:, :keep]
                    if min(source_size, size) % 2 == 0:
                        result[:, keep-1] *= .5 if source_size < size else 2.
                    return torch.fft.irfft(result, n=size, dim=-1, norm='forward')
                native = resample(union, count)
                restored = resample(native, union.shape[1])
                if not torch.allclose(restored, union, rtol=1e-5, atol=1e-6):
                    raise ValueError('HSE sampling grid would discard admitted signal content')
                union = native
        return union.unsqueeze(-1), {}

    def encode(self, x: torch.Tensor, *, file_id=None, sample_rates=None,
               incremental=None, availability=None, start_indices_L=None,
               start_indices_C=None) -> torch.Tensor:
        """Return backbone tokens without consulting any classification head.

        Unseen targets supply acquisition metadata directly, not a fabricated
        source file/system identity. Caller controls eval mode and freezing.
        """
        tokens = self._embed(x, file_id, sample_rates=sample_rates,
                             incremental=incremental, availability=availability,
                             start_indices_L=start_indices_L, start_indices_C=start_indices_C)
        return self._encode(tokens)

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

    def forward(self, x: torch.Tensor, file_id=False, task_id=False, return_feature=False,
                *, incremental=None, availability=None):
        """Forward pass through embedding, backbone, and task head."""
        self.shape = x.shape
        x = self._embed(x, file_id, incremental=incremental, availability=availability)
        if return_feature:
            feature = self._encode(x)
            output = self._head(feature, file_id, task_id)
            return output, feature

        x = self._encode(x)
        return self._head(x, file_id, task_id)

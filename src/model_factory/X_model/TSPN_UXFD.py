"""TSPN_UXFD

Compatibility wrapper for UXFD merge.

This model intentionally stays close to the upstream UXFD `TSPN.py` structure:
`SignalProcessingLayer → FeatureExtractorlayer → Classifier`.

Implementation note:
- Default behavior reuses the existing `src/model_factory/X_model/TSPN.py` code path.
- When enabled via `model.uxfd.*` config, it assembles optional UXFD modules under
  `src/model_factory/X_model/UXFD/` (best-effort; keeps the entrypoint stable).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .TSPN import Model as _TSPNModel
from .UXFD.fusion import FusionConfig, build_fusion
from .UXFD.fuzzy import (
    FuzzyConfig,
    FuzzyReasoner,
    FuzzyTrace,
    P05F0Decision,
)
from .UXFD.neurosymbolic import LogicConfig, LogicReasoner
from .UXFD.operator_attention import OperatorAttention1D, OperatorAttentionConfig
from .UXFD.signal_processing_2d import STFTTimeFrequency
from .UXFD.signal_processing_2d.stft_tfr import STFTConfig


@dataclass(frozen=True)
class FuzzyTraceOutput:
    """Complete same-forward reconstruction record for the additive control."""

    logits: torch.Tensor
    non_fuzzy_logits: torch.Tensor
    fuzzy_scale: float
    fuzzy_trace: FuzzyTrace

    def scaled_rule_contributions(self) -> torch.Tensor:
        return self.fuzzy_trace.rule_contributions * float(self.fuzzy_scale)

    def reconstruct_logits(self) -> torch.Tensor:
        return self.non_fuzzy_logits + self.scaled_rule_contributions().sum(dim=1)

    def reconstruction_residual(self) -> torch.Tensor:
        return self.logits - self.reconstruct_logits()


@dataclass(frozen=True)
class P05FeatureLogitOutput:
    """Features and additive-control logits from one shared model forward."""

    reduced_features: torch.Tensor
    logits: torch.Tensor


class Model(_TSPNModel):
    """UXFD-aligned TSPN with optional module assembly.

    Config (optional):
    ```yaml
    model:
      name: TSPN_UXFD
      type: X_model
      uxfd:
        enable_sp2d: true
        sp2d:
          n_fft: 128
          hop_length: 64
    ```
    """

    def __init__(self, args: Any, metadata: Any = None):
        _validate_num_classes(args)
        super().__init__(args, metadata)
        self._uxfd_enable_sp2d = bool(_get_attr(args, "uxfd.enable_sp2d", False))
        self._uxfd_enable_fuzzy = bool(_get_attr(args, "uxfd.fuzzy.enable", False))
        self._uxfd_enable_operator_attention = bool(
            _get_attr(args, "uxfd.operator_attention.enable", False)
        )
        self._uxfd_enable_logic = bool(_get_attr(args, "uxfd.logic.enable", False))

        self._uxfd_sp2d: Optional[nn.Module] = None
        self._uxfd_2d_proj: Optional[nn.Module] = None
        self._uxfd_fusion: Optional[nn.Module] = None
        self._uxfd_fuzzy: Optional[nn.Module] = None
        self._uxfd_fuzzy_scale: float = 1.0
        self._uxfd_operator_attention: Optional[nn.Module] = None
        self._uxfd_logic: Optional[nn.Module] = None
        self._uxfd_logic_scale: float = 1.0

        if self._uxfd_enable_sp2d:
            cfg = _build_stft_cfg(args)
            if not cfg.magnitude:
                raise ValueError(
                    "TSPN_UXFD SP2D requires uxfd.sp2d.magnitude=true; "
                    "complex real/imag output is not supported by the U1 pooling and fusion contract."
                )
            self._uxfd_sp2d = STFTTimeFrequency(cfg).to(self.args.device)
            self._uxfd_2d_proj = nn.Linear(
                int(self.args.in_channels), int(self.channel_for_classifier)
            ).to(self.args.device)
            fusion_cfg = _build_fusion_cfg(args)
            self._uxfd_fusion = build_fusion(int(self.channel_for_classifier), cfg=fusion_cfg).to(
                self.args.device
            )

        if self._uxfd_enable_fuzzy:
            fuzzy_cfg = _build_fuzzy_cfg(args)
            self._uxfd_fuzzy = FuzzyReasoner(
                dim_in=int(self.channel_for_classifier),
                num_classes=int(self.args.num_classes),
                cfg=fuzzy_cfg,
            ).to(self.args.device)
            self._uxfd_fuzzy_scale = float(getattr(fuzzy_cfg, "logit_scale", 1.0))

        if self._uxfd_enable_operator_attention:
            op_cfg = _build_operator_attention_cfg(args)
            self._uxfd_operator_attention = OperatorAttention1D(
                in_channels=int(getattr(self.args, "in_channels", 1)),
                cfg=op_cfg,
            ).to(self.args.device)

        if self._uxfd_enable_logic:
            logic_cfg = _build_logic_cfg(args)
            self._uxfd_logic = LogicReasoner(
                dim_in=int(self.channel_for_classifier),
                num_classes=int(self.args.num_classes),
                cfg=logic_cfg,
            ).to(self.args.device)
            self._uxfd_logic_scale = float(getattr(logic_cfg, "logit_scale", 1.0))

    def forward(self, x: torch.Tensor, data_id=None, task_id=None) -> torch.Tensor:
        return self.forward_with_features(
            x,
            data_id=data_id,
            task_id=task_id,
        ).logits

    def forward_with_features(
        self,
        x: torch.Tensor,
        data_id=None,
        task_id=None,
    ) -> P05FeatureLogitOutput:
        """Return the shared feature tensor and additive-control logits."""

        features = self._forward_features(x)
        logits = self._forward_logits_from_features(features)
        return P05FeatureLogitOutput(
            reduced_features=features,
            logits=logits,
        )

    def _forward_logits_from_features(self, features: torch.Tensor) -> torch.Tensor:
        logits = self._forward_non_fuzzy_logits(features)
        if self._uxfd_enable_fuzzy:
            assert self._uxfd_fuzzy is not None
            fuzzy_logits = self._uxfd_fuzzy(features)
            logits = logits + self._uxfd_fuzzy_scale * fuzzy_logits
        return logits

    def forward_with_fuzzy_trace(
        self,
        x: torch.Tensor,
        *,
        rule_mask: Optional[torch.Tensor] = None,
        consequent_permutation: Optional[torch.Tensor] = None,
        data_id=None,
        task_id=None,
    ) -> FuzzyTraceOutput:
        """Return the additive-control prediction and exact fuzzy trace."""

        if not self._uxfd_enable_fuzzy or self._uxfd_fuzzy is None:
            raise RuntimeError(
                "forward_with_fuzzy_trace requires model.uxfd.fuzzy.enable=true."
            )
        features = self._forward_features(x)
        non_fuzzy_logits = self._forward_non_fuzzy_logits(features)
        fuzzy_trace = self._uxfd_fuzzy.forward_with_trace(
            features,
            rule_mask=rule_mask,
            consequent_permutation=consequent_permutation,
        )
        logits = non_fuzzy_logits + self._uxfd_fuzzy_scale * fuzzy_trace.fuzzy_logits
        return FuzzyTraceOutput(
            logits=logits,
            non_fuzzy_logits=non_fuzzy_logits,
            fuzzy_scale=float(self._uxfd_fuzzy_scale),
            fuzzy_trace=fuzzy_trace,
        )

    def forward_f0(
        self,
        x: torch.Tensor,
        *,
        rule_to_class: torch.Tensor,
        conflict_threshold: float,
        rule_mask: Optional[torch.Tensor] = None,
        consequent_override: Optional[torch.Tensor] = None,
        data_id=None,
        task_id=None,
    ) -> P05F0Decision:
        """Issue P05 F0 directly from memberships and minimum-t-norm rules.

        The classifier and logic heads are deliberately not called. They remain
        available to the additive control path used by B0/B1-style comparisons.
        """

        if not self._uxfd_enable_fuzzy or self._uxfd_fuzzy is None:
            raise RuntimeError("forward_f0 requires model.uxfd.fuzzy.enable=true.")
        features = self._forward_features(x)
        return self._uxfd_fuzzy.forward_f0(
            features,
            rule_to_class=rule_to_class,
            conflict_threshold=conflict_threshold,
            rule_mask=rule_mask,
            consequent_override=consequent_override,
        )

    def _forward_non_fuzzy_logits(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.clf(features)
        if self._uxfd_enable_logic:
            assert self._uxfd_logic is not None
            logic_logits = self._uxfd_logic(features)
            logits = logits + self._uxfd_logic_scale * logic_logits
        return logits

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        features_1d = self._forward_1d_features(x)
        if not self._uxfd_enable_sp2d:
            return features_1d

        assert self._uxfd_sp2d is not None
        assert self._uxfd_2d_proj is not None
        assert self._uxfd_fusion is not None
        x2d = self._uxfd_sp2d(x)
        pooled = x2d.mean(dim=(1, 2))
        projected = self._uxfd_2d_proj(pooled)
        return self._uxfd_fusion(features_1d, projected)

    def _forward_1d_features(self, x: torch.Tensor) -> torch.Tensor:
        if self._uxfd_enable_operator_attention:
            assert self._uxfd_operator_attention is not None
            x, _weights = self._uxfd_operator_attention(x)
        for layer in self.signal_processing_layers:
            x = layer(x)
        return self.feature_extractor_layers(x)

    def get_uxfd_debug_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "enable_sp2d": bool(self._uxfd_enable_sp2d),
            "enable_fuzzy": bool(self._uxfd_enable_fuzzy),
            "enable_operator_attention": bool(self._uxfd_enable_operator_attention),
            "enable_logic": bool(self._uxfd_enable_logic),
        }

        try:
            if self._uxfd_enable_operator_attention and self._uxfd_operator_attention is not None:
                op_attn = self._uxfd_operator_attention
                weights = getattr(op_attn, "last_attention_weights", None)
                ops = getattr(op_attn, "operators", None)
                if weights is not None and hasattr(weights, "mean"):
                    mean_w = weights.mean(dim=0).detach().cpu().tolist()
                else:
                    mean_w = None
                state["operator_attention"] = {"operators": ops, "attention_mean": mean_w}
        except Exception:
            pass

        return state


def _get_attr(obj: Any, dotted: str, default: Any) -> Any:
    cur = obj
    for part in dotted.split("."):
        if cur is None or not hasattr(cur, part):
            return default
        cur = getattr(cur, part)
    return cur


def _validate_num_classes(args: Any) -> None:
    num_classes = getattr(args, "num_classes", None)
    if isinstance(num_classes, bool) or not isinstance(num_classes, Integral):
        if isinstance(num_classes, dict):
            detail = f"dict keys={list(num_classes.keys())}"
        else:
            detail = type(num_classes).__name__
        raise ValueError(
            "TSPN_UXFD U1 requires args.num_classes to be an integer. "
            f"Got {detail}; dict-valued multi-dataset heads are out of scope for U1."
        )
    if int(num_classes) <= 0:
        raise ValueError(f"TSPN_UXFD requires a positive num_classes, got {num_classes}.")


def _build_stft_cfg(args: Any) -> STFTConfig:
    # Prefer explicit config, otherwise derive a safe default from `in_dim`.
    in_dim = int(getattr(args, "in_dim", 256) or 256)
    default_n_fft = max(16, min(256, in_dim))
    default_hop = max(1, default_n_fft // 2)

    sp2d_cfg = _get_attr(args, "uxfd.sp2d", None)
    if sp2d_cfg is None:
        return STFTConfig(n_fft=default_n_fft, hop_length=default_hop)

    cfg_dict = {}
    if hasattr(sp2d_cfg, "__dict__"):
        cfg_dict = dict(sp2d_cfg.__dict__)
    elif isinstance(sp2d_cfg, dict):
        cfg_dict = dict(sp2d_cfg)

    merged = dict(asdict(STFTConfig(n_fft=default_n_fft, hop_length=default_hop)))
    allowed = set(merged.keys())
    merged.update({k: v for k, v in cfg_dict.items() if k in allowed and v is not None})
    merged["n_fft"] = max(16, min(int(merged["n_fft"]), in_dim))
    merged["hop_length"] = max(1, min(int(merged["hop_length"]), merged["n_fft"]))
    if merged.get("win_length") is not None:
        merged["win_length"] = max(1, min(int(merged["win_length"]), merged["n_fft"]))
    return STFTConfig(**merged)


def _build_fusion_cfg(args: Any) -> FusionConfig:
    fusion_obj = _get_attr(args, "uxfd.fusion", None)
    if fusion_obj is None:
        return FusionConfig()

    fusion_type = None
    if hasattr(fusion_obj, "type"):
        fusion_type = getattr(fusion_obj, "type")
    elif isinstance(fusion_obj, dict):
        fusion_type = fusion_obj.get("type")

    if fusion_type is None:
        return FusionConfig()

    return FusionConfig(fusion_type=str(fusion_type))


def _build_fuzzy_cfg(args: Any) -> FuzzyConfig:
    fuzzy_obj = _get_attr(args, "uxfd.fuzzy", None)
    if fuzzy_obj is None:
        return FuzzyConfig()

    cfg_dict = {}
    if hasattr(fuzzy_obj, "__dict__"):
        cfg_dict = dict(fuzzy_obj.__dict__)
    elif isinstance(fuzzy_obj, dict):
        cfg_dict = dict(fuzzy_obj)

    base = FuzzyConfig()
    allowed = set(base.__dict__.keys())
    merged = {k: v for k, v in cfg_dict.items() if k in allowed and v is not None}
    return FuzzyConfig(**{**base.__dict__, **merged})


def _build_operator_attention_cfg(args: Any) -> OperatorAttentionConfig:
    op_obj = _get_attr(args, "uxfd.operator_attention", None)
    if op_obj is None:
        return OperatorAttentionConfig()

    cfg_dict = {}
    if hasattr(op_obj, "__dict__"):
        cfg_dict = dict(op_obj.__dict__)
    elif isinstance(op_obj, dict):
        cfg_dict = dict(op_obj)

    base = OperatorAttentionConfig()
    allowed = set(base.__dict__.keys())
    merged = {k: v for k, v in cfg_dict.items() if k in allowed and v is not None}
    if isinstance(merged.get("operators"), str):
        merged["operators"] = [s.strip() for s in merged["operators"].split(",") if s.strip()]
    return OperatorAttentionConfig(**{**base.__dict__, **merged})


def _build_logic_cfg(args: Any) -> LogicConfig:
    logic_obj = _get_attr(args, "uxfd.logic", None)
    if logic_obj is None:
        return LogicConfig()

    cfg_dict = {}
    if hasattr(logic_obj, "__dict__"):
        cfg_dict = dict(logic_obj.__dict__)
    elif isinstance(logic_obj, dict):
        cfg_dict = dict(logic_obj)

    base = LogicConfig()
    allowed = set(base.__dict__.keys())
    merged = {k: v for k, v in cfg_dict.items() if k in allowed and v is not None}
    return LogicConfig(**{**base.__dict__, **merged})

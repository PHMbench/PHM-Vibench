"""
PromptLibrary: metadata-conditioned prompt generators for PHM-Vibench.

This module implements the prompt candidate generation logic described in the
updated three-stage specification. It supports three generator families:

1. **Random Dictionary**: Learnable prompt vectors per system (and optionally
   per domain) that are initialised randomly and updated during training.
2. **Physics-only**: Deterministic prompts derived from physical metadata via
   the existing SystemPromptEncoder (no linear mixing).
3. **Hybrid (Physics + Dictionary)**: Learnable combinations that blend the
   physics-derived prompt with system/domain dictionaries using either linear
   addition or a small MLP mixer.

The library returns a batch of prompt candidates for each input sample, along
with intermediate components to support downstream regularisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .SystemPromptEncoder import SystemPromptEncoder


@dataclass
class PromptLibraryOutput:
    """Container for prompt library outputs."""

    prompts: torch.Tensor  # (B, M_p, prompt_dim)
    physics_prompt: Optional[torch.Tensor] = None  # (B, prompt_dim)
    system_prompt: Optional[torch.Tensor] = None  # (B, M_p, prompt_dim)
    domain_prompt: Optional[torch.Tensor] = None  # (B, prompt_dim)


class PromptLibrary(nn.Module):
    """
    Metadata-conditioned prompt library supporting random, physics-only, and hybrid prompts.

    Args:
        prompt_dim: Dimensionality of each prompt candidate.
        num_prompts: Number of prompt candidates per system (``M_p``).
        library_type: One of ``{"random", "physics", "hybrid"}``.
        max_dataset_ids: Maximum dataset index (exclusive) expected in metadata.
        max_domain_ids: Maximum domain index (exclusive) expected in metadata.
        use_domain_prompts: Whether to maintain domain-level vectors ``p_{s,d}``.
        combination: Hybrid mixing strategy ``{"linear", "mlp"}``.
        physics_encoder: Optional custom encoder that maps metadata to physics prompts.
        mlp_hidden_ratio: Hidden width multiplier for the MLP combiner.
        device: Optional device to place parameters on initialisation.
    """

    VALID_TYPES = {"random", "physics", "hybrid"}
    VALID_COMBINATIONS = {"linear", "mlp"}

    def __init__(
        self,
        prompt_dim: int,
        num_prompts: int,
        library_type: str = "random",
        max_dataset_ids: int = 64,
        max_domain_ids: int = 64,
        use_domain_prompts: bool = True,
        combination: str = "linear",
        physics_encoder: Optional[nn.Module] = None,
        mlp_hidden_ratio: float = 2.0,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        if library_type not in self.VALID_TYPES:
            raise ValueError(f"Unsupported library_type '{library_type}'. "
                             f"Expected one of {sorted(self.VALID_TYPES)}")

        if combination not in self.VALID_COMBINATIONS:
            raise ValueError(f"Unsupported combination '{combination}'. "
                             f"Expected one of {sorted(self.VALID_COMBINATIONS)}")

        if num_prompts < 1:
            raise ValueError("num_prompts must be >= 1")

        self.prompt_dim = prompt_dim
        self.num_prompts = num_prompts
        self.library_type = library_type
        self.max_dataset_ids = max_dataset_ids
        self.max_domain_ids = max_domain_ids
        self.use_domain_prompts = use_domain_prompts
        self.combination = combination
        self.device = device

        # Physics encoder (defaults to SystemPromptEncoder if not provided)
        if physics_encoder is None:
            self.physics_encoder = SystemPromptEncoder(
                prompt_dim=prompt_dim,
                max_dataset_ids=max_dataset_ids,
                max_domain_ids=max_domain_ids,
            )
        else:
            self.physics_encoder = physics_encoder

        # Learnable dictionaries for random / hybrid modes.
        if library_type in {"random", "hybrid"}:
            self.system_prompts = nn.Parameter(
                torch.randn(max_dataset_ids, num_prompts, prompt_dim, device=device) * 0.02
            )
        else:
            self.register_parameter("system_prompts", None)

        if library_type in {"random", "hybrid"} and use_domain_prompts:
            self.domain_prompts = nn.Parameter(
                torch.randn(max_dataset_ids, max_domain_ids, prompt_dim, device=device) * 0.02
            )
        else:
            self.register_parameter("domain_prompts", None)

        # Hybrids require an additional combiner.
        if library_type == "hybrid":
            if combination == "linear":
                self.hybrid_linear = nn.Linear(prompt_dim, prompt_dim, bias=False)
            else:
                input_dim = prompt_dim * (2 + int(use_domain_prompts))
                hidden_dim = max(prompt_dim, int(prompt_dim * mlp_hidden_ratio))
                self.hybrid_mlp = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, prompt_dim),
                )
        else:
            self.register_module("hybrid_linear", None)
            self.register_module("hybrid_mlp", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialise learnable parameters."""
        if self.system_prompts is not None:
            nn.init.normal_(self.system_prompts, mean=0.0, std=0.02)
        if self.domain_prompts is not None:
            nn.init.normal_(self.domain_prompts, mean=0.0, std=0.02)
        if getattr(self, "hybrid_linear", None) is not None:
            nn.init.xavier_uniform_(self.hybrid_linear.weight)
        if getattr(self, "hybrid_mlp", None) is not None:
            for module in self.hybrid_mlp:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self, metadata_dict: Dict[str, torch.Tensor]) -> PromptLibraryOutput:
        """
        Generate prompt candidates conditioned on metadata.

        Args:
            metadata_dict: Dictionary containing at least ``Dataset_id`` and ``Domain_id``.

        Returns:
            PromptLibraryOutput containing stacked prompts and auxiliary components.
        """
        dataset_ids, domain_ids = self._extract_ids(metadata_dict)

        physics_prompt = self._compute_physics_prompt(metadata_dict)
        system_prompt = self._gather_system_prompts(dataset_ids)
        domain_prompt = self._gather_domain_prompts(dataset_ids, domain_ids)

        if self.library_type == "physics":
            prompts = physics_prompt.unsqueeze(1)
            if self.num_prompts > 1:
                prompts = prompts.repeat(1, self.num_prompts, 1)
            return PromptLibraryOutput(
                prompts=prompts,
                physics_prompt=physics_prompt,
            )

        if self.library_type == "random":
            prompts = system_prompt
            if prompts is None:
                raise RuntimeError("System prompts are not initialised for random mode.")
            return PromptLibraryOutput(
                prompts=prompts,
                system_prompt=prompts,
                domain_prompt=domain_prompt,
            )

        # Hybrid branch.
        prompts = self._combine_hybrid_prompts(
            physics_prompt=physics_prompt,
            system_prompt=system_prompt,
            domain_prompt=domain_prompt,
        )

        return PromptLibraryOutput(
            prompts=prompts,
            physics_prompt=physics_prompt,
            system_prompt=system_prompt,
            domain_prompt=domain_prompt,
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _extract_ids(self, metadata_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract dataset and domain identifiers."""
        if "Dataset_id" not in metadata_dict or "Domain_id" not in metadata_dict:
            raise KeyError("metadata_dict must contain 'Dataset_id' and 'Domain_id'")

        dataset_ids = metadata_dict["Dataset_id"].long()
        domain_ids = metadata_dict["Domain_id"].long()

        if dataset_ids.min() < 0 or dataset_ids.max() >= self.max_dataset_ids:
            raise ValueError(
                f"'Dataset_id' out of bounds [0, {self.max_dataset_ids})."
            )
        if domain_ids.min() < 0 or domain_ids.max() >= self.max_domain_ids:
            raise ValueError(
                f"'Domain_id' out of bounds [0, {self.max_domain_ids})."
            )
        return dataset_ids, domain_ids

    def _compute_physics_prompt(self, metadata_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute physics-derived prompt via the configured encoder."""
        physics_prompt = self.physics_encoder(metadata_dict)
        if physics_prompt.shape[-1] != self.prompt_dim:
            raise ValueError(
                f"Physics encoder returned dim {physics_prompt.shape[-1]} != prompt_dim {self.prompt_dim}"
            )
        return physics_prompt

    def _gather_system_prompts(self, dataset_ids: torch.Tensor) -> Optional[torch.Tensor]:
        """Gather system-level dictionary prompts."""
        if self.system_prompts is None:
            return None
        prompts = self.system_prompts[dataset_ids]  # (B, M_p, prompt_dim)
        return prompts

    def _gather_domain_prompts(
        self,
        dataset_ids: torch.Tensor,
        domain_ids: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Gather domain-specific prompt vectors."""
        if self.domain_prompts is None:
            return None
        domain_vectors = self.domain_prompts[dataset_ids, domain_ids]  # (B, prompt_dim)
        return domain_vectors

    def _combine_hybrid_prompts(
        self,
        physics_prompt: torch.Tensor,
        system_prompt: Optional[torch.Tensor],
        domain_prompt: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Blend physics and dictionary prompts according to the configured strategy."""
        if system_prompt is None:
            raise RuntimeError("Hybrid mode requires system prompts.")

        if self.combination == "linear":
            base = self.hybrid_linear(physics_prompt)  # (B, prompt_dim)
            base = base.unsqueeze(1)  # (B, 1, prompt_dim)
            combined = base + system_prompt  # broadcast to (B, M_p, prompt_dim)
            if domain_prompt is not None:
                combined = combined + domain_prompt.unsqueeze(1)
            return combined

        # MLP mixing
        inputs = [physics_prompt.unsqueeze(1).expand_as(system_prompt), system_prompt]
        if domain_prompt is not None:
            inputs.append(domain_prompt.unsqueeze(1).expand_as(system_prompt))
        stacked = torch.cat(inputs, dim=-1)  # (B, M_p, prompt_dim * (#components))
        combined = self.hybrid_mlp(stacked)
        return combined


def build_prompt_library(
    prompt_dim: int,
    num_prompts: int,
    library_type: str,
    max_dataset_ids: int,
    max_domain_ids: int,
    use_domain_prompts: bool,
    combination: str,
    physics_encoder: Optional[nn.Module] = None,
    mlp_hidden_ratio: float = 2.0,
    device: Optional[torch.device] = None,
) -> PromptLibrary:
    """
    Convenience builder for PromptLibrary with defaults mirroring config fields.
    """
    return PromptLibrary(
        prompt_dim=prompt_dim,
        num_prompts=num_prompts,
        library_type=library_type,
        max_dataset_ids=max_dataset_ids,
        max_domain_ids=max_domain_ids,
        use_domain_prompts=use_domain_prompts,
        combination=combination,
        physics_encoder=physics_encoder,
        mlp_hidden_ratio=mlp_hidden_ratio,
        device=device,
    )


__all__ = ["PromptLibrary", "PromptLibraryOutput", "build_prompt_library"]


"""
PromptInjector: Apply prompt candidates to token matrices `U`.

The injector supports the three injection families described in the updated
specification:

- **Prefix Tokens**: Replicate prompt vectors as learnable prefix tokens and
  concatenate them with the original tokens.
- **Additive / FiLM Modulation**: Broadcast prompt vectors across the token
  dimension and apply additive or affine modulation.
- **Feature Concatenation**: Concatenate prompt features along the channel
  dimension (optionally projecting back to the original width).
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn


InjectionMode = Literal["prefix", "add", "film", "concat"]


class PromptInjector(nn.Module):
    """
    Inject prompt candidates into token matrices.

    Args:
        token_dim: Dimensionality of the base token embeddings (``d_out``).
        prompt_dim: Dimensionality of prompt vectors.
        mode: Injection strategy ({``prefix``, ``add``, ``film``, ``concat``}).
        prefix_length: Number of prefix tokens to prepend for ``prefix`` mode.
        project_concat: Whether to project concatenated features back to
            ``token_dim`` to preserve backbone compatibility.
        eps: Numerical epsilon for stability when computing affine parameters.
    """

    def __init__(
        self,
        token_dim: int,
        prompt_dim: int,
        mode: InjectionMode = "add",
        prefix_length: int = 2,
        project_concat: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.prompt_dim = prompt_dim
        self.mode = mode
        self.prefix_length = prefix_length
        self.project_concat = project_concat
        self.eps = eps

        # Shared projections
        self.prompt_to_token = nn.Linear(prompt_dim, token_dim)

        if mode == "film":
            self.gamma_proj = nn.Linear(prompt_dim, token_dim)
            self.beta_proj = nn.Linear(prompt_dim, token_dim)
        else:
            self.register_module("gamma_proj", None)
            self.register_module("beta_proj", None)

        if mode == "concat" and project_concat:
            self.concat_proj = nn.Linear(token_dim + prompt_dim, token_dim)
        else:
            self.register_module("concat_proj", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.prompt_to_token.weight)
        if self.prompt_to_token.bias is not None:
            nn.init.zeros_(self.prompt_to_token.bias)

        if self.gamma_proj is not None:
            nn.init.xavier_uniform_(self.gamma_proj.weight)
            nn.init.zeros_(self.gamma_proj.bias)
        if self.beta_proj is not None:
            nn.init.xavier_uniform_(self.beta_proj.weight)
            nn.init.zeros_(self.beta_proj.bias)

        if self.concat_proj is not None:
            nn.init.xavier_uniform_(self.concat_proj.weight)
            nn.init.zeros_(self.concat_proj.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        prompt_candidates: torch.Tensor,
    ) -> Tuple[torch.Tensor, int, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Inject prompts into token matrices.

        Args:
            tokens: Base token tensor of shape ``(B, N_p, token_dim)``.
            prompt_candidates: Prompt tensor of shape ``(B, M_p, prompt_dim)``.

        Returns:
            injected_tokens: Tensor of shape ``(B, M_p, N', D')`` containing
                prompt-conditioned tokens for each candidate.
            effective_dim: Output feature dimension ``D'`` (useful for sanity checks).
            attention_mask: Optional attention mask for prefix/concat modes.
            position_ids: Optional position IDs for prefix/concat modes.
        """
        if tokens.dim() != 3:
            raise ValueError("tokens must have shape (B, N_p, token_dim)")
        if prompt_candidates.dim() != 3:
            raise ValueError("prompt_candidates must have shape (B, M_p, prompt_dim)")
        if tokens.size(0) != prompt_candidates.size(0):
            raise ValueError("Batch size mismatch between tokens and prompt_candidates")

        mode = self.mode
        if mode not in {"prefix", "add", "film", "concat"}:
            raise ValueError(f"Unsupported injection mode '{mode}'")

        base = tokens.unsqueeze(1).expand(-1, prompt_candidates.size(1), -1, -1)

        if mode == "prefix":
            return self._inject_prefix(base, prompt_candidates)
        if mode == "add":
            injected, dim = self._inject_add(base, prompt_candidates)
            return injected, dim, None, None
        if mode == "film":
            injected, dim = self._inject_film(base, prompt_candidates)
            return injected, dim, None, None
        if mode == "concat":
            return self._inject_concat(base, prompt_candidates)

        raise RuntimeError("Unhandled injection mode")

    # ------------------------------------------------------------------
    # Injection strategies
    # ------------------------------------------------------------------
    def _inject_prefix(
        self,
        base_tokens: torch.Tensor,
        prompts: torch.Tensor,
    ) -> Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
        """
        Replicate prompt vectors as prefix tokens.
        """
        B, M, N_p, token_dim = base_tokens.shape
        prefix = self.prompt_to_token(prompts)  # (B, M, token_dim)
        prefix = prefix.unsqueeze(2).repeat(1, 1, max(self.prefix_length, 1), 1)
        injected = torch.cat([prefix, base_tokens], dim=2)

        # Create attention mask for prefix tokens (1 for visible tokens)
        N_prime = injected.size(2)
        attention_mask = torch.ones(B, M, N_prime, dtype=torch.long, device=base_tokens.device)

        # Create position IDs for proper positioning
        position_ids = torch.arange(N_prime, device=base_tokens.device).view(1, 1, -1).expand(B, M, -1)

        return injected, injected.size(-1), attention_mask, position_ids

    def _inject_add(
        self,
        base_tokens: torch.Tensor,
        prompts: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """
        Add broadcasted prompt vectors to tokens.
        """
        prompt_proj = self.prompt_to_token(prompts)  # (B, M, token_dim)
        prompt_proj = prompt_proj.unsqueeze(2)
        injected = base_tokens + prompt_proj
        return injected, injected.size(-1)

    def _inject_film(
        self,
        base_tokens: torch.Tensor,
        prompts: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """
        Apply FiLM-style affine modulation to tokens.
        """
        if self.gamma_proj is None or self.beta_proj is None:
            raise RuntimeError("FiLM projections not initialised.")

        gamma = self.gamma_proj(prompts).unsqueeze(2)
        beta = self.beta_proj(prompts).unsqueeze(2)
        injected = (1.0 + gamma) * base_tokens + beta
        return injected, injected.size(-1)

    def _inject_concat(
        self,
        base_tokens: torch.Tensor,
        prompts: torch.Tensor,
    ) -> Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
        """
        Concatenate prompt features along the channel dimension.
        """
        prompt_expanded = prompts.unsqueeze(2).expand(
            -1, -1, base_tokens.size(2), -1
        )
        concatenated = torch.cat([base_tokens, prompt_expanded], dim=-1)
        if self.concat_proj is not None:
            injected = self.concat_proj(concatenated)
        else:
            injected = concatenated

        # For concat mode, we also need attention_mask and position_ids
        B, M, N_p, _ = injected.shape
        attention_mask = torch.ones(B, M, N_p, dtype=torch.long, device=base_tokens.device)
        position_ids = torch.arange(N_p, device=base_tokens.device).view(1, 1, -1).expand(B, M, -1)

        return injected, injected.size(-1), attention_mask, position_ids


__all__ = ["PromptInjector", "InjectionMode"]


"""Physical-duration and Nyquist-aware heterogeneous signal embedding.

The P08 method deliberately excludes dataset identity.  A signal is represented
by fixed-duration patches, an explicit shared/private spectral split, and a
continuous prompt derived from measurable acquisition quantities only.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model_factory.ISFM.system_utils import normalize_fs


class HSE_prompt(nn.Module):
    """Embed heterogeneous signals without dataset-identity features.

    Parameters are read from the model config.  ``physical_patch_duration_s``
    fixes the physical support of every token, while
    ``physical_patch_points`` fixes the neural input size after interpolation.
    ``shared_band_hz`` is a source-derived cutoff supplied by the fold protocol.
    """

    _PROMPT_FEATURES = (
        "log_sampling_rate",
        "log_window_duration",
        "log_nyquist",
        "observable_shared_fraction",
    )

    def __init__(self, args):
        super().__init__()

        self.physical_patch_duration_s = float(
            getattr(args, "physical_patch_duration_s", 0.01)
        )
        self.physical_patch_points = int(
            getattr(args, "physical_patch_points", getattr(args, "patch_size_L", 256))
        )
        self.use_physical_duration = bool(
            getattr(args, "use_physical_duration", True)
        )
        self.fixed_raw_token_points = int(
            getattr(args, "fixed_raw_token_points", self.physical_patch_points)
        )
        # Preserve the historical attribute used by model introspection.
        self.patch_size_L = self.physical_patch_points
        self.patch_size_C = int(getattr(args, "patch_size_C", 1))
        self.num_patches = int(getattr(args, "num_patches", 64))
        self.output_dim = int(getattr(args, "output_dim", 128))

        self.shared_band_hz = float(getattr(args, "shared_band_hz", 5000.0))
        self.use_band_projection = bool(
            getattr(args, "use_band_projection", True)
        )
        self.prompt_reference_fs_hz = float(
            getattr(args, "prompt_reference_fs_hz", 10000.0)
        )
        self.prompt_reference_duration_s = float(
            getattr(
                args,
                "prompt_reference_duration_s",
                self.physical_patch_duration_s,
            )
        )

        self.use_prompt = bool(getattr(args, "use_prompt", True))
        self.prompt_dim = int(getattr(args, "prompt_dim", 64))
        self.prompt_combination = str(getattr(args, "prompt_combination", "add"))
        self.freeze_prompts_in_finetuning = bool(
            getattr(args, "freeze_prompts_in_finetuning", False)
        )

        self._validate_config()

        patch_input_dim = self.physical_patch_points * (self.patch_size_C + 1)
        self.patch_encoder = nn.Sequential(
            nn.Linear(patch_input_dim, self.output_dim),
            nn.SiLU(),
            nn.Linear(self.output_dim, self.output_dim),
        )
        self.band_encoder = nn.Sequential(
            nn.Linear(2, self.output_dim),
            nn.SiLU(),
            nn.Linear(self.output_dim, self.output_dim),
        )

        if self.use_prompt:
            self.prompt_encoder = nn.Sequential(
                nn.Linear(len(self._PROMPT_FEATURES), self.prompt_dim),
                nn.SiLU(),
                nn.LayerNorm(self.prompt_dim),
                nn.Linear(self.prompt_dim, self.prompt_dim),
            )
            if self.prompt_combination == "add":
                self.prompt_proj = (
                    nn.Identity()
                    if self.prompt_dim == self.output_dim
                    else nn.Linear(self.prompt_dim, self.output_dim)
                )
            else:
                self.concat_proj = nn.Linear(
                    self.output_dim + self.prompt_dim, self.output_dim
                )

        self.final_norm = nn.LayerNorm(self.output_dim)
        self.dropout = nn.Dropout(float(getattr(args, "dropout", 0.1)))

        # Diagnostics are detached snapshots for tests and run manifests.  They
        # are not consumed by the forward computation.
        self.last_raw_patch_points: Optional[torch.Tensor] = None
        self.last_patch_starts: Optional[torch.Tensor] = None
        self.last_band_fractions: Optional[torch.Tensor] = None
        self.last_prompt_features: Optional[torch.Tensor] = None

        self._init_parameters()

    def _validate_config(self) -> None:
        if self.physical_patch_duration_s <= 0:
            raise ValueError("physical_patch_duration_s must be positive")
        if self.physical_patch_points < 2:
            raise ValueError("physical_patch_points must be at least 2")
        if self.fixed_raw_token_points < 2:
            raise ValueError("fixed_raw_token_points must be at least 2")
        if self.patch_size_C < 1 or self.num_patches < 1 or self.output_dim < 1:
            raise ValueError("patch_size_C, num_patches, and output_dim must be positive")
        if self.shared_band_hz <= 0:
            raise ValueError("shared_band_hz must be positive")
        if self.prompt_reference_fs_hz <= 0 or self.prompt_reference_duration_s <= 0:
            raise ValueError("continuous-prompt reference scales must be positive")
        if self.prompt_combination not in {"add", "concat"}:
            raise ValueError("prompt_combination must be 'add' or 'concat'")
        if self.use_prompt and self.prompt_dim < 1:
            raise ValueError("prompt_dim must be positive when use_prompt is true")

    def _init_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def set_training_stage(self, stage: str) -> None:
        stage = stage.lower()
        if stage in {"pretraining", "pretrain"}:
            stage = "pretrain"
        elif stage in {"finetuning", "finetune"}:
            stage = "finetune"
        else:
            raise ValueError(f"unsupported training stage: {stage}")

        self.training_stage = stage
        if self.use_prompt:
            requires_grad = not (
                stage == "finetune" and self.freeze_prompts_in_finetuning
            )
            for parameter in self.prompt_encoder.parameters():
                parameter.requires_grad = requires_grad

    def forward(
        self,
        x: torch.Tensor,
        fs: Union[torch.Tensor, float],
        dataset_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return ``[batch, num_patches, output_dim]`` embeddings.

        ``dataset_ids`` remains in the signature only to detect legacy callers.
        Passing it is an error, which prevents silent reintroduction of the
        forbidden dataset-ID prompt.
        """
        if dataset_ids is not None:
            raise ValueError(
                "dataset_ids are forbidden inputs for the P08 continuous-metadata prompt"
            )
        if x.ndim != 3:
            raise ValueError(f"x must have shape [B, L, C], got {tuple(x.shape)}")
        if not torch.is_floating_point(x):
            raise TypeError("x must be a floating-point tensor")

        batch_size, signal_length, _ = x.shape
        fs_tensor = normalize_fs(
            fs, batch_size=batch_size, device=x.device, as_column=False
        ).to(dtype=x.dtype)
        if not torch.isfinite(fs_tensor).all() or torch.any(fs_tensor <= 0):
            raise ValueError("all sampling rates must be finite and positive")

        signal_embeddings, band_fractions, raw_points = self._process_signal_patches(
            x, fs_tensor
        )
        if self.use_band_projection:
            signal_embeddings = signal_embeddings + self.band_encoder(band_fractions)

        prompt_features = self._continuous_prompt_features(
            fs_tensor, signal_length=signal_length
        )
        if self.use_prompt:
            prompt_vectors = self.prompt_encoder(prompt_features)
            if self.prompt_combination == "add":
                signal_embeddings = signal_embeddings + self.prompt_proj(
                    prompt_vectors
                ).unsqueeze(1)
            else:
                expanded = prompt_vectors.unsqueeze(1).expand(
                    -1, self.num_patches, -1
                )
                signal_embeddings = self.concat_proj(
                    torch.cat((signal_embeddings, expanded), dim=-1)
                )

        output = self.dropout(self.final_norm(signal_embeddings))
        if not torch.isfinite(output).all():
            raise FloatingPointError("HSE_prompt produced non-finite embeddings")

        self.last_raw_patch_points = raw_points.detach()
        self.last_band_fractions = band_fractions.detach()
        self.last_prompt_features = prompt_features.detach()
        return output

    def _continuous_prompt_features(
        self, fs_tensor: torch.Tensor, signal_length: int
    ) -> torch.Tensor:
        eps = torch.finfo(fs_tensor.dtype).eps
        nyquist_hz = fs_tensor / 2.0
        window_duration_s = signal_length / fs_tensor
        shared_hz = torch.minimum(
            nyquist_hz,
            torch.full_like(nyquist_hz, self.shared_band_hz),
        )
        features = torch.stack(
            (
                torch.log(fs_tensor.clamp_min(eps) / self.prompt_reference_fs_hz),
                torch.log(
                    window_duration_s.clamp_min(eps)
                    / self.prompt_reference_duration_s
                ),
                torch.log(nyquist_hz.clamp_min(eps) / self.shared_band_hz),
                shared_hz / nyquist_hz.clamp_min(eps),
            ),
            dim=-1,
        )
        if not torch.isfinite(features).all():
            raise FloatingPointError("continuous prompt features are non-finite")
        return features

    def _select_channels(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[-1]
        if channels >= self.patch_size_C:
            return x[..., : self.patch_size_C]
        repeats = (self.patch_size_C + channels - 1) // channels
        return x.repeat(1, 1, repeats)[..., : self.patch_size_C]

    def _process_signal_patches(
        self, x: torch.Tensor, fs_tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, signal_length, _ = x.shape
        selected = self._select_channels(x)
        token_rows = []
        band_rows = []
        raw_point_counts = []
        start_rows = []

        if self.use_physical_duration:
            time_seconds = torch.linspace(
                0.0,
                self.physical_patch_duration_s,
                self.physical_patch_points,
                device=x.device,
                dtype=x.dtype,
            )
            normalized_time = time_seconds / self.prompt_reference_duration_s
        else:
            # Fixed-point controls must not receive a hidden physical-duration
            # coordinate.  Keeping a zero column preserves the patch encoder
            # shape without leaking sampling-rate information.
            normalized_time = torch.zeros(
                self.physical_patch_points, device=x.device, dtype=x.dtype
            )

        if torch.all(fs_tensor == fs_tensor[0]):
            return self._process_uniform_rate_patches(
                selected=selected,
                fs_value=float(fs_tensor[0].detach().cpu().item()),
                normalized_time=normalized_time,
            )

        for batch_index in range(batch_size):
            fs_value = float(fs_tensor[batch_index].detach().cpu().item())
            if self.use_physical_duration:
                raw_points = max(
                    2,
                    int(
                        math.floor(
                            self.physical_patch_duration_s * fs_value + 0.5
                        )
                    ),
                )
            else:
                raw_points = self.fixed_raw_token_points
            if raw_points > signal_length:
                raise ValueError(
                    "physical patch exceeds the available signal window: "
                    f"duration={self.physical_patch_duration_s}s, fs={fs_value}Hz, "
                    f"requires={raw_points}, available={signal_length}"
                )
            raw_point_counts.append(raw_points)

            max_start = signal_length - raw_points
            if self.num_patches == 1:
                starts = torch.tensor(
                    [max_start // 2], dtype=torch.long, device=x.device
                )
            else:
                available_starts = max_start + 1
                if self.num_patches > available_starts:
                    raise ValueError(
                        "unique physical-patch starts are impossible: "
                        f"requested={self.num_patches}, available={available_starts}, "
                        f"raw_points={raw_points}, signal_length={signal_length}"
                    )
                indices = torch.arange(
                    self.num_patches, dtype=torch.long, device=x.device
                )
                denominator = 2 * (self.num_patches - 1)
                starts = (
                    2 * indices * max_start + (self.num_patches - 1)
                ) // denominator

            if torch.unique(starts).numel() != self.num_patches:
                raise RuntimeError("physical-patch start construction is not unique")
            start_rows.append(starts)

            sample_tokens = []
            sample_bands = []
            for start in starts.tolist():
                raw_patch = selected[
                    batch_index, start : start + raw_points, :
                ]
                resampled = F.interpolate(
                    raw_patch.transpose(0, 1).unsqueeze(0),
                    size=self.physical_patch_points,
                    mode="linear",
                    align_corners=False,
                ).squeeze(0).transpose(0, 1)
                time_column = normalized_time.unsqueeze(-1)
                patch_with_time = torch.cat((resampled, time_column), dim=-1)
                sample_tokens.append(patch_with_time.flatten())
                if self.use_band_projection:
                    sample_bands.append(
                        self._band_fractions(raw_patch, fs_hz=fs_value)
                    )
                else:
                    sample_bands.append(
                        torch.zeros(2, device=x.device, dtype=x.dtype)
                    )

            token_rows.append(torch.stack(sample_tokens, dim=0))
            band_rows.append(torch.stack(sample_bands, dim=0))

        flat_tokens = torch.stack(token_rows, dim=0)
        band_fractions = torch.stack(band_rows, dim=0)
        embeddings = self.patch_encoder(flat_tokens)
        raw_points_tensor = torch.tensor(
            raw_point_counts, dtype=torch.long, device=x.device
        )
        self.last_patch_starts = torch.stack(start_rows, dim=0).detach()
        return embeddings, band_fractions, raw_points_tensor

    def _process_uniform_rate_patches(
        self,
        selected: torch.Tensor,
        fs_value: float,
        normalized_time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorized evidence path for an exact-rate bucket."""
        batch_size, signal_length, _ = selected.shape
        if self.use_physical_duration:
            raw_points = max(
                2,
                int(
                    math.floor(
                        self.physical_patch_duration_s * fs_value + 0.5
                    )
                ),
            )
        else:
            raw_points = self.fixed_raw_token_points
        if raw_points > signal_length:
            raise ValueError(
                "physical patch exceeds the available signal window: "
                f"duration={self.physical_patch_duration_s}s, fs={fs_value}Hz, "
                f"requires={raw_points}, available={signal_length}"
            )

        max_start = signal_length - raw_points
        if self.num_patches == 1:
            starts = torch.tensor(
                [max_start // 2], dtype=torch.long, device=selected.device
            )
        else:
            available_starts = max_start + 1
            if self.num_patches > available_starts:
                raise ValueError(
                    "unique physical-patch starts are impossible: "
                    f"requested={self.num_patches}, available={available_starts}, "
                    f"raw_points={raw_points}, signal_length={signal_length}"
                )
            indices = torch.arange(
                self.num_patches, dtype=torch.long, device=selected.device
            )
            denominator = 2 * (self.num_patches - 1)
            starts = (
                2 * indices * max_start + (self.num_patches - 1)
            ) // denominator
        if torch.unique(starts).numel() != self.num_patches:
            raise RuntimeError("physical-patch start construction is not unique")

        offsets = torch.arange(raw_points, device=selected.device)
        gather_indices = starts.unsqueeze(1) + offsets.unsqueeze(0)
        raw_patches = selected[:, gather_indices, :]
        flat_patches = raw_patches.permute(0, 1, 3, 2).reshape(
            batch_size * self.num_patches, self.patch_size_C, raw_points
        )
        resampled = F.interpolate(
            flat_patches,
            size=self.physical_patch_points,
            mode="linear",
            align_corners=False,
        ).reshape(
            batch_size,
            self.num_patches,
            self.patch_size_C,
            self.physical_patch_points,
        ).permute(0, 1, 3, 2)
        time_column = normalized_time.view(1, 1, -1, 1).expand(
            batch_size, self.num_patches, -1, -1
        )
        flat_tokens = torch.cat((resampled, time_column), dim=-1).flatten(2)

        if self.use_band_projection:
            spectrum = torch.fft.rfft(raw_patches, dim=2)
            power = spectrum.abs().square().mean(dim=-1)
            frequencies = torch.fft.rfftfreq(
                raw_points, d=1.0 / fs_value, device=selected.device
            ).to(dtype=selected.dtype)
            shared_mask = frequencies <= min(self.shared_band_hz, fs_value / 2.0)
            eps = torch.finfo(power.dtype).eps
            total = power.sum(dim=-1).clamp_min(eps)
            shared = power[..., shared_mask].sum(dim=-1) / total
            private = power[..., ~shared_mask].sum(dim=-1) / total
            band_fractions = torch.stack((shared, private), dim=-1)
        else:
            band_fractions = torch.zeros(
                batch_size,
                self.num_patches,
                2,
                device=selected.device,
                dtype=selected.dtype,
            )

        self.last_patch_starts = starts.unsqueeze(0).expand(
            batch_size, -1
        ).detach()
        raw_points_tensor = torch.full(
            (batch_size,), raw_points, dtype=torch.long, device=selected.device
        )
        return self.patch_encoder(flat_tokens), band_fractions, raw_points_tensor

    def _band_fractions(self, raw_patch: torch.Tensor, fs_hz: float) -> torch.Tensor:
        spectrum = torch.fft.rfft(raw_patch, dim=0)
        power = spectrum.abs().square().mean(dim=-1)
        frequencies = torch.fft.rfftfreq(
            raw_patch.shape[0], d=1.0 / fs_hz, device=raw_patch.device
        ).to(dtype=raw_patch.dtype)
        observable_shared_hz = min(self.shared_band_hz, fs_hz / 2.0)
        shared_mask = frequencies <= observable_shared_hz
        private_mask = ~shared_mask
        eps = torch.finfo(power.dtype).eps
        total_power = power.sum().clamp_min(eps)
        shared_fraction = power[shared_mask].sum() / total_power
        private_fraction = power[private_mask].sum() / total_power
        return torch.stack((shared_fraction, private_fraction))

    def get_model_info(self) -> dict:
        info = {
            "model_type": "HSE_prompt_physical_duration_nyquist",
            "physical_patch_duration_s": self.physical_patch_duration_s,
            "physical_patch_points": self.physical_patch_points,
            "use_physical_duration": self.use_physical_duration,
            "fixed_raw_token_points": self.fixed_raw_token_points,
            "patch_size_C": self.patch_size_C,
            "num_patches": self.num_patches,
            "output_dim": self.output_dim,
            "shared_band_hz": self.shared_band_hz,
            "shared_band_policy": "source_fold_min_nyquist",
            "use_band_projection": self.use_band_projection,
            "use_prompt": self.use_prompt,
            "prompt_features": list(self._PROMPT_FEATURES),
            "dataset_identity_consumed": False,
            "total_parameters": sum(p.numel() for p in self.parameters()),
        }
        if self.use_prompt:
            info.update(
                {
                    "prompt_dim": self.prompt_dim,
                    "prompt_combination": self.prompt_combination,
                    "prompt_parameters": sum(
                        p.numel() for p in self.prompt_encoder.parameters()
                    ),
                }
            )
        return info
